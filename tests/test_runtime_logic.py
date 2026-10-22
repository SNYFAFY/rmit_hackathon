"""Offline CPU regression tests using randomly initialized tiny models only."""
import ast
import copy
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import f1_score
from transformers import Trainer, TrainerCallback, TrainingArguments, XLMRobertaConfig, XLMRobertaForSequenceClassification

from btc_worker import BtcTrainer, LABELS, checked_logits, metrics, prediction_frame, validate_model_source
from scripts.analyze_validation import confidence_metrics, validate_rows

torch.set_num_threads(2)


def tiny(dropout=0.0):
    torch.manual_seed(42)
    return XLMRobertaForSequenceClassification(XLMRobertaConfig(
        vocab_size=80, hidden_size=24, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, max_position_embeddings=64, num_labels=3,
        hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
        classifier_dropout=dropout, id2label=dict(enumerate(LABELS)),
        label2id=dict(zip(LABELS, range(3))), pad_token_id=1,
    ))


def dataset():
    return Dataset.from_dict({'input_ids': [[0, 5 + i, 10, 2] for i in range(6)],
                              'attention_mask': [[1] * 4 for _ in range(6)], 'labels': [0, 2, 1, 1, 0, 2]})


def notebook_definitions():
    """Extract only definitions; never execute notebook cells or mutate notebook."""
    nb = json.loads((ROOT / 'Training.ipynb').read_text(encoding='utf-8'))
    wanted = {'SecurityTrainer', 'IndexedPromptDataset', 'compute_metrics', 'score_macro'}
    nodes = []
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            for node in ast.parse(''.join(c['source'])).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
                    nodes.append(node)
    namespace = dict(Trainer=Trainer, Dataset=torch.utils.data.Dataset, np=np, torch=torch, F=F,
                     f1_score=f1_score, re=re, LOSS_MODE='ce', LABEL_SMOOTHING=0.02,
                     LEARNING_RATE=2e-5, LAYERWISE_LR_DECAY=0.92,
                     CLASSIFIER_LR_MULTIPLIER=1.3, WEIGHT_DECAY=0.02)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'Training.ipynb:definitions_only', 'exec'), namespace)
    return namespace


class GradCheck(TrainerCallback):
    def __init__(self):
        self.steps = 0
        self.nonzero_lora = False
        self.nonzero_head = False

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self.steps += 1
        for name, p in model.named_parameters():
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise AssertionError(f'Nonfinite gradient: {name}')
                if p.grad.abs().sum() > 0:
                    self.nonzero_lora |= 'lora_' in name
                    self.nonzero_head |= 'classifier' in name


class RuntimeTests(unittest.TestCase):
    def test_source_and_mapping_guards(self):
        cfg = dict(training_data='BTC_ONLY', model_id='FacebookAI/xlm-roberta-large',
                   model_path='FacebookAI/xlm-roberta-large', labels=LABELS, precision='bf16')
        validate_model_source(cfg)
        for key, value in [('model_path', 'external_adapter'), ('model_id', 'FacebookAI/xlm-roberta-base'),
                           ('labels', LABELS[::-1]), ('training_data', 'external'), ('precision', 'oops')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_model_source({**cfg, key: value})

    def test_submission_ids_and_logits(self):
        ids = pd.Series(['0009', '0002', 'NA'])
        sample = pd.DataFrame({'id': ids})
        logits = np.array([[3, 1, 0], [0, 1, 3], [0, 3, 1]], dtype=float)
        frame = prediction_frame(ids, sample, logits)
        self.assertEqual(frame.id.tolist(), ids.tolist())
        self.assertEqual(frame.label.tolist(), [LABELS[0], LABELS[2], LABELS[1]])
        for invalid in (sample.iloc[::-1], sample.iloc[:-1], pd.DataFrame({'id': ['0009'] * 3})):
            with self.assertRaises(ValueError):
                prediction_frame(ids, invalid, logits)
        for invalid in (logits[:, :2], logits[:2], np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                checked_logits(invalid, 3)
        np.testing.assert_equal(checked_logits((logits,), 3), logits)

    def test_metrics_and_confidence(self):
        y = np.array([0, 1, 2, 2])
        logits = np.array([[8, 0, 0], [0, 8, 0], [8, 0, 0], [0, 0, 8]])
        expected = f1_score(y, logits.argmax(1), labels=[0, 1, 2], average='macro')
        self.assertAlmostEqual(metrics(SimpleNamespace(predictions=(logits,), label_ids=y))['macro_f1'], expected)
        result, probs = confidence_metrics(y, logits)
        self.assertAlmostEqual(result['macro_f1'], expected)
        np.testing.assert_allclose(probs.sum(1), 1)
        self.assertEqual(result['incorrect_confidence']['count_ge_0_99'], 1)
        uniform, _ = confidence_metrics(np.array([0, 1, 2]), np.zeros((3, 3)))
        self.assertAlmostEqual(uniform['nll'], math.log(3))
        self.assertAlmostEqual(uniform['multiclass_brier'], 2 / 3)
        self.assertAlmostEqual(uniform['ece'], 0)
        shifted, _ = confidence_metrics(y, logits + 10000)
        self.assertAlmostEqual(shifted['nll'], result['nll'])
        extreme, _ = confidence_metrics(np.array([2]), np.array([[10000, 0, -10000]]))
        self.assertEqual(extreme['nll'], 20000)
        for bad in (np.full((4, 3), np.inf), logits[:, :2]):
            with self.assertRaises(ValueError):
                confidence_metrics(y, bad)

    def test_validation_split_identity(self):
        official = pd.DataFrame({'id': ['a', 'b', 'c'], 'label': LABELS,
                                 'language': ['en', 'fr', 'pt'], 'text': ['A', 'B', 'C']})
        rows, train_ids = official.iloc[[2, 0]].copy(), official.iloc[[1]][['id']].copy()
        validate_rows(rows, train_ids, official)
        with self.assertRaises(ValueError):
            validate_rows(rows, official.iloc[[0]][['id']], official)
        wrong = rows.copy()
        wrong['label'] = wrong.label.iloc[::-1].to_numpy()
        with self.assertRaises(ValueError):
            validate_rows(wrong, train_ids, official)

    def test_lora_gradients_frozen_backbone_save_reload_eval(self):
        base = tiny(dropout=0.2)
        initial = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=2, lora_alpha=4,
            lora_dropout=0.05, target_modules=['query', 'value'], modules_to_save=['classifier'], bias='none'))
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all('lora_' in n or 'modules_to_save' in n for n in before))
        callback = GradCheck()
        with tempfile.TemporaryDirectory() as temp:
            args = TrainingArguments(output_dir=temp, use_cpu=True, max_steps=2,
                per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=2,
                gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                learning_rate=1e-3, save_strategy='no', report_to='none', disable_tqdm=True)
            trainer = BtcTrainer(model=model, args=args, train_dataset=dataset(), callbacks=[callback],
                                 class_weights=torch.tensor([0.5, 1., 2.]))
            result = trainer.train()
            self.assertTrue(np.isfinite(result.training_loss))
            self.assertEqual(trainer.state.global_step, 2)
            self.assertEqual(callback.steps, 2)
            self.assertTrue(callback.nonzero_lora and callback.nonzero_head)
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before))
            self.assertTrue(all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen))
            states = []
            handle = model.register_forward_pre_hook(lambda m, _: states.append((m.training, torch.is_grad_enabled())))
            first = trainer.predict(dataset()).predictions
            second = trainer.predict(dataset()).predictions
            handle.remove()
            self.assertTrue(states and all(s == (False, False) for s in states))
            np.testing.assert_allclose(first, second, atol=1e-7)
            batch = {k: torch.tensor(dataset()[k]) for k in ('input_ids', 'attention_mask')}
            model.eval()
            with torch.inference_mode():
                manual = model(**batch).logits.numpy()
            np.testing.assert_allclose(first, manual, atol=1e-7)
            trainer.save_model(str(Path(temp) / 'adapter'))
            fresh = tiny(dropout=0.2)
            fresh.load_state_dict(initial)
            loaded = PeftModel.from_pretrained(fresh, str(Path(temp) / 'adapter')).eval()
            with torch.inference_mode():
                reloaded = loaded(**batch).logits.numpy()
            np.testing.assert_allclose(first, reloaded, atol=1e-7)

    def test_accumulation_matches_full_batch_worker_and_notebook(self):
        definitions = notebook_definitions()
        self.assertIn('SecurityTrainer', definitions, 'Notebook changed; update review/tests after Claude handoff.')
        for trainer_class in (BtcTrainer, definitions['SecurityTrainer']):
            states = []
            for batch_size, accumulation in ((6, 1), (2, 3)):
                model = tiny()
                with tempfile.TemporaryDirectory() as temp:
                    args = TrainingArguments(output_dir=temp, use_cpu=True, max_steps=1,
                        per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
                        max_grad_norm=0, save_strategy='no', report_to='none', disable_tqdm=True,
                        lr_scheduler_type='constant', seed=42, data_seed=42)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    trainer = trainer_class(model=model, args=args, train_dataset=dataset(),
                                            optimizers=(optimizer, None))
                    trainer.train()
                    states.append(copy.deepcopy(model.state_dict()))
            with self.subTest(trainer=trainer_class.__name__):
                for name in states[0]:
                    torch.testing.assert_close(states[0][name], states[1][name], atol=2e-7, rtol=1e-5)

    def test_notebook_metric_and_index_order(self):
        defs = notebook_definitions()
        y = np.array([2, 0, 1])
        logits = np.array([[0, 0, 4], [4, 0, 0], [0, 4, 0]])
        self.assertEqual(defs['score_macro'](y, logits), 1)
        self.assertEqual(defs['compute_metrics'](SimpleNamespace(predictions=logits, label_ids=y))['macro_f1'], 1)
        records = [{'input_ids': [i]} for i in range(3)]
        ds = defs['IndexedPromptDataset'](records, [2, 0, 1], y)
        self.assertEqual([ds[i]['input_ids'][0] for i in range(3)], [2, 0, 1])

    def test_length_grouped_prediction_order_regression(self):
        fixed_trainer = notebook_definitions()['SecurityTrainer']
        lengths = [5, 13, 4, 11, 6, 9]
        ds = Dataset.from_dict({'input_ids': [[0] + [5 + i] * (n - 2) + [2] for i, n in enumerate(lengths)],
                                'attention_mask': [[1] * n for n in lengths], 'labels': [0, 2, 1, 1, 0, 2]})

        def collate(rows):
            width = max(len(r['input_ids']) for r in rows)
            return {'input_ids': torch.tensor([r['input_ids'] + [1] * (width - len(r['input_ids'])) for r in rows]),
                    'attention_mask': torch.tensor([r['attention_mask'] + [0] * (width - len(r['input_ids'])) for r in rows]),
                    'labels': torch.tensor([r['labels'] for r in rows])}

        for trainer_class, grouped, expect_order in ((Trainer, True, False), (fixed_trainer, True, True),
                                                     (BtcTrainer, False, True)):
            model = tiny().eval()
            with torch.inference_mode():
                manual = np.concatenate([model(**{k: v for k, v in collate([r]).items() if k != 'labels'}).logits.numpy()
                                         for r in ds])
            with tempfile.TemporaryDirectory() as temp:
                args = TrainingArguments(output_dir=temp, use_cpu=True, per_device_eval_batch_size=2,
                    group_by_length=grouped, report_to='none', disable_tqdm=True, seed=42)
                trainer = trainer_class(model=model, args=args, data_collator=collate)
                result = trainer.predict(ds)
                matches = np.allclose(result.predictions, manual, atol=1e-7, rtol=1e-6)
                with self.subTest(trainer=trainer_class.__name__, grouped=grouped):
                    self.assertEqual(matches, expect_order)
                    if expect_order:
                        np.testing.assert_equal(result.label_ids, np.asarray(ds['labels']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
