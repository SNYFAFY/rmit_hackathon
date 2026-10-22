"""Fine-tune XLM-R large bằng CSV BTC, không có giai đoạn dữ liệu ngoài."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)


APPROVED_MODEL = 'FacebookAI/xlm-roberta-large'
LABELS = ['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION']


def validate_model_source(cfg):
    # A model_id declaration alone does not constrain from_pretrained(model_path).
    if cfg.get('training_data') != 'BTC_ONLY' or cfg.get('model_id') != APPROVED_MODEL:
        raise ValueError('Only BTC_ONLY and the approved original XLM-R large are allowed.')
    if cfg.get('model_path') != APPROVED_MODEL:
        raise ValueError('model_path must be the original approved Hub ID; local/custom weights need provenance review.')
    if cfg.get('labels') != LABELS:
        raise ValueError('Label order must be BENIGN=0, JAILBREAK=1, PROMPT_INJECTION=2.')
    if cfg.get('precision') not in ('bf16', 'fp16'):
        raise ValueError('precision must be bf16 or fp16.')


def checked_logits(predictions, rows):
    logits = np.asarray(predictions[0] if isinstance(predictions, tuple) else predictions, dtype=np.float32)
    if logits.shape != (rows, 3) or not np.isfinite(logits).all():
        raise ValueError(f'Expected finite logits ({rows}, 3); got {logits.shape}.')
    return logits


def prediction_frame(ids, sample, logits, names=LABELS):
    logits = checked_logits(logits, len(ids))
    if not ids.is_unique or not sample.id.is_unique or ids.tolist() != sample.id.tolist():
        raise ValueError('Prediction IDs must be unique and match sample order exactly.')
    return pd.DataFrame({'id': ids.to_numpy(), 'label': [names[i] for i in logits.argmax(-1)]})


def base_model(cfg,cpu=False):
    names=cfg['labels']
    model=AutoModelForSequenceClassification.from_pretrained(cfg['model_path'],
        num_labels=3,id2label=dict(enumerate(names)),label2id={name:i for i,name in enumerate(names)},
        dtype=torch.float32 if cpu else (torch.bfloat16 if cfg['precision']=='bf16' else torch.float16),
        low_cpu_mem_usage=True,attn_implementation='sdpa')
    model.classifier.float()
    return model


def metrics(output):
    logits=output.predictions[0] if isinstance(output.predictions,tuple) else output.predictions
    return {'macro_f1':f1_score(output.label_ids,logits.argmax(-1),labels=[0,1,2],average='macro',zero_division=0)}


class BtcTrainer(Trainer):
    def __init__(self,*args,class_weights=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.class_weights=class_weights
        self.model_accepts_loss_kwargs=False

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        labels=inputs.pop('labels')
        output=model(**inputs)
        weight=self.class_weights.to(output.logits.device) if self.class_weights is not None else None
        loss=torch.nn.functional.cross_entropy(output.logits.float(),labels,weight=weight)
        if not torch.isfinite(loss):raise RuntimeError('Loss không hữu hạn; dừng để kiểm tra log.')
        return (loss,output) if return_outputs else loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['smoke','train','predict'],required=True)
    parser.add_argument('--resume',default=None)
    parser.add_argument('--cpu-test',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    validate_model_source(cfg)
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==cfg['worker_sha256']
    cpu=args.cpu_test
    if not cpu:
        assert torch.cuda.is_available(), 'Kiểm tra kernel, PyTorch cu128 và driver.'
        assert torch.cuda.device_count()>=1
        if cfg['precision']=='bf16':assert torch.cuda.is_bf16_supported()
        torch.cuda.set_device(0)
        print('GPU:',torch.cuda.get_device_name(0),'precision:',cfg['precision'],flush=True)
    run=Path(cfg['run_dir']);prepared=Path(cfg['prepared_dir']);names=cfg['labels']
    if args.resume:
        checkpoint=Path(args.resume).resolve()
        assert args.stage=='train' and checkpoint.parent==(run/'train').resolve()
        assert checkpoint.name.startswith('checkpoint-') and (checkpoint/'trainer_state.json').exists()
    set_seed(cfg['seed'])
    tokenizer=AutoTokenizer.from_pretrained(cfg['model_path'])
    collator=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8)
    common=dict(use_cpu=cpu,fp16=not cpu and cfg['precision']=='fp16',
        bf16=not cpu and cfg['precision']=='bf16',per_device_eval_batch_size=cfg['batch_size'],
        dataloader_num_workers=0,report_to='none')
    if args.stage=='predict':
        assert (run/'train_completed.json').exists()
        model=PeftModel.from_pretrained(base_model(cfg,cpu),str(run/'btc_adapter'))
        trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(run/'inference'),**common),
            data_collator=collator,processing_class=tokenizer)
        result=trainer.predict(load_from_disk(str(prepared/'test_ds')))
        ids=pd.read_csv(prepared/'test_ids.csv',dtype={'id':str},keep_default_na=False).id
        sample=pd.read_csv(prepared/'sample_submission.csv',dtype={'id':str},keep_default_na=False)
        logits=checked_logits(result.predictions,len(ids))
        frame=prediction_frame(ids,sample,logits,names)
        assert len(frame)==len(sample)==cfg['test_rows'] and frame.id.is_unique
        assert frame.id.tolist()==sample.id.tolist() and set(frame.label)<=set(names)
        frame.to_csv(run/'submission.csv',index=False);np.save(run/'test_logits.npy',logits)
        print(frame.label.value_counts(),flush=True)
        return
    model=get_peft_model(base_model(cfg,cpu),LoraConfig(task_type=TaskType.SEQ_CLS,
        r=8,lora_alpha=16,lora_dropout=0.05,target_modules=['query','value'],
        modules_to_save=['classifier'],bias='none'))
    for p in model.parameters():
        if p.requires_grad:p.data=p.data.float()
    model.config.use_cache=False
    model.print_trainable_parameters()
    tr=load_from_disk(str(prepared/'train_ds'));va=load_from_disk(str(prepared/'valid_ds'))
    counts=np.bincount(np.asarray(tr['labels']),minlength=3)
    smoke=args.stage=='smoke'
    if smoke:
        # Exercise the longest actual BTC rows; never train on generated prompts.
        lengths=np.asarray([len(ids) for ids in tr['input_ids']])
        selected=np.argsort(lengths,kind='stable')[-min(len(tr),max(16,2*cfg['batch_size'])):]
        tr=tr.select(selected.tolist())
    if not smoke and (run/'btc_adapter'/'adapter_config.json').exists() and not args.resume:
        raise RuntimeError('Adapter đã tồn tại. Dùng kết quả cũ, resume checkpoint hoặc RUN_NAME mới.')
    weights=None
    if cfg['class_weights']:
        if np.any(counts==0):
            raise ValueError('Class weights require all three labels in the training split.')
        w=np.sqrt(counts.sum()/(3*counts));weights=torch.tensor(w/w.mean(),dtype=torch.float32)
    training_args=TrainingArguments(output_dir=str(run/args.stage),num_train_epochs=cfg['epochs'],
        max_steps=2 if smoke else -1,learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=cfg['batch_size'],gradient_accumulation_steps=1 if smoke else cfg['grad_accum'],
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        optim='adamw_torch',weight_decay=0.01,warmup_ratio=0.06,max_grad_norm=1.0,
        eval_strategy='no' if smoke else 'epoch',save_strategy='no' if smoke else 'epoch',
        load_best_model_at_end=not smoke,metric_for_best_model='macro_f1',greater_is_better=True,
        save_total_limit=2,save_safetensors=True,logging_steps=1 if smoke else 20,
        logging_nan_inf_filter=False,seed=cfg['seed'],data_seed=cfg['seed'],**common)
    trainer=BtcTrainer(model=model,args=training_args,train_dataset=tr,eval_dataset=va,
        data_collator=collator,processing_class=tokenizer,compute_metrics=metrics,class_weights=weights,
        callbacks=[] if smoke else [EarlyStoppingCallback(early_stopping_patience=2)])
    if not cpu:torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();trainer.train(resume_from_checkpoint=args.resume)
    elapsed=time.perf_counter()-start
    peak=None if cpu else torch.cuda.max_memory_allocated()/2**30
    print('Phút:',round(elapsed/60,2),'Peak allocated GB:',peak,flush=True)
    if smoke:
        (run/'smoke_passed.json').write_text(json.dumps({'elapsed_seconds':elapsed,'peak_gb':peak}))
        return
    trainer.save_model(str(run/'btc_adapter'));tokenizer.save_pretrained(run/'btc_adapter');trainer.save_state()
    result=trainer.predict(va)
    logits=checked_logits(result.predictions,len(va))
    y=result.label_ids;pred=logits.argmax(-1)
    report=classification_report(y,pred,labels=[0,1,2],target_names=names,zero_division=0,output_dict=True)
    (run/'validation_report.json').write_text(json.dumps(report,indent=2))
    np.save(run/'validation_logits.npy',logits)
    frame=pd.read_csv(prepared/'validation_rows.csv',keep_default_na=False)
    assert len(frame)==len(pred)
    frame['prediction']=[names[i] for i in pred];frame.to_csv(run/'validation_predictions.csv',index=False)
    rows=[]
    for lang in sorted(frame.language.unique()):
        mask=frame.language.to_numpy()==lang
        rows.append({'language':lang,'count':int(mask.sum()),
            'macro_f1':f1_score(y[mask],pred[mask],labels=[0,1,2],average='macro',zero_division=0)})
    pd.DataFrame(rows).to_csv(run/'validation_by_language.csv',index=False)
    pd.DataFrame(confusion_matrix(y,pred,labels=[0,1,2]),index=names,columns=names).to_csv(run/'confusion_matrix.csv')
    pd.DataFrame(trainer.state.log_history).to_csv(run/'train_logs.csv',index=False)
    (run/'train_completed.json').write_text(json.dumps({'best_checkpoint':trainer.state.best_model_checkpoint,
        'best_metric':trainer.state.best_metric,'elapsed_seconds':elapsed,'peak_gb':peak,'config':cfg},indent=2))
    print(classification_report(y,pred,labels=[0,1,2],target_names=names,zero_division=0,digits=4),flush=True)


if __name__=='__main__':main()
