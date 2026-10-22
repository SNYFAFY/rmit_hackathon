"""Analyze unmodified validation logits; no calibration, thresholds, or test labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

LABELS = ['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION']


def confidence_metrics(y, logits, bins=15):
    y = np.asarray(y)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(y), 3) or not len(y) or not np.isfinite(logits).all():
        raise ValueError('Expected nonempty, finite [N,3] raw logits.')
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError('Expected integer labels 0/1/2.')
    if bins < 1:
        raise ValueError('bins must be positive.')
    z = logits - logits.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs)
    pred = logits.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y
    bin_ids = np.minimum((conf * bins).astype(int), bins - 1)
    bin_rows = []
    ece = 0.0
    for i in range(bins):
        mask = bin_ids == i
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        avg = float(conf[mask].mean()) if count else None
        if count:
            ece += count / len(y) * abs(acc - avg)
        bin_rows.append({'lower': i / bins, 'upper': (i + 1) / bins,
                         'count': count, 'accuracy': acc, 'mean_confidence': avg})
    f1 = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)
    result = {'count': len(y), 'macro_f1': float(f1.mean()), 'per_class_f1': dict(zip(LABELS, f1.tolist())),
              'nll': float(-log_probs[np.arange(len(y)), y].mean()),
              'multiclass_brier': float(np.square(probs - np.eye(3)[y]).sum(axis=1).mean()),
              'ece': float(ece), 'ece_bins': bin_rows,
              'confusion_matrix_true_rows': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}
    for name, mask in [('correct', correct), ('incorrect', ~correct)]:
        result[f'{name}_confidence'] = {'count': int(mask.sum()),
            'mean': float(conf[mask].mean()) if mask.any() else None,
            'median': float(np.median(conf[mask])) if mask.any() else None,
            'count_ge_0_9': int((conf[mask] >= 0.9).sum()), 'count_ge_0_99': int((conf[mask] >= 0.99).sum())}
    return result, probs


def validate_rows(rows, train_ids, official):
    if not rows.id.is_unique or not train_ids.id.is_unique or not official.id.is_unique:
        raise ValueError('Duplicate IDs.')
    if set(rows.id) & set(train_ids.id):
        raise ValueError('Train/validation ID overlap.')
    if set(rows.id) | set(train_ids.id) != set(official.id):
        raise ValueError('Saved split IDs must partition the official training CSV.')
    truth = official.set_index('id').loc[rows.id]
    for col in ('label', 'language', 'text'):
        if col not in rows or rows[col].tolist() != truth[col].tolist():
            raise ValueError(f'Validation {col} or row order differs from official rows.')
    if not rows.label.isin(LABELS).all():
        raise ValueError('Unknown label.')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--logits', type=Path, required=True)
    p.add_argument('--rows', type=Path, required=True, help='Validation rows in exact logits order.')
    p.add_argument('--train-ids', type=Path, required=True, help='Saved, finalized training split IDs, CSV id column.')
    p.add_argument('--official-train', type=Path, default=Path(__file__).resolve().parents[1] / 'data/train.csv')
    p.add_argument('--run-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory; never overwrites an analysis.')
    a = p.parse_args()
    cfg = json.loads(a.run_config.read_text(encoding='utf-8'))
    if cfg.get('model_id', cfg.get('MODEL_ID')) != 'FacebookAI/xlm-roberta-large':
        raise ValueError('Run does not declare the approved large backbone.')
    if cfg.get('labels', cfg.get('LABELS')) != LABELS:
        raise ValueError('Run label mapping does not match.')
    if cfg.get('training_data') != 'BTC_ONLY':
        raise ValueError('Run must declare BTC_ONLY provenance.')
    rows, train_ids, official = [pd.read_csv(f, dtype={'id': str}, keep_default_na=False)
                                 for f in (a.rows, a.train_ids, a.official_train)]
    validate_rows(rows, train_ids, official)
    logits = np.load(a.logits, allow_pickle=False)
    y = rows.label.map(dict(zip(LABELS, range(3)))).to_numpy()
    result, probs = confidence_metrics(y, logits)
    result['per_language'] = {}
    for lang in sorted(rows.language.unique()):
        mask = rows.language.to_numpy() == lang
        result['per_language'][lang] = confidence_metrics(y[mask], logits[mask])[0]
    result['input_sha256'] = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in (a.logits, a.rows, a.train_ids, a.official_train, a.run_config)}
    result['definitions'] = {'nll': 'Mean negative log probability of the true class, natural log.',
        'brier': 'Mean sum over all 3 classes of (p_k - one_hot_k)^2, range [0,2].',
        'ece': '15 equal-width confidence bins [lower,upper), last includes 1; sum n_bin/N * abs(accuracy - mean confidence).',
        'confidence': 'Exactly one stable softmax over class axis=1 of supplied raw logits.',
        'provenance_limit': 'Split/row/config checks and input hashes do not prove which checkpoint produced logits; verify original run log and model source before interpretation.'}
    rows = rows.copy()
    rows['prediction'] = np.asarray(LABELS)[logits.argmax(axis=1)]
    rows['confidence'] = probs.max(axis=1)
    if 'prediction' in pd.read_csv(a.rows, nrows=0).columns:
        original = pd.read_csv(a.rows, dtype={'id': str}, keep_default_na=False)
        if original.prediction.tolist() != rows.prediction.tolist():
            raise ValueError('Saved predictions disagree with raw-logit argmax; check mapping/stale output.')
    for i, label in enumerate(LABELS):
        rows[f'p_{label.lower()}'] = probs[:, i]
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    rows.to_csv(a.output / 'validation_confidence.csv', index=False)
    rows.loc[rows.prediction != rows.label].sort_values('confidence', ascending=False).to_csv(a.output / 'errors_by_confidence.csv', index=False)
    print(json.dumps({k: result[k] for k in ('count', 'macro_f1', 'nll', 'multiclass_brier', 'ece')}, indent=2))


if __name__ == '__main__':
    main()
