"""Read-only audit. Optional CUDA arithmetic probe, never loads a pretrained model."""
import argparse
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda-probe', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import torch

    packages = ['torch', 'transformers', 'peft', 'accelerate', 'datasets', 'huggingface-hub',
                'safetensors', 'sentencepiece', 'protobuf', 'numpy', 'pandas', 'scikit-learn',
                'jupyterlab', 'ipykernel', 'scipy', 'matplotlib']
    installed = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in metadata.distributions()}
    out = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'root': str(ROOT),
           'executable': sys.executable, 'python': sys.version, 'platform': platform.platform(),
           'packages': {name: installed.get(name, 'MISSING') for name in packages},
           'pip_check': command([sys.executable, '-m', 'pip', 'check']),
           'gpu_processes': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv']),
           'nvidia_smi': command(['nvidia-smi']), 'pid': os.getpid()}
    disk = shutil.disk_usage(ROOT)
    out['disk'] = {'total_bytes': disk.total, 'free_bytes': disk.free}
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total_physical', 'available_physical',
                    'total_pagefile', 'available_pagefile', 'total_virtual', 'available_virtual', 'extended_virtual')]
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            out['ram'] = {'total_bytes': memory.total_physical, 'available_bytes': memory.available_physical}
    out['torch'] = {'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
                    'arch_list': torch.cuda.get_arch_list()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        out['torch'].update(name=props.name, capability=list(torch.cuda.get_device_capability(0)),
                            total_memory_bytes=props.total_memory, free_total_bytes=list(torch.cuda.mem_get_info()),
                            bf16=torch.cuda.is_bf16_supported())
    if args.cuda_probe:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable.')
        proc = out['gpu_processes']
        if proc['exit_code'] or len(proc['stdout'].strip().splitlines()) != 1:
            raise RuntimeError('GPU is occupied or process query failed; probe skipped.')
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(128, 128, device='cuda', requires_grad=True)
        w = torch.randn(128, 128, device='cuda', requires_grad=True)
        reference = x.detach().float() @ w.detach().float()
        out['cuda_arithmetic'] = {}
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
                y = x @ w
                loss = y.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(loss) and torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all())
            out['cuda_arithmetic'][str(dtype)] = {'loss': loss.item(), 'finite_loss_and_gradients': finite,
                                                 'max_abs_error': (y.float() - reference).abs().max().item()}
            if not finite:
                raise RuntimeError(f'Nonfinite CUDA arithmetic with {dtype}.')
            x.grad = w.grad = None
        out['cuda_probe_peak_bytes'] = torch.cuda.max_memory_allocated()

    frames = {}
    out['data'] = {}
    for name in ('train', 'test', 'sample_submission'):
        path = ROOT / 'data' / f'{name}.csv'
        frame = pd.read_csv(path, dtype={'id': str}, keep_default_na=False)
        frames[name] = frame
        out['data'][name] = {'path': str(path), 'sha256': sha256(path), 'rows': len(frame),
                             'columns': list(frame.columns), 'unique_ids': bool(frame.id.is_unique)}
    tr, te, sa = (frames[x] for x in ('train', 'test', 'sample_submission'))
    out['data'].update(labels=tr.label.value_counts().to_dict(), languages=tr.language.value_counts().to_dict(),
                       train_test_id_overlap=len(set(tr.id) & set(te.id)),
                       test_sample_same_order=te.id.tolist() == sa.id.tolist(),
                       normalized_train_duplicates=int(tr.text.str.strip().str.casefold().str.replace(r'\s+', ' ', regex=True).duplicated().sum()))
    notebooks = {}
    for name in ('Training.ipynb', 'xlmr_large_btc.ipynb'):
        path = ROOT / name
        nb = json.loads(path.read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')
        notebooks[name] = {'sha256': sha256(path), 'cells': len(nb['cells']),
                           'calls_btc_worker': 'btc_worker.py' in code,
                           'code_cells': [{'index': i, 'execution_count': c.get('execution_count'),
                                           'source_sha256': hashlib.sha256(''.join(c.get('source', [])).encode()).hexdigest(),
                                           'outputs_present': bool(c.get('outputs'))}
                                          for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']}
    out['notebooks'] = notebooks
    out['worker_sha256'] = sha256(ROOT / 'btc_worker.py')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
