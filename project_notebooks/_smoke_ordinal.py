"""Smoke test for the converted notebook: run cells 0-9 up to (but not including)
the CV loop, then train_ordinal on a small slice of fold 0 for 2 epochs, verify
checkpoints + OOF arrays, verify the resume-complete path still collects OOF,
run the recording aggregation + ordinal_oof_metrics, then clean smoke artifacts.
"""
import io
import contextlib
import json
import os

nb = json.load(open('neural_network_3class.ipynb', encoding='utf-8'))
g: dict = {}


def run(i: int) -> None:
    src = ''.join(nb['cells'][i]['source'])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, f'cell{i}', 'exec'), g)
    out = buf.getvalue()
    tail = out[-800:] if len(out) > 800 else out
    if tail.strip():
        print(f'--- cell {i} ---\n{tail}'.encode('ascii', errors='replace').decode('ascii'))


for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:  # everything except the CV loop / eval / figures
    run(i)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

dataset, FOLDS, cv_pool = g['dataset'], g['FOLDS'], g['cv_pool']
train_subset, val_subset = g['train_subset'], g['val_subset']

# small slice of fold 0
tr_pos = [cv_pool[p] for p in FOLDS[0][0]][:400]
va_pos = [cv_pool[p] for p in FOLDS[0][1]][:120]
g['train_subset'] = Subset(dataset, tr_pos)
g['val_subset'] = Subset(dataset, va_pos)

NAME = 'smoke__default__f0'
hist, oof = g['train_ordinal'](NAME, num_epochs=2)
assert len(hist['val_loss']) == 2, f"expected 2 epochs, got {len(hist['val_loss'])}"
assert len(oof['p1']) == len(va_pos) == len(oof['path']) == len(oof['true_rank'])
assert set(np.unique(oof['true_rank'])) <= {0, 1, 2}
assert os.path.exists(f"run_state_3class/{NAME}_latest.pth.tar")
assert os.path.exists(f"run_state_3class/{NAME}_best.pth.tar")
print('OK: smoke train_ordinal 2 epochs, checkpoints + OOF arrays verified')

# resume-complete path: same call must skip training but STILL collect OOF
hist2, oof2 = g['train_ordinal'](NAME, num_epochs=2)
assert len(hist2['val_loss']) == 2, 'resume retrained unexpectedly'
assert len(oof2['p1']) == len(va_pos)
print('OK: resume-complete path skips training, still collects OOF')

# recording aggregation + metrics on the smoke OOF
# aggregate_recordings lives in the eval cell (eval1oof), which the smoke run
# does not execute — extract just the function source and exec it standalone.
import re  # noqa: E402

eval_src = ''.join(next(c['source'] for c in nb['cells'] if c['id'] == 'eval1oof'))
fn_src = 'def aggregate_recordings' + eval_src.split('def aggregate_recordings', 1)[1].split('\nEVAL_COLS', 1)[0]
ns = {'np': np}
exec(compile(fn_src, 'eval1oof', 'exec'), ns)
rec = ns['aggregate_recordings']({k: np.asarray(v) for k, v in oof.items()})
assert len(rec['true']) == len(set(oof['path']))
m = g['ordinal_oof_metrics'](rec['true'], rec['pred'], rec['p1'], rec['p2'])
assert 0.0 <= m['macro_f1'] <= 1.0 and m['rank_mae'] >= 0.0
print('OK: aggregate_recordings + ordinal_oof_metrics on smoke OOF:', m)

# cleanup smoke artifacts
for f in (f"run_state_3class/{NAME}_latest.pth.tar", f"run_state_3class/{NAME}_best.pth.tar"):
    if os.path.exists(f):
        os.remove(f)
print('OK: smoke artifacts cleaned')
