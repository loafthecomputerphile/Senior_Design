"""Smoke check for Phase 1 changes: removed per-class aug gates, proportional
time mask, oversample-to-balance logic (mirrors the notebook's oversample_balance).
Run from project_notebooks/ with the project venv."""
import json
import numpy as np
import torch

# 1) WaveformAugmentor: gate removed -> augmentation fires for every label
from modules.preprocessing.audio_process import WaveformAugmentor, SpecAugmentations

aug = WaveformAugmentor(16_000)
aug.train()
x = torch.arange(16_000, dtype=torch.float32).unsqueeze(0) / 16_000
# no-op rate = P(all 4 technique coin flips miss) — ~0.14-0.15 observed; same for
# every label now, pass bound 0.30 unchanged
for label in (0, 1, 2):
    noops = sum(torch.equal(aug(x.clone(), label), x) for _ in range(200))
    assert noops / 200 <= 0.30, f"label {label}: no-op rate {noops/200:.2f} — gate not removed?"
    print(f"1) label {label}: no-op rate {noops/200:.2f} (expected ~0.14-0.15)")

# 2) SpecAugmentations: proportional time mask (floor 5), TimeStretch path gone
spec = SpecAugmentations()
spec.train()
for T, expected in ((32, 5), (157, max(5, 157 // 20))):
    xt = torch.randn(1, 3, 104, T, dtype=torch.complex64)
    out = spec(xt)  # label=None path
    assert out.shape[-1] == T, f"T changed at {T}: {out.shape}"  # no stretch -> fixed T
    assert spec.time_mask.mask_param == expected, (T, spec.time_mask.mask_param)
print("2) proportional time mask OK (T=32 -> 5, T=157 -> 7), output T unchanged (stretch gone)")

# 3) oversample_balance logic (same code as the folds cell)
def oversample_balance(labels_seq, seed=42):
    positions = list(range(len(labels_seq)))
    by_cls = {0: [], 1: [], 2: []}
    for p in positions:
        by_cls[labels_seq[p]].append(p)
    target = max(len(v) for v in by_cls.values())
    out = []
    for idxs in by_cls.values():
        if not idxs:
            continue
        reps, rem = divmod(target, len(idxs))
        out.extend(idxs * reps + idxs[:rem])
    np.random.default_rng(seed).shuffle(out)
    return out

labels = [0] * 10 + [1] * 3 + [2] * 1
ob = oversample_balance(labels)
cnt = np.bincount([labels[i] for i in ob], minlength=3)
assert cnt.tolist() == [10, 10, 10], cnt
val = list(range(3))
assert len(oversample_balance(val)) == 3  # already balanced -> untouched
print("3) oversample_balance: 14 -> 30 positions, balanced [10,10,10]")

# 4) edited notebook cells compile
nb = json.load(open("neural_network_3class.ipynb", encoding="utf-8"))
edited = {"d6c374cf", "dda0e9f7", "23e32579"}
for cell in nb["cells"]:
    if cell.get("id") in edited:
        src = "".join(cell["source"])
        compile(src, cell["id"], "exec")
print(f"4) edited notebook cells compile: {sorted(edited)}")

print("smoke OK")
