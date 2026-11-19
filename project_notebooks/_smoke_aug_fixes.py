"""Smoke checks for the aug-overhaul fixes (fix 1..N append here as numbered
sections). Run from project_notebooks/ with plain `python`.

NOTE: `from modules.nn import ...` does NOT work under plain python (torchvision
missing), so CORAL cumulative targets are inlined as
torch.stack(((rank > 0).float(), (rank > 1).float()), dim=1).
"""
import json

import torch

# =========================================================================
# Section 1: mixup_waveforms (fix 1)
# =========================================================================
from modules.preprocessing.audio_process import mixup_waveforms

torch.manual_seed(0)
rank = torch.tensor([0, 1, 2, 0])
x = torch.randn(4, 16000)

# 1a) property: shapes, bounds, ordinal monotonicity over 50 draws
for _ in range(50):
    xm, t = mixup_waveforms(x, rank, alpha=0.2)
    assert xm.shape == x.shape and t.shape == (4, 2)
    assert (t >= 0).all() and (t <= 1).all() and (t[:, 1] <= t[:, 0]).all()
print("1a) mixup properties: shapes (4,16000)/(4,2), targets in [0,1], col1<=col0 over 50 draws")

# 1b) lam=1.0 edge: waveforms unchanged, targets == hard cumulative targets
hard = torch.stack(((rank > 0).float(), (rank > 1).float()), dim=1)
xm, t = mixup_waveforms(x, rank, alpha=0.2, lam=1.0)
assert torch.equal(xm, x)
assert torch.equal(t, hard)
print("1b) lam=1.0 edge: waveforms unchanged, targets == hard cumulative targets")

# =========================================================================
# Section 2: edited notebook cells compile (fix 1: train_ordinal cell)
# =========================================================================
nb = json.load(open("neural_network_3class.ipynb", encoding="utf-8"))
edited = {"23e32579", "f94fb1b7"}  # train_ordinal + audio_process import cell
for cell in nb["cells"]:
    if cell.get("id") in edited:
        src = "".join(cell["source"])
        compile(src, cell["id"], "exec")
# wiring sanity: the flag and helper call are actually in the train_ordinal cell
tr_src = next("".join(c["source"]) for c in nb["cells"] if c.get("id") == "23e32579")
assert "mixup_alpha: float = 0.0" in tr_src
assert "mixup_waveforms(waveforms, rank, alpha=mixup_alpha)" in tr_src
assert "loss = criterion(outputs, mix_targets)" in tr_src
assert tr_src.count("coral_targets(") == 2  # train else-branch + val loop only (OOF uses none)
print(f"2) edited notebook cells compile + mixup wiring present: {sorted(edited)}")

# =========================================================================
# Section 3: balanced WeightedRandomSampler (fix 2)
# =========================================================================
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# 3a) sampler math: inverse-frequency weights over toy labels rebalance draws.
# labels [0]*10+[1]*3+[2]*1, weights 1/bincount(labels)[labels], 30_000 draws ->
# each class should land within 15% of 10_000.
toy_labels = torch.tensor([0] * 10 + [1] * 3 + [2] * 1)
toy_weights = (1.0 / torch.bincount(toy_labels).float())[toy_labels]
torch.manual_seed(42)
draws = list(WeightedRandomSampler(toy_weights, num_samples=30_000, replacement=True))
draw_counts = torch.bincount(toy_labels[torch.tensor(draws)], minlength=3).float()
for cls in range(3):
    assert abs(draw_counts[cls] - 10_000) <= 0.15 * 10_000, (
        f"class {cls}: {draw_counts[cls].item():.0f} not within 15% of 10_000"
    )
print(f"3a) sampler math balanced: counts {draw_counts.tolist()} (each within 15% of 10_000)")

# 3b) DataLoader takes exactly one of shuffle/sampler: sampler-only works,
# shuffle+sampler together raises ValueError.
toy_ds = TensorDataset(torch.arange(14).float(), toy_labels)
sampler = WeightedRandomSampler(toy_weights, num_samples=len(toy_labels), replacement=True)
list(DataLoader(toy_ds, batch_size=4, sampler=sampler))  # sampler-only: fine
try:
    DataLoader(toy_ds, batch_size=4, shuffle=True, sampler=sampler)
    raised = False
except ValueError:
    raised = True
assert raised, "DataLoader accepted both shuffle and sampler (must raise ValueError)"
print("3b) DataLoader exclusivity: sampler-only OK, shuffle+sampler raises ValueError")

# 3c) integration pin: train_ordinal cell 23e32579 is actually wired to the sampler
tr_src = next("".join(c["source"]) for c in nb["cells"] if c.get("id") == "23e32579")
assert "balanced_sampler" in tr_src, "23e32579 missing balanced_sampler flag"
assert "WeightedRandomSampler" in tr_src, "23e32579 missing WeightedRandomSampler"
print("3c) integration pin: cell 23e32579 contains WeightedRandomSampler + balanced_sampler")

print("smoke OK")
