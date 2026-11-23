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

# =========================================================================
# Section 4: speed perturbation wired into forward (fix 3)
# =========================================================================
from modules.preprocessing.audio_process import WaveformAugmentor

# 4a) default tech_probs include speed at 0.3 (previously missing -> dead code)
default_aug = WaveformAugmentor(16_000)
assert default_aug.tech_probs["speed"] == 0.3, default_aug.tech_probs

# 4b) speed-only augmentor: shape always (1, 16000), and speed actually fires
# (factors [0.9, 1.0, 1.1] -> ~2/3 of trials should change the waveform)
torch.manual_seed(7)
speed_aug = WaveformAugmentor(16_000)
speed_aug.tech_probs = {"gain": 0.0, "noise": 0.0, "time_shift": 0.0, "speed": 1.0}
speed_aug.train()
xs = torch.randn(1, 16000)
changed = 0
for _ in range(50):
    out = speed_aug(xs.clone(), 0)
    assert out.shape == (1, 16000), f"shape changed: {tuple(out.shape)}"
    changed += int(not torch.equal(out, xs))
assert changed >= 20, f"only {changed}/50 outputs differ from input (speed not applied?)"
print(f"4) speed perturbation: 50/50 outputs shape (1,16000), {changed}/50 differ from input (>=20 required)")

# =========================================================================
# Section 5: peak-based noise scaling (fix 4)
# =========================================================================
torch.manual_seed(11)
noise_aug = WaveformAugmentor(16_000)
noise_aug.tech_probs = {"gain": 0.0, "noise": 1.0, "time_shift": 0.0, "speed": 0.0}
noise_aug.train()

# Two signals with the SAME peak 1.0: one with 75% trailing zeros (like the
# stored trim_tile segments), one with no padding. RMS-based noise scales with
# sqrt(mean square), so the padded signal gets ~0.5x the noise of the full one
# (ratio ~0.5); peak-based noise gives ratio ~1.0.
two_pi = 2.0 * 3.141592653589793
padded = torch.cat([
    torch.sin(torch.arange(4000, dtype=torch.float32) * (two_pi / 1000)),
    torch.zeros(12000),
]).unsqueeze(0)
full = torch.sin(torch.arange(16000, dtype=torch.float32) * (two_pi / 1000)).unsqueeze(0)
assert abs(padded.abs().max().item() - 1.0) < 1e-3
assert abs(full.abs().max().item() - 1.0) < 1e-3

def _median_delta_max(x: torch.Tensor, trials: int = 20) -> float:
    vals = sorted(
        (noise_aug(x.clone(), 0) - x).abs().max().item() for _ in range(trials)
    )
    return vals[trials // 2]

d_pad = _median_delta_max(padded)
d_full = _median_delta_max(full)
ratio = d_pad / d_full
assert d_pad > 0.0, "padded signal received no noise at all"
assert 0.75 <= ratio <= 1.25, (
    f"padded/unpadded noise ratio {ratio:.3f} outside [0.75, 1.25] "
    "(old RMS code gives ~0.5 with 75% zero padding)"
)
print(f"5) peak-based noise: padded delta {d_pad:.4f}, full delta {d_full:.4f}, "
      f"ratio {ratio:.3f} in [0.75, 1.25] (old RMS code ~0.5)")

# 5b) pad-masked noise (final-review fix 1): noise must stay inside the signal's
# nonzero support. randn_like over the ~75% zero-pad makes trim_tile's wi != 0
# probe run to the last sample (nz[-1] == 15999), silently disabling trim+tile
# for augmented train samples while val/test (no augmentor) always tile.
from modules.preprocessing.audio_process import trim_tile

torch.manual_seed(13)
tail_aug = WaveformAugmentor(16_000)
tail_aug.tech_probs = {"gain": 0.0, "noise": 1.0, "time_shift": 0.0, "speed": 0.0}
tail_aug.train()
_pad_sig = torch.cat([torch.randn(4000), torch.zeros(12000)]).unsqueeze(0)
_pad_sig *= 1.0 / _pad_sig.abs().max()  # peak 1.0, content in first quarter
L = int(torch.nonzero(_pad_sig[0] != 0, as_tuple=False)[-1]) + 1
assert L == 4000, L
for _ in range(20):
    _out = tail_aug(_pad_sig.clone(), 0)
    assert _out.shape == _pad_sig.shape
    assert (_out[..., L:] == 0).all(), (
        "noise sprayed into the trailing zero-pad (trim_tile's wi != 0 would misfire)"
    )
_trimmed = trim_tile(_out.clone())
assert torch.equal(_trimmed[0, :L], _out[0, :L]), "trim_tile changed the content"
assert torch.equal(_trimmed[0, L:2 * L], _out[0, :L]), (
    "trim_tile did not tile (tail not a repeat of content) - trim+tile still disabled?"
)
print(f"5b) pad-masked noise: tail beyond index {L} exactly zero over 20 trials; "
      "trim_tile(out) trims to L and tiles the content")

# =========================================================================
# Section 6: SpecAugment 2+2 policy, freq width 40 (fix 5)
# =========================================================================
from modules.preprocessing.audio_process import SpecAugmentations

# 6a) freq width is real now: 40 of 513 bins (~8%), not 5 (~1% noop)
spec = SpecAugmentations()
assert spec.freq_mask.mask_param == 40, spec.freq_mask.mask_param
print("6a) freq_mask.mask_param == 40")

# 6b) 2+2 masks: with p=1.0 the masks virtually always bite (old single
# freq-mask-at-5 on 104 bins was a near-noop)
torch.manual_seed(3)
spec.p = 1.0
spec.train()
x = torch.randn(1, 3, 104, 32, dtype=torch.complex64)
ref = x.abs().pow(2)
diff = 0
for _ in range(50):
    out = spec(x.clone())
    assert out.shape == x.shape, f"output shape changed: {tuple(out.shape)} (T must stay 32)"
    diff += int(not torch.equal(out, ref))
assert diff >= 45, f"masks changed output in only {diff}/50 trials (near-noop?)"
print(f"6) SpecAugment 2+2: T preserved (32), masks changed output in {diff}/50 trials (>=45 required)")

# =========================================================================
# Section 7: fix 6 - none control arm, F1 checkpoint selection, v2 run names
# =========================================================================
import ast as _ast

def _cell_src(cell_id: str) -> str:
    return "".join(next(c["source"] for c in nb["cells"] if c.get("id") == cell_id))

# 7a) control arm FIRST in AUG_CONFIGS: exactly 4 keys, "none" -> (False, False)
_split_src = _cell_src("aug1split")
_cfg = None
for _node in _ast.parse(_split_src).body:
    if isinstance(_node, _ast.AnnAssign) and getattr(_node.target, "id", None) == "AUG_CONFIGS":
        _cfg = _ast.literal_eval(_node.value)
assert _cfg is not None, "AUG_CONFIGS not found in aug1split"
assert list(_cfg) == ["none", "wave_only", "spec_only", "both"], list(_cfg)
assert _cfg["none"] == (False, False), _cfg["none"]
print("7a) control arm: AUG_CONFIGS has exactly 4 keys, 'none': (False, False) first")

# 7b) v2 run names + flag opt-in in the run cell (new code must never resume
# old-code checkpoints -> a2__ prefix, not aug__)
_train_src = _cell_src("aug2train")
assert "f\"a2__{cfg}__default\"" in _train_src, "aug2train missing a2__ run-name prefix"
assert "aug__{cfg}__default" not in _train_src, "aug2train still uses old aug__ prefix"
assert ("train_ordinal(name, min_epoch=8, wave_aug=w, spec_aug=s, "
        "mixup_alpha=0.2, balanced_sampler=True)") in _train_src, "aug2train call not upgraded"
print("7b) run cell: a2__ prefix + mixup_alpha=0.2 + balanced_sampler=True")

# 7c) F1-based checkpoint selection in the trainer cell: save-best keyed to
# val macro-F1, early-stop plateau still keyed to val loss (loss branch keeps
# ONLY best_val_loss/plateau bookkeeping - the save moved out)
_tr_src = _cell_src("23e32579")
assert "best_val_f1 = " in _tr_src, "best_val_f1 not derived after _resume_with_best"
assert "if val_macro_f1 > best_val_f1 + 1e-6:" in _tr_src, "F1 save branch missing"
assert "_atomic_torch_save(state, _ckpt_best_path(name))" in _tr_src
_loss_branch = (
    "        if val_epoch_loss < best_val_loss - 1e-6:\n"
    "            best_val_loss = val_epoch_loss\n"
    "            plateau = 0\n"
    "        else:\n"
    "            plateau += 1\n"
)
assert _loss_branch in _tr_src, "val-loss branch no longer pure plateau bookkeeping"
print("7c) checkpoint selection: best_val_f1 derived; save-best on F1, plateau on loss")

# 7d) every notebook code cell compiles
for _i, _c in enumerate(nb["cells"]):
    if _c["cell_type"] == "code":
        compile("".join(_c["source"]), _c.get("id", f"cell{_i}"), "exec")
print(f"7d) all {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')} notebook code cells compile")

# 7e) run-name bump must reach the downstream cells: aug3test loads the a2__
# checkpoints, aug4figs reads the a2__ CSVs and colors all 4 arms
_test_src = _cell_src("aug3test")
assert 'f"a2__{cfg}__default"' in _test_src, "aug3test still uses old aug__ checkpoint names"
assert "aug__{cfg}__default" not in _test_src, "aug3test still references old aug__ names"
_figs_src = _cell_src("aug4figs")
assert "a2__{cfg}__default" in _figs_src, "aug4figs still reads old aug__ CSV names"
assert "aug__{cfg}__default" not in _figs_src, "aug4figs still references old aug__ names"
import re as _re
_colors = _re.search(r"_AUG_COLORS(?::\s*dict\[str,\s*str\])?\s*=\s*\{([^}]*)\}", _figs_src)
assert _colors, "_AUG_COLORS not found in aug4figs"
_color_keys = _re.findall(r'"([a-z_]+)":', _colors.group(1))
assert _color_keys == ["none", "wave_only", "spec_only", "both"], _color_keys
assert _figs_src.count("plt.subplots(1, 4") == 2, (
    "aug4figs CM/hist figures must have 4 axes (one per arm, incl. none)"
)
print("7e) downstream cells: aug3test + aug4figs on a2__ names, 4 colors, 4-axis figures")

# 7f) final-review fix 2: a2 artifact names everywhere downstream
for _cid in ("aug3test", "aug4figs"):
    _s = _cell_src(_cid)
    assert "aug_test_state" not in _s, f"{_cid} still references aug_test_state.pt"
    assert "aug_test_eval" not in _s, f"{_cid} still references aug_test_eval.csv"
    assert "aug_default_" not in _s, f"{_cid} still references aug_default_* figures"
    assert "a2_test_state" in _s, f"{_cid} missing a2_test_state.pt"
print("7f) a2 artifact names: no aug_test_state / aug_test_eval / aug_default_ refs remain")

# 7g) final-review fix 3: best_val_f1 must be derived from the _best checkpoint
# ON DISK, not from the resumed (latest) history — _resume_with_best restores
# _latest only, so after a crash between the _latest and _best saves the latest
# history can run AHEAD of the stale _best file.
assert ('torch.load(_ckpt_best_path(name), weights_only=True, map_location="cpu")["history"]' in _tr_src
        or "torch.load(_ckpt_best_path(name), weights_only=True, map_location=\"cpu\")[\"history\"]" in _tr_src), (
    "best_val_f1 not derived from the on-disk _best checkpoint's history"
)
assert 'max(history["val_F1"])' not in _tr_src, (
    "best_val_f1 still derived from the latest-checkpoint history (crash-window bug back)"
)
_resume_idx = _tr_src.index("_resume_with_best(")
_disk_idx = _tr_src.index('torch.load(_ckpt_best_path(name), weights_only=True, map_location="cpu")["history"]')
assert _resume_idx < _disk_idx, "disk-based best_val_f1 derivation must follow _resume_with_best"
print("7g) checkpoint invariant: best_val_f1 derived from on-disk _best history (after resume)")

print("smoke OK")
