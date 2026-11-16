import sys, torch
from torch.utils.data import Dataset
sys.path.insert(0, r"C:\Users\drewq\Documents\code\GitHub\senior_design\Senior_Design\project_notebooks")
from modules.dataset.make_dataset import dataset_split_stratified_by_recording, stratified_recording_folds

class FakeDS(Dataset):
    def __init__(self, labels): self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return torch.zeros(1)

# 20 recordings x 3 segments: class 0 = rec00..09, class 1 = rec10..19
SEG_PER_REC = 3
paths = [f"rec{i:02d}.wav" for i in range(20)] * SEG_PER_REC
paths = sorted(paths)  # interleave siblings like concatenate does
labels = torch.tensor([0 if int(p[3:5]) < 10 else 1 for p in paths])
ds = FakeDS(labels)

# ---- stratified grouped split: grouping + composition + determinism ----
tr, va, te = dataset_split_stratified_by_recording(ds, paths, 0.7, 0.2, seed=42)

def recs(s): return {paths[i].split(".")[0] for i in s.indices}
overlap = (recs(tr) & recs(va)) | (recs(tr) & recs(te)) | (recs(va) & recs(te))
assert not overlap, f"leak: {overlap}"
assert len(tr) + len(va) + len(te) == 60
assert set(tr.indices) | set(va.indices) | set(te.indices) == set(range(60))

# per-class split counts within one recording (3 segments) of the class targets
for cls in (0, 1):
    seg_total = 10 * SEG_PER_REC
    target = {"train": 0.7 * seg_total, "val": 0.2 * seg_total, "test": 0.1 * seg_total}
    for name, s in (("train", tr), ("val", va), ("test", te)):
        got = sum(1 for i in s.indices if int(labels[i]) == cls)
        assert abs(got - target[name]) <= SEG_PER_REC, (cls, name, got, target[name])

tr2, va2, te2 = dataset_split_stratified_by_recording(ds, paths, 0.7, 0.2, seed=42)
assert tr.indices == tr2.indices and va.indices == va2.indices

# ---- folds: partition + no recording validated twice + both classes per fold ----
folds = stratified_recording_folds(paths, labels, n_splits=5, seed=42)
assert len(folds) == 5
assert sorted(p for _, va_pos in folds for p in va_pos) == list(range(60))
val_recs = [{paths[p].split(".")[0] for p in va_pos} for _, va_pos in folds]
assert sum(len(rs) for rs in val_recs) == 20, "a recording landed in two folds"
for _, va_pos in folds:
    assert {int(labels[p]) for p in va_pos} == {0, 1}, "val fold lost a class"
print("OK: stratified split (grouping, composition within 1 recording, deterministic) "
      "+ 5 folds (partition, each recording validated once, both classes per fold)")
