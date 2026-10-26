import sys, numpy as np, torch
from torch.utils.data import TensorDataset, Subset
sys.path.insert(0, r"C:\Users\drewq\Documents\code\GitHub\senior_design\Senior_Design\project_notebooks")
from modules.dataset.make_dataset import dataset_split_by_recording

# 20 recordings x 5 segments, labels embedded in features
paths = [f"rec{i:02d}.wav" for i in range(20)] * 5
paths = sorted(paths)  # interleave siblings like concatenate does
data = torch.arange(100).unsqueeze(1).float()
ds = TensorDataset(data)
tr, va, te = dataset_split_by_recording(ds, paths, seed=42)

def recs(s): return {paths[i].split(".")[0] for i in s.indices}
overlap = (recs(tr) & recs(va)) | (recs(tr) & recs(te)) | (recs(va) & recs(te))
assert not overlap, f"leak: {overlap}"
assert len(tr) + len(va) + len(te) == 100
assert len(tr) == 80, len(tr)
assert len(set(tr.indices) | set(va.indices) | set(te.indices)) == 100
# same seed -> same split
tr2, va2, te2 = dataset_split_by_recording(ds, paths, seed=42)
assert tr.indices == tr2.indices and va.indices == va2.indices
print("OK: no recording overlap, sizes 80/10/10, deterministic")
