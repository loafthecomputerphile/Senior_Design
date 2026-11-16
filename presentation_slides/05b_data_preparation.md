# Slide 5 — Data Preparation: From Raw Audio to Network Input
**(paste screenshot of `make_dataset.py` here — `project_notebooks/modules/dataset/make_dataset.py`)**

**On-slide text (keep short):**

```
raw recording → segment coughs → fixed length → balance caps → .npz on disk
```

- One recording yields **many** labeled segments (one cough per clip)
- Source path stored with every segment → split by **recording**, not by segment
- Split happens at the *recording* level — segments from one file never cross train/val/test

**How the network eats it:**
- `NpzDataPipeline` → tensors `(waveform, int label)`; label mapping healthy=0, COVID=1, symptomatic=2
- `TaskDataset` re-labels for the cascade: stage 1 = {COVID, symptomatic} vs healthy; stage 2 = COVID vs symptomatic, healthy dropped
- `AugmentedSubset` applies augmentation **only on the training split** — per-class probabilities (COVID 90%, symptomatic 70%, healthy 40%)
- `binary_labels` / `binary_labels_grouped` produce the 1-vs-rest targets at loss/metrics time, so raw 3-class labels stay intact through the pipeline

**Verbally:**
- Why segmentation matters: coughs are short events inside long clips; the network trains on cough events, not background noise — same segmentation logic runs in the backend (`/evaluate` → segment → gate → model)
- Why the caps: `max_per_clip` kills near-duplicate segments from one recording; `max_per_label` caps the majority class so healthy (74%) can't dominate
- Why recording-wise split (the leakage fix): random split put sibling segments of the same speaker into train AND test — the model memorized the voice, not the cough. All splits are seeded and persisted (`persist_indicies`) so every experiment is reproducible
- Why augmentation happens on-the-fly in `__getitem__`, not on disk: every epoch sees a different perturbation, and disk stays small
- Everything lands in one `.npz` (`final_processed_dataset.npz`) so training loads instantly without re-decoding audio

One-liner: "every segment carries its source recording's ID, so the data pipeline itself enforces the no-leakage rule."
