# Artifacts README

This zip collects every log, CSV and figure produced by the two halves of the project:
the **YAMNet cough-verification work** and the **training experiments** built on the
filtered dataset. Dates refer to file modification times; everything was produced in
`project_notebooks/` (all paths below are relative to it).

## 1. YAMNet cough verification

**Question:** can we trust CoughVid's `cough_detected` column? It is a heuristic
probability from the dataset authors, and we found audible coughs in files it scored
high *and* files it scored 0. So we re-verified every file with an independent model.

**Process** (see `yamnet_cough_filter.ipynb`, executed top-to-bottom; the notebook in
this zip contains all printed logs and inline figures as cell outputs):

1. Load the CoughVid v3 metadata CSV (`data/tabular_form/tabular_form/coughvid_v3.csv`,
   34,434 rows) with polars; build wav paths from `audio_name`.
2. Segment each wav into cough regions with `segment_cough_fast`
   (`modules/preprocessing/audio_process.py`) — the same energy-based hysteresis
   segmentation the training pipeline uses, so verification sees exactly what training
   would see. Segments shorter than 1s are zero-padded.
3. Score every segment with **PyTorch YAMNet** (`w-hc/torch_audioset`, AudioSet 521
   classes). The "Cough" class index is looked up at runtime from the official
   `yamnet_class_map.csv` (**index 42**; index 41 is "Snort" — an early run used the
   wrong index and was discarded). Score = max sigmoid probability over the segment's
   0.96s frames.
4. **File score** = highest segment score in the file: one clear cough anywhere labels
   the whole file. Files where segmentation finds no cough-like energy score 0.0
   (production parity: they yield no training segments anyway).
5. Scoring ran in chunks of 512 files on CUDA, flushing progress to
   `yamnet/yamnet_scores.parquet` every 8 chunks (interrupt-safe; ~34k files).
6. Sanity checks: files the dataset calls strong coughs (`cough_detected > 0.95`) score
   high, files it rejects (`< 0.01`) score low; the model's top-5 predicted classes for
   a strong-cough file include "Cough".
7. Threshold selection: the max-score distribution is smooth (no knee), so the
   threshold was **matched to the original filter's keep-count** (`cough_detected >
   0.8`), making the before/after comparison same-size and apples-to-apples.
   Default in code: `YAMNET_THRESHOLD = 0.3`.

**Files:**
- `yamnet/yamnet_scores.parquet` — one row per wav: `wav_path`,
  `cough_detected_yamnet` (max segment score, [0,1]). Null = unreadable file.
- `yamnet/yamnet_class_map.csv` — official AudioSet/YAMNet class map (downloaded from
  the tensorflow/models repo), source of the cough index.
- `yamnet_cough_filter.ipynb` — the notebook above with all logs/figures.
- `notes.md` — working notes.

**Pipeline integration** (committed code, not in this zip):
`modules/preprocessing/cough_verify.py` exposes `segment_cough_scores()` +
`YAMNET_THRESHOLD`; `modules/dataset/make_dataset.py` takes `yamnet_threshold=` and
drops any segment scoring below it before feature fitting — a 15-file test kept 17/43
segments at threshold 0.3.

**Standalone reproduction:** `make_dataset_standalone.py` re-creates the dataset
exactly as the **original** (pre-YAMNet) pipeline did — same metadata filter
(`cough_detected > 0.8` + complete metadata), same `segment_cough_fast` segmentation,
same 1s pad/truncate, same npz layout — as one self-contained file (numpy/polars/
soundfile only, no torch or project modules needed). Verified: 12,885 files →
47,842 segments, class counts printed. It does **not** apply any YAMNet filtering;
pass `--cough-threshold` to change the metadata cutoff.

## 2. Training experiments

**Process:** models were trained on the segment dataset produced by `make_dataset()`
(spectrogram/MFCC feature variants; see `modules/preprocessing/audio_process.py` and
`modules/nn/`). Experiment runs, metric aggregation and figure generation were driven
by `_append_cells.py`, which writes one CSV per task/feature-variant and renders the
PNGs in `figures/`.

**CSVs** (root of the zip):
- `*_experiment_stats.csv` — per-epoch training/validation metrics for each task ×
  feature variant. Prefixes: unmarked = pre-filter runs; `trained_*` = retrained runs.
  Tasks: `covid_symp_vs_healthy` (stage 1), `covid_vs_symp` (stage 2). Feature
  variants: `wave_only`, `spec_only`, `both`.
- `trained_experiment_summary.csv` — aggregate best/final metrics per trained run.
- `*_eval_results.csv`, `*_stage2_standalone_eval.csv` — cascade evaluation results
  (stage-1 gate + stage-2 classify, untrained vs retrained).

**Figures** (`figures/`):
- `*_history.png` — loss/accuracy curves per training run.
- `*_test_curves.png`, `*_curves.png` — test-set ROC/PR-style curves per task/variant.
- `*_cm.png` — stage-1/stage-2 confusion matrices per feature variant
  (`trained_*` = retrained versions).
- `cascade_3class_*.png` — full 3-class cascade performance per feature variant.
- `both (wave + spec).png`, `wave only.png`, `spec only.png` — feature-variant
  comparison summary plots.

## Not included
- `data/final_processed_dataset.npz` (2.9 GB processed segments) and the raw
  CoughVid audio — regenerate via the notebook + `make_dataset()`.
- Source code: lives in the repo (`modules/`, `_append_cells.py`, notebooks).
