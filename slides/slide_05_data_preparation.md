# Slide: Data Preparation & the Neural-Network Pipeline

Source: `project_notebooks/modules/dataset/make_dataset.py`

## Bullet content

- **One-pass build script (`make_dataset()`)**: reads the labeled dataframe (CoughVID + sympathetic-cough datasets), decodes each recording with `soundfile` (float32), downsamples to mono, and resamples to the model's target sample rate on the fly.
- **Cough segmentation**: every recording is passed through `segment_cough_fast` (energy-based cough detector, 0.1 s padding / min 0.1 s length) so training uses only the cough region — not silence, speech, or background noise.
- **Fixed-length enforcement**: each extracted cough segment is cropped/padded by `EnforceFixedLength` to a uniform length so all segments can be stacked into a single NumPy array.
- **Class balance controls**:
  - `max_per_clip` — caps near-duplicate segments coming from a single recording.
  - `max_per_label` — randomly subsamples each class after concatenation so healthy/COVID/symptomatic counts are controlled.
- **Persisted artifact**: everything is written to `project_notebooks/final_processed_dataset.npz` as `{healthy, COVID-19, symptomatic}_{data, labels, paths}` blocks, plus a total `length` key. Raw audio is stored as waveform, not spectrograms — feature extraction happens later in the neural-network notebook.

## How the network consumes it

- **`NpzDataPipeline`** (`make_dataset.py:134`) loads the `.npz`, concatenates the three per-class blocks into one tensor, and maps labels to integers (0=healthy, 1=COVID-19, 2=symptomatic). Returns `(waveform, label)` tensors ready for the training loop.
- **Leak-safe splitting**: `dataset_split_by_recording()` groups all segments by *source recording*, so sibling coughs from the same clip can't appear in both train and validation — a common silent-inflation-of-accuracy bug. Splits are saved with `persist_indicies()` and reloaded with `load_indicies()` for reproducibility.
- **`AugmentedSubset`** wraps any split and applies `WaveformAugmentor` only to training data at fetch time (on-the-fly augmentation, no extra disk).
- **`TaskDataset`** turns the 3-class dataset into the binary problems the models actually train on:
  - `covid_symp_vs_healthy` — all samples, COVID+symptomatic merged as positive.
  - `covid_vs_symp` — healthy filtered out entirely, COVID vs symptomatic only.
  - Labels are relabeled at the loss step via `TASK_TARGETS`, so augmentation dictionaries still see the raw 3-class labels.

## Slide image

- `![dataset pipeline](../project_notebooks/figures/<figure-showing-segmented-coughs-or-class-counts>.png)` — pick from `project_notebooks/figures/`.

## Screenshot suggestions

- `make_dataset.py` lines 27–66 (`make_dataset` + `process_row`) — shows segmentation and resampling.
- `make_dataset.py` lines 219–257 (`dataset_split_by_recording`) — highlight the leak-safe split, good talking point.
- `make_dataset.py` lines 306–313 (`TaskDataset` docstring) — shows how binary tasks are derived.
- Console output of `print_class_counts` showing healthy/COVID/symptomatic totals for the class-balance bullet.
