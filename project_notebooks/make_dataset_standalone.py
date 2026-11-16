"""Standalone dataset builder for CoughVid v3 — the ORIGINAL pipeline, no YAMNet.

Reproduces exactly what modules/dataset/make_dataset.py did before the YAMNet
segment filter existed: filter the metadata table, segment each wav into coughs,
pad/truncate segments to 1s, and write one npz with per-class arrays.

Self-contained: only needs numpy, polars, soundfile, tqdm (no torch, no modules/).

Run from project_notebooks/:
    python make_dataset_standalone.py                       # defaults below
    python make_dataset_standalone.py --out my_npz.npz --cough-threshold 0.8

npz layout (identical to the original):
    {label}_data   float32 [n, sample_rate]  padded/truncated 1s waveforms
    {label}_paths  str    [n]                source recording path
    {label}_labels str    [n]                status label per segment
    length         int                       total segment count
"""
import argparse
import numpy as np
import polars as pl
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is a progress bar only — run without it if missing
    def tqdm(x, **k): return x

LABELS = ["healthy", "COVID-19", "symptomatic"]


# ---------------- segmentation (verbatim port of segment_cough_fast) ----------------
# Energy-based hysteresis segmentation: coughs are bursts above a high RMS threshold
# that end after `tolerance` consecutive samples below a low threshold.

def _runs(cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start indices and lengths of every True run in a boolean array."""
    padded = np.concatenate(([False], cond, [False])).view(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return edges[0::2], edges[1::2] - edges[0::2]


def segment_cough_fast(x, fs, cough_padding=0.2, min_cough_len=0.2,
                       th_l_multiplier=0.1, th_h_multiplier=2, tol_multiplier=0.01):
    """Vectorized cough segmentation — same thresholds/padding/min-length rules as
    the original per-sample loop (segment_cough in audio_process.py)."""
    power = np.square(x)
    rms = np.sqrt(np.mean(power))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    padding = round(fs * cough_padding)
    min_cough_samples = round(fs * min_cough_len)
    tolerance = round(tol_multiplier * fs)
    n = len(x)

    below_starts, below_lens = _runs(power < seg_th_l)
    above_starts = _runs(power > seg_th_h)[0]

    coughSegments = []
    carry = 0           # below-threshold counter carried across coughs (loop quirk)
    active_until = -1   # cough still in progress through this index
    ri = 0              # cursor into below-runs
    for s in above_starts:
        if s <= active_until:
            continue
        while ri < len(below_starts) and below_starts[ri] <= s:
            ri += 1
        trigger = None
        j = ri
        while j < len(below_starts):
            rs, rl = below_starts[j], below_lens[j]
            if j == ri and rs == s + 1 and carry > tolerance:
                trigger = s + 1
                carry += 1
                break
            if rl > tolerance:
                trigger = rs + tolerance
                carry = tolerance + 1
                break
            j += 1  # run too short to trip the counter; it resets after the run
        if trigger is None:
            # cough persists to end of file — keep it if the final sample is still loud
            if s < n - 1 and power[-1] >= seg_th_l:
                cough_start = max(0, s - padding)
                if n - cough_start - 2 * padding > min_cough_samples:
                    coughSegments.append(x[cough_start:n])
            break
        cough_start = max(0, s - padding)
        cough_end = min(trigger + padding, n - 1)
        active_until = trigger
        if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
            coughSegments.append(x[cough_start:cough_end + 1])
    return coughSegments


def fit_length(segment: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros / truncate to exactly target_samples (EnforceFixedLength)."""
    out = np.zeros(target_samples, dtype=segment.dtype)
    out[:min(len(segment), target_samples)] = segment[:target_samples]
    return out


def print_class_counts(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    for label, n in counts.items():
        print(f"{label}: {n} ({n / total:.1%})")
    print(f"total: {total}")


def make_dataset(df: pl.DataFrame, sample_rate: int, out: str,
                 max_workers: int = 8, max_per_clip: int | None = None) -> None:
    """df must have `audio_name` (real wav path) and `status` columns; rows with
    null status / missing metadata must be filtered before calling (as in the
    original pipeline)."""

    master = {k: {"data": [], "paths": [], "labels": []} for k in LABELS}
    target_samples = sample_rate  # 1.0s segments

    def process_row(row: dict):
        label, file_path = row["status"], row["audio_name"]
        wave, rate = sf.read(file_path, dtype="float32")
        if wave.ndim > 1:
            wave = wave[:, 0]
        if rate != sample_rate:
            # original path used torchaudio.functional.resample (kaiser); use it when
            # torch is installed, else fall back to linear interp (close enough for speech)
            try:
                import torch, torchaudio.functional as AF
                wave = AF.resample(torch.tensor(wave), rate, sample_rate).numpy()
            except ImportError:
                n_out = round(len(wave) * sample_rate / rate)
                wave = np.interp(
                    np.linspace(0, len(wave) - 1, n_out), np.arange(len(wave)), wave
                ).astype(np.float32)

        waves = segment_cough_fast(wave, sample_rate, cough_padding=0.1, min_cough_len=0.1)
        fitted = np.array([fit_length(w, target_samples) for w in waves])
        labels = np.full(len(waves), label, dtype=object)
        paths = np.full(len(waves), file_path, dtype=object)

        # per-clip cap kills near-duplicate segments from one recording
        if max_per_clip is not None and len(fitted) > max_per_clip:
            keep = np.sort(np.random.default_rng(0).choice(len(fitted), size=max_per_clip, replace=False))
            fitted, labels, paths = fitted[keep], labels[keep], paths[keep]
        return label, fitted, labels, paths

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(tqdm(pool.map(process_row, df.iter_rows(named=True)),
                            total=len(df), desc="processing audio"))

    rng = np.random.default_rng(0)
    out_data = {}
    total = 0
    for label, data, labels, paths in results:
        if len(data):
            master[label]["data"].append(data)
            master[label]["labels"].append(labels)
            master[label]["paths"].append(paths)

    for label in LABELS:
        if not master[label]["data"]:
            continue
        data = np.concatenate(master[label]["data"], axis=0)
        labels = np.concatenate(master[label]["labels"], axis=0)
        paths = np.concatenate(master[label]["paths"], axis=0)
        out_data[f"{label}_data"] = data
        out_data[f"{label}_labels"] = labels
        out_data[f"{label}_paths"] = paths
        total += len(data)

    out_data["length"] = total
    np.savez(out, **out_data)
    print_class_counts({k: len(out_data[f"{k}_data"]) for k in LABELS if f"{k}_data" in out_data})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="data/tabular_form/tabular_form/coughvid_v3.csv")
    ap.add_argument("--wav-dir", default="data/public_dataset_v3_wav")
    ap.add_argument("--out", default="final_processed_dataset.npz")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--cough-threshold", type=float, default=0.8,
                    help="keep rows with cough_detected above this (original filter)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-per-clip", type=int, default=None)
    args = ap.parse_args()

    df = pl.read_csv(args.csv, schema_overrides={"age": pl.Float32})
    # original filter: dataset's own cough probability + complete metadata
    df = df.filter(
        (pl.col("cough_detected") > args.cough_threshold)
        & pl.col("age").is_not_null()
        & pl.col("gender").is_not_null()
        & pl.col("status").is_not_null()
    )
    # audio_name is a webm/uuid name in the CSV — point it at the converted wavs
    df = df.with_columns(
        pl.col("audio_name")
        .map_elements(lambda n: str(Path(args.wav_dir) / (n.split(".")[0] + ".wav")),
                      return_dtype=pl.String)
        .alias("audio_name")
    )
    print(f"{len(df)} rows after filtering (cough_detected > {args.cough_threshold})")
    make_dataset(df, args.sample_rate, args.out,
                 max_workers=args.workers, max_per_clip=args.max_per_clip)


if __name__ == "__main__":
    main()
