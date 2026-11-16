"""Whole-file (no-segmentation) dataset builder — Phase 2.

One npz row = ONE RECORDING, not a segmented cough:
  >5s  -> middle 5s (center crop)
  <5s  -> right-pad zeros to 5s
  <2s  -> dropped
Optional YAMNet keep-gate: a recording is kept only if its max 0.96s-frame
cough score clears `yamnet_threshold` (reuse of cough_verify verbatim).

Decode via soundfile (no torchcodec/FFmpeg DLLs) + ThreadPoolExecutor — same
pattern as make_dataset.py, but without the segmentation step. npz layout is
identical to make_dataset.py, so NpzDataPipeline loads it unchanged:
    {label}_data   float32 [n, max_seconds * sample_rate]
    {label}_paths  str     [n]   source recording path
    {label}_labels str     [n]   status label
    length         int           total recording count
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm.auto import tqdm

from modules.dataset.make_dataset import LABELS, print_class_counts
from modules.preprocessing.cough_verify import segment_cough_scores
from modules.preprocessing.dataframe_process import filter_coughvid_df


def _load_wave(path: str, sample_rate: int) -> np.ndarray:
    """Fast C decode (soundfile) -> mono -> resample if needed."""
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != sample_rate:
        wave = AF.resample(torch.from_numpy(wave), rate, sample_rate).numpy()
    return wave


def _fit_window(
    wave: np.ndarray, min_samples: int, target_samples: int
) -> np.ndarray | None:
    """Whole-recording window: middle `target_samples` if longer, right-pad to
    it if shorter, None if below `min_samples` (dropped)."""
    if len(wave) < min_samples:
        return None
    if len(wave) > target_samples:
        start = (len(wave) - target_samples) // 2
        wave = wave[start:start + target_samples]
    return np.pad(wave, (0, target_samples - len(wave)))


def make_dataset_whole(
    df: pl.DataFrame,
    sample_rate: int,
    out: str,
    max_workers: int = 8,
    yamnet_threshold: float | None = 0.1,
    min_seconds: float = 2.0,
    max_seconds: float = 5.0,
) -> None:
    """Build the whole-file npz from a df with `wav_path` (or `audio_name`) and
    `status` columns. yamnet_threshold=None disables the cough keep-gate."""
    min_samples = round(min_seconds * sample_rate)
    target_samples = round(max_seconds * sample_rate)
    path_col = "wav_path" if "wav_path" in df.columns else "audio_name"
    df = df.select([path_col, "status"])

    failed: list[str] = []
    gate = [0, 0]  # [windows scored, kept after YAMNet gate]

    def process_row(row: dict) -> tuple[str, str, np.ndarray | None]:
        label, file_path = row["status"], row[path_col]
        if label not in LABELS:
            return label, file_path, None
        try:
            wave = _fit_window(
                _load_wave(file_path, sample_rate), min_samples, target_samples
            )
        except Exception as e:
            failed.append(f"{file_path}: {e}")
            return label, file_path, None
        if wave is None:
            return label, file_path, None
        # whole wave = one "segment" for segment_cough_scores: max over its frames
        if yamnet_threshold is not None:
            gate[0] += 1
            if segment_cough_scores([wave], sample_rate)[0] <= yamnet_threshold:
                return label, file_path, None
            gate[1] += 1
        return label, file_path, wave

    # ponytail: threads like make_dataset — decode + YAMNet forward release the GIL
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(tqdm(
            pool.map(process_row, df.iter_rows(named=True)),
            total=len(df), desc="processing whole files",
        ))

    if yamnet_threshold is not None:
        print(f"yamnet whole-file gate @ {yamnet_threshold}: kept {gate[1]}/{gate[0]} recordings")
    if failed:
        print(f"WARNING: {len(failed)} file(s) failed to load, e.g.:")
        for p in failed[:5]:
            print(f"  {p}")

    out_data: dict[str, np.ndarray] = {}
    total = 0
    for label in LABELS:
        hits = [(p, w) for lab, p, w in results if lab == label and w is not None]
        if not hits:
            continue
        out_data[f"{label}_data"] = np.stack([w for _, w in hits])
        out_data[f"{label}_paths"] = np.array([p for p, _ in hits], dtype=object)
        out_data[f"{label}_labels"] = np.full(len(hits), label, dtype=object)
        total += len(hits)

    out_data["length"] = total
    np.savez(out, **out_data)
    print_class_counts({k: len(out_data[f"{k}_data"]) for k in LABELS if f"{k}_data" in out_data})


def nonexpert_df(df_all: pl.DataFrame, df_expert: pl.DataFrame) -> pl.DataFrame:
    """Non-expert rows: self-reported status with complete metadata
    (cough_detected > 0.8), minus every recording the expert file already
    covers — anti-join keeps the two npz outputs disjoint."""
    return (
        df_all.with_columns(pl.col("cough_detected").cast(pl.Float64))
        .pipe(filter_coughvid_df)
        .join(df_expert.select("wav_path"), on="wav_path", how="anti")
        .filter(pl.col("status").is_in(LABELS))
    )
