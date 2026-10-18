"""Verify segment_cough_fast matches segment_cough on real audio files and is faster.
Run from project_notebooks/: python test_segment_cough_fast.py
"""
import sys, time

sys.path.insert(0, ".")

import numpy as np
import polars as pl
import soundfile as sf
import torch
import torchaudio.functional as F

from modules.preprocessing.audio_process import segment_cough, segment_cough_fast
from modules.preprocessing.dataframe_process import filter_coughvid_df

SAMPLE_RATE: int = 16_000
N_FILES: int = 300


def make_path(audio_name: str) -> str:
    return f"data/public_dataset_v3_wav/{audio_name.split('.')[0]}.wav"


def mask_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) pairs of True runs in a mask."""
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).view(np.int8)))
    return list(zip(edges[0::2], edges[1::2] - 1))


def main() -> None:
    df: pl.DataFrame = filter_coughvid_df(
        pl.read_csv("data/tabular_form/tabular_form/coughvid_v3.csv", schema_overrides={"age": pl.Float32})
    )

    files: list[tuple[str, np.ndarray]] = []
    for name in df["audio_name"]:
        if len(files) >= N_FILES:
            break
        path: str = make_path(name)
        try:
            # ponytail: torchaudio.load needs FFmpeg/torchcodec DLLs absent in this env;
            # soundfile reads wav natively. Swap back if torchaudio decoding is fixed.
            x, rate = sf.read(path, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
        except Exception:
            continue
        if rate != SAMPLE_RATE:
            x = F.resample(torch.tensor(x), rate, SAMPLE_RATE).numpy()
        files.append((path, x))

    assert len(files) >= 8, f"only found {len(files)} loadable files (need >= 8)"

    print(f"{'file':<28} {'n_old':>5} {'n_fast':>6} {'match':>5} {'t_old':>8} {'t_fast':>7} {'speedup':>7}")
    total_old: float = 0.0
    total_fast: float = 0.0
    all_match: bool = True

    for path, x in files:
        t0 = time.perf_counter()
        segs_old, mask_old = segment_cough(x, SAMPLE_RATE, cough_padding=0.1, min_cough_len=0.1)
        total_old += time.perf_counter() - t0

        t0 = time.perf_counter()
        segs_fast, mask_fast = segment_cough_fast(x, SAMPLE_RATE, cough_padding=0.1, min_cough_len=0.1)
        total_fast += time.perf_counter() - t0

        # identical if: same segment count, same segment lengths, same mask (boundaries)
        lengths_match: bool = [len(s) for s in segs_old] == [len(s) for s in segs_fast]
        mask_match: bool = np.array_equal(mask_old, mask_fast)
        match: bool = lengths_match and mask_match
        all_match &= match
        speedup: float = total_old / max(total_fast, 1e-9)

        print(
            f"{path.split('/')[-1]:<28} {len(segs_old):>5} {len(segs_fast):>6} "
            f"{str(match):>5} {total_old:>8.3f} {total_fast:>7.3f} {speedup:>6.1f}x"
        )
        if not match:
            b_old: list = mask_bounds(mask_old)
            b_fast: list = mask_bounds(mask_fast)
            print(f"    old bounds: {b_old}")
            print(f"    fast bounds: {b_fast}")

    print(f"\ntotal old: {total_old:.3f}s  fast: {total_fast:.3f}s  overall speedup: {total_old / max(total_fast, 1e-9):.1f}x")
    assert all_match, "segment_cough_fast disagrees with segment_cough on at least one file"
    assert total_fast < total_old, "segment_cough_fast is not faster"
    print("PASS: identical segments on all files, and faster")


if __name__ == "__main__":
    main()