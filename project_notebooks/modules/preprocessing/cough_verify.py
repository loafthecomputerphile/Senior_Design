"""YAMNet-based cough verification: score individual cough segments.

Each segment is scored with the max over its 0.96s YAMNet frame cough
probabilities, so a segment counts as a cough if ANY frame inside it sounds
like one. Used by make_dataset() to drop non-cough segments before training.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torchaudio.functional as AF

# module-level knob: segments scoring below this are dropped by make_dataset().
# Tune via the keep-count-matched sweep in yamnet_cough_filter.ipynb.
YAMNET_THRESHOLD: float = 0.3

TARGET_SR = 16000
FRAME_SAMPLES = 15360        # 0.96s at 16k — YAMNet's window
MIN_SEG_SAMPLES = TARGET_SR  # pad segments to >= 1s so YAMNet gets a full frame

# process is all this pipeline needs.
_model = None
_transform = None
_cough_idx: int | None = None


def _load_model() -> tuple:
    """Lazy-load YAMNet + the cough class index. CUDA if available, else CPU."""
    global _model, _transform, _cough_idx
    if _model is None:
        from torch_audioset.yamnet import yamnet
        from torch_audioset.data.torch_input_processing import WaveformToInput

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _model = yamnet(pretrained=True).to(device).eval()
        _transform = WaveformToInput()

        # Never hardcode the cough index: it comes from the official class map
        # (index 42 — 41 is "Snort", an easy off-by-one).
        csv_path = Path('data/yamnet_class_map.csv')
        if not csv_path.exists():
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv',
                csv_path,
            )
        import polars as pl
        class_map = pl.read_csv(csv_path)
        _cough_idx = class_map.filter(pl.col('display_name') == 'Cough')['index'][0]
    return _model, _transform, _cough_idx


def _seg_to_frames(seg: npt.NDArray, sr: int) -> torch.Tensor:
    """One segment -> YAMNet input frames [T, 1, 96, 64].

    - resample to 16k (YAMNet's rate)
    - zero-pad to >= 1s: short coughs must fill at least one 0.96s frame
    - pad the final partial 0.96s frame: WaveformToInput silently DROPS the
      tail, so a cough at the end of a segment would be lost without this
    """
    model, transform, _ = _load_model()  # noqa: F841 — warms the singleton on first call
    s = torch.from_numpy(np.ascontiguousarray(seg))
    if sr != TARGET_SR:
        s = AF.resample(s, sr, TARGET_SR)
    if len(s) < MIN_SEG_SAMPLES:
        s = torch.nn.functional.pad(s, (0, MIN_SEG_SAMPLES - len(s)))
    tail = len(s) % FRAME_SAMPLES
    if tail:
        s = torch.nn.functional.pad(s, (0, FRAME_SAMPLES - tail))
    return _transform(s.unsqueeze(0), TARGET_SR)


@torch.no_grad()
def segment_cough_scores(waves: list[npt.NDArray], sr: int, batch_frames: int = 256) -> np.ndarray:
    """Max YAMNet cough probability per segment; waves[i] -> scores[i] in [0, 1].

    All frames from all segments go through the model in one batched pass.
    Thread-safe: forward passes under no_grad are fine from make_dataset's pool.
    """
    model, _, cough_idx = _load_model()
    frames = [_seg_to_frames(w, sr) for w in waves]
    big = torch.cat(frames)
    device = next(model.parameters()).device

    scores = torch.empty(len(big), device=device)
    for i in range(0, len(big), batch_frames):
        out = model(big[i:i + batch_frames].to(device), to_prob=True)  # [B, 521] sigmoid probs
        scores[i:i + batch_frames] = out[:, cough_idx]

    # max over each segment's frames (frames are contiguous per segment)
    per_seg = []
    start = 0
    for f in frames:
        per_seg.append(scores[start:start + len(f)].max().item())
        start += len(f)
    return np.array(per_seg, dtype=np.float32)
