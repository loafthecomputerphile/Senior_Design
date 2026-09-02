from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias

import torch
import matplotlib.pyplot as plt
import torchaudio.transforms as T

if TYPE_CHECKING:
    import numpy.typing as npt
    
    
Axis: TypeAlias =  tuple[float, float, float, float]

__all__ = ["plot_waveform", "plot_spectrogram", "plot_fbank"]

def plot_waveform(waveform: torch.Tensor, sr: int, title="Waveform", ax: Axis | None = None) -> None:
    waveform: npt.NDArray = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis: torch.Tensor = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram: torch.Tensor, title: str | None = None, ylabel: str = "freq_bin", ax: Axis | None = None) -> None:
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    power_to_db = T.AmplitudeToDB("power", 80.0)
    ax.imshow(power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank: npt.ArrayLike, title: str| None = None) -> None:
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")