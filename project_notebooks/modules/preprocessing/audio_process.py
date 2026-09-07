from __future__ import annotations
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F
from torchaudio.transforms import (
    FrequencyMasking,
    MelScale,
    Resample,
    Spectrogram,
    TimeMasking,
    TimeStretch,
    MFCC,
    AmplitudeToDB
)

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["segment_cough", "compute_SNR", "MelSpectrogramPipeline", "AugmentedMelSpectrogramPipeline"]

'''
SOURCE: 
Orlandic, L., Teijeiro, T. & Atienza, D. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. Sci Data 8, 156 (2021). https://doi.org/10.1038/s41597-021-00937-4
'''


#Use old segmentation
def segment_cough(x: npt.NDArray,fs: float, cough_padding: float = 0.2, min_cough_len: float =0.2, th_l_multiplier: float = 0.1, th_h_multiplier: int = 2) -> tuple[list[npt.NDArray], npt.NDArray]:
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power
    
    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator
    
    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress"""
                
    cough_mask: npt.NDArray = np.array([False]*len(x))
    

    #Define hysteresis thresholds
    rms: np.float64 = np.sqrt(np.mean(np.square(x)))
    seg_th_l: float = th_l_multiplier * rms
    seg_th_h: float =  th_h_multiplier*rms

    #Segment coughs
    coughSegments: list = []
    padding: float = round(fs*cough_padding)
    min_cough_samples: float = round(fs*min_cough_len)
    cough_start: int = 0
    cough_end: int = 0
    cough_in_progress: bool = False
    tolerance: float = round(0.01*fs)
    below_th_counter: int = 0
    
    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample<seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i+padding if (i+padding < len(x)) else len(x)-1
                    cough_in_progress = False
                    if (cough_end+1-cough_start-2*padding>min_cough_samples):
                        coughSegments.append(x[cough_start:cough_end+1])
                        cough_mask[cough_start:cough_end+1] = True
            elif i == (len(x)-1):
                cough_end=i
                cough_in_progress = False
                if (cough_end+1-cough_start-2*padding>min_cough_samples):
                    coughSegments.append(x[cough_start:cough_end+1])
            else:
                below_th_counter = 0
        else:
            if sample>seg_th_h:
                cough_start = i-padding if (i-padding >=0) else 0
                cough_in_progress = True
    
    return coughSegments, cough_mask

def compute_SNR(x: npt.NDArray, fs: float) -> float | int:
    """Compute the Signal-to-Noise ratio of the audio signal x (np.array) with sampling frequency fs (float)"""
    segments, cough_mask = segment_cough(x,fs)
    RMS_signal: float | int = 0 if len(x[cough_mask])==0 else np.sqrt(np.mean(np.square(x[cough_mask])))
    RMS_noise: float = np.sqrt(np.mean(np.square(x[~cough_mask])))
    SNR: float | int = 0 if (RMS_signal==0 or np.isnan(RMS_noise)) else 20*np.log10(RMS_signal/RMS_noise)
    return SNR


class ComplexToPower(torch.nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().pow(2)
    


class MelSpectrogramPipeline(torch.nn.Module):
    
    def __init__(self, resample_freq: int = 16_000, n_fft: int=1024, n_mel: int = 256) -> None:
        super().__init__()
        self.resample_freq: int = resample_freq
        self.current_input_freq: int | None = None
        self.resampler: Resample | None = None

        self.spec: Spectrogram = Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale: MelScale = MelScale(
            n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1
        )
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_min: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x-x_min)/(x_max-x_min+1e-8)      

    def forward(self, waveform: torch.Tensor, input_freq: int) -> torch.Tensor:
        if input_freq != self.current_input_freq:
            self.current_input_freq = input_freq
            if input_freq != self.resample_freq:
                self.resampler = Resample(orig_freq=input_freq, new_freq=self.resample_freq).to(waveform.device)
            else:
                self.resampler = None

        resampled = self.resampler(waveform) if self.resampler is not None else waveform
        spec = self.spec(resampled)
        return self._normalize(self.mel_scale(spec))
    
    
class AugmentedMelSpectrogramPipeline(torch.nn.Module):
    
    def __init__(self, resample_freq: int = 16_000, n_fft: int=1024, n_mel: int = 256, stretch_factor: float = 0.8, req_mask: int = 5, time_mask: int = 5) -> None:
        super().__init__()
        self.resample_freq: int = resample_freq
        self.current_input_freq: int | None = None
        self.resampler: Resample = None

        self.spec: Spectrogram = Spectrogram(n_fft=n_fft, power=2)
        self.spec_aug: torch.nn.Sequential = torch.nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True, n_freq=n_fft // 2 + 1),
            ComplexToPower(),
            FrequencyMasking(freq_mask_param=req_mask),
            TimeMasking(time_mask_param=time_mask),
        )
        self.mel_scale: MelScale = MelScale(
            n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1
        )
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_min: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x-x_min)/(x_max-x_min+1e-8)     

    def forward(self, waveform: torch.Tensor, input_freq: int) -> torch.Tensor:
        if input_freq != self.current_input_freq:
            self.current_input_freq = input_freq
            if input_freq != self.resample_freq:
                self.resampler = Resample(orig_freq=input_freq, new_freq=self.resample_freq).to(waveform.device)
            else:
                self.resampler = None

        resampled = self.resampler(waveform) if self.resampler is not None else waveform
        spec = self.spec(resampled)
        spec = self.spec_aug(spec)
        return self._normalize(self.mel_scale(spec))
    
    
class MFCCPipeline(torch.nn.Module):
    
    def __init__(self, resample_freq: int = 16_000, n_mfcc: int = 20, n_fft: int=1024, n_mel: int = 256) -> None:
        super().__init__()
        self.resample_freq: int = resample_freq
        self.current_input_freq: int | None = None
        self.resampler: Resample | None = None
        
        self.mfcc = MFCC(
            sample_rate=resample_freq,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": n_fft // 2,
                "n_mels": n_mel,
            },
        )
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_min: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x-x_min)/(x_max-x_min+1e-8)
    
    
    def forward(self, waveform: torch.Tensor, input_freq: int) -> torch.Tensor:
        if input_freq != self.current_input_freq:
            self.current_input_freq = input_freq
            if input_freq != self.resample_freq:
                self.resampler = Resample(orig_freq=input_freq, new_freq=self.resample_freq).to(waveform.device)
            else:
                self.resampler = None

        resampled = self.resampler(waveform) if self.resampler is not None else waveform
        
        return self._normalize(self.mfcc(resampled))