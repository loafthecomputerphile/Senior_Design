from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable

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
    AmplitudeToDB,
    GriffinLim
)

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "segment_cough", "compute_SNR", "MelSpectrogramPipeline", 
    "segment_cough_robust", "SpecAugmentations", "MFCCPipeline", 
    "ComombinedSpectMFCCPipeline"
]

'''
SOURCE: 
Orlandic, L., Teijeiro, T. & Atienza, D. The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms. Sci Data 8, 156 (2021). https://doi.org/10.1038/s41597-021-00937-4
'''


#Use old segmentation
def segment_cough(x: npt.NDArray,fs: float, cough_padding: float = 0.2, min_cough_len: float =0.2, th_l_multiplier: float = 0.1, th_h_multiplier: int = 2, tol_multiplier: float = 0.01) -> tuple[list[npt.NDArray], npt.NDArray]:
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
    tolerance: float = round(tol_multiplier*fs)
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




def segment_cough_robust(
    x: npt.NDArray,
    fs: float, 
    cough_padding: float = 0.2, 
    min_cough_len: float = 0.2, 
    th_l_multiplier: float = 0.1, 
    th_h_multiplier: float = 2.0
) -> tuple[list[npt.NDArray], npt.NDArray]:
    
    cough_mask: npt.NDArray = np.zeros(len(x), dtype=bool)
    power = x**2
    
    # 1. FIX: Use the 90th percentile of energy instead of pure mean to isolate active bursts
    # This prevents long silent stretches from completely deflating your thresholds
    active_rms = np.sqrt(np.percentile(power, 90))
    
    # Safety fallback: if the audio file is completely near-silent, use standard mean
    if active_rms < 1e-5:
        active_rms = np.sqrt(np.mean(power))
        
    seg_th_l: float = th_l_multiplier * active_rms
    seg_th_h: float = th_h_multiplier * active_rms
    
    # 2. Setup structural variables as strict integers
    padding: int = int(round(fs * cough_padding))
    min_cough_samples: int = int(round(fs * min_cough_len))
    tolerance: int = int(round(0.01 * fs))
    
    coughSegments: list = []
    cough_start: int = 0
    cough_end: int = 0
    cough_in_progress: bool = False
    below_th_counter: int = 0
    
    # 3. Step through the timeline
    for i, sample in enumerate(power):
        if cough_in_progress:
            if sample < seg_th_l:
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = min(i + padding, len(x) - 1)
                    cough_in_progress = False
                    
                    segment_length = cough_end + 1 - cough_start
                    if segment_length >= min_cough_samples:
                        coughSegments.append(x[cough_start:cough_end + 1])
                        cough_mask[cough_start:cough_end + 1] = True
            else:
                below_th_counter = 0
                
            if i == (len(x) - 1) and cough_in_progress:
                cough_end = i
                cough_in_progress = False
                segment_length = cough_end + 1 - cough_start
                if segment_length >= min_cough_samples:
                    coughSegments.append(x[cough_start:cough_end + 1])
                    cough_mask[cough_start:cough_end + 1] = True
        else:
            if sample > seg_th_h:
                cough_start = max(0, i - padding)
                cough_in_progress = True
                below_th_counter = 0
                
    return coughSegments, cough_mask




def compute_SNR(x: npt.NDArray, fs: float) -> float | int:
    """Compute the Signal-to-Noise ratio of the audio signal x (np.array) with sampling frequency fs (float)"""
    _, cough_mask = segment_cough(x,fs)
    RMS_signal: float | int = 0 if len(x[cough_mask])==0 else np.sqrt(np.mean(np.square(x[cough_mask])))
    RMS_noise: float = np.sqrt(np.mean(np.square(x[~cough_mask])))
    SNR: float | int = 0 if (RMS_signal==0 or np.isnan(RMS_noise)) else 20*np.log10(RMS_signal/RMS_noise)
    return SNR


class ComplexToPower(torch.nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs().pow(2)
    
    
class ReverseSpectrogram:
    
    aug_spec: torch.Tensor
    n_fft: int

    def post_aug_waveform(self, waveform: torch.Tensor, n_iter: int | None) -> torch.Tensor:
        return GriffinLim(
            n_fft=self.n_fft, hop_length=self.n_fft//2,
            n_iter=n_iter if n_iter is not None else 32
        )(waveform)


class EnforceFixedLength(torch.nn.Module):
    
    def __init__(self, fs: float, max_duration: float = 1.0) -> None:
        super().__init__()
        self.target_samples: int = int(max_duration * fs)
        
    def forward(self, segment: torch.Tensor) -> torch.Tensor:
        current_samples: torch.Tensor = segment.shape[-1]
        segment = segment[..., :self.target_samples]
        
        padding_needed: int = max(0, self.target_samples - current_samples)
        
        if padding_needed > 0:
            return F.pad(segment, (0, padding_needed), mode='constant', value=0.0)
            
        return segment


    
class BaseSpecAug(torch.nn.Module):
    
    _n_freq_op: Callable[[int], int] = lambda x: x
    _n_freq: int | None = None
    
    @property
    def n_freq(self) -> int:
        if self._n_freq is None:
            raise ValueError()
        return self._n_freq_op(self._n_freq)

    @n_freq.setter
    def n_freq(self, n: int) -> None:
        self._n_freq = n
        
        

    
class SpecAugmentations(BaseSpecAug):
    
    def __init__(self, stretch_factor: float = 0.8, req_mask: int = 5, time_mask: int = 5, transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        
        self._n_freq_op: Callable[[int], int] = lambda x: x // 2 + 1
            
        self.transform: Callable[[torch.Tensor], torch.Tensor] | None = transform
        self.pipeline: Callable[..., nn.Sequential] = lambda x: nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True, n_freq=self.n_freq),
            ComplexToPower(),
            FrequencyMasking(freq_mask_param=req_mask),
            TimeMasking(time_mask_param=time_mask),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform:
            x = self.transform(x)
        return self.pipeline(1)(x)
        
        
        
    
class MelSpectrogramPipeline(torch.nn.Module, ReverseSpectrogram):
    
    def __init__(self, resample_freq: int = 16_000, n_fft: int = 1024, n_mel: int = 256, transforms: BaseSpecAug | None = None) -> None:
        super().__init__()
        self.resample_freq: int = resample_freq
        self.current_input_freq: int | None = None
        self.resampler: Resample = None
        self.transforms: BaseSpecAug | None = transforms
        self.n_fft = n_fft
        if self.transforms:
            self.transforms.n_freq = n_fft

        self.spec: Spectrogram = Spectrogram(n_fft=n_fft, power=2)
        self.mel_spec: MelScale = MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)
        
    def make_transform_waveform(self, waveform: torch.Tensor, n_iter: int | None = None) -> torch.Tensor:
        return self.post_aug_waveform(
            self.transforms(self.spec(waveform)), n_iter
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_min: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x-x_min)/(x_max-x_min+1e-8)     

    def forward(self, waveform: torch.Tensor, input_freq: int) -> torch.Tensor:
        spec: torch.Tensor = self.spec(waveform)
        if self.transforms:
            spec = self.transforms(spec)
        return self._normalize(self.mel_spec(spec))
    
    
class MFCCPipeline(torch.nn.Module, ReverseSpectrogram):
    
    def __init__(self, n_mfcc: int = 20, n_fft: int=1024, n_mel: int = 256, transforms: BaseSpecAug | None = None) -> None:
        super().__init__()
        self.transforms: BaseSpecAug | None = transforms
        self.n_fft = n_fft
        if self.transforms:
            self.transforms.n_freq = n_fft
            
        self.amp_to_db: AmplitudeToDB = AmplitudeToDB()
                
        self.spec: Spectrogram = Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale: MelScale = MelScale(
            n_mels=n_mel, n_stft=n_fft // 2 + 1
        )
        
        self.dct_mat: torch.Tensor = F.create_dct(
            n_mfcc=n_mfcc, n_mels=n_mel, norm="ortho"
        )
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_min: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x-x_min)/(x_max-x_min+1e-8)
    
    def make_transform_waveform(self, waveform: torch.Tensor, n_iter: int | None = None) -> torch.Tensor:
        return self.post_aug_waveform(
            self.transforms(self.spec(waveform)), n_iter
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec: torch.Tensor = self.spec(waveform)
        if self.transforms:
            spec = self.transforms(spec)
        spec = self.mel_scale(spec)
        spec = self.amp_to_db(spec)
        
        mfcc: torch.Tensor = torch.matmul(
            spec.transpose(-1, -2), self.dct_mat
        ).transpose(-1, -2)
        return self._normalize(mfcc)
    

def n_fft_n_freq_op(n: int) -> int:
    return n // 2 +1


class ComombinedSpectMFCCPipeline(torch.nn.Module, ReverseSpectrogram):
    
    def __init__(self, n_mfcc: int = 20, n_fft: int=1024, n_mel: int = 256, transforms: BaseSpecAug | None = None) -> None:
        super().__init__()
        self.current_input_freq: int | None = None
        self.transforms: BaseSpecAug | None = transforms
        self.n_fft = n_fft
        if self.transforms:
            self.transforms.n_freq = n_fft
        
        
        self.amp_to_db: AmplitudeToDB = AmplitudeToDB()
        
        self.spec: Spectrogram = Spectrogram(n_fft=n_fft, power=2)
        self.mel_scale: MelScale = MelScale(
            n_mels=n_mel, n_stft=n_fft // 2 + 1
        )
        
        self.dct_mat = F.create_dct(
            n_mfcc=n_mfcc, n_mels=n_mel, norm="ortho"
        )
        
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_val: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val + 1e-8)      
        
    def make_transform_waveform(self, waveform: torch.Tensor, n_iter: int | None = None) -> torch.Tensor:
        return self.post_aug_waveform(
            self.transforms(self.spec(waveform)), n_iter
        )
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        
        spec: torch.Tensor = self.spec(waveform)
        if self.transforms:
            self.aug_spec = self.transforms(spec)
        mel = self.mel_scale(self.aug_spec)
        spec = self.amp_to_db(mel)
        
        mfcc: torch.Tensor = torch.matmul(
            spec.transpose(-1, -2), self.dct_mat
        ).transpose(-1, -2)
        
        norm_mfcc: torch.Tensor = self._normalize(mfcc) 
        norm_spec: torch.Tensor = self._normalize(mel)
        
        return torch.cat([norm_spec, norm_mfcc], dim=-2)