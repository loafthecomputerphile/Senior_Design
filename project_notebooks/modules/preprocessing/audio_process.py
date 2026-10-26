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
    GriffinLim, SpeedPerturbation, ComputeDeltas
)

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "segment_cough", "compute_SNR", "MelSpectrogramPipeline", 
    "segment_cough_robust", "SpecAugmentations", "MFCCPipeline", 
    "SpectMFCCPipeline", "EnforceFixedLength", "segment_cough_fast"
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




def _runs(cond: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Start indices and lengths of every True run in a boolean array."""
    padded: npt.NDArray = np.concatenate(([False], cond, [False])).view(np.int8)
    edges: npt.NDArray = np.flatnonzero(np.diff(padded))
    return edges[0::2], edges[1::2] - edges[0::2]


def segment_cough_fast(x: npt.NDArray,fs: float, cough_padding: float = 0.2, min_cough_len: float =0.2, th_l_multiplier: float = 0.1, th_h_multiplier: int = 2, tol_multiplier: float = 0.01) -> tuple[list[npt.NDArray], npt.NDArray]:
    """Vectorized equivalent of segment_cough: same thresholds, padding, min-length and
    end-of-file rules, but O(#coughs) Python steps instead of O(#samples).

    ponytail: exact-boundary parity is asserted by test_segment_cough_fast.py on real files."""
    power: npt.NDArray = np.square(x)
    rms: np.float64 = np.sqrt(np.mean(power))
    seg_th_l: float = th_l_multiplier * rms
    seg_th_h: float = th_h_multiplier * rms

    padding: int = round(fs * cough_padding)
    min_cough_samples: int = round(fs * min_cough_len)
    tolerance: int = round(tol_multiplier * fs)
    n: int = len(x)

    # Cough ends: the original loop trips once `tolerance + 1` consecutive samples sit
    # below the low threshold, i.e. at index run_start + tolerance.
    below_starts, below_lens = _runs(power < seg_th_l)

    # Cough starts: rising edges above the high threshold
    above_starts: npt.NDArray = _runs(power > seg_th_h)[0]

    coughSegments: list = []
    cough_mask: npt.NDArray = np.zeros(n, dtype=bool)
    # Replicates two quirks of the original per-sample loop:
    #  - below_th_counter is never reset when a cough starts, only by samples >= th_l,
    #    so a cough whose very next sample is below th_l ends immediately.
    #  - a cough stops being "in progress" at the trigger index (not the padded end),
    #    so the next cough may start inside the previous one's padded tail.
    carry: int = 0          # counter value carried across coughs (frozen between coughs)
    active_until: int = -1  # cough still in progress through this index
    ri: int = 0             # cursor into below-runs
    for s in above_starts:
        if s <= active_until:
            continue
        while ri < len(below_starts) and below_starts[ri] <= s:
            ri += 1
        trigger: int | None = None
        j: int = ri
        while j < len(below_starts):
            rs: int = below_starts[j]
            rl: int = below_lens[j]
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
            # Cough persists to end of file; the original only appends it if the final
            # sample is at/above the low threshold and the start isn't the last sample.
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
            cough_mask[cough_start:cough_end + 1] = True

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

    def __init__(self, fs: float, max_duration: float = 1.0, to_numpy: bool = False) -> None:
        super().__init__()

        self.target_samples: int = int(max_duration * fs)
        self.to_numpy: bool = to_numpy

    def forward(self, segment: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        current_samples: int = segment.shape[-1]

        if isinstance(segment, np.ndarray):
            output: np.ndarray = np.zeros(self.target_samples, dtype=segment.dtype)
            samples_to_copy: int = min(current_samples, self.target_samples)
            output[:samples_to_copy] = segment[:samples_to_copy]
            return output

        output: torch.Tensor = torch.zeros(
            self.target_samples,
            dtype=segment.dtype,
            device=segment.device
        )

        samples_to_copy: int = min(current_samples, self.target_samples)
        output[:samples_to_copy] = segment[:samples_to_copy]

        if self.to_numpy:
            return output.numpy()

        return output

    
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
        self.p: float = 0.5
        # Same per-class augment probabilities as WaveformAugmentor (healthy, COVID-19, symptomatic)
        self.augment_probs: dict[int, float] = {0: 0.40, 1: 0.90, 2: 0.70}

        self.pipeline: Callable[..., nn.Sequential] = lambda x: nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True, n_freq=self.n_freq),
            ComplexToPower(),
        )

        self.freq_mask: FrequencyMasking = FrequencyMasking(freq_mask_param=req_mask)
        self.time_mask: TimeMasking = TimeMasking(time_mask_param=time_mask)

    def _should_apply(self, p: float | None = None) -> bool:
        return bool(torch.rand(()) < (self.p if p is None else p))

    def forward(self, x: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
        if self.transform:
            x = self.transform(x)
        # TimeStretch + power conversion always run (the downstream mel step needs a power spectrogram)
        x = self.pipeline(1)(x)
        if not self.training:
            return x
        if label is None:
            # No labels available: fall back to a single coin flip for the whole batch
            if self._should_apply():
                x = self.time_mask(self.freq_mask(x))
            return x
        # Per-sample gating by class, same probabilities as WaveformAugmentor
        for i in range(x.shape[0]):
            if torch.rand(()) >= self.augment_probs[int(label[i])]:
                continue
            if not self._should_apply():
                continue
            x[i] = self.time_mask(self.freq_mask(x[i : i + 1]))[0]
        return x
        
        
        
    
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

        self.spec: Spectrogram = Spectrogram(n_fft=n_fft, power=None if self.transforms else 2)
        self.mel_spec: MelScale = MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)
        
    def make_transform_waveform(self, waveform: torch.Tensor, n_iter: int | None = None) -> torch.Tensor:
        return self.post_aug_waveform(
            self.transforms(self.spec(waveform)), n_iter
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_min: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x-x_min)/(x_max-x_min+1e-8)     

    def forward(self, waveform: torch.Tensor, input_freq: int, label: torch.Tensor | None = None) -> torch.Tensor:
        spec: torch.Tensor = self.spec(waveform)
        if self.transforms:
            spec = self.transforms(spec, label)
        return self._normalize(self.mel_spec(spec))
    
    
class MFCCPipeline(torch.nn.Module, ReverseSpectrogram):
    
    def __init__(self, n_mfcc: int = 20, n_fft: int=1024, n_mel: int = 256, transforms: BaseSpecAug | None = None) -> None:
        super().__init__()
        self.transforms: BaseSpecAug | None = transforms
        self.n_fft = n_fft
        if self.transforms:
            self.transforms.n_freq = n_fft
            
        self.amp_to_db: AmplitudeToDB = AmplitudeToDB()
                
        self.spec: Spectrogram = Spectrogram(n_fft=n_fft, power=None if self.transforms else 2)
        self.mel_scale: MelScale = MelScale(
            n_mels=n_mel, n_stft=n_fft // 2 + 1
        )
        
        self.register_buffer(
            "dct_mat",
            F.create_dct(
                n_mfcc=n_mfcc,
                n_mels=n_mel,
                norm="ortho",
            ),
        )
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x_min: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x-x_min)/(x_max-x_min+1e-8)
    
    def make_transform_waveform(self, waveform: torch.Tensor, n_iter: int | None = None) -> torch.Tensor:
        return self.post_aug_waveform(
            self.transforms(self.spec(waveform)), n_iter
        )
        
    def normalize_and_clamp_mfcc(self, waveform):
        spec: torch.Tensor = self.spec(waveform)
        if self.transforms:
            spec = self.transforms(spec)
        spec = self.mel_scale(spec)
        spec = self.amp_to_db(spec)
        
        mfcc_tensor: torch.Tensor = torch.matmul(
            spec.transpose(-1, -2), self.dct_mat
        ).transpose(-1, -2)
            
        
        # 1. Z-Score (CMVN) along the time axis (dim=-1)
        mean = mfcc_tensor.mean(dim=-1, keepdim=True)
        std = mfcc_tensor.std(dim=-1, keepdim=True)
        mfcc_norm = (mfcc_tensor - mean) / (std + 1e-8)

        # 2. Clamp extreme acoustic spikes at +/- 3 standard deviations
        mfcc_clamped = torch.clamp(mfcc_norm, min=-3.0, max=3.0)
        
        # 3. Min-Max scale the clamped boundaries cleanly to 0.0 - 1.0
        img_scaled = (mfcc_clamped - (-3.0)) / (3.0 - (-3.0))

        return img_scaled
    
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


class SpectMFCCPipeline(torch.nn.Module, ReverseSpectrogram):
    
    def __init__(self, n_mfcc: int = 40, n_fft: int = 1024, n_mel: int = 64, combind: bool = False, transforms: BaseSpecAug | None = None) -> None:
        super().__init__()
        self.current_input_freq: int | None = None
        self.transforms: BaseSpecAug | None = transforms
        self.n_fft = n_fft
        self.combind: bool = combind
        if self.transforms:
            self.transforms.n_freq = n_fft
        
        self.amp_to_db: AmplitudeToDB = AmplitudeToDB()
        
        self.spec: Spectrogram = Spectrogram(n_fft=n_fft, power=None if self.transforms else 2)
        self.mel_scale: MelScale = MelScale(
            n_mels=n_mel, n_stft=n_fft // 2 + 1
        )
        
        self.register_buffer(
            "dct_mat",
            F.create_dct(
                n_mfcc=n_mfcc,
                n_mels=n_mel,
                norm="ortho",
            ),
        )
        
        self.compute_deltas = ComputeDeltas()
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_val: torch.NumberType = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val: torch.NumberType = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (x - min_val) / (max_val - min_val + 1e-8)      
        
    def make_transform_waveform(self, waveform: torch.Tensor, n_iter: int | None = None) -> torch.Tensor:
        return self.post_aug_waveform(
            self.transforms(self.spec(waveform)), n_iter
        )
        
    def normalize_and_clamp_mfcc(self, mfcc_tensor):
        # 1. Z-Score (CMVN) along the time axis (dim=-1)
        mean = mfcc_tensor.mean(dim=-1, keepdim=True)
        std = mfcc_tensor.std(dim=-1, keepdim=True)
        mfcc_norm = (mfcc_tensor - mean) / (std + 1e-8)

        # 2. Clamp extreme acoustic spikes at +/- 3 standard deviations
        mfcc_clamped = torch.clamp(mfcc_norm, min=-3.0, max=3.0)
        
        # 3. Min-Max scale the clamped boundaries cleanly to 0.0 - 1.0
        img_scaled = (mfcc_clamped - (-3.0)) / (3.0 - (-3.0))

        return img_scaled
        
    def forward(self, waveform: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:

        spec: torch.Tensor = self.spec(waveform)
        if self.transforms:
            spec = self.transforms(spec, label)
        mel = self.mel_scale(spec)
        spec = self.amp_to_db(mel)
        
        mfcc: torch.Tensor = torch.matmul(
            spec.transpose(-1, -2), self.dct_mat
        ).transpose(-1, -2)
        

        norm_mfcc: torch.Tensor = self.normalize_and_clamp_mfcc(mfcc)
        norm_spec: torch.Tensor = self._normalize(mel)
        
        three_chan_spec = self.to_three_channel_images(norm_spec, norm_spec)
        three_chan_mfcc = self.to_three_channel_images(norm_mfcc, norm_mfcc)

        if self.combind:
            # Sticking them vertically (height axis) as asked in your snippet
            return torch.cat([three_chan_spec, three_chan_mfcc], dim=-2)
        
        return three_chan_spec, three_chan_mfcc
    
    def to_three_channel_images(self, norm_spec: torch.Tensor, norm_mfcc: torch.Tensor) -> torch.Tensor:
        """
        Combines 1-channel Mel Spectrogram and 1-channel MFCC into a 
        standard 3-channel image format using the Mel's Delta stream.
        """
        # Ensure tensors have an explicit channel dimension: [..., 1, height, width]
        if norm_spec.ndim == 2:  # [height, width] -> [1, height, width]
            norm_spec = norm_spec.unsqueeze(0)
            norm_mfcc = norm_mfcc.unsqueeze(0)
        elif norm_spec.ndim == 3 and norm_spec.shape[0] != 1: # Batched [B, H, W] -> [B, 1, H, W]
            norm_spec = norm_spec.unsqueeze(1)
            norm_mfcc = norm_mfcc.unsqueeze(1)

        # 1. Compute the Delta velocity mapping from the Mel matrix
        delta_map = self.compute_deltas(norm_spec)
        
        # 2. Min-max scale the delta layer to normalize its distribution to 0.0 - 1.0
        delta_min = delta_map.amin(dim=(-2, -1), keepdim=True)
        delta_max = delta_map.amax(dim=(-2, -1), keepdim=True)
        norm_delta = (delta_map - delta_min) / (delta_max - delta_min + 1e-8)

        # 3. Concatenate along the channel axis (dim=-3)
        # Channel 0: Mel Spectrogram | Channel 1: MFCC | Channel 2: Mel Delta
        three_channel_img = torch.cat([norm_spec, norm_mfcc, norm_delta], dim=-3)
        return three_channel_img
    
    
    
class WaveformAugmentor(nn.Module):

    def __init__(self, sample_rate: int, p: float = 0.5) -> None:
        super().__init__()

        self.sample_rate: int = sample_rate
        self.p: float = p

        self.speed: SpeedPerturbation = SpeedPerturbation(
            orig_freq=sample_rate,
            factors=[0.95, 1.0, 1.05],
        )
        
        self.augment_probs: dict[int, float] = {
            0: 0.40,  # healthy
            1: 0.90,  # COVID-19
            2: 0.70,  # symptomatic
        }

    def _should_apply(self, p: float | None = None) -> bool:
        return bool(
            torch.rand(()) < (self.p if p is None else p)
        )

    def _gain(self, x: torch.Tensor) -> torch.Tensor:
        gain: torch.Tensor = torch.empty(
            (), device=x.device, dtype=x.dtype,
        ).uniform_(0.8, 1.2)

        return x * gain

    def _noise(self, x: torch.Tensor) -> torch.Tensor:
        rms: torch.Tensor = torch.sqrt(
            x.square().mean(dim=-1, keepdim=True) + 1e-8
        )

        noise_level: torch.Tensor = torch.empty(
            (), device=x.device, dtype=x.dtype,
        ).uniform_(0.002, 0.01)

        return x + torch.randn_like(x) * rms * noise_level

    def _time_shift(self, x: torch.Tensor) -> torch.Tensor:
        max_shift: int = int(self.sample_rate * 0.05)

        shift: torch.Tensor = torch.randint(
            -max_shift, max_shift + 1, (),
        ).item()

        if shift == 0:
            return x

        output: torch.Tensor = torch.zeros_like(x)

        if shift > 0:
            output[..., shift:] = x[..., :-shift]
        else:
            shift = abs(shift)
            output[..., :-shift] = x[..., shift:]

        return output

    def _speed_perturb(self, x: torch.Tensor) -> torch.Tensor:
        original_length: int = x.shape[-1]

        x, _ = self.speed(x)

        # Restore the original fixed length.
        if x.shape[-1] > original_length:
            x = x[..., :original_length]
        elif x.shape[-1] < original_length:
            x = torch.nn.functional.pad(
                x, (0, original_length - x.shape[-1]),
            )

        return x

    def forward(self, waveform: torch.Tensor, label: torch.NumberType) -> torch.Tensor:

        if not self.training:
            return waveform
        
        if torch.rand(()) > self.augment_probs[int(label)]:
            return waveform

        if self._should_apply():
            waveform = self._gain(waveform)

        if self._should_apply(0.3):
            waveform = self._noise(waveform)

        if self._should_apply():
            waveform = self._time_shift(waveform)

        return waveform