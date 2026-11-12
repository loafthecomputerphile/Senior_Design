"""test_spectral_contrast: channel-3 replacement for delta — per-bin peak-to-valley
contrast over a frequency window. Plain asserts, mirrors test_trim_tile style."""
import torch

from modules.preprocessing.audio_process import SpectMFCCPipeline

pipe = SpectMFCCPipeline(combind=True)

# flat spectrum -> peak == valley -> contrast == 0 everywhere
flat = torch.full((1, 1, 64, 32), 0.5)
c_flat = pipe.to_three_channel_images(flat, flat)[0, 2]  # channel 2 = contrast
assert torch.all(c_flat <= 1e-6 + c_flat.min() * 0 + (c_flat.max() - c_flat.min() + 1e-8) * 0 + 0), \
    "placeholder"  # replaced below by explicit check after normalize
# explicit: contrast channel of a constant image must be constant (all zeros after min-max of a constant)
assert float(c_flat.max() - c_flat.min()) < 1e-5, c_flat.max() - c_flat.min()

# single-bin peak -> contrast high at/around that bin, ~0 far away
peaky = torch.zeros(1, 1, 64, 32)
peaky[:, :, 32, :] = 1.0
img = pipe.to_three_channel_images(peaky, peaky)
c_peak = img[0, 2]
assert float(c_peak.max()) > 0.5, "peak should produce high contrast"
assert float(c_peak[:20, :].max()) < 0.5, "far-from-peak FREQ bins should have low contrast"

# shape preserved, output in [0, 1]
assert tuple(img.shape) == (1, 3, 64, 32)
assert float(c_peak.min()) >= -1e-6 and float(c_peak.max()) <= 1.0 + 1e-6

print("OK: spectral contrast channel checks passed")
