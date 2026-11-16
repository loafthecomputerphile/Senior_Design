# Slide 6 — Our Conv Block: Two Cheap Steps Beat One Expensive One
**(paste screenshot of `DepthwiseSeparableBlock` — `project_notebooks/modules/nn/base_nn.py:56`)**

**On-slide text:**

A normal convolution does 2 jobs at once:
- scans the image with a 3×3 window (**spatial patterns**)
- mixes all input channels together (**channel mixing**)

Our block splits them:
1. **Depthwise 3×3** — one small filter per channel, spatial patterns only
2. **Pointwise 1×1** — mixes the filtered channels, sets the output size

```
Cost, 3→32 channels:   normal = 864 weights   |   ours = 123   (~7× fewer)
Cost, 64→128 channels: normal = 73,728      |   ours = 8,768 (~8× fewer)
```

Fewer parameters → less capacity to memorize → better generalization on small data.

Verbally:
- The savings grow with depth: a normal conv multiplies the channel counts together; we add them instead
- Each block: depthwise → BatchNorm → ReLU → pointwise → BatchNorm → ReLU → MaxPool (halve size, keep strongest responses)
- Stacking 3 blocks builds the hierarchy: raw acoustic events → cough motifs → segment-level signature
- This is the MobileNet trick — decompose-then-recombine loses almost nothing in practice, and cough spectrograms don't need heavy machinery early on
- Directly relevant to our story: the old 3-class model overfit; this block is part of the fix

One-liner: "we never need spatial filtering and channel mixing at the same time, so we never pay for doing both at once."
