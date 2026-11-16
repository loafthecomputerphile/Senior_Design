# Slide 7b — The Three Models We Trained
**(paste screenshot of `base_nn.py` model classes here — `project_notebooks/modules/nn/base_nn.py:79`)**

**On-slide text:**

| Model | Idea | Params |
|---|---|---|
| **Default CNN** | 3× DepthwiseSeparable blocks → GAP → small head | ~0.1M |
| **Branched CNN** | two parallel branches (log-Mel + MFCC) → pool → merge → head | ~0.15M |
| **ResNet50** | ImageNet-pretrained backbone, new linear head | ~23M (head retrained) |

**Verbally:**
- **Default CNN:** the workhorse. DepthwiseSeparable blocks (3→32→64→128 channels) extract features, then Global Average Pooling squashes every feature map to one number — the classifier head is tiny (~33k params), so there's nothing to memorize with. Input: one image-like representation (wave or spectrogram).
- **Branched CNN:** the winner. Coughs carry two complementary views: the spectrogram (energy over time×frequency) and MFCCs (perceptual coefficients). Each gets its own conv branch, then GAP on both — this also lets the branches have different H×W sizes — then concat (128→128) and the same small head. Each modality gets dedicated filters instead of sharing a single first layer.
- **ResNet50:** the transfer-learning baseline. ImageNet weights, frozen or fine-tuned, last layer swapped for our classes. Result: overfits instantly (train loss 0.13→0.04 in 20 epochs while val loss climbs 0.13→0.46) — features tuned for photos of cats don't transfer to cough acoustics on small data.
- All three use the same **FocalLoss** (class-weighted, gamma=2) so the minority class isn't ignored, and LeakyReLU throughout (no dead neurons on quiet audio).
- Same training loop for all: seeded split, early stopping on val loss, checkpointing on best val loss.

One-liner: "three architectures, one honest split — the purpose-built small CNN beat the 23M-parameter pretrained giant."
