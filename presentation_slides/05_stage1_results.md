# Slide 5 — Stage 1: Sick vs Healthy
**Images (side by side):**
- `project_notebooks/figures/trained_default__covid_symp_vs_healthy__both_history.png`
- `project_notebooks/figures/trained_resnet__covid_symp_vs_healthy__both_history.png`

Big number: **Val AUC ~0.56 — all three architectures converge to the same ceiling**

Small table:

| Model | Val AUC | Epochs |
|---|---|---|
| Default | 0.57 | 37 (early stop) |
| Branched | 0.56 | 54 |
| ResNet50 | — | 23 (overfit, early-stopped) |

Verbally: no more flatline — metrics move epoch to epoch (F1 > 0.51, PR-AUC ~0.30), unlike the 3-class runs. But models still learn the class prior more than acoustic signal. ResNet diverges immediately: train loss 0.13→0.04 while val loss climbs 0.13→0.46 — pretrained features don't fit cough audio. This ceiling is why the cascade bottleneck (next slide) exists.
