# Slide 5 — Stage 2: COVID vs Symptomatic (the win)
**Images (side by side):**
- `project_notebooks/figures/trained_branched_stage2_covid_vs_symp_both_cm.png`
- `project_notebooks/figures/stage2_covid_vs_symp_test_curves.png`

One table, small (best **validation** AUC, from `project_notebooks/logs/trained_models_experiment_2026-10-25.txt`):

| | Val AUC | Test ROC-AUC | COVID recall |
|---|---|---|---|
| **Branched** | **0.623** | **0.637** | **0.61 (81% @ 0.5)** |
| Default | 0.532 | 0.599 | 0.47 |
| ResNet50 | 0.627* | 0.584 | 0.50 |

\* ResNet's val AUC peaked at epoch 6 and didn't hold on test — classic overfit; early stopping caught it at epoch 23.

Verbally: branched wins, pretrained ResNet overfit (train 0.13→0.04, val 0.13→0.46), misses go the clinically safe direction, ~0.70 AUC is the literature ceiling for honest splits.
