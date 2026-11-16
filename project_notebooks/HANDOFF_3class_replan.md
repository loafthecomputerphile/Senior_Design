# Handoff: 3-class risk model — replan

Date: 2026-11-10
Context: `neural_network_3class.ipynb`, `final_processed_dataset.npz`, logs in `logs_3class/`, metrics in `three_class__*_experiment_stats.csv`.

## Diagnosis (evidence-based, do not re-litigate)

- **Val set is 57 segments.** Every val_acc in the CSVs is a multiple of 1.754% = n/57. With COVID at ~5%, val holds ~3 COVID segments. Val AUC swinging 0.48–0.60 between epochs is noise from 3 positives. Current single-split metrics are unmeasurable — all past "didn't work" conclusions were read off a broken gauge.
- **No real signal found yet.** Val AUC never exceeds 0.60 in 60 epochs (chance = 0.5). Val loss doubles 0.44 → 1.4 after ~epoch 15 while train acc climbs to 63% — memorization, no transfer.
- **Current val_acc (~43%) is worse than the majority-class baseline (74% always-healthy).** Extreme class weighting (1/freq, normalized ×3) pushed the model off the baseline into noise. Weighted sampling was not "broken" — it was unmeasurable + too aggressive.
- **Data scale:** 9,480 healthy / 2,593 symptomatic / 623 COVID **recordings** (segments: 31,522 / 8,724 / 2,118). Effective sample size = recording count, not segments (sibling segments correlated).
- **Network is not the main problem.** From-scratch CNN at 623 COVID recordings is expected to plateau. (Keep pretrained-feature option open, not in scope for this pass.)

## Plan for next session

### Scope rule (user decision — do not deviate)
The work targets **`neural_network.ipynb` (the default notebook)**. **Convert it in place** — no appended-only cells, no halfway edits:
- Every training loop in it (`train_binary` cell, cascade stage-1/stage-2 cells, multi-model `train_model` cell) is modified to the new scheme.
- Binary one-vs-rest experiments are **replaced** by the ordinal model; dead binary cells are removed, not left behind.
- Cascade stage-1/stage-2 roles collapse into a single ordinal model per aug config.
- Multi-model block (DefaultCNN / BranchedCoughCNN / ResNet50) trains the ordinal model instead.
- Eval/plot cells are rewritten to recording-level OOF metrics, ordinal confusion matrix, and rank-MAE.
- `neural_network_3class.ipynb` stays as-is for now (it's the old 3-class baseline).

### 1. Fix the gauge first: recording-level CV harness
- Use `stratified_recording_folds` (in `modules/dataset/make_dataset.py`) for 5-fold CV split **by recording**, inside `neural_network.ipynb`.
- Pool all folds' out-of-fold predictions.
- **Aggregate segment probabilities → recording level** (mean prob per recording) before computing metrics.
- Report macro-F1, macro AUC, per-class recall at recording level across folds.
- This replaces the 80/10/10 single split for all evaluation. One-time build; every later decision uses it.

### 2. Inverse-frequency class weighting (user decision — overrides earlier sqrt suggestion)
- Use **inverse frequency** weights (`1/counts`, normalized) in the loss, measured at recording level where practical.
- Evaluate it *with the CV harness from step 1* — past "weighted sampling didn't work" conclusions are unreliable (measured on the 57-segment val). Keep inverse, but drop it if recording-level CV shows it below unweighted.
- No SMOTE (meaningless on 1-second waveforms — interpolated coughs aren't coughs). Do not revisit.

### 3. Reframe as ordinal risk: low < med < high
Target mapping: healthy = low, symptomatic = med, COVID = high.
- Primary approach: **ordinal loss (CORAL-style cumulative logits)** — P(risk > low), P(risk > med) with shared weights.
- Fallback if CORAL is fiddly in this codebase: **two-stage** — (a) binary sick-vs-healthy (12,147 vs 9,480 — much more learnable), (b) COVID-vs-symptomatic only among predicted-sick. Report stage (b) uncertainty honestly.
- Report ordered metrics: macro-F1 plus MAE of the ordinal distance (|pred − true| in rank steps) — being off by one rank is the error that matters.

## Guardrails
- Stop training around epoch ~15 unless val loss still falling (past runs: val loss doubles after that).
- Success bar for pretrained-feature-free pass: recording-level macro AUC meaningfully > 0.5 on CV. If it lands 0.55–0.6, that's the honest finding, not a bug — next escalation is frozen YAMNet/PANNs embeddings + small head (`yamnet_cough_filter.ipynb` already exists).
- Keep seeds/checkpoint resume behavior as-is (notebook already resumes from disk without wiping).

## File pointers
- Default notebook (the conversion target, 27 cells): `neural_network.ipynb`
  - cell 0: imports (already has `stratified_recording_folds`)
  - cell 4: experiment setup (`num_epochs=60`, `SEED=42`, augmentor imports)
  - cell 6: recording-wise split → becomes CV fold builder
  - cell 7: per-class inverse-frequency weights (keep, rewire to ordinal)
  - cell 9: `train_binary` (~223 lines) → becomes `train_ordinal` (CORAL head, CV fold loop)
  - cells 11–19: cascade experiment loop, sanity asserts, eval/plots → rewrite for ordinal OOF metrics
  - cells 21–26: multi-model block (`_make_model`, `train_model`, TRAINED_RUNS loop, eval, plots) → ordinal per model
- Split/fold logic: `modules/dataset/make_dataset.py` (`dataset_split_by_recording`, `stratified_recording_folds`)
- Old 3-class baseline (leave untouched): `neural_network_3class.ipynb`
- Baseline stats to beat: `three_class__*_experiment_stats.csv`
- Logs: `logs_3class/three_class_experiment.txt`
