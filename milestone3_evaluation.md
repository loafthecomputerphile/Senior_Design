# Milestone 3 Evaluation — Covid-19 Detection

**Submission checklist:**

1. Submit softcopy (PDF) with signature via Canvas by Milestone 1 date (September 28th).
2. Ask your Advisor to submit scores with signature to Dr. Chan.
3. Post softcopy (PDF) on your project website (https://covidwav.io).

> Sections 9 and 10 of the official form (client meetings and client feedback) are omitted: this project has no separate client — the faculty advisor serves as the client contact.

---

# 1. Project Title, Team Members

**Project Title:** Covid-19 Detection

* Drew Quashie (dquashie2024@my.fit.edu)
* Amanda Ogbonna (kogbonna2025@my.fit.edu)
* Loleyi Oluwatomisin (ooluwatomisi2023@my.fit.edu)
* Richard Alonso Garcia (ralonsogarci2023@my.fit.edu)

## 2. Faculty Advisor

**Name:** Dr. Zahra Nematzadeh
**Email:** znematzadeh@fit.edu

## 3. Client

Client is Faculty Advisor; Dr. Zahra Nematzadeh (znematzadeh@fit.edu) — Faculty Advisor, Florida Tech

## 4. Progress of current Milestone (progress matrix)

| Task | Completion % | Drew Quashie | Amanda Ogbonna | Loleyi Oluwatomisin | Richard Alonso Garcia | To do |
| --- | --- | --- | --- | --- | --- | --- |
| Evaluate and integrate additional data sources to grow the training set | 100% | x | x | x | x | Record the decision to keep the current corpus; revisit only with reliably labeled sick audio |
| Improve preprocessing and feature extraction for the cascade models | 100% | x | x | x | x | Add YAMNet embeddings and test longer samples |
| Rebalance the training data and revisit the class-weighting strategy | 90% | x | x | x | x | Re-check rebalancing under recording-level CV before adopting it |
| Refine stage 1 of the cascade: thresholds, architecture, and sick-class recall | 100% | x | x | x | x | Carry the tuned CORAL thresholds into the backend gate |
| Retrain and re-evaluate the full cascade at the improved operating point | 95% | x | x | x | x | Re-run the ensemble evaluation after the embedding and window-length changes |

## 5. Discussion (at least a paragraph) of each accomplished task (and obstacles) for the current Milestone

**Evaluate and integrate additional data sources to grow the training set**

We surveyed the usual public cough corpora and decided not to add another one this milestone. Most of them label the symptomatic class from self-reported symptoms, which is the same problem we already have: in the CoughVid re-review by Orlandic et al., only 32 of 581 recordings labeled symptomatic were confirmed COVID-positive. Adding more recordings with that label would grow the dataset while making the class less trustworthy, so we kept the existing corpus and spent the time on measurement instead. The recording-level cross-validation pool holds 10,682 recordings, and the held-out test set holds 1,156. We used YAMNet as an optional cough-quality filter on the audio we already have, not as a source of new samples. The obstacle here is data quality rather than data volume, and it is still the limiting factor on every result below.

**Improve preprocessing and feature extraction for the cascade models**

The existing pipeline produced mel and MFCC channels. We added spectral contrast as a third channel in `SpectMFCCPipeline` and now train with no augmentation, giving the network a measure of peak-to-valley structure in the spectrum alongside the two original representations. We also replaced the single random split with 5-fold `StratifiedGroupKFold` over source recording, and kept the grouped split through every later experiment so segments from one recording cannot land in both training and validation. SpecAugment uses two frequency masks and two time masks per sample, with time stretching left off because changing the time dimension per sample breaks ragged-batch collation and speed variation is natural in cough recordings. The obstacle is that we have no isolated spectral-contrast ablation in the result tables, so we can say the channel was added but not that it caused a gain. The augmentation arms also failed to help: on the held-out set the `a2__none` arm, which has waveform and spectrogram augmentation off but still uses mixup at 0.2, reached macro F1 0.3205, while the waveform-only, spectrogram-only, and combined arms reached 0.1545, 0.2182, and 0.1118. The headline results come from the 15 CV folds, which used mixup at 0.25 and no waveform or spectrogram augmentation.

**Rebalance the training data and revisit the class-weighting strategy**

We tried the standard rebalancing levers this milestone: mixup (Beta(0.25) in the 15 CV folds, 0.2 in the augmentation arms), `WeightedRandomSampler`, and `BCEWithLogitsLoss` on the ordinal targets. `coral_pos_weights` is implemented and unit-tested, but we left it out of the trainer because combining per-class loss weights with oversampling corrects the imbalance twice. The 5-fold runs used fixed oversampling duplication; the sampling and mixup flags were exercised in the four augmentation arms on a fixed split. That split is the obstacle to reading those arms as a clean win: the sampler and mixup combination did not beat the control, and the mixed arms fell to macro F1 0.1118 at worst. Rebalancing is implemented and available behind flags, but the recording-level CV results do not support turning it on for the headline run. We are keeping the levers in the code and using the same CV gauge to judge any later change.

**Refine stage 1 of the cascade: thresholds, architecture, and sick-class recall**

The biggest change was dropping the two-stage cascade. Its first stage kept routing COVID recordings to healthy before the second stage could see them, so we replaced it with one CORAL ordinal model that predicts low, medium, or high risk through cumulative logits and learned biases. On the network side, `DepthwiseSeparableBlock` now ends with a spatial attention module and uses a 1x1 expansion before the depthwise convolution. Each CNN stage also has a projected skip path that adds the stage input back with a ReLU6 residual (`_branch_and_skips` and `_res_forward`), so the skip goes around the block rather than inside it. We swept the two CORAL thresholds on validation rather than keeping the 0.5 defaults. Pooled out-of-fold results over 10,682 recordings give macro F1 0.3265 (default CNN), 0.3295 (branched), and 0.3392 (ResNet50), with high-risk recall still low at 0.0057, 0.0265, and 0.0435. The obstacle is the threshold tradeoff. On the held-out test set, moving to a high-recall operating point lifted high-risk recall from 0.2787 to 0.5902 but pushed rank MAE from 0.7811 to 1.077. We report both numbers together.

**Retrain and re-evaluate the full cascade at the improved operating point**

We retrained all three architectures across 5 folds, giving 15 fold runs, then evaluated the best checkpoint per model on the 1,156-recording test set. Best-fold test macro F1 was 0.3132 for the default CNN, 0.2575 for the branched model, and 0.3214 for ResNet50. We then built a weighted ensemble of 10 checkpoints and tuned its thresholds on validation to 0.50 and 0.30. At the default 0.50/0.50 thresholds the ensemble scored test macro F1 0.3289 with zero high-risk recall; at the tuned thresholds it scored 0.3669 with high-risk recall 0.3115 and macro AUC 0.5836. Thresholds chosen on our own validation OOF for each model separately reached 0.3436 (default), 0.3464 (branched), 0.3482 (ResNet50), and 0.3460 (ensemble), so threshold selection is doing real work here. The obstacle is the cost of that recall: test accuracy drops from 54.33% to 52.51% and rank MAE rises from 0.4784 to 0.622. 

## 6. Discussion (at least a paragraph) of contribution of each team member to the current Milestone

**Drew Quashie**

<!-- TODO: add Drew Quashie's milestone 3 contribution paragraph -->

**Amanda Ogbonna**

<!-- TODO: add Amanda Ogbonna's milestone 3 contribution paragraph -->

**Loleyi Oluwatomisin**

<!-- TODO: add Loleyi Oluwatomisin's milestone 3 contribution paragraph -->

**Richard Alonso Garcia**

<!-- TODO: add Richard Alonso Garcia's milestone 3 contribution paragraph -->

## 7. Plan for the next Milestone (task matrix) — Milestone 3 (November 23rd)

| Task | Drew Quashie | Amanda Ogbonna | Loleyi Oluwatomisin | Richard Alonso Garcia |
| --- | --- | --- | --- | --- |
| Add medical embeddings (YAMNet) as a new input to our networks | x | x | x | x |
| Train and evaluate on longer samples of data | x | x | x | x |
| Continue work on the backend | x | x | x | x |
| Clean up the codebase | x | x | x | x |

## 8. Discussion (at least a paragraph) of each planned task for the next Milestone

**Add medical embeddings (YAMNet) as a new input to our networks**

The current models see mel, MFCC, and spectral contrast, all computed from short waveforms. Next we will extract YAMNet embeddings from the same recordings and feed them in as a fourth input, either concatenated with the spectral channels or through a small embedding branch. YAMNet is already used as a cough filter in the project, so the work is to reuse that path as a feature source rather than a cough detector. We will keep the recording-wise 5-fold split and compare against the current baseline with macro F1, macro AUC, and high-risk recall. The check is a same-split, same-metric comparison, not a new split that would make the numbers incomparable.

**Train and evaluate on longer samples of data**

Cough recordings in the current pipeline are cut into short segments, which limits how much of a cough pattern the network can see. We will build a longer-window variant while keeping each segment tied to its source recording, so the grouped folds and the recording-level aggregation stay valid. The longer windows mean fewer training segments and heavier batches, so we will measure that cost alongside the metrics and keep the shorter-window pipeline available for comparison. Any change is evaluated on the same held-out set with the same CORAL thresholds before we treat it as an improvement.

**Continue work on the backend**

The tuned CORAL ensemble and its 0.50/0.30 thresholds are currently experiment artifacts. Next we will wire the selected checkpoints and threshold values into the evaluation endpoint so uploaded audio gets the same low/medium/high risk output the notebook reports. That includes returning the confidence values the gate needs, handling the model-loading path cleanly, and updating the smoke test to cover the new response fields. The check is an end-to-end request through the running API with a known audio file, compared against the notebook output for that recording.

**Clean up the codebase**

This milestone left several parallel experiment paths in the notebooks and modules: the old cascade cells, stale channel comments, multiple augmentation arms, and empty or superseded artifacts. We will remove the dead paths, bring the run names and figure outputs in line with the current CORAL experiments, and make sure the preprocessing and model construction code matches the comments. We will keep the small checks for spectral contrast and the CORAL metrics, since those pin behavior we rely on. The check is a clean run of the current 5-fold pipeline from the documented entry point after the cleanup, with the same metrics as before.

## 11. Date(s) of meeting(s) with Faculty Advisor during the current milestone

<!-- TODO: add faculty advisor meeting dates -->

## 12. Faculty Advisor feedback on each task for the current Milestone

**Train & Fine-Tune CNN Models & Transfer Learning Architectures**

<!-- Faculty Advisor feedback -->

**Initial Model Evaluation & Performance Reports**

<!-- Faculty Advisor feedback -->

**Develop FastAPI Backend Endpoints**

<!-- Faculty Advisor feedback -->

**Build React Web UI Components for Prediction Results and Risk Feedback**

<!-- Faculty Advisor feedback -->

**Faculty Advisor Signature:** ___________________________

**Date:** _______________________________

---

## 13. Evaluation by Faculty Advisor

*Return this page to Dr. Chan (HC 209) or email scores to pkc@cs.fit.edu.*

**Score (0–10) for each member** — circle one score; two adjacent scores may be circled for a .25 increment, or write any real number between 0 and 10.

**1. Drew Quashie**

`0  1  2  3  4  5  5.5  6  6.5  7  7.5  8  8.5  9  9.5  10`

**Score:** __________

**2. Amanda Ogbonna**

`0  1  2  3  4  5  5.5  6  6.5  7  7.5  8  8.5  9  9.5  10`

**Score:** __________

**3. Loleyi Oluwatomisin**

`0  1  2  3  4  5  5.5  6  6.5  7  7.5  8  8.5  9  9.5  10`

**Score:** __________

**4. Richard Alonso Garcia**

`0  1  2  3  4  5  5.5  6  6.5  7  7.5  8  8.5  9  9.5  10`

**Score:** __________

**Faculty Advisor Signature:** ___________________________

**Date:** _______________________________
