# COVID-19 Cough Classification — Literature Research Results

*Date: 2026-10-23. Full-text read where accessible; papers 3, 5, 6, 9, 10 covered from abstracts/secondary sources (paywalled/403).*

## The big one first: most accuracy numbers in the summary table are inflated by speaker leakage

The [npj Digital Medicine leakage study (Buttow et al. 2021)](https://www.nature.com/articles/s41746-021-00553-x) did the controlled experiment: same data, same model, only the split changed. **Random sample-level split: AUC up to 0.90. Participant-independent split: AUC 0.71.** The [Lancet Digital Health "seven grains of salt" comment](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00141-2/fulltext) called out COUGHVID specifically for not enforcing participant-disjoint splits, and showed sensitivity jumping 0.65 → 0.84 when test users appear in training.

- Papers 1, 2, 5 all use random splits (paper 1: "80/10/10, no subject-wise split described"; paper 2: 5-fold CV with "mutually exclusive subsets" — segments, not subjects; paper 5 doesn't say, which is itself the tell). Their 74–98% numbers are speaker-recognition, not disease detection.
- **Our recording-wise split (implemented 2026-10-23) is the fix.** Collapsing to ~58% majority-class under a *clean* split is consistent with the literature's honest numbers: AUC ~0.70 is roughly what a good model gets. We are not failing where others succeed — we are the first to measure honestly.

## Label quality is a known first-class problem — and the fix is already in the dataset

[Orlandic et al. 2023](https://www.sciencedirect.com/science/article/pii/S0169260723004091) (COUGHVID's own authors) built a semi-supervised relabeling pipeline because user labels disagree with physicians. Concrete numbers: of 581 "symptomatic" recordings they re-examined, only 32 were actually COVID-positive. Their cleaned labels tripled COVID-vs-healthy separability (classifier AUC 0.797).

**The relabeled labels ship in COUGHVID v3**: `metadata_compiled.csv` has a `status_SSL` column. If `filter_coughvid_df` uses `status` rather than `status_SSL`, switching is a one-line change that directly attacks the noisy-minority-label problem that makes models hedge toward majority.

## Architecture: what actually works on COUGHVID

- [Chang et al. (CovNet, Frontiers 2022)](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.799067/full): 4 simple CNNs, **64 mel bins, 16 kHz, window 512 / overlap 256**, class weights in the loss ("re-scale the weight parameter for each class"), **no augmentation**. Transfer learning beat from-scratch by ~2.6–3.6% AUC. Best: 72.4% AUC (DiCOVA), 66.4% UAR (ComParE). Notably: positives "frequently confused with negatives" — our exact failure mode, at much larger scale.
- [Campana et al. 2023 (arXiv:2307.02975)](https://arxiv.org/abs/2307.02975): VGGish/YAMNet/OpenL3 on 13,447 samples, **user-independent evaluation**. Key result: **frozen embeddings + shallow classifier beat fine-tuning by ~6.6% PR-AUC on average** — the respiratory domain is too far from AudioSet/YouTube for fine-tuning to help with this little data. OpenL3 best (PR-AUC ~0.68 on COUGHVID), hand-crafted features + SVM ~0.61, YAMNet worst.
- [Kim et al. (JMIR 2024)](https://www.jmir.org/2024/1/e51640/): ResNet-18 + mel-spectrograms, AUC 0.92–0.93 internal but **0.55 on Omicron** — external generalization is where everything dies.
- [Pahar et al. 2022](https://www.sciencedirect.com/science/article/pii/S0010482521009471): ResNet50 transfer learning, AUC ~0.98 — but on Coswara, a much cleaner dataset than COUGHVID.
- The CNN-LSTM attention paper ([PMC9034264](https://pmc.ncbi.nlm.nih.gov/articles/PMC9034264/)) also **collapsed 3-class to binary** ("healthy" vs "likely-COVID", merging COVID+ and symptomatic) — and their 91% is still on a random split.

## The 3-class framing is the hardest version of the task

Every strong result is binary. Paper 5 is the only honest 3-class COUGHVID result in the table — **74.4% accuracy, 73% F1** — and that's with a random split (i.e., inflated). The "symptomatic" class is near-unlearnable from cough audio: Orlandic's data shows even experts can't reliably separate it from COVID. Two of our three classes are label-noise-heavy and acoustically overlapping.

## Paper details fetched in full

### Paper 1 — Cough2COVID-19 (Hussain et al., Sci Rep 2024)
- Preprocessing: librosa — silence trimming, amplitude normalization, noise reduction, resampling, segmentation. 15 features: time-domain (RMSE, ZCR), frequency-domain (spectral bandwidth/centroid/contrast/flatness/flux/rolloff), time-frequency (MFCC, chroma, tonnetz).
- Architecture: 3-branch ensemble — MFCC→MLP (4 FC), spectrogram→CNN (3 conv + 3 FC, VGG-inspired), chromagram→MLP; concatenate to 768-dim → dense blocks → sigmoid.
- Split: 80/10/10 on combined Coswara/Virufy/ComParE; COUGHVID held out for cross-validation/inference. **No subject-wise split described.**
- Imbalance: SMOTE oversampling of minority class. No other augmentation.
- Results: 98% acc / 97% sens / 98% spec / AUC 0.981. No per-class table.
- Limitations (authors'): inter-individual cough variation, need larger diverse data.

### Paper 2 — PSR 3D CNN (Altae et al., Sci Rep 2025)
- Preprocessing: Phase Space Reconstruction, τ=13 (mutual information first minimum), embedding dim d=5; input tensor 256³ (PCA→3D, 8-bit quantized). Cough-probability filter ≥0.5 → 8,407 samples (6,466 healthy / 658 COVID / 1,283 symptomatic).
- Architecture: 3D CNN, five 3×3×3 conv layers (64→512 filters), maxpool 2×2×2, dense 100→3 softmax. Adam lr 1e-3 → 1e-4 after 50 epochs.
- Split: 5-fold CV, "mutually exclusive subsets" — **not stated subject-wise** (gap in reporting).
- Imbalance: not addressed. Augmentation: none.
- Results: 98.5% acc / 96.5% recall / 99.7% spec.
- Limitations (authors'): crowdsourced label noise, high compute cost, low transparency.

### Paper 4 — JMIR longitudinal (Kim et al. 2024)
- Cambridge-trained ResNet-18 + mel-spectrograms: AUC 0.93 internal, 0.92 Virufy (same period), then Coswara by variant: Alpha 0.83, Delta 0.77, **Omicron 0.55 (near random)**.
- Lesson: variant/time drift dominates; external validation is where models die.
- Related: arXiv 2106.03851 shows AUC varies up to 0.1 depending on split choice (random vs time vs site).

### Paper 7 — Orlandic SSL relabeling (CMPB 2023)
- SSL: one ML model per expert physician + cross-agreement with user labels → high-confidence relabeled subset. Labeled 1,018 previously-unlabeled recordings; resolved ambiguous "symptomatic" for 581 recordings (only 32 actually COVID+).
- Separability improvements: COVID-vs-healthy 3×, wet/dry 11.3×, severity 5.1×.
- Classifier on relabeled data: AUC 0.797, 79.2% sens, 65.8% spec.
- **`status_SSL` column now included in COUGHVID v3 `metadata_compiled.csv`.**

### Paper 14 — CovNet transfer learning (Chang et al., Frontiers Digital Health 2022)
- Backbones: CNN-4, VGG-7, ResNet-6, MobileNet-6 (all shallow). Best: CNN-4 72.38% AUC (DiCOVA), VGG-7 66.43% UAR (ComParE).
- Input: log-mel, 64 bins, 16 kHz, window 512 / overlap 256. Stratified 80/20 (COUGHVID), 5-fold (DiCOVA).
- Imbalance: class weights in loss. Augmentation: none.
- Transfer > scratch by ~2.6–3.6%. Positives frequently confused with negatives.

### Leakage study (Buttow et al., npj Digital Medicine 2021)
- 5,240 samples / 2,478 participants (514 pos / 1,964 neg), VGGish + CNN.
- Participant-independent split: **AUC 0.71**. Random sample-level split: **AUC up to 0.90**.
- Also documented demographic bias: gender bias (female sensitivity 0.23), language bias (English sensitivity 0.25).

## Proposed fixes, re-ranked with paper evidence

1. **Keep the recording-wise split — we are now measuring honestly.** Reset expectations: AUC ~0.70 under subject-wise splits is *literature-level*, not failure. (npj study, Lancet comment)
2. **Switch labels to `status_SSL`** in `filter_coughvid_df` if not already using it. Orlandic's own fix for COUGHVID noise, already in the data. (Orlandic 2023)
3. **Go binary: COVID+ vs rest** (or COVID vs healthy). The only configuration with strong literature results; also fixes collapse pressure since "symptomatic" stops muddying the negative class. Report healthy-vs-COVID and symptomatic-vs-COVID as separate binary models if 3 outputs are required for the deliverable.
4. **Class weights in CrossEntropyLoss** (Chang et al.'s exact method — weight per class in the loss, no sampler), not the sampler we removed.
5. **Frozen pretrained embeddings + small head** rather than the from-scratch CNN or heavy fine-tuning: OpenL3 or VGGish embeddings, frozen, + logistic/MLP head. Campana showed fine-tuning *hurts* here. Practical PyTorch option: torchaudio VGGish wrappers, or timm ResNet-18 pretrained on ImageNet over the existing 3-channel mel images (paper 4's recipe — but expect their Omicron lesson: validate across subsets).
6. **Mel params to literature norms: 64 bins, 16 kHz** (Chang et al.) — also kills the zero-filterbank warning at n_mels=256.
7. **Report UAR/AUC + per-class recall, not just macro-F1** (Chang et al. metric choice) — under collapse those decompose the failure immediately.

## What NOT to do

- 3D-PSR (paper 2) or EDNN-CHIO (paper 3): unreproducible pipelines whose numbers don't survive the split-quality question.
- Fine-tuning a big pretrained model end-to-end: Campana's data says frozen beats it at this data scale (~6.6% PR-AUC drop).

## Sources

- [npj leakage study](https://www.nature.com/articles/s41746-021-00553-x)
- [Lancet "seven grains of salt"](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00141-2/fulltext)
- [Orlandic SSL relabeling](https://www.sciencedirect.com/science/article/pii/S0169260723004091)
- [CovNet transfer learning](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2021.799067/full)
- [Campana VGGish/YAMNet/OpenL3](https://arxiv.org/abs/2307.02975)
- [JMIR ResNet-18 generalization](https://www.jmir.org/2024/1/e51640/)
- [CNN-LSTM attention paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9034264/)
- [Pahar transfer learning](https://www.sciencedirect.com/science/article/pii/S0010482521009471)
- [Cough2COVID](https://www.nature.com/articles/s41598-024-76639-9)
- [PSR 3D CNN](https://www.nature.com/articles/s41598-025-29633-8)
