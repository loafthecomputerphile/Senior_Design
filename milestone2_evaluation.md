# Milestone 2 Evaluation — Covid-19 Detection

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
| Train & Fine-Tune CNN Models & Transfer Learning Architectures | 80% | x | x | x |  | Finish cascade stage-2 training; compare transfer learning variants |
| Initial Model Evaluation & Performance Reports | 60% | x | x |  |  | Complete both cascade runs; generate full per-class reports |
| Develop FastAPI Backend Endpoints | 90% | x | x |  |  | Wire real model artifacts into gate and inference stubs |
| Build React Web UI Components for Prediction Results and Risk Feedback | 30% |  |  |  | x | Results screen, inconclusive variant, integration with /evaluate |

## 5. Discussion (at least a paragraph) of each accomplished task (and obstacles) for the current Milestone

**Train & Fine-Tune CNN Models & Transfer Learning Architectures**

The first approach was a single 3-class CNN trained directly on the three labels (healthy, symptomatic, COVID-19), with joint MFCC + log-Mel features, and three input variants tested (waveform only, spectrogram only, both). Every variant produced the same result: the model collapsed to the majority class. The validation F1 sat at 0.2459 for all 60 epochs in each run while accuracy stayed at 58.43%, which is exactly the healthy-class share of the dataset (31,522 of 42,364 recordings, 74.4%). Training loss decreased while validation loss did not, so the runs were memorizing the training set rather than learning anything transferable.

Selecting augmentation methods also turned out to be harder than expected because of how unusual cough audio is as a signal: standard audio augmentations (time stretch, pitch shift, noise) can distort the acoustic features that distinguish a sick cough from a healthy one, so aggressive augmentation pushed the model toward the majority-class answer instead of helping it generalize.

The collapse had a second cause we found and fixed: the original random split let recordings from the same speaker appear in train, validation, and test sets. The literature warned us about exactly this; Buttow et al. (2021, npj Digital Medicine) showed AUC dropping from 0.90 to 0.71 when a cough model moves from a random sample-level split to a participant-independent one. We rebuilt the split to be recording-wise (implemented 2026-10-23), so no speaker crosses between sets.

The larger pivot was architectural. The direct 3-class setup asks one model to learn the COVID-vs-symptomatic boundary, which is the one boundary the literature says is not learnable from cough audio: Orlandic et al. (2023, Computer Methods and Programs in Biomedicine) re-reviewed 581 "symptomatic" CoughVid recordings with physicians and found only 32 were actually COVID-positive, meaning the symptomatic class is defined by self-reported symptoms rather than acoustics. We restructured into a two-model cascade: model 1 is binary, (COVID + symptomatic) vs healthy, with the large healthy majority stabilizing training; model 2 is binary, COVID vs symptomatic, run only on recordings model 1 flags as sick. Their combined outputs keep the 3-class output the app requires. Training uses a class-weighted loss for imbalance (the method Chang et al. 2022 used on CoughVid), and the recording-wise split applies to both models. At the time of writing the cascade models are mid-training and evaluation is ongoing.

**Initial Model Evaluation & Performance Reports**

Evaluation is currently concentrated on the first stage of the cascade. The early results confirm the pivot did its job: the model no longer leans entirely on the largest class. Where the direct 3-class runs sat frozen at 58.43% accuracy and 0.2459 F1, the (COVID+symptomatic)-vs-healthy binary model reaches an F1 above 0.51 with validation AUC around 0.56 and PR-AUC around 0.29, and the metrics move epoch to epoch instead of flatlining. Absolute numbers are still modest, but that matches the honest literature baseline: participant-independent splits on CoughVid-style data land near AUC 0.70 at best, and the 90%+ figures published with random splits are leakage artifacts. This is why we report per-model AUC and PR-AUC alongside F1.

The obstacle that dominated evaluation was the leakage problem described above: before the recording-wise split, our test numbers were not measuring anything meaningful, since the model could recognize speakers rather than cough characteristics. The second evaluation obstacle is the COVID-vs-symptomatic confusion itself, which stage 2 of the cascade now owns in isolation; its runs are still in progress, so its performance report is not ready yet. The remaining work for this milestone is completing both cascade runs, then generating the full performance reports (accuracy, precision, recall, F1, ROC-AUC per class) that feed the confidence-gate calibration.

**Develop FastAPI Backend Endpoints**

The backend is a FastAPI app with two routers. Auth: `POST /signup`, `POST /login` (bearer token), and `GET /me`. Evaluation: `POST /evaluate` takes a multipart audio upload from an authenticated user, resamples it to 16 kHz mono WAV, runs the XGBoost cough gate first (rejection is a 422), then the PyTorch model, and stores the result in SQLite; `GET /evaluations`, `GET /evaluations/{id}`, and `GET /export` serve the stored history back. Models load lazily from environment-overridable paths, so the API layer never hardcodes model locations. A framework-free smoke test (`test_smoke.py`) verifies the whole set against a live server: signup, login, wrong-password rejection, upload to `/evaluate`, history listing, export, and auth enforcement. All smoke tests pass.

Two pieces are deliberately stubbed: the XGBoost gate scoring and the PyTorch inference call return placeholder values because the model artifacts are not yet exported from the training pipeline; the HTTP contract, validation, gate ordering, and persistence are real and slot the models in behind the same function signatures. One deviation from the design document: results are currently stored server-side per user account, while the design targets stateless inference with browser-side storage (SNFR2); this was the intermediate step that made the endpoints testable end-to-end, and it gets resolved when persistence moves client-side in Milestone 3.

**Build React Web UI Components for Prediction Results and Risk Feedback**

Paragraph....

## 6. Discussion (at least a paragraph) of contribution of each team member to the current Milestone

**Drew Quashie**

I revamped the training pipeline for the two-model cascade approach: model 1 on (COVID + symptomatic) vs healthy and model 2 on COVID vs symptomatic, with Amanda and Loleyi helping me create the new subsets that split the dataset for the two binary cascade tasks. I also managed the experiment run for the augmentations, configuring the per-class augmentation probabilities (augmenting COVID-19 samples 90% of the time, symptomatic 70%, and healthy only 40%, so the minority classes get the most exposure) and running the waveform-vs-spectrogram input comparison (wave only, spec only, and both branches) across full 60-epoch training runs. On top of that I started making the code for the mass training runs across both cascade models and their input variants, and I redid the base convolution block with the adaptive version.

**Amanda Ogbonna**

For this milestone, I worked on expanding the research portion of our project based on feedback from our faculty advisor. I went back through the papers we collected and added more detailed information about which algorithms performed the best, the limitations of each study, dataset sizes, whether the datasets used speaker-independent evaluation splits, and what each study's results actually rested on. Working through that literature is what surfaced the problem with our original 3-class setup: the COVID-vs-symptomatic boundary a single 3-class model has to learn is the exact boundary the papers show cannot be learned from cough audio, since the symptomatic class is defined by self-reported symptoms rather than acoustics (Orlandic et al. 2023 re-reviewed 581 "symptomatic" CoughVid recordings with physicians and found only 32 were actually COVID-positive). Based on that finding I recommended the pivot to the two-model cascade design, and together with Loleyi I helped create the new dataset subsets that split the data into the two binary tasks the cascade needs: the binary label-grouping utilities that remap the 3-class labels into each binary task's targets, and the subset construction for the stage-1 (COVID + symptomatic) vs healthy and stage-2 COVID vs symptomatic datasets. I also added the SQLite database layer for the backend, including the users and evaluations tables the endpoints read and write.

**Loleyi Oluwatomisin**

My work this milestone centered on the dataset restructuring for the two-model cascade. They helped create the new dataset subsets that split the data into the two binary tasks the cascade needs: the (COVID + symptomatic) vs healthy grouping for stage 1 and the COVID vs symptomatic subset for stage 2. This included the binary label-grouping utilities that remap the 3-class labels (healthy / COVID-19 / symptomatic) into each binary task's targets: the one-vs-group mapping for stage 1's merged sick class and the per-label filtering that builds stage 2's standalone dataset from only the sick recordings. They also supported the augmentation experiment runs.

**Richard Alonso Garcia**

Contribution...

## 7. Plan for the next Milestone (task matrix) — Milestone 3 (November 23rd)

| Task | Drew Quashie | Amanda Ogbonna | Loleyi Oluwatomisin | Richard Alonso Garcia |
| --- | --- | --- | --- | --- |
| Integrate React frontend, FastAPI backend, and PyTorch model for end-to-end real-time predictions | x | x | x | x |
| Implement the longitudinal cough tracking visualization dashboard | x |  |  | x |
| Complete full system testing, performance optimization, and bug resolution | x | x | x | x |
| Finalize all project deliverables: demonstration video, poster, and final project presentation and publish paper | x | x | x | x |

## 8. Discussion (at least a paragraph) of each planned task for the next Milestone

**Integrate React frontend, FastAPI backend, and PyTorch model for end-to-end real-time predictions**

This task connects the three components built so far into one working flow: the React recorder captures three coughs, uploads the single recording to `POST /v1/analyze`, and the FastAPI backend runs its pipeline (validation and quality gate, XGBoost cough-presence gate, segmentation into three 1-second windows, MFCC + log-Mel feature extraction, three CNN forward passes, majority vote with a mean-sigmoid tie-break). The JSON response (risk level, confidence percentage, vote breakdown, per-cough classes and signal strengths) is rendered on the results screen with the mandatory pre-screening disclaimer and risk-matched recommendations. Inconclusive conditions (poor quality, no cough detected, confidence below 0.50) are surfaced with their specific reason and a Try Again action rather than an error. The target is capture-to-result in under 6 seconds end-to-end, with server inference under 1.5 s.

**Implement the longitudinal cough tracking visualization dashboard**

Every saved analysis is written to the browser's IndexedDB; the server keeps nothing after inference, so all persistence is local. The trends dashboard plots risk level and confidence over time (the design targets the last 30 days) with an activity summary, so users can see how their results change across days. It also provides CSV and JSON export of the full history so a user can hand their record to a clinician without any server-side account or storage. History management includes listing past analyses on the dashboard and a delete-all option in settings.

**Complete full system testing, performance optimization, and bug resolution**

Testing covers the full pipeline against the requirements: input validation and the quality gate on bad inputs (too short, silent, clipped, non-cough audio), correct voting and risk mapping behavior, the confidence gate, browser compatibility (Chrome, Edge, Firefox, Safari) and responsive layout from phone-width to desktop, and accessibility requirements such as risk conveyed by color and text together. Performance work checks the measured latency of each pipeline stage against the 1.5 s inference budget and the 6 s end-to-end target, and confirms server statelessness (no audio, features, or results retained). Bugs found during integration and testing are fixed before the final deliverables.

**Finalize all project deliverables: demonstration video, poster, and final project presentation and publish paper**

The remaining course deliverables are produced once the integrated system is stable: a demonstration video showing the capture-to-result flow, the final poster, the final project presentation, and the publishable paper covering the approach, model, and evaluation results.

## 11. Date(s) of meeting(s) with Faculty Advisor during the current milestone

1. Thursday October 1st 11:15am — Olin Engineering Complex room 353
2. Thursday October 8th 11:45am — Olin Engineering Complex room 353
3. Thursday October 15th 12:00pm — Olin Engineering Complex room 353
4. Thursday October 22th 12:00pm — Olin Engineering Complex room 353

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
