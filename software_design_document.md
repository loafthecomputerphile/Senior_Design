# COVID-19 Cough Detection Web Application

## Software Design Document

**Version 1.0**

Drew Quashie, Amanda Ogbonna, Loleyi Oluwatomisin, Richard Alonso Garcia
Faculty Advisor: Dr. Zahra Nematzadeh

---

## Table of Contents

1.0 Introduction
2.0 System Overview
3.0 System Architecture
4.0 Data Design
5.0 Component Design
6.0 Human Interface Design
7.0 Algorithm Design
8.0 Performance and Optimization
9.0 References

---

## 1.0 Introduction

### 1.1 Purpose

This software design document describes the architecture and system design of the COVID-19 Cough Detection Web Application, a browser-based solution for preliminary COVID-19 risk pre-screening using recorded cough audio, server-side convolutional neural network inference, and on-device longitudinal health tracking.

### 1.2 Background and Objectives

During pandemic surges, individuals with mild or uncertain symptoms often visit testing centers for reassurance, overloading laboratories and clinical staff. Early, accessible pre-screening can direct people who are likely negative toward self-monitoring while surfacing likely positive cases to care sooner.

The primary objectives are:

- Provide a pre-screening risk assessment from a single recorded cough using ordinary device microphones — no specialized hardware.
- Deliver near-real-time feedback from browser capture to displayed result.
- Support longitudinal self-tracking so users (and, via sharing, clinicians) can observe trends over time.
- Commit to honest result framing: a pre-screening signal, never a diagnosis.

### 1.3 Definitions and Acronyms

| Acronym | Term |
| --- | --- |
| CNN | Convolutional Neural Network |
| MFCC | Mel-Frequency Cepstral Coefficients |
| STFT | Short-Time Fourier Transform |
| SNR | Signal-to-Noise Ratio |
| API | Application Programming Interface |
| ONNX | Open Neural Network Exchange |
| UI/UX | User Interface / User Experience |

| Term | Definition |
| --- | --- |
| Risk Level | Categorized pre-screening assessment (Low / Medium / High) derived from model output |
| Confidence Score | Probability measure indicating prediction certainty |
| Segment | Fixed-length (1 s) excerpt extracted from a user's raw cough recording |
| Feature Map | Matrix of MFCC or Mel-spectrogram coefficients derived from one segment |
| Inconclusive | Result withheld because input quality or model confidence is below threshold |

---

## 2.0 System Overview

The application comprises four major subsystems that collaborate to deliver the complete experience.

### 2.1 Audio Capture Module

Handles all aspects of microphone acquisition in the browser via the Web Audio API and `getUserMedia`: permission request, live input-level metering, recording start/stop, and capture guidance. The user records a single free-form cough; no fixed number of coughs or timed window is imposed. Basic client-side checks (silence detection, gross clipping) provide immediate feedback, while full quality validation occurs server-side after upload.

### 2.2 Machine Learning Inference Engine

The core analytical component. The FastAPI backend receives the raw audio upload, runs the quality gate, segments the clip to the model's fixed 1 s window, extracts the joint MFCC + Mel-spectrogram feature set, and executes a PyTorch CNN/ResNet classifier. The engine returns a three-class probability distribution (Negative / Uncertain / Positive) plus input-quality metadata.

### 2.3 Results Processing and Display

Interprets raw model output into user-facing information: maps class probabilities to Low/Medium/High risk levels with a minimum-confidence gate, generates appropriate clinical recommendations, and formats results with mandatory disclaimers. Renders the longitudinal dashboard from locally stored history.

### 2.4 Data Management System

Responsible for browser-side persistence of analysis history, trend series, and user preferences using IndexedDB. All health data remains on the user's device; nothing is retained on the server after inference completes. The system supports manual export (CSV/JSON) so a user can share their trend record with a clinician.

---

## 3.0 System Architecture

### 3.1 Architectural Design

The system follows a three-tier client–server architecture with a browser device layer:

```
┌─────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER (React, browser)                    │
│  Dashboard · Recorder · Results · Trends · Settings      │
├─────────────────────────────────────────────────────────┤
│  BROWSER DEVICE LAYER                                   │
│  Web Audio API · getUserMedia · IndexedDB               │
├────────────────────────┬────────────────────────────────┤
│  APPLICATION LOGIC     │  REST API (HTTPS, multipart)    │
│  (FastAPI)             │  POST /analyze                  │
├────────────────────────┼────────────────────────────────┤
│  ML LAYER (PyTorch)    │  DATA PERSISTENCE (server)     │
│  Quality gate · Segment│  Stateless — no health data    │
│  · Feature extract ·   │  stored post-inference         │
│  Classifier            │                                │
└────────────────────────┴────────────────────────────────┘
```

The Presentation Layer owns all user interaction and local persistence. The Application Logic Layer validates requests and coordinates the ML pipeline. The ML Layer performs analytical computation. The server is **stateless with respect to health data**: an inference request in, a result out, nothing retained.

### 3.2 Decomposition Description

- **React ↔ FastAPI:** single primary endpoint, `POST /analyze`, accepts multipart audio (WAV/OGG/WEBM) and returns JSON: class probabilities, risk level, confidence, segment-quality report, and trace ID. Versioned under `/v1`.
- **FastAPI ↔ ML Layer:** model loaded once at process start; the API layer never touches tensors directly beyond passing the preprocessed payload to the inference service.
- **React ↔ IndexedDB:** all history and trend reads/writes go through the Storage Manager; no other component touches IndexedDB.
- **Future boundary:** the ML Layer's feature-extraction + classifier pair is intentionally kept free of FastAPI dependencies so it can be exported to ONNX and embedded in the browser (see §8.2).

---

## 4.0 Data Design

### 4.1 Data Description

The system manages four primary data types:

- **Audio Data:** raw microphone captures in browser-native formats (WEBM/OGG, transcoded or accepted as WAV), typically 1–5 s and under 500 KB. Used transiently: uploaded, analyzed, discarded by the server; optionally kept locally at user discretion.
- **Feature Tensors:** MFCC and Mel-spectrogram matrices computed from the 1 s segment — transient in memory on the server only, never persisted.
- **Model Data:** trained PyTorch weights (`.pt`) with a planned ONNX export (`.onnx`) for the future client-side path.
- **Analysis Records:** structured JSON records — timestamp, risk level, class probabilities, confidence, segment quality metrics, and optional user notes — stored in IndexedDB on the user's device.

### 4.2 Data Dictionary

| Data Element | Format | Size | Description |
| --- | --- | --- | --- |
| Raw Cough Audio | WEBM/WAV | 50–500 KB | Single free-form cough capture |
| Quality Report | JSON | <2 KB | SNR, clipping ratio, duration, cough-presence score |
| Feature Tensors (MFCC + Mel) | float32 arrays | ~100–300 KB | 1 s segment feature maps, server RAM only |
| Trained Model | .pt | 10–90 MB | Full PyTorch checkpoint (training only) |
| Inference Model | .onnx | 5–50 MB | Exported weights (serving / future browser use) |
| Analysis Record | JSON (IndexedDB) | <2 KB | Result metadata + quality metrics + user note |
| User Preferences | JSON (IndexedDB) | <2 KB | Retention settings, guidance dismissal flags |

### 4.3 Data Storage Design

IndexedDB object stores:

| Store | Key | Value |
| --- | --- | --- |
| `analyses` | auto-increment id | Analysis Record (indexed by `timestamp`) |
| `preferences` | fixed key | User Preferences |

Retention is user-configurable (default: keep records indefinitely until manually deleted; bulk-delete and export actions provided). The server keeps **no** copy of audio, features, or results.

---

## 5.0 Component Design

### 5.1 Presentation Layer Components

**Main UI Component**
- Responsibility: application navigation and dashboard rendering.
- Key methods:
  - `navigateToScreen(screenName)` — handles view transitions.
  - `displayDashboard()` — renders recent analyses summary and entry points.
  - `handleUserPreferences()` — settings for retention, export, guidance.

**Recorder UI Component**
- Responsibility: microphone interface with capture guidance.
- Key methods:
  - `initializeRecorder()` — requests `getUserMedia`, wires Web Audio context.
  - `showLiveLevelMeter()` — visual input meter confirming mic activity.
  - `startRecording()` / `stopRecording()` — capture a single free-form cough; client-side silence/clipping check with immediate recapture prompt.
  - `uploadForAnalysis(blob)` — POST to `/v1/analyze`, manages loading and error states.

**Results UI Component**
- Responsibility: presentation of assessment outcome.
- Key methods:
  - `displayRiskAssessment(result)` — Low/Medium/High badge, confidence bar, probability breakdown.
  - `presentMedicalDisclaimer()` — mandatory pre-screening-only notice.
  - `generateRecommendations(riskLevel)` — actionable guidance per level.
  - `saveLocally(result)` — writes Analysis Record via Storage Manager.

**Trends UI Component**
- Responsibility: longitudinal dashboard.
- Key methods:
  - `renderTrendChart(records)` — risk level and confidence over time.
  - `exportHistory(format)` — CSV/JSON export for clinician sharing.

### 5.2 Application Logic Layer Components

**Analysis Request Controller**
- Responsibility: HTTP contract and validation.
- Key methods:
  - `acceptUpload(request)` — size/type/duration limits, abuse controls.
  - `validateRequest(audioMeta)` — reject empty, oversized, or truncated payloads before ML work begins.
  - `formatResponse(result)` — assemble JSON contract, map internal errors to client-safe messages.

**Inference Orchestrator**
- Responsibility: coordinates the ML pipeline end to end.
- Key methods:
  - `executeAnalysisPipeline(audio)` — quality gate → segmentation → feature extraction → classification → risk mapping.
  - `monitorPerformanceMetrics()` — per-stage latency logging.
  - `handlePipelineErrors()` — distinguish input problems (4xx, "recapture") from system problems (5xx, retry).

**Result Processor**
- Responsibility: interpretation and formatting of model output.
- Key methods:
  - `interpretModelOutput(probs)` — three-class distribution → risk level.
  - `applyConfidenceGate(confidence)` — below-threshold → Inconclusive.
  - `formatForDisplay(result)` — assemble the user-facing result object.

### 5.3 Machine Learning Layer Components

**Model Loader**
- Responsibility: model lifecycle on the server.
- Key methods:
  - `loadModel()` — load `.pt` (or `.onnx`) once at startup.
  - `verifyModelIntegrity()` — checksum + smoke-test inference on a reference tensor.
  - `optimizeInferenceSettings()` — `eval()` mode, `torch.inference_mode()`, optional half-precision on GPU.

**Audio Preprocessor**
- Responsibility: raw clip → analyzable segment.
- Key methods:
  - `validateQuality(clip)` — SNR estimate, clipping ratio, cough-presence/duration check; fails → Inconclusive with reason.
  - `extractSegment(clip)` — locate the cough event and cut the fixed 1 s window (segment length from the 98.5th-percentile length experiment; see §7.1).
  - `normalize(segment)` — amplitude normalization and resampling to the training sample rate.

**Feature Extractor**
- Responsibility: segment → model input.
- Key methods:
  - `computeMelSpectrogram(segment)` — log-Mel power spectrogram.
  - `computeMFCCs(segment)` — cepstral coefficients from log-Mel energies.
  - `fuseFeatures(mel, mfcc)` — stack into the joint representation the classifier consumes (fusion point fixed at training time; see §7.2).

**Prediction Interpreter**
- Responsibility: model output → assessment.
- Key methods:
  - `getClassProbabilities(logits)` — softmax over Negative / Uncertain / Positive.
  - `calculateConfidence(probs)` — certainty metric used by the confidence gate.
  - `assignRiskLevel(probs)` — threshold table mapping to Low / Medium / High / Inconclusive.

### 5.4 Device and Persistence Layer Components

**Microphone Interface (browser)**
- Responsibility: device audio access abstraction.
- Key methods:
  - `acquireMicPermission()` — `getUserMedia` with graceful denial handling.
  - `configureCapture()` — sample rate and channel selection.
  - `releaseMic()` — teardown after stop or navigation.

**Storage Manager (browser)**
- Responsibility: local persistence and retrieval.
- Key methods:
  - `saveAnalysis(record)` — IndexedDB insert.
  - `retrieveHistory(range)` — records for dashboard and trends.
  - `exportHistory(format)` — CSV/JSON for sharing.
  - `pruneHistory(policy)` — retention enforcement.

---

## 6.0 Human Interface Design

### 6.1 Overview of User Interface

The interface is a responsive single-page React application prioritizing a clear information hierarchy: capture is one tap away, results are legible at a glance, and trends are always available but never alarming. All result screens carry the pre-screening disclaimer; risk levels are conveyed by color **and** text (accessibility), never color alone.

### 6.2 Screen Images and Mockups

**Main Dashboard Screen**

```
[COVID Cough Check]                    [Settings]

  ┌──────────────────────────────┐
  │      [ 🎤 RECORD COUGH ]     │
  └──────────────────────────────┘

  Recent analyses
  Sep 24  ·  LOW      ·  78%
  Sep 23  ·  MEDIUM   ·  71%
  Sep 21  ·  LOW      ·  82%

  [ View Full Trends ]   [ Export History ]
```

**Recorder Screen**

```
[Back]        Record Your Cough        [Help]

        One loud, clear cough.
        Hold the device 6-8 inches away
        in a quiet room.

             ● ● ● ● ●   (live level meter)

        [        RECORD        ]

        Tip: if the meter doesn't move,
        check microphone permissions.
```

**Results Screen**

```
[Back]       Analysis Complete          [Save] [Share]

  IMPORTANT: Pre-screening tool only — not a diagnosis.
  Always confirm with an official COVID-19 test.

  Result:  MEDIUM RISK
  Confidence: 74%   [████████░░]

  Model probabilities:
    Negative 22%   Uncertain 4%   Positive 74%

  Recommendations:
    • Consider taking an official test
    • Wear a mask around others until confirmed
    • Monitor symptoms over the next 48 hours

  Quality: OK (SNR good, cough detected)

  [Save Result]      [New Analysis]      [Learn More]
```

**Inconclusive Result (variant)**

```
  Result:  INCONCLUSIVE
  We couldn't analyze that recording
  (low SNR / cough not detected).
  Please recapture in a quieter room.
  [Try Again]
```

**Longitudinal Trends Screen**

```
[Back]        Your Trends                [Export]

  Risk level over time (last 30 days)
  ▁▁▂▃▅▇▅▃▂▁▁  ← daily risk line chart
  Confidence % overlay

  Coughs recorded: 12      Streak: 4 days

  [Export CSV]   [Export JSON]   [Delete All Data]
```

---

## 7.0 Algorithm Design

### 7.1 Audio Processing Pipeline

```
BEGIN Audio Processing Algorithm
 1. RECEIVE raw audio clip from client upload
 2. VALIDATE clip bounds: duration, byte size, decodable format
 3. VALIDATE quality metrics:
      - Estimate SNR (spectral gating noise floor)
      - Check clipping ratio
      - Detect cough presence and its duration
 4. IF quality inadequate THEN RETURN Inconclusive(reason)
 5. LOCATE cough event on the timeline
      - Energy-based onset detection within the clip
 6. CUT fixed 1-second segment centered on the cough event
      (segment length = 98.5th percentile of CoughVid cough
       durations ≈ 0.98 s, rounded to 1.0 s)
      IF clip shorter than 1 s THEN pad with silence
 7. RESAMPLE to training sample rate (22.05 kHz mono)
 8. NORMALIZE amplitude to [-1, 1]
 9. EXTRACT log-Mel spectrogram and MFCCs over the segment
10. APPLY training-time normalization (mean/std from training set)
11. RETURN fused feature tensors
END Audio Processing Algorithm
```

### 7.2 Model Training and Label Engineering

**The label problem (open design item).** CoughVid provides COVID-positive / COVID-negative style status labels plus metadata (symptoms, age range, gender, etc.), but the product spec requires three classes: Negative, Uncertain, Positive. The third class does not exist natively. Two candidate constructions:

- **Plan A — confident-band labeling (primary):** train the binary Negative/Positive classifier, and define the *Uncertain* class at inference time as the probability band between two thresholds on `P(positive)` (e.g. `[0.35, 0.65]`). The "three-class model" is then a binary model with a calibrated abstention band — the simplest construction that is honest about the labels available.
- **Plan B — symptom-derived third class (fallback):** use CoughVid metadata to label recordings with inconclusive or mixed symptom reports (neither confirmed COVID+ nor clean negative) as *Uncertain* and train a true 3-way softmax. Risk: noisy labels, since symptom metadata is self-reported.

**Plan A is the intended approach**, with Plan B reserved if the abstention band proves poorly calibrated. This choice is flagged for advisor review.

```
BEGIN Training Algorithm
 INPUT: CoughVid clips + labels
 1. FILTER clips: duration bounds, SNR floor, dedupe
 2. SPLIT train/val/test by speaker (no speaker leakage)
 3. FOR each training clip:
      - segment to 1 s (per §7.1)
      - AUGMENT (random, per-epoch):
          TimeMasking, FrequencyMasking, TimeStretch,
          additive ambient noise, pitch shift
      - EXTRACT joint MFCC + Mel features
 4. TRAIN CNN/ResNet classifier with class-weighted loss
      (handles residual class imbalance)
 5. EVALUATE on held-out test split:
      Accuracy, Precision, Recall, F1, ROC-AUC
 6. CALIBRATE confidence (temperature scaling) so probability
    bands are meaningful for the abstention gate
 7. EXPORT best checkpoint to ONNX
END Training Algorithm
```

### 7.3 Risk Classification Algorithm

```
BEGIN Risk Classification Algorithm
 INPUT: probs = (p_neg, p_unc, p_pos), quality_report
 IF quality_report failed THEN
    RETURN "INCONCLUSIVE - " + quality_report.reason
 END IF

 confidence = MAX(probs)
 IF confidence < 0.50 THEN
    RETURN "INCONCLUSIVE - low model confidence"
 END IF

 IF p_pos >= 0.65 THEN
    risk = "HIGH"
    recommendation = "Take an official COVID-19 test soon and
                      contact a healthcare provider if symptomatic"
 ELSE IF p_pos >= 0.35 THEN
    risk = "MEDIUM"
    recommendation = "Consider testing; mask up around others;
                      monitor symptoms over the next 48 hours"
 ELSE
    risk = "LOW"
    recommendation = "Continue normal precautions; re-check if
                      symptoms develop"
 END IF

 RETURN (risk, recommendation, probs, confidence)
END Risk Classification Algorithm
```

*(Thresholds 0.65 / 0.35 are the initial abstention band from §7.2 Plan A; final values set by calibration.)*

---

## 8.0 Performance and Optimization

### 8.1 Performance Targets

- **Capture to result:** < 5 seconds end-to-end (upload + preprocessing + inference + response) on a typical broadband connection.
- **Server inference:** < 1.5 seconds per analysis, excluding network transfer.
- **API payload:** < 500 KB upload per analysis, < 10 KB JSON response.
- **Concurrent analyses:** sustain 10 concurrent inference requests without failure on target hardware.
- **Client footprint:** initial bundle < 3 MB compressed; history of 1 year (≈365 records) < 2 MB in IndexedDB.
- **Availability of local history:** dashboard renders from IndexedDB in < 500 ms regardless of network state (history works offline).

### 8.2 Optimization Strategies

**Inference efficiency:** load the model once at process start; run under `torch.inference_mode()`; export to ONNX and serve via ONNX Runtime for CPU consistency; keep preprocessing (feature extraction) in the same worker to avoid serialization overhead.

**Pipeline latency:** accept compressed browser audio directly (avoid client-side re-encode to WAV); run quality gating before feature extraction so bad inputs fail fast; log per-stage timings against the 1.5 s inference budget.

**Future client-side inference (noted, not committed):** the ML Layer is kept free of FastAPI dependencies specifically so the ONNX export can later run in-browser via ONNX Runtime Web, eliminating upload latency entirely and enabling fully offline analysis. This is a post-v2 path; the v2 architecture is server-side as documented in §3.1.

**Storage:** single JSON-blob records per analysis keep IndexedDB access simple; export duplicates data out rather than requiring a sync protocol. Retention pruning runs lazily on app start.

---

## 9.0 References

1. CoughVid, "Coughvid Dataset," ETH Zurich, 2020, https://github.com/coughlanb/CoughVID.
2. "Convolutional Neural Network," Wikipedia, Wikimedia Foundation, 2023, https://en.wikipedia.org/wiki/Convolutional_neural_network.
3. PyTorch, "PyTorch Documentation," Meta AI, 2024, https://pytorch.org/docs/stable/.
4. FastAPI, "FastAPI Documentation," 2024, https://fastapi.tiangolo.com/.
5. Su, Y., Zhang, Y., Wang, P., & Madani, K. (2019). Environment Sound Classification Using a Two-Stream CNN with Decision-Level Fusion. *Sensors*, 19(7), 1733. https://doi.org/10.3390/s19071733
6. Qu, Y., Li, S., Qin, Z., & Lu, X. (2022). Acoustic scene classification based on three-dimensional multi-channel feature-correlated deep learning networks. *Scientific Reports*, 12, 17863. https://doi.org/10.1038/s41598-022-17863-z
7. Karim, S. H. T., et al. (2024). A Multi-Feature Fusion Approach for Dialect Identification using 1D CNN. *JOIV*, 8(3), 1246. https://doi.org/10.62527/joiv.8.3.2146
8. Chu, H.-C., Zhang, Y.-L., & Chiang, H.-C. (2023). A CNN Sound Classification Mechanism Using Data Augmentation. *Sensors*, 23(15), 6972. https://doi.org/10.3390/s231506972
9. Aggarwal, A., et al. (2022). Two-Way Feature Extraction for Speech Emotion Recognition. *Sensors*, 22(6), 2378. https://doi.org/10.3390/s22062378
10. World Health Organization, "COVID-19 testing," 2023, https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports.
