# 1. Project Title: Covid-19 Detection

## 2. Team Members:

* Drew Quashie
* Amanda Ogbonna
* Loleyi Oluwatomisin
* Richard Alonso Garcia

## 3. Faculty Advisor:
  ### &emsp; Name:  Dr. Zahra Nematzadeh
  ### &emsp;  Email: znematzadeh@fit.edu
#### &emsp;
## 4. Client:

* Medical Professionals


* Potential Covid Patients



## 5. Meeting Dates

1. Tuesday August 25th 11:15am

* Olin Engineering Complex room 353

## 6. Goals and Motivations:

* Allow users to detect COVID-19 early using accessible devices like smartphones or computers.


* Develop a web app that can predict COVID-19 infections by analyzing coughing sounds.


* Provide real-time risk assessments and longitudinal health tracking to assist both patients and medical professionals.



## 7. Approach:

* **Phase 1: Data Preprocessing & Augmentation:** Extract Mel-Frequency Cepstral Coefficients (MFCCs) and spectrogram images from raw audio signals in the CoughVid dataset (~2,800 recordings). Apply signal normalization, SpecAugment, pitch shifting, time shifting/stretching, and additive noise to improve robustness. Partition data into 70% training, 15% validation, and 15% test sets with k-fold cross-validation.


* **Phase 2: Deep Learning Model Development:** Build, train, and fine-tune Convolutional Neural Networks (CNNs) and leverage transfer learning architectures (e.g., ResNet) using spectrogram inputs to classify COVID-19 status.


* **Phase 3: Model Evaluation:** Measure performance using Accuracy, Precision, Recall, F1 Score, and ROC-AUC metrics on an independent test dataset.


* **Phase 4: Web Application Integration:** Develop a full-stack web application capable of capturing audio in real time, running feature extraction, serving backend inference predictions, and displaying risk feedback alongside guidance on next steps.



## 8. Novel Features/Functionalities

* **Web-Based Audio Capture:** Browser-accessible interface enabling users to record cough sounds directly without requiring specialized hardware.


* **Automated Risk Assessment & Guidance:** Real-time output providing risk levels (Low, Medium, High) paired with clinical recommendations (e.g., self-isolation, contacting a healthcare provider).


* **Longitudinal Cough Tracking Dashboard:** Interactive health tracking visualization monitoring daily cough frequencies and health trends over time.



## 9. Algorithms and Tools

| Domain | Tool / Technique |
| --- | --- |
| **Frontend Framework** | React, Web Audio API |
| **Backend & API** | Python, FastAPI |
| **Machine Learning** | PyTorch, Librosa (Feature Extraction) |
| **Model Architectures** | Convolutional Neural Networks (CNNs), ResNet (Transfer Learning)|
| **Dataset** | CoughVid Dataset (2,800 audio recordings with status and metadata)|
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1 Score, ROC-AUC|

## 10. Technical Challenges

* **Audio Noise Variability:** High ambient background noise and microphone hardware discrepancies across user web browsers require robust audio preprocessing, filtering, and data augmentation.


* **Real-Time Inference Latency:** Optimizing the pipeline between browser-based web recording, audio feature conversion, and server-side PyTorch execution to ensure near-instant feedback for the user.
* **Dataset Class Imbalance & Generalization:** Addressing potential sample distribution biases in cough audio to maintain model accuracy across diverse demographic profiles and symptom severities.



## 11. Milestone 1 (September 28th)

* Complete project planning, requirements documentation, and system architecture design.


* Set up CoughVid dataset pipeline: audio filtering, feature extraction (MFCCs/Spectrograms), and normalization.


* Implement data augmentation techniques (additive noise, pitch/time shifting, SpecAugment).


* Build baseline PyTorch CNN model structure and React frontend audio recording component.



## 12. Milestone 2 (October 26th)

* Train and fine-tune advanced CNN models and transfer learning architectures (e.g., ResNet).


* Complete initial model evaluation and generate performance reports (F1 Score, ROC-AUC).


* Develop FastAPI backend endpoints for handling client audio uploads and running inference pipelines.
* Build initial React web user interface components for displaying prediction results and risk feedback.



## 13. Milestone 3 (November 23rd)

* Integrate React frontend, FastAPI backend, and PyTorch model for end-to-end real-time predictions.


* Implement the longitudinal cough tracking visualization dashboard.


* Complete full system testing, performance optimization, and bug resolution.


* Finalize all project deliverables: demonstration video, poster, and final project presentation and publish paper.



## 14. Milestone 1 Task Matrix

| Task Description | Assigned Team Member | Status |
| --- | --- | --- |
| Requirements & Project Plan Documentation | Amanda Ogbonna | Completed |
| CoughVid Audio Data Preprocessing & MFCC/Spectrogram Pipeline | Drew Quashie | In Progress |
| Audio Augmentation Pipeline (Noise, Pitch & Time Shifts) | Drew Quashie | In Progress |
| Baseline PyTorch CNN Architecture Development | Loleyi Oluwatomisin Amanda Ogbonna | Pending |
| React Audio Recording Interface Setup | Richard Alonso Garcia | Pending |

## 15. Faculty Approval

**Faculty Advisor Name:** Dr. Zahra Nematzadeh

**Signature:** ___________________________

**Date:** _______________________________