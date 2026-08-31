# 1. Project Title: Covid-19 Detection

## 2. Team Members:

* Drew Quashie (dquashie2024@my.fit.edu)
* Amanda Ogbonna (kogbonna2025@my.fit.edu)
* Loleyi Oluwatomisin (ooluwatomisi2023@my.fit.edu)
* Richard Alonso Garcia (ralonsogarci2023@my.fit.edu)

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


* Many persons during the panademic would go to testing centers thinking they were sick thus overloading the system. Our product will help lower the load on essential workers by providing an easy pre screaning method


* This will be achieved by providing real-time risk assessments and longitudinal health tracking to assist both patients and medical professionals.



## 7. Approach:

* **Web-Based Audio Capture:** The patient can record cough sounds directly without requiring specialized hardware.


* **Automated Risk Assessment & Guidance:** The patient will be provided risk levels (Low, Medium, High) paired with clinical recommendations (e.g., self-isolation, contacting a healthcare provider).


* **Longitudinal Cough Tracking Dashboard:** The patient can recieve interactive health tracking visualization that monitors daily cough frequencies and health trends over time.



## 8. Novel Features/Functionalities

* The Cough Tracking Dashboard is unique as it allows users to gauge their condition over time while allowing health professions see patient's daily health progress to enable more streamlined testing of patients




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