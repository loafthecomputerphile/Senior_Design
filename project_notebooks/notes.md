# Meeting notes


## 10th Sept:
- get notes on why to jointly use MFCC and Mel spectrogram
- link with papers to support this
- add audio play of raw, segmented, and augmented cough
- segment by length ( do experimentally, most likely will use the mean after processing every high confidence wav file)


### make a flow chart of our steps:
- the overall archetecture of our project
  - currently have load -> segment -> augment
-   have to add in noise cutting


### 10 recent papers for covid detection
year, topic, table, drawbacks, furture work, and output








## 1. Complementary Feature Sets

* **Mel Spectrograms**  
  Keep the complete 2D time-frequency grid intact. Standard 2D convolutional kernels easily pick up on continuous energy distributions, pitch contours, harmonics, and formant shifts across adjacent bins (Su et al., 2019; Aggarwal et al., 2022).

* **MFCCs**  
  Applying a Discrete Cosine Transform (DCT) to log-Mel energies decorrelates the frequency channels. This compresses overall vocal tract shape and timbre into a few low-order coefficients while filtering out fine pitch variations (Su et al., 2019).

---

## 2. Overcoming CNN Spatial Assumptions

Standard 2D CNN filters assume that adjacent pixels are locally correlated. While Mel spectrograms fit this visual model, MFCCs present a decorrelated, orthogonal view. 

Combining both prevents the network from over-relying solely on local pixel continuity, balancing broad spatial energy maps with explicit timbre representations (Qu et al., 2022; Chu et al., 2023).

---

## 3. Better Noise Handling and Generalization

Higher-order MFCC coefficients degrade quickly in noisy environments, but Mel spectrograms retain broader ambient context across frequency bands. 

Fusing both feature streams helps models generalize better and maintain classification accuracy in noisy conditions compared to single-feature networks (Su et al., 2019; Karim et al., 2024).

---

## References

* **Aggarwal, A., Srivastava, A., & Agarwal, A. et al.** (2022). Two-Way Feature Extraction for Speech Emotion Recognition Using Deep Learning. *Sensors*, 22(6), 2378. https://doi.org/10.3390/s22062378
* **Chu, H.-C., Zhang, Y.-L., & Chiang, H.-C.** (2023). A CNN Sound Classification Mechanism Using Data Augmentation. *Sensors*, 23(15), 6972. https://doi.org/10.3390/s23156972
* **Karim, S. H. T., Ghafoor, K. J., Abdulrahman, A. O., & Hama Rawf, K. M.** (2024). A Multi-Feature Fusion Approach for Dialect Identification using 1D CNN. *JOIV: International Journal on Informatics Visualization*, 8(3), 1246. https://doi.org/10.62527/joiv.8.3.2146
* **Qu, Y., Li, X., Qin, Z., & Lu, Q.** (2022). Acoustic scene classification based on three-dimensional multi-channel feature-correlated deep learning networks. *Scientific Reports*, 12, 17863. https://doi.org/10.1038/s41598-022-17863-z
* **Su, Y., Zhang, K., Wang, J., & Madani, K.** (2019). Environment Sound Classification Using a Two-Stream CNN Based on Decision-Level Fusion. *Sensors*, 19(7), 1733. https://doi.org/10.3390/s19071733
"""

___



































## To be transfered into WORD file

### Phase 1
- we did an experiment to figure out the distribution of cough sound lengths. it was rightly skewed and as such we use d the 98.5 percentile. due to that the length of a cough segment was 0.98s and we rounded it to 1s
- using this segment params we tested it on an actual audio and it was succeful
- we are currently using TimeMasking, FrequencMaxing and TimeStrech. we should allow for more audio augmentatiuons along with intermediate class variable to test augmentation's effect on the cough sound
- MFCCs get the fine grade timbre and coals of the audio whule the SPectrogram, pitch, harmonics