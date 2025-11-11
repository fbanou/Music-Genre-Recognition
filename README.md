# 🎵 Music Genre Recognition

A **music genre classification** system that uses **Gaussian Mixture Models (GMMs)** and **Mel-Frequency Cepstral Coefficients (MFCCs)** for audio feature extraction. The project aims to classify musical tracks into different genres (e.g., Blues, Classical, Reggae) based on their audio features.

---

## ✨ Features

- 🎶 **Audio Preprocessing**: Extracts **MFCCs** from audio files to represent spectral features of the music
- 📊 **Model Training**: Trains a **Gaussian Mixture Model (GMM)** for each music genre using the Expectation-Maximization (EM) algorithm
- 🔍 **Genre Classification**: Classifies music genres based on the highest log-likelihood of the GMMs
- 📈 **Evaluation**: Measures performance using **accuracy** and **confusion matrix**

---

## ⚙ Technologies

- **Language**: Python  
- **Libraries**: librosa, numpy, scipy, joblib, scikit-learn  
- **Machine Learning**: Gaussian Mixture Models (GMM)  
- **Feature Extraction**: Mel-Frequency Cepstral Coefficients (MFCCs)  
