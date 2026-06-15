# MARS | Deepfake Audio Detection Suite

This repository implements a complete Machine Learning and Deep Learning system to detect **Deepfake Audio (AI-Generated)** vs. **Genuine Audio (Human speech)**. 

The system achieves high-performance classification, satisfying the strict requirements:
- **Overall Accuracy**: &ge; 80%
- **Equal Error Rate (EER)**: &le; 12%
- **F1-Score**: &ge; 80%
- **Per-Class Accuracy**: &ge; 75%

---
Live deployment is not currently hosted. Run locally using:

python -m streamlit run app/app.py

## 🚀 Key Features

1. **Dual-Model Architecture**:
   - **Machine Learning Baseline**: Random Forest Classifier trained on statistical summaries of acoustic features.
   - **Deep Learning Model**: A 2D Convolutional Neural Network (CNN) trained on Mel-Frequency Cepstral Coefficients (MFCC) spectrogram matrices.
2. **Robust Feature Extraction**: Extracts MFCCs, Delta MFCCs, Delta-Delta MFCCs, Spectral Centroid, Spectral Rolloff, Zero Crossing Rate, and RMS energy.
3. Dataset Support: Supports both generated datasets for rapid prototyping and the Fake-or-Real (FoR) dataset for real-world deepfake audio detection experiments.
4. **Interactive Dashboard**: A premium Streamlit web app with player integration, visual waveform/spectrogram rendering, and real-time confidence scores.
## Dataset

This project was trained and evaluated using a subset of the Fake-or-Real (FoR) Dataset (for-norm training split), containing genuine human speech and AI-generated speech samples. A balanced subset of 500 genuine and 500 deepfake recordings was used for experimentation and model evaluation.
---

## 📂 Project Structure

```
D:\MARS_ML\
├── app\
│   └── app.py                # Streamlit Web Application
├── docs\
│   ├── performance_report.md # Performance metrics report
│   ├── rf_confusion_matrix.png
│   └── cnn_confusion_matrix.png
├── notebooks\
│   └── deepfake_audio_detection.ipynb # Jupyter Notebook containing full running code
├── scripts\
│   ├── prepare_dataset.py    # Simulated dataset generator
│   └── test_audio.py         # Command-line audio inference tool
├── src\
│   ├── __init__.py
│   ├── features.py           # Librosa-based acoustic feature extraction
│   ├── models.py             # PyTorch CNN model architecture
│   └── train.py              # Model training and evaluation script
├── .gitignore
├── requirements.txt          # Python dependencies
└── README.md                 # Project description and guide
```

---

## ⚙️ Preprocessing & Feature Extraction

Audio signals are resampled to a uniform rate of **16,000 Hz** and processed to extract the following acoustic properties:
- **MFCCs (13 or 20 coefficients)**: Captures the spectral envelope representing the shape of the vocal tract.
- **Delta & Delta-Delta MFCCs**: Capture the dynamic changes and trajectories of speech spectral properties.
- **Spectral Centroid**: Computes the "center of mass" of the spectrum, capturing brightness or robotic "hiss" differences.
- **Spectral Rolloff**: The frequency below which a certain percentage (e.g. 85%) of total spectral energy lies.
- **Zero Crossing Rate (ZCR)**: The rate at which the signal changes signs, useful for identifying synthetic high-frequency noise.
- **RMS Energy**: Measures signal power over time.

---

## 🤖 Model Architectures

### 1. Machine Learning (Random Forest)
A Random Forest Classifier with 100 estimators is trained on a 1D feature vector of summary statistics (mean and standard deviation) computed across all time-frames. This serves as a fast, robust, and highly interpretable baseline.

### 2. Deep Learning (PyTorch CNN)
A 2D Convolutional Neural Network processes raw MFCC spectrograms of shape `(batch, 1, 20, 128)`. The architecture consists of:
- Three Conv2D blocks with Batch Normalization, ReLU activations, and MaxPool2D layers.
- A fully connected dense layer with Dropout (0.5) to prevent overfitting.
- A single raw logit output with Binary Cross-Entropy loss.

---

## 📊 Verification Metrics

Training evaluation results on the validation set:

| Model | Accuracy | Equal Error Rate (EER) | F1-Score | Genuine Accuracy | Deepfake Accuracy |
|---|---|---|---|---|---|
| **Random Forest** | 92.0% | 5.0% | 92.38% | 87.0% | 97.0% |
| **PyTorch CNN** | 95.0% | 6.0% | 95.15% | 92.0% | 98.0% |

*Note: Results are reported on a subset of the Fake-or-Real (FoR) dataset using the normalized training split. Performance may vary when evaluated on the complete dataset or cross-dataset benchmarks such as ASVspoof 2019.*

---

## 🛠️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare/Generate Dataset
Generate a simulated dataset (optional):
```bash
python scripts/prepare_dataset.py
```
To train on the real Kaggle dataset: Download and place the .wav files inside data/genuine and data/deepfake.

### 3. Run Training Pipeline
Train both the Random Forest and PyTorch models, evaluate metrics, and save checkpoints:
```bash
python -m src.train
```

### 4. Test Single Audio File
Test a new WAV file via the CLI:
```bash
python scripts/test_audio.py --audio D:\MARS_ML\data\deepfake\deepfake_000.wav --model_type rf
python scripts/test_audio.py --audio D:\MARS_ML\data\genuine\genuine_000.wav --model_type cnn
```

### 5. Launch Streamlit Web App
Run the interactive dashboard locally:
```bash
streamlit run app/app.py
```
*The app will automatically open at `http://localhost:8501` in your browser.*
