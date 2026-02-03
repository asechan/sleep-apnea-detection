# Sleep Apnea Detection from Audio using Deep Learning

End-to-end deep learning system for detecting sleep apnea events from overnight breathing audio recordings using CNNs and mel-spectrogram features.

This project processes raw sleep audio into time-aligned clinical predictions and computes a full **Apnea-Hypopnea Index (AHI)** score for sleep severity classification.

---

## Features

- Raw audio → Mel Spectrogram → CNN pipeline
- Subject-independent training and validation
- Apple Silicon (MPS) acceleration support
- Automatic epoch segmentation (30s windows)
- Event merging and clinical AHI scoring
- Real-time inference ready
- Clean, modular Python codebase

---

## Project Structure
sleep-apnea/
├── src/
│   ├── data_sync_apsaa.py
│   ├── epoch_labeler.py
│   ├── make_dataset.py
│   ├── train_cnn.py
│   └── infer_apnea.py
├── models/
│   └── cnn.py
├── data/
│   ├── raw/        # Not tracked (APSAA dataset)
│   ├── processed/ # Generated epochs + labels
│   └── features/  # Mel spectrograms
├── checkpoints/   # Saved models (not tracked)
└── README.md
---

## Dataset

This project is designed for the **APSAA Sleep Audio Dataset**.

Due to licensing restrictions, the dataset is **not included**.  
Download it separately and place it in: data/raw/

---

## Setup

### Create Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Sync Dataset
```bash
python src/data_sync_apsaa.py --raw_dir data/raw --out_csv data/manifest_apsaa.csv
```
### Generate Epochs & Labels
```bash
python src/epoch_labeler.py
```
### Extract Features
```bash
python src/make_dataset.py
```
### Train Model
```bash
python src/train_cnn.py --epochs 25
```
### Run Inference
```bash
python src/infer_apnea.py your_audio.wav
```
### Output Example
```bash
===== SLEEP APNEA REPORT =====
Total sleep time: 7.19 hours
Detected events: 42
AHI: 5.83
Severity: Mild
```
### Model
	•	Architecture: CNN on Mel-Spectrograms
	•	Input: 64xT Mel Features
	•	Output: Binary apnea probability
	•	Threshold tuned via validation F1-score
### Hardware Support
	•	CPU
	•	Apple Silicon (MPS)
	•	CUDA (optional)
License

MIT License
