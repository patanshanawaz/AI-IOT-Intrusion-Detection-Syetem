<<<<<<< HEAD
# AI-IoT-IDS
AI-driven IoT Intrusion Detection System using machine learning to detect normal and malicious network traffic (DoS, probe, etc.). Includes preprocessing, RandomForest model, evaluation metrics, confusion matrix visualization, and optional Flask API for real-time inference.
=======
# AI-driven Intrusion Detection System for IoT Networks

## Author
Patan Shanawaz, B.Tech Computer Science Engineering – Cybersecurity  
Currently studying at BestIU

## 📝 Project Summary
This project implements a **baseline AI-driven Intrusion Detection System (IDS)** for **IoT networks**. It leverages machine learning to detect **network anomalies and attacks** such as DoS, probe, and other malicious traffic. The framework demonstrates practical **cybersecurity skills** and IoT security applications.

**Key Features:**
- Preprocessing of IoT network traffic datasets
- Baseline **RandomForest model** for multi-class attack detection
- Evaluation metrics: **Accuracy, Precision, Recall, F1-score**
- Confusion matrix visualization
- Optional **Flask API** for real-time inference / deployment
- Cloud / edge deployment-ready structure

---

## 🔗 Dataset Instructions

### Option A — NSL-KDD Dataset (Recommended)
1. Download the NSL-KDD train/test CSVs and place them in:
   - `data/raw/nsl_train.csv`
   - `data/raw/nsl_test.csv`

2. **Download Links / Mirror:**
   - [KDDTrain+.txt](https://www.unb.ca/cic/datasets/nsl.html) → Save as `data/raw/nsl_train.csv`
   - [KDDTest+.txt](https://www.unb.ca/cic/datasets/nsl.html) → Save as `data/raw/nsl_test.csv`

> **Note:** Ensure the files are CSV formatted. If using raw `.txt` files, remove headers and separate columns by commas.

### Option B — Fallback Synthetic Dataset
If NSL-KDD cannot be downloaded immediately, run the synthetic dataset generator:

```bash
python data/sample_generate.py
>>>>>>> 9cc61ec (Add local project files and dataset)
