# 🛸 deep-space-sentinel

> AI-powered anomaly detection and autonomous decision support system for deep space missions — combining Isolation Forest, One-Class SVM, PyTorch Autoencoder, and LSTM models on NASA's CMAPSS turbofan dataset to predict engine failures and trigger mission-critical autonomous responses.

---

## 📌 Overview

**deep-space-sentinel** simulates an onboard AI system capable of monitoring spacecraft engine telemetry in real time, detecting degradation anomalies, predicting Remaining Useful Life (RUL), and issuing autonomous mission decisions — from routine monitoring all the way to emergency shutdown protocols.

Built on NASA's CMAPSS FD001 dataset, this project explores how multi-model ensemble anomaly detection and deep learning can support autonomous decision-making in environments where ground-station communication latency makes human-in-the-loop responses impractical.

---

## 🗂️ Project Phases

| Phase | Description |
|-------|-------------|
| **Phase 1** | EDA, Feature Engineering, Rolling Statistics |
| **Phase 2** | Statistical Baseline, Isolation Forest, One-Class SVM, PyTorch Autoencoder, LSTM Classifier, LSTM RUL Regressor, Model Evaluation |
| **Phase 3** | Autonomous Decision Layer — Rule-Based Logic, AI Risk Scoring, Decision Confidence Estimation |
| **Phase 4** | Interactive Visualization Dashboards (Plotly + Matplotlib) |

---

## 📊 Dataset

**NASA CMAPSS Turbofan Engine Degradation Simulation Dataset (FD001)**

- 100 training engines, 100 test engines
- 21 sensor readings + 3 operational settings per cycle
- Ground-truth RUL values provided for test set
- Failure threshold: **RUL ≤ 30 cycles**

🔗 [Kaggle Link](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

Place the following files in the project root before running:
```
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

---

## 🧠 Models

### Anomaly Detection
| Model | Approach |
|-------|----------|
| **Z-Score Baseline** | Statistical thresholding on normalized sensor means |
| **Isolation Forest** | Ensemble-based unsupervised anomaly detection |
| **One-Class SVM** | Kernel-based boundary learning on normal data |
| **PyTorch Autoencoder** | Reconstruction-error anomaly scoring with early stopping |

### Sequence Models
| Model | Task |
|-------|------|
| **LSTM Classifier** | Binary failure prediction from 30-cycle sensor sequences |
| **LSTM Regressor** | Remaining Useful Life (RUL) regression (capped at 125 cycles) |

---

## ⚙️ Autonomous Decision Layer

The decision system integrates outputs from multiple models to produce a unified **risk score (0–100)** and a corresponding autonomous action:

| Risk Level | Score Range | Action |
|------------|-------------|--------|
| LOW | 0–19 | Nominal Operations |
| MODERATE | 20–39 | Increase Monitoring Frequency |
| ELEVATED | 40–59 | Switch to Redundant Subsystem |
| HIGH | 60–79 | Alert Ground Station + Safe Mode |
| CRITICAL | 80–100 | Emergency Shutdown + Alert Earth |

Decision **confidence** is also estimated based on inter-model agreement and margin from decision boundaries.

---

## 📈 Dashboards

Three interactive/static dashboards are generated:

- **`mission_control_dashboard.html`** — Risk scores, anomaly scores, risk level distribution, decision confidence (Plotly)
- **`live_telemetry_dashboard.html`** — Per-engine sensor trends and RUL curves (Plotly)
- **`model_summary.png`** — Comparative model performance bar charts (Matplotlib)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/deep-space-sentinel.git
cd deep-space-sentinel
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset files
Download the CMAPSS FD001 files from Kaggle and place them in the project root.

### 4. Run the notebook
```bash
jupyter notebook Main.ipynb
```

Run all cells sequentially. GPU acceleration is used automatically if available.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **PyTorch** — Autoencoder, LSTM Classifier, LSTM Regressor
- **scikit-learn** — Isolation Forest, One-Class SVM, metrics
- **Pandas / NumPy** — Data processing and feature engineering
- **Plotly** — Interactive mission dashboards
- **Matplotlib / Seaborn** — Static plots and model evaluation visuals

---

## 📋 Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
plotly
jupyter
```
