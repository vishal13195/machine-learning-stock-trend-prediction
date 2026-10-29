# P-396: Machine Learning-Based Stock Market Trend Prediction System (RELIANCE)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()
[![Platform](https://img.shields.io/badge/Platform-VS%20Code%20%7C%20Google%20Colab-orange.svg)]()
[![Frameworks](https://img.shields.io/badge/Engines-PyTorch%20%7C%20TensorFlow%20%7C%20Scikit--Learn%20%7C%20XGBoost-blueviolet.svg)]()

An end-to-end quantitative machine learning and deep learning software pipeline designed to forecast the **Next-Day Directional Trend (UP/DOWN)** of Indian equity securities, specifically **Reliance Industries Limited (`RELIANCE.NS`)** on the **National Stock Exchange of India (NSE)**.

---

## 📌 Project Overview

* **Project ID:** P-396
* **Academic Year:** 2024–2026
* **Department:** School of Computer Science & Engineering
* **Institution:** IILM University, Greater Noida

### 👥 Project Team
* **Vineet Gupta** (Enrollment: 2410030229)
* **Vishal Mansuriya** (Enrollment: 2410030220)
* **Nayan Pandey** (Enrollment: 2410030227)
* **Amar Singh Yadav** (Enrollment: 2410030145)

**Project Guide:** Ms. Ritu Dewan, Assistant Professor, School of CSE  

---

## 🚀 Key Features

* **Real Historical Market Data:** Ingests 11+ years (2015 to 2026, 2,911 trading sessions) of genuine daily OHLCV prices from Yahoo Finance API with automatic offline snapshot caching.
* **Quantitative Technical Indicators:** Automatically computes 7 mathematical indicators across trend, momentum, and volatility:
  * Simple Moving Average (**SMA 10**)
  * Exponential Moving Average (**EMA 10**)
  * Wilder's Smoothed Relative Strength Index (**RSI 14**)
  * Moving Average Convergence Divergence (**MACD 12, 26**) & **MACD Signal (9)**
  * Bollinger Bands (**Upper & Lower 20, 2$\sigma$**)
* **Strict Anti-Leakage Chronological Partitioning:** Enforces a non-shuffled temporal split (70% Train, 15% Validation, 15% Test) with train-isolated feature scalers to prevent look-ahead bias.
* **Multi-Model Benchmarking:** Implements and tunes 5 distinct algorithms across Raw OHLCV and Technical Indicator feature spaces (10 configurations total):
  1. Decision Tree Classifier
  2. Random Forest Ensemble
  3. Support Vector Machine (SVM with RBF Kernel)
  4. Extreme Gradient Boosting (XGBoost)
  5. Long Short-Term Memory (LSTM) Recurrent Neural Network
* **Dual-Engine Deep Learning (Universal Compatibility):** Seamlessly executes using **PyTorch** in local environments (VS Code/Windows) and **TensorFlow / Keras** in cloud runtimes (Google Colab) with zero configuration needed.
* **Automated Production Artifacts:** Automatically exports evaluation reports (`model_comparison_results.csv`), high-resolution visual plots, and live forward trading signals (`final_output.json`).
* **Self-Verification Test Suite:** Runs 6 internal automated unit tests verifying mathematical correctness, temporal monotonicity, target causality, and zero future data leakage.

---

## 🏗️ System Architecture

```
[ Yahoo Finance API (RELIANCE.NS) ]
                 │
                 ▼
┌────────────────────────────────────────┐
│  Layer 1: Ingestion & Hygiene Engine   │
│  • Automated daily OHLCV download      │
│  • Local snapshot caching (offline)    │
│  • Data validation & holiday filtering │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Layer 2: Quantitative Feature Engine  │
│  • 7 Rolling Technical Indicators      │
│  • Next-day directional target (0 / 1) │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Layer 3: Temporal Anti-Leakage Split  │
│  • Strict Chronological Partitioning   │
│  • 70% Train / 15% Val / 15% Test      │
│  • Train-Fitted Standard Scalers       │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Layer 4: Multi-Model Machine Learning │
│  • Decision Tree, Random Forest, SVM   │
│  • XGBoost (Gradient Boosted Trees)    │
│  • Dual-Engine LSTM (PyTorch & TF)     │
│  • Hyperparameter Validation Tuning    │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Layer 5: Statistical Eval & Inference │
│  • Out-of-sample Test Evaluation (432d)│
│  • 95% Confidence Intervals & p-values │
│  • Next-Day Trade Signal (JSON Export) │
│  • Automated Visual Benchmark Plots    │
└────────────────────────────────────────┘
```

---

## 📊 Experimental Results & Benchmarks

The models were rigorously evaluated on an untouched, out-of-sample chronological test set of **432 trading days** (January 2025 to October 2026):

* **Majority Baseline:** **49.1%** (Accuracy of always predicting the training majority class)
* **Random Chance:** **50.0%**

### Performance Summary

| Model | Feature Set | Accuracy | Precision | Recall | F1-Score | vs. Baseline | $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Top Model)** | **Technical Indicators** | **54.4%** | **60.9%** | **19.8%** | **0.299** | **+5.3%** | **$p = 0.015$** *(Sig)* |
| **Random Forest** | Raw OHLCV | **53.2%** | **75.0%** | 7.1% | 0.129 | +4.2% | $p = 0.046$ *(Sig)* |
| **XGBoost** | Raw OHLCV | **53.0%** | **69.6%** | 7.5% | 0.136 | +3.9% | $p = 0.056$ |
| **Decision Tree** | Raw OHLCV | **52.1%** | 100.0% | 2.4% | 0.046 | +3.0% | $p = 0.114$ |
| **SVM (RBF Kernel)** | Technical Indicators | **50.9%** | 50.0% | 4.2% | 0.078 | +1.9% | $p = 0.235$ |
| **LSTM Neural Net** | Technical Indicators | **50.2%** | 49.5% | 74.5% | 0.595 | +1.2% | $p = 0.332$ |

### Key Experimental Findings
1. **Tree Ensembles Outperform Neural Networks:** XGBoost and Random Forest consistently outperformed the sequential LSTM on tabular daily price data.
2. **Indicator Boost:** Incorporating technical indicators improved XGBoost test accuracy from 53.0% to 54.4%.
3. **Statistical Significance:** XGBoost achieved a Binomial test $p$-value of **0.015** ($< 0.05$), confirming the directional edge over baseline is statistically genuine.
4. **Top Predictive Indicators:** Normalized feature importance identified **SMA(10)** (24.2%), **RSI(14)** (14.7%), and **Volume** (8.3%) as the highest-ranked features.

---

## 📁 Repository Structure

```
Major Project v1/
├── stock_trend_pipeline_universal.ipynb    # Interactive Jupyter Notebook (VS Code & Colab)
├── stock_trend_pipeline.py                 # Standalone Headless / CLI Script
├── requirements.txt                        # Project dependencies
├── README.md                               # Project documentation (this file)
├── DESIGN_AND_ARCHITECTURE_DOCUMENT.md    # System Design & Architecture Document
├── TECHNICAL_DOCUMENTATION.md              # Technical User Manual & API Guide
├── data_snapshot_yahoo.csv                 # Cached 11-year Reliance Market History
└── outputs/                                # Auto-generated Production Artifacts
    ├── model_comparison_results.csv        # Tabular Test Metrics for all 10 models
    ├── model_comparison_chart.png          # High-resolution benchmark comparison chart
    ├── feature_importances.csv             # Feature importance weights
    ├── feature_importance_chart.png        # Top feature contribution plot
    ├── confusion_matrices.txt              # True/False Positives & Negatives breakdown
    ├── chronological_split_dates.txt       # Train/Val/Test calendar date bounds
    └── final_output.json                   # Machine-readable next-day live trade signal
```

---

## ⚡ Quick-Start Guide

### 1. Prerequisites
* Python 3.10 – 3.12 (64-bit)
* Pip package manager

### 2. Installation
Clone or navigate to the project directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Execution Pathways

#### Pathway A: VS Code Interactive Notebook
1. Open `stock_trend_pipeline_universal.ipynb` in VS Code.
2. Select your Python kernel in the top-right.
3. Click **Run All** to execute all cells and inspect inline plots.

#### Pathway B: Google Colab (One-Click Cloud)
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `stock_trend_pipeline_universal.ipynb`.
3. Click **Runtime** $\rightarrow$ **Run all**.

#### Pathway C: Standalone Command Line / Terminal
```bash
python stock_trend_pipeline.py
```
*Execution takes approximately 25–30 seconds on a standard quad-core CPU.*

---

## 🔮 Next-Day Forward Signal Sample

At the end of each run, the pipeline automatically writes the forward signal to `outputs/final_output.json`:
```json
{
  "last_data_date": "2026-10-19",
  "model": "XGBoost",
  "feature_set": "indicator",
  "p_up": 0.544,
  "prediction": "UP"
}
```

---

