# 🫀 ECG Multimodal Diagnosis

*A unified framework for multimodal ECG analysis and dialysis-related cardiac risk evaluation*

---

## 📘 Overview

This repository hosts a modular research framework that integrates **electrocardiogram (ECG) data processing**, **deep feature extraction**, and **clinical multimodal fusion** for patient-level diagnosis and risk stratification.

The system is built around the **PTB-XL** public ECG dataset as a verification benchmark, and is designed for extension to **dialysis patient cohorts** for real-world clinical validation.

---

## 🧩 Project Architecture

```
ecg_multimodal_diagnosis/
│
├── data/                 # All raw & intermediate data sources
│   ├── ptbxl/            # PTB-XL dataset for baseline validation
│   ├── paperecg/         # image to digital signal converter 
│   ├── clinical/         # Future integration: dialysis EHR & lab data
│   └── intermediate/     # Cached and processed signals
│
├── modules/              # Core model components (feature extractors, fusion)
├── pipelines/            # End-to-end scripts (load → inference → evaluate)
├── configs/              # Experiment and model configuration YAMLs
├── notebooks/            # Exploratory analysis and visualization
├── results/              # Metrics, feature embeddings, confusion matrices
└── tests/                # Unit and integration tests
```

---

## ⚙️ Features

* ✅ **PTB-XL loader** with parallel I/O and caching for large-scale ECGs
* ✅ **ECGFounder integration** for deep feature extraction (1024-D embeddings)
* ✅ **Feature validation metrics:** mean/std, cosine similarity, inter-class distance
* ✅ **t-SNE & PCA visualization** for model interpretability
* 🩸 **Clinical data fusion layer** (in progress) for dialysis-related cardiovascular risk prediction

---

## 🚀 Quick Start

```bash
# 1. Activate environment
conda activate ECG_FOUNDER

# 2. Run PTB-XL data loader (sample)
python data/ptbxl/readingdata.py

# 3. Launch feature validation
python pipelines/feature_validation.py
```

---

## 🧠 Research Goal

This project aims to create a **trustworthy multimodal diagnostic framework** that can:

* Reproduce published ECG embeddings (e.g., PaperECG / ECG-FounDer)
* Verify pipeline correctness using PTB-XL as a reference
* Extend to **dialysis patients** for real-world clinical cardiac risk prediction

---

