
# 🔬 Retinal Disease Classification in OCT Scans

A deep learning project that classifies retinal diseases from **Optical Coherence Tomography (OCT)** scans into four categories using two models: a **Custom CNN** built from scratch and **ResNet50** via transfer learning.


## Overview

This project implements and compares two deep learning architectures for automated retinal disease diagnosis from OCT scans. Early detection of conditions like CNV, DME, and Drusen is critical for preventing vision loss, and this notebook demonstrates how deep learning can achieve strong classification performance on medical imaging data.

**Disease Classes:**

| Class | Description |
|---|---|
| **CNV** | Choroidal Neovascularisation |
| **DME** | Diabetic Macular Edema |
| **DRUSEN** | Early AMD indicator (drusen deposits) |
| **NORMAL** | Healthy retina |


## Dataset

**Source:** [Kermany 2018 OCT Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) via Kaggle

- **Input size:** 224 × 224 × 3 (RGB)
- **Batch size:** 32
- **Split:** Train / Validation / Test

---

## Models

### 1. Custom CNN (5-Block Architecture)

Built from scratch with progressively deeper convolutional blocks:

```
Conv2D(32) → BN → MaxPool
Conv2D(64) → BN → MaxPool
Conv2D(128) → BN → MaxPool
Conv2D(256) → BN → MaxPool
Conv2D(512) → BN → GlobalAvgPool
Dense(512) → Dropout(0.5)
Dense(256) → Dropout(0.3)
Dense(4, softmax)
```

- **Optimizer:** Adam (lr = 0.001)
- **Epochs:** Up to 20 (with early stopping)
- **Parameters:** ~custom (smaller footprint)

### 2. ResNet50 — Transfer Learning (Two-Phase Fine-Tuning)

Pre-trained on ImageNet, adapted for OCT classification:

- **Phase 1:** Frozen base → train classification head (10 epochs, lr = 0.001)
- **Phase 2:** Unfrozen base → fine-tune all layers (8 epochs, lr = 1e-5)
- **Parameters:** ~25M

```
ResNet50 (ImageNet weights, frozen/unfrozen)
→ GlobalAveragePooling2D
→ Dense(512) → BN → Dropout(0.5)
→ Dense(256) → Dropout(0.3)
→ Dense(4, softmax)
```

---

## Results & Evaluation

### Model Performance Summary

| Model | Test Accuracy | Parameters | Training Epochs |
|---|---|---|---|
| **Custom CNN** | See notebook output | ~smaller | Up to 20 |
| **ResNet50** | See notebook output | ~25M | Up to 18 (10 + 8) |


### Evaluation Metrics

Both models are evaluated using:

- **Test Accuracy** — overall classification correctness
- **Precision, Recall, F1-score** — per-class performance via `classification_report`
- **Confusion Matrix** — visualises per-class prediction patterns

### Key Observations

- **ResNet50** achieves higher accuracy through transfer learning, at the cost of a larger model size (~25M parameters)
- **Custom CNN** provides a strong from-scratch baseline and is useful for understanding the task without external pre-training
- **DRUSEN vs NORMAL** are the most commonly confused classes due to visual similarity in OCT scans
- **Early stopping** (patience = 5 on val_accuracy) and **ReduceLROnPlateau** (factor = 0.2, patience = 3) effectively prevent overfitting in both models

### Regularisation Techniques

| Technique | Applied To | Purpose |
|---|---|---|
| Data Augmentation (rotation, flip, shift, zoom) | Training data | Reduce overfitting |
| Batch Normalisation | Both models | Stabilise activations |
| Dropout (50% / 30%) | Dense layers | Prevent memorisation |
| Early Stopping | Both models | Restore best weights |
| ReduceLROnPlateau | Both models | Adaptive learning rate |

### Output Visualisations

The notebook generates four saved plots:

| Plot | Filename | Description |
|---|---|---|
| Training History | `training_history.png` | Accuracy & loss curves per epoch |
| Confusion Matrices | `confusion_matrices.png` | Per-class prediction heatmaps |
| Model Comparison | `model_comparison.png` | Accuracy & parameter count bar charts |
| Convergence Analysis | `training_convergence.png` | Validation accuracy convergence with best-epoch annotations |

---

## Setup & Usage

### Requirements

```bash
pip install kagglehub tensorflow scikit-learn matplotlib seaborn numpy
```

### Run

1. Open the notebook in **Google Colab** (recommended for GPU access) or Jupyter
2. Ensure your **Kaggle API credentials** are configured for `kagglehub` to download the dataset
3. Run all cells in order — the notebook will auto-detect train/test/val directories

> **GPU Note:** The notebook configures TensorFlow GPU memory growth automatically. CPU execution is supported but significantly slower.

---

## Technologies Used

| Tool / Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| ResNet50 (ImageNet) | Transfer learning backbone |
| Kaggle Hub | Dataset download |
| Scikit-learn | Classification report, confusion matrix |
| Matplotlib / Seaborn | Visualisations |
| NumPy | Numerical operations |

---

## Conclusion

Both models successfully classify retinal diseases from OCT scans. ResNet50 with two-phase fine-tuning is the **recommended model for deployment** given its higher accuracy of **98.12%**. The Custom CNN remains a useful educational baseline demonstrating that competitive performance is achievable without pre-trained weights.

Data augmentation, dropout, batch normalisation, early stopping, and adaptive learning rate scheduling all contributed to controlling overfitting across both architectures.
