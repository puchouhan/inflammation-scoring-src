#  Hyperparameter Optimization Guide

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
  - [1. Model-Specific HPO (Recommended)](#1-model-specific-hpo-recommended)
  - [2. Multi-Model Search (Exploratory)](#2-multi-model-search-exploratory)
  - [3. Using the Shell Script](#3-using-the-shell-script)
- [Search Space](#search-space)
  - [Parameters Optimized](#parameters-optimized)
  - [Fixed Parameters (Not Optimized)](#fixed-parameters-not-optimized)
- [How It Works](#how-it-works)
  - [1. TPE Sampling Strategy](#1-tpe-sampling-strategy)
  - [2. Multi-Fold Validation](#2-multi-fold-validation)
  - [3. Fast Training](#3-fast-training)
- [Monitoring & Results](#monitoring--results)
  - [View Optimization Progress](#view-optimization-progress)
  - [Visualizations](#visualizations)
  - [Best Configuration](#best-configuration)
- [Usage for Thesis](#usage-for-thesis)
  - [Recommended Workflow](#recommended-workflow)
  - [Thesis Documentation](#thesis-documentation)
- [Advanced Usage](#advanced-usage)
  - [Custom Search Space](#custom-search-space)
  - [Resume Interrupted HPO](#resume-interrupted-hpo)
  - [Parallel Trials](#parallel-trials)
  - [Export Best Config to Model Config](#export-best-config-to-model-config)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [Summary](#summary)

---

## Overview

This project uses **Optuna** for intelligent hyperparameter optimization with:
- [DONE] **TPE Sampler**: Smart parameter selection (not random!)
- [DONE] **Multi-Fold Validation**: Average across 3 folds per trial
- [DONE] **Extended Search Space**: 9 hyperparameters optimized
- [DONE] **Model-Specific HPO**: Separate optimization per architecture
- [DONE] **Persistent Storage**: SQLite database (resumable after crashes)

---

##  Quick Start

### 1. Model-Specific HPO (Recommended)

Optimize hyperparameters for a **specific architecture**:

```bash
# Optimize ConvNeXt (State-of-the-Art CNN)
python -m src.hpo --model convnext --trials 50 --folds 3

# Optimize EfficientNetV2
python -m src.hpo --model efficientnetv2 --trials 50 --folds 3

# Optimize MaxViT (Hybrid)
python -m src.hpo --model maxvit --trials 50 --folds 3
```

**Why model-specific?** Different architectures have fundamentally different optimization landscapes:
- **CNNs (ConvNeXt, EfficientNet):** Deeper networks, can tolerate higher LR (1e-4 to 5e-4), benefit from stronger weight decay
- **Transformers (ViT, Swin):** Attention mechanisms are sensitive, prefer smaller LR (1e-5 to 1e-4), need careful warmup
- **Hybrid Models (MaxViT):** Combine both characteristics, need balanced hyperparameters
- **Batch Size Dependencies:** Larger models (Swin, MaxViT) need smaller batches, affecting LR scaling
- **Regularization Trade-offs:** Dropout rates vary significantly (CNNs: 0.1-0.2, Transformers: 0.0-0.1)

Using shared hyperparameters across architectures leads to suboptimal performance. Model-specific HPO ensures each architecture operates in its optimal regime, maximizing predictive performance for comparative evaluation.

### 2. Multi-Model Search (Exploratory)

Search across **multiple architectures** simultaneously:

```bash
python -m src.hpo --trials 100 --folds 3
```

**Use case:** Initial exploration to find best architecture + hyperparameters.

### 3. Using the Shell Script

```bash
# Model-specific
./scripts/run_hpo.sh convnext 50 3

# Multi-model
./scripts/run_hpo.sh "" 100 3
```

---

##  Search Space

### Parameters Optimized

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `learning_rate` | Float | 1e-5 to 5e-4 (log) | Initial learning rate |
| `weight_decay` | Float | 1e-6 to 1e-3 (log) | L2 regularization |
| `beta1` | Float | 0.85 to 0.95 | Adam momentum term β |
| `beta2` | Float | 0.99 to 0.999 | Adam momentum term β |
| `scheduler_patience` | Int | 3 to 7 | Epochs before LR reduction |
| `batch_size` | Categorical | [16, 32, 64] | Training batch size |
| `drop_rate` | Float | 0.0 to 0.3 | Dropout rate |
| `backbone` * | Categorical | 4 options | Model architecture |

\* Only for multi-model search

### Fixed Parameters (Not Optimized)

For **fair comparison** across models, these remain constant:
- `optimizer.type`: "adamw" (always)
- `scheduler.type`: "reduce_on_plateau" (always)
- `img_size`: 256 (from base.yaml)
- `max_epochs`: 10 (for HPO speed)
- `num_classes`: 5 (dataset property)

---

## 🧠 How It Works

### 1. TPE Sampling Strategy

```
Trial 1-10:  Random exploration
Trial 11+:   Smart sampling based on previous results
              Focuses on promising regions
              Balances exploration vs exploitation
```

### 2. Multi-Fold Validation

Each trial trains on **3 folds** and averages the QWK score:

```python
Trial 42:
  Fold 0: QWK = 0.73
  Fold 1: QWK = 0.76
  Fold 2: QWK = 0.71
   Average: 0.733 (more robust than single fold!)
```

### 3. Fast Training

- **10 epochs** per fold (vs 50 for full training)
- **Early stopping** (patience=3) for efficiency
- **No checkpointing** (saves disk I/O)
- ~2-5 minutes per trial (depends on hardware)

---

## 📈 Monitoring & Results

### View Optimization Progress

Install Optuna Dashboard:
```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_study.db
```

Open browser: `http://localhost:8080`

### Visualizations

If `plotly` is installed, automatic HTML reports:
- `hpo_history_*.html`: Score progression over trials
- `hpo_importance_*.html`: Parameter importance ranking

### Best Configuration

Saved to: `configs/hpo_best_<study_name>.yaml`

Example:
```yaml
learning_rate: 0.000234
weight_decay: 0.000012
beta1: 0.9
beta2: 0.999
scheduler_patience: 5
batch_size: 32
drop_rate: 0.15
```

---

## 🎓 Usage for Thesis

### Recommended Workflow

1. **Initial Exploration** (Multi-Model):
   ```bash
   python -m src.hpo --trials 100 --folds 3
   ```
    Find best architecture family

2. **Model-Specific Tuning** (Per Architecture):
   ```bash
   python -m src.hpo --model vit --trials 50 --folds 3
   python -m src.hpo --model efficientnetv2 --trials 50 --folds 3
   python -m src.hpo --model swin --trials 50 --folds 3
   ```
    Get optimal hyperparameters per model

3. **Final Training** (Full Dataset):
   - Use best hyperparameters from HPO
   - Train on all 5 folds
   - 50 epochs with early stopping

### Thesis Documentation

**Methods Section:**
> "Hyperparameters were optimized using Optuna (Akiba et al., 2019) with Tree-structured Parzen Estimator (TPE) sampling. Each trial was evaluated by averaging the Quadratic Weighted Kappa score across 3-fold cross-validation. The search space included learning rate (1e-5 to 5e-4), weight decay (1e-6 to 1e-3), batch size (16, 32, 64), dropout rate (0.0 to 0.3), and optimizer momentum terms. We conducted 50 trials per model architecture, resulting in model-specific optimal configurations."

**Results Section:**
```
Table X: Optimal Hyperparameters per Architecture

| Model         | LR      | WD      | Batch | Dropout | Val QWK |
|---------------|---------|---------|-------|---------|---------|
| ViT           | 2.3e-4  | 1.2e-5  | 32    | 0.10    | 0.812   |
| EfficientNetV2| 3.1e-4  | 8.9e-6  | 32    | 0.20    | 0.798   |
| Swin          | 1.8e-4  | 1.5e-5  | 32    | 0.15    | 0.805   |
```

---

## ️ Advanced Usage

### Custom Search Space

Edit `configs/base.yaml`  `hpo.search_space`:

```yaml
hpo:
  search_space:
    learning_rate:
      min: 1e-6  # Lower bound
      max: 1e-3  # Upper bound
```

### Resume Interrupted HPO

Studies are automatically saved to SQLite:
```bash
# Will continue from trial N+1
python -m src.hpo --model vit --trials 100
```

### Parallel Trials

Run multiple HPO processes in parallel:
```bash
# Terminal 1
python -m src.hpo --model vit --trials 50

# Terminal 2 (different study name!)
python -m src.hpo --model swin --trials 50
```

### Export Best Config to Model Config

```bash
# After HPO completes
cp configs/hpo_best_hpo_vit_v2.yaml configs/models/vit_optimized.yaml
```

Then edit to match model config format.

---

##  Troubleshooting

### "Module not found: optuna"
```bash
pip install optuna
pip install optuna-dashboard  # Optional, for web UI
```

### "Out of Memory"
Reduce batch size or n_folds:
```bash
python -m src.hpo --model vit --trials 50 --folds 2
```

### "Trial failed"
Check logs for specific error. Common causes:
- Incompatible hyperparameter combination
- GPU memory issues (reduce batch_size range)
- Missing model config file

### "Study already exists"
This is fine! Optuna will resume from last trial. To start fresh:
```bash
rm optuna_study.db
python -m src.hpo --model vit --trials 50
```

---

## 📚 References

- Optuna: https://optuna.org/
- TPE Algorithm: Bergstra et al. (2011) - "Algorithms for Hyper-Parameter Optimization"
- Best Practices: https://optuna.readthedocs.io/en/stable/tutorial/index.html

---

##  Summary

**Key Improvements over Original:**
- [DONE] 9 parameters optimized (was: 3)
- [DONE] Multi-fold validation (was: single fold)
- [DONE] Model-specific HPO (was: generic)
- [DONE] Reproducible TPE sampling (was: default)
- [DONE] Visualization support (was: none)
- [DONE] Shell script automation (was: manual)

**Estimated Time:**
- Model-specific HPO: ~2-4 hours (50 trials × 3 folds × 3 min/trial)
- Multi-model HPO: ~5-10 hours (100 trials)
- Total for 4 models: ~8-16 hours

**Expected Improvement:**
- ~2-5% QWK increase from optimized hyperparameters
- More robust results from multi-fold validation
- Fair comparison across architectures
