# Configuration Structure

## Table of Contents
- [Directory Structure](#directory-structure)
- [Design Philosophy](#design-philosophy)
  - [Shared Parameters (base.yaml)](#shared-parameters-baseyaml)
  - [Directory Configuration (base.yaml)](#directory-configuration-baseyaml)
  - [Model-Specific Parameters (models/*.yaml)](#model-specific-parameters-modelsyaml)
- [Usage](#usage)
  - [Option 1: Using the utility function (Recommended)](#option-1-using-the-utility-function-recommended)
  - [Option 2: Manual loading](#option-2-manual-loading)
- [Best Practices](#best-practices)
- [Adding a New Model](#adding-a-new-model)
- [Configuration Validation](#configuration-validation)
- [References](#references)

---

This directory contains a **hybrid configuration system** for model training:

##  Directory Structure

```
configs/
├── base.yaml              # Shared parameters for ALL models (fair comparison)
├── baseline.yaml          # Original paper baseline for comparison
├── environment.yml        # Conda environment definition
├── utils.py              # Configuration loading utilities
├── README.md             # This file
└── models/               # Model-specific configurations
    ├── densenet.yaml     # Baseline CNN
    ├── efficientnetv2.yaml
    ├── regnety.yaml
    ├── convnext.yaml     # SOTA CNN
    ├── swin.yaml
    ├── maxvit.yaml       # Hybrid (CNN+Transformer)
    ├── vit.yaml          # Vision Transformer
    ├── convit.yaml       # Convolutional ViT
    ├── tnt.yaml          # Transformer-in-Transformer
    ├── simclr.yaml       # Self-supervised (SimCLR)
    └── dino.yaml         # Self-supervised (DINO)
```

##  Design Philosophy

### Shared Parameters (base.yaml)
**These remain IDENTICAL across all models for fair scientific comparison:**
- `learning_rate`: 1e-4
- `weight_decay`: 1e-5
- `batch_size`: 32
- `img_size`: 256 (default -- see Model-Specific Overrides below)
- `max_epochs`: 50
- `patience`: 10 (Early Stopping)

**Loss Function Configuration (`data.loss`):**
```yaml
data:
  loss:
    name: ordinal_smoothing      # OrdinalLabelSmoothingLoss
    smoothing_factor: 0.1         # epsilon for neighbor distribution
```
- `ordinal_smoothing`: Distributes epsilon probability mass to neighboring classes, respecting the ordinal nature of inflammation scores (0 < 1 < 2 < 3). Recommended for this task.
- `cross_entropy`: Standard CrossEntropyLoss with `ignore_index` for the artifact class. Falls back to this if `name` is not `ordinal_smoothing`.
- Only classes 0-3 are used for ordinal loss calculation; Ignore class (index 4) is masked out before loss computation.
- See `src/models/loss.py` for implementation and `docs/TRAINING_CONCEPTS_EXPLAINED.md` for detailed explanation.

**CRITICAL:** Changing shared parameters requires retraining ALL models to ensure fair comparison. Any modification to these hyperparameters invalidates previous results for comparative analysis.

### Directory Configuration (base.yaml)
**All paths are centrally configured for portability:**
- `directories.experiments_dir`: Main experiment output directory (default: `experiments`)
- `directories.runs_dir`: Legacy runs directory (default: `runs`)
- `directories.checkpoints_subdir`: Checkpoint subdirectory name (default: `checkpoints`)
- `directories.figures_subdir`: Figures/metrics subdirectory name (default: `figures`)
- `directories.test_evaluation_subdir`: Test evaluation subdirectory (default: `test_evaluation`)

### Model-Specific Parameters (models/*.yaml)
**These can differ per architecture for optimal performance:**
- `backbone`: TIMM model name (e.g., `convnext_tiny`, `maxvit_tiny_224`)
- `drop_rate`: Dropout rate
- `drop_path_rate`: Stochastic depth
- Architecture-specific: `patch_size`, `window_size`, `embed_dim`, etc.

### Model-Specific img_size Overrides

Some architectures have **hard constraints on input resolution** due to their patch/window design.
Model-specific YAML files can override `data.img_size` via `deep_merge`, which only affects that model's runs.

| Model | img_size | Reason |
|---|---|---|
| DenseNet, ConvNeXt, EfficientNetV2, RegNetY | 256 (default) | Fully convolutional, resolution-agnostic |
| ViT (vit_small_patch16_224) | **224** | 224/16 = 14x14 patches (ganzzahlig). 256/16 = 16 patches moeglich, aber pretrained positional embeddings sind auf 14x14 trainiert |
| Swin (swin_tiny_patch4_window7_224) | **224** | 224/4 = 56 patches, 56/7 = 8 windows (ganzzahlig). 256/4 = 64, 64/7 = 9.14 -- nicht ganzzahlig, AssertionError |
| MaxViT, ConViT, TNT | 224 | Fixed positional embeddings trained at 224px |

**Mechanism:** Model YAML sets `data.img_size: 224`, which overrides `base.yaml`'s 256 through `deep_merge()`.
Both data transforms (Albumentations) and model forward pass receive the same resolution.

**Scientific note:** Different input resolutions across architectures are standard practice in the literature.
Each architecture uses its native/optimal resolution. This is documented per model in the run analysis files.

### Metrics Configuration (NEW in 2026)

All models automatically track **30+ metrics per fold** organized into 4 categories:

**1. Core Performance Metrics (Always Computed)**
```yaml
# Automatically tracked during training
metrics:
  primary: "qwk"              # Quadratic Weighted Kappa (checkpoint selection)
  secondary: ["accuracy", "f1_macro"]
  monitor: "val_kappa"        # Early stopping metric
```

**2. Per-Class Metrics (Auto-Logged)**
```yaml
# No configuration needed - automatically computed
# - f1_class_0, f1_class_1, f1_class_2, f1_class_3
# - precision_class_X, recall_class_X for X in [0,3]
```

**3. Efficiency Metrics (Auto-Computed Post-Training)**
```yaml
# Computed after validation, saved to fold_X_efficiency.json
efficiency:
  compute: true               # Default: true
  batch_size: 32              # Inference batch size
  warmup_iterations: 10       # GPU warmup before timing
  timing_iterations: 100      # Iterations for accurate timing
```

**4. Calibration Metrics (Auto-Computed Post-Training)**
```yaml
# Computed after validation, saved to fold_X_calibration.json
calibration:
  compute: true               # Default: true
  n_bins: 10                  # Number of bins for ECE calculation
  save_curve: true            # Save calibration curve data
```

**No manual configuration required!** All metrics are automatically computed and saved during training.

##  Usage

### Option 1: Using the utility function (Recommended)
```python
from configs.utils import load_config

# Load merged config for specific model
config = load_config("convnext")
print(config['model']['backbone'])        # convnext_tiny
print(config['training']['learning_rate']) # 0.0001
```

### Option 2: Manual loading
```python
import yaml

# Load base config
with open('configs/base.yaml') as f:
    config = yaml.safe_load(f)

# Load model-specific config
with open('configs/models/vit.yaml') as f:
    model_config = yaml.safe_load(f)

# Merge (model config overrides base)
config.update(model_config)
```

##  Best Practices

1. **Never change shared parameters** unless you retrain ALL models
   - Ensures fair comparison in thesis/papers
   - Changes to `learning_rate` require full re-run

2. **Model-specific parameters can be tuned** independently
   - Dropout rates, architecture details
   - Document changes in model config files

3. **Add notes to model configs**
   - Parameter counts, architecture insights
   - Alternative backbone options

## 🆕 Adding a New Model

1. Create `configs/models/newmodel.yaml`:
```yaml
model:
  backbone: "newmodel_base"
  pretrained: true
  drop_rate: 0.1
  # ... model-specific params
  
  notes:
    - "Brief description"
    - "Parameter count"
```

2. Add to `base.yaml`:
```yaml
models_to_train:
  - newmodel
```

3. Use in training:
```python
config = load_config("newmodel")
```

##  Configuration Validation

Check available models:
```python
from configs.utils import list_available_models
print(list_available_models())
```

Validate config:
```python
config = load_config("vit")
assert 'model' in config
assert 'training' in config
assert config['training']['learning_rate'] == 1e-4
```

---

## 🐍 Environment Configuration

### environment.yml

The `environment.yml` file defines the complete Conda environment for the project, including:

- **Python version**: 3.9+
- **Deep Learning frameworks**: PyTorch, torchvision
- **Scientific computing**: NumPy, SciPy, pandas
- **Image processing**: Pillow, OpenCV
- **Visualization**: Matplotlib, seaborn
- **Model libraries**: timm (PyTorch Image Models)
- **Utilities**: PyYAML, tqdm, Jupyter

**Setup:**
```bash
# Create environment from file
conda env create -f configs/environment.yml

# Activate environment
conda activate inflammation_env

# Update existing environment
conda env update -f configs/environment.yml --prune
```

**Why in configs/?**
- Centralized configuration management
- Version controlled alongside model configs
- Clear separation from code
- Easier to find and maintain

---

## 🔗 References

- **base.yaml**: Common training parameters
- **baseline.yaml**: Heinemann et al. (2018) PLOS ONE paper metrics
- **environment.yml**: Conda environment definition
- **models/**: Individual architecture configurations
- **utils.py**: Config loading and merging logic
