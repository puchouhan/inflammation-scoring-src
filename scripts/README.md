# Scripts Overview

## Table of Contents
- [Available Scripts](#available-scripts)
- [Execution Order](#execution-order)
- [Usage Examples](#usage-examples)
- [Common Issues](#common-issues)

---

## Available Scripts

### run_hpo.sh
**Purpose:** Execute hyperparameter optimization for single or multiple models

**Usage:**
```bash
# Model-specific HPO
bash scripts/run_hpo.sh maxvit 50 3
# Arguments: <model_name> <n_trials> <n_folds>

# Multi-model HPO (empty model name)
bash scripts/run_hpo.sh "" 100 3
```

**Parameters:**
- `model_name`: Model to optimize (densenet, maxvit, etc.) or empty for multi-model
- `n_trials`: Number of Optuna trials (default: 50)
- `n_folds`: Number of CV folds for each trial (default: 3)

**Output:**
- HPO results saved to `configs/hpo_best_*.yaml`
- Optuna database: `optuna_study.db`
- Progress logged to console

**Execution Time:**
- Single model: ~3-6 hours (50 trials × 3 folds × 2-5 min/trial)
- Multi-model: ~10-20 hours (depends on search space)

---

### remove_emojis.py
**Purpose:** Remove emojis from markdown documentation files

**Usage:**
```bash
python scripts/remove_emojis.py
```

**What it does:**
1. Scans all `.md` files in project (excluding `systemdoku/`)
2. Replaces emojis with text equivalents:
   - ✅ → [DONE]
   - ⚠️ → WARNING:
   - 🔍 → SEARCH:
   - 📋 → NOTE:
   - etc.
3. Creates backup: `file.md.backup`
4. Updates files in-place

**When to use:**
- Preparing documentation for professional/academic submission
- Ensuring compatibility with LaTeX/PDF converters
- Cleaning up documentation for version control

---

### replace_prints.py
**Purpose:** Migrate print() statements to logging framework

**Usage:**
```bash
# Scan all Python files
python scripts/replace_prints.py

# Dry run (preview changes)
python scripts/replace_prints.py --dry-run

# Specific file
python scripts/replace_prints.py --file src/train.py
```

**What it does:**
1. Finds all `print()` statements in Python files
2. Analyzes context (error, warning, info, debug)
3. Replaces with appropriate logging calls:
   ```python
   # Before
   print("Training started...")
   print(f"⚠️  Warning: {message}")
   
   # After
   logger.info("Training started...")
   logger.warning(f"Warning: {message}")
   ```
4. Adds `import logging` and `logger = logging.getLogger(__name__)` if missing

**When to use:**
- Migrating legacy code to logging framework
- Standardizing output across project
- Preparing for production deployment

---

### find_models.py.archived
**Purpose:** (ARCHIVED) Search for available TIMM models

**Note:** This script has been archived. Use TIMM's built-in search instead:
```python
import timm
print(timm.list_models('*vit*'))  # Search for ViT models
print(timm.list_models('resnet*'))  # Search for ResNet models
```

**Why archived:**
- TIMM provides better built-in search
- Model registry constantly updated
- Direct API more reliable

---

## Execution Order

For a complete workflow from scratch:

```bash
# 1. Setup environment (one-time)
conda env create -f configs/environment.yml
conda activate inflammation_env

# 2. Data preprocessing (one-time)
python src/data/preprocess_stains.py

# 3. Optional: Run HPO for models (one-time or as needed)
bash scripts/run_hpo.sh maxvit 50 3
bash scripts/run_hpo.sh convnext 50 3

# 4. Training (main workflow)
jupyter notebook src/notebooks/Master_Runner.ipynb
# Or: python src/train_runner.py

# 5. Optional: Clean documentation
python scripts/remove_emojis.py

# 6. Optional: Migrate to logging (if adding new code)
python scripts/replace_prints.py
```

---

## Usage Examples

### Example 1: Quick Model Training
```bash
# Skip HPO, use defaults
# Edit base.yaml: hpo.mode = "skip"
python src/train.py --model maxvit
```

### Example 2: Optimized Training
```bash
# Run HPO first
bash scripts/run_hpo.sh maxvit 50 3

# Train with optimized params
# Edit base.yaml: hpo.mode = "use_existing"
python src/train.py --model maxvit
```

### Example 3: Full Experiment Pipeline
```bash
# HPO for all models
for model in densenet convnext maxvit swin; do
  bash scripts/run_hpo.sh $model 50 3
done

# Train all with Master Runner
jupyter notebook src/notebooks/Master_Runner.ipynb
```

### Example 4: Documentation Cleanup
```bash
# Remove emojis for publication
python scripts/remove_emojis.py

# Migrate logging
python scripts/replace_prints.py --dry-run  # Preview
python scripts/replace_prints.py            # Execute
```

---

## Common Issues

### Issue: Permission Denied
```
bash: scripts/run_hpo.sh: Permission denied
```
**Solution:**
```bash
chmod +x scripts/run_hpo.sh
```

### Issue: HPO Database Locked
```
sqlite3.OperationalError: database is locked
```
**Solution:**
```bash
# Kill existing HPO processes
pkill -f "python -m src.hpo"

# Or remove database and restart
rm optuna_study.db
bash scripts/run_hpo.sh maxvit 50 3
```

### Issue: Script Not Found
```
python: can't open file 'scripts/remove_emojis.py'
```
**Solution:**
```bash
# Ensure you're in project root
cd /path/to/master_thesis_inflammation

# Run from project root
python scripts/remove_emojis.py
```

### Issue: Import Error in Scripts
```
ModuleNotFoundError: No module named 'src'
```
**Solution:**
```bash
# Ensure environment activated
conda activate inflammation_env

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run as module
python -m scripts.remove_emojis
```

---

## Best Practices

1. **Always backup before running scripts:**
   ```bash
   cp important_file.md important_file.md.backup
   ```

2. **Use dry-run mode when available:**
   ```bash
   python scripts/replace_prints.py --dry-run
   ```

3. **Run HPO overnight:**
   ```bash
   # Use nohup to prevent interruption
   nohup bash scripts/run_hpo.sh maxvit 50 3 > hpo_maxvit.log 2>&1 &
   ```

4. **Monitor HPO progress:**
   ```bash
   # Install dashboard
   pip install optuna-dashboard
   
   # Launch (separate terminal)
   optuna-dashboard sqlite:///optuna_study.db
   
   # Open: http://localhost:8080
   ```

5. **Version control after script execution:**
   ```bash
   git add -A
   git commit -m "Applied remove_emojis.py to all docs"
   ```

---

## Script Development Guidelines

If adding new scripts:

1. **Add shebang line:**
   ```bash
   #!/usr/bin/env bash
   ```
   or
   ```python
   #!/usr/bin/env python
   ```

2. **Make executable:**
   ```bash
   chmod +x scripts/new_script.sh
   ```

3. **Add to this README:**
   - Purpose
   - Usage
   - Parameters
   - Output
   - Example

4. **Include error handling:**
   ```bash
   set -e  # Exit on error
   set -u  # Exit on undefined variable
   ```

5. **Add help text:**
   ```bash
   if [ "$#" -lt 1 ]; then
     echo "Usage: $0 <model_name> <n_trials> <n_folds>"
     exit 1
   fi
   ```

---

**Last Updated:** January 6, 2026  
**Total Scripts:** 4 (3 active, 1 archived)

