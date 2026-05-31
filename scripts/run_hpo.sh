#!/bin/bash
# Hyperparameter Optimization Scripts
# Usage: ./scripts/run_hpo.sh [model_name] [n_trials]

set -e

echo "============================================"
echo "    HYPERPARAMETER OPTIMIZATION"
echo "============================================"
echo ""

# Default values
MODEL=${1:-""}
TRIALS=${2:-50}
FOLDS=${3:-3}

# Activate environment (adjust path as needed)
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  Warning: No virtual environment detected"
fi

# Navigate to project root
cd "$(dirname "$0")/.."

if [ -z "$MODEL" ]; then
    echo "Running multi-model HPO (searches across architectures)"
    echo "Trials: $TRIALS"
    echo "Folds per trial: $FOLDS"
    echo ""
    python -m src.hpo --trials $TRIALS --folds $FOLDS
else
    echo "Running model-specific HPO for: $MODEL"
    echo "Trials: $TRIALS"
    echo "Folds per trial: $FOLDS"
    echo ""
    python -m src.hpo --model $MODEL --trials $TRIALS --folds $FOLDS
fi

echo ""
echo "============================================"
echo "    HPO COMPLETE"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  - SQLite database: optuna_study.db"
echo "  - Best config: configs/hpo_best_*.yaml"
echo "  - Visualizations: hpo_*.html (if plotly installed)"
echo ""
echo "To view results in Optuna Dashboard:"
echo "  optuna-dashboard sqlite:///optuna_study.db"
echo ""
