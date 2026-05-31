#!/bin/bash
# Verification Script: Check if HPO parameters are being used
# ============================================================
# Usage: bash scripts/verify_hpo_usage.sh densenet

MODEL_NAME=${1:-densenet}

echo "================================================================"
echo "HPO PARAMETER VERIFICATION FOR MODEL: $MODEL_NAME"
echo "================================================================"
echo ""

# Step 1: Check if HPO results file exists
echo "[1] Checking for HPO results file..."
HPO_FILE=""
for pattern in "hpo_best_hpo_${MODEL_NAME}_v2.yaml" "hpo_best_hpo_${MODEL_NAME}_master_run.yaml" "hpo_best_${MODEL_NAME}.yaml"; do
    if [ -f "configs/$pattern" ]; then
        HPO_FILE="configs/$pattern"
        break
    fi
done

if [ -z "$HPO_FILE" ]; then
    echo "   WARNING: No HPO results file found for $MODEL_NAME"
    echo "   Expected locations:"
    echo "     - configs/hpo_best_hpo_${MODEL_NAME}_v2.yaml"
    echo "     - configs/hpo_best_hpo_${MODEL_NAME}_master_run.yaml"
    echo "     - configs/hpo_best_${MODEL_NAME}.yaml"
    echo ""
    echo "   STATUS: HPO has NOT been run yet for this model"
    echo "   ACTION: Run training with --enable-hpo --hpo-mode use_existing"
    exit 1
else
    echo "   FOUND: $HPO_FILE"
    echo ""
fi

# Step 2: Display HPO parameters
echo "[2] HPO-Optimized Parameters:"
echo "   ----------------------------------------"
cat "$HPO_FILE" | grep -E "learning_rate|weight_decay|batch_size|beta1|beta2|scheduler_patience|drop_rate" | sed 's/^/   /'
echo "   ----------------------------------------"
echo ""

# Step 3: Check base.yaml default parameters for comparison
echo "[3] Base Config Default Parameters (for comparison):"
echo "   ----------------------------------------"
echo "   learning_rate: $(grep -A 1 'learning_rate:' configs/base.yaml | tail -1 | awk '{print $2}')"
echo "   weight_decay: $(grep -A 1 'weight_decay:' configs/base.yaml | tail -1 | awk '{print $2}')"
echo "   batch_size: $(grep 'batch_size:' configs/base.yaml | awk '{print $2}')"
echo "   ----------------------------------------"
echo ""

# Step 4: Instructions for verification during training
echo "[4] How to verify during training:"
echo "   ----------------------------------------"
echo "   When you run training, look for this section in the logs:"
echo ""
echo "   =========================================================================="
echo "   HPO PARAMETERS DETECTED"
echo "   =========================================================================="
echo ""
echo "   Optimized parameters loaded:"
echo "     Learning Rate: X.XXe-XX"
echo "     Weight Decay: X.XXe-XX"
echo "     Batch Size: XX"
echo "     ..."
echo "   =========================================================================="
echo ""
echo "   If you see this section, HPO parameters ARE being used!"
echo "   ----------------------------------------"
echo ""

# Step 5: Grep training logs for HPO usage
echo "[5] Checking recent training logs for HPO usage..."
LATEST_EXP=$(ls -td experiments/2026-* 2>/dev/null | head -1)
if [ -n "$LATEST_EXP" ]; then
    echo "   Latest experiment: $LATEST_EXP"
    if [ -f "$LATEST_EXP/training.log" ]; then
        if grep -q "HPO PARAMETERS DETECTED" "$LATEST_EXP/training.log"; then
            echo "   STATUS: HPO parameters WERE used in this run"
            echo ""
            echo "   Extracted HPO section:"
            grep -A 10 "HPO PARAMETERS DETECTED" "$LATEST_EXP/training.log" | head -15
        else
            echo "   WARNING: HPO parameters NOT detected in training log"
            echo "   This might mean:"
            echo "     - Training used default parameters (HPO was skipped)"
            echo "     - Log file incomplete"
        fi
    else
        echo "   No training.log found in latest experiment"
    fi
else
    echo "   No experiments found yet"
fi

echo ""
echo "================================================================"
echo "VERIFICATION COMPLETE"
echo "================================================================"
