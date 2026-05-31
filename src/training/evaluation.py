"""
Evaluation utilities for model selection and test set evaluation.

This module contains functions for:
- Evaluating best models on the held-out test set
- Selecting the best model based on cross-validation results

These are core training/evaluation utilities, not reporting functions.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, confusion_matrix, classification_report
import json
from src.models.model_factory import ModelFactory
from src.data.inflammation_dataset import InflammationDataset
from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)

def evaluate_on_test_set(run_dir: Path, config: Dict[str, Any], models_to_train: List[str], verbose: bool = True) -> pd.DataFrame:
    """
    Evaluate best model (per architecture) on held-out test set.
    
    Loads the best checkpoint per fold from checkpoints.json and evaluates
    on the held-out val/ folder (test set). Uses the best single fold
    checkpoint for each model.
    
    Args:
        run_dir: Path to experiment run directory (e.g., experiments/run_id/model_name)
        config: Full config dict
        models_to_train: List of model names
        verbose: If True, log detailed output
        
    Returns:
        DataFrame with test set metrics for each model
    """
    from src.data.inflammation_dataset import get_test_dataloader
    from src.models.base_model import InflammationModel
    
    test_loader = get_test_dataloader(config)
    results = []
    
    for model_name in models_to_train:
        model_dir = run_dir / model_name
        checkpoint_file = model_dir / "checkpoints.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"No checkpoints.json for model: {model_name} at {checkpoint_file}")
            continue
        
        with open(checkpoint_file, "r") as f:
            checkpoint_meta = json.load(f)
        
        # Find best fold checkpoint by val_qwk
        folds = checkpoint_meta.get("folds", [])
        if not folds:
            logger.warning(f"No fold data in checkpoints.json for {model_name}")
            continue
        
        best_fold = max(folds, key=lambda f: f.get("val_qwk", 0.0))
        checkpoint_path = best_fold["checkpoint_path"]
        
        try:
            model = InflammationModel.load_from_checkpoint(checkpoint_path, cfg=config)
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    outputs = model(images)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            # Metrics (exclude ignore class from evaluation)
            ignore_idx = config.get('ignore_class_index', 4)
            valid_mask = np.array(all_labels) != ignore_idx
            y_true = np.array(all_labels)[valid_mask]
            y_pred = np.array(all_preds)[valid_mask]
            # Clamp predictions to valid range 0-3
            y_pred = np.clip(y_pred, 0, 3)
            
            test_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
            test_acc = accuracy_score(y_true, y_pred)
            test_f1 = f1_score(y_true, y_pred, average="macro")
            cm = confusion_matrix(y_true, y_pred)
            report = classification_report(y_true, y_pred, digits=4)
            
            results.append({
                "Model": model_name,
                "Test QWK": test_kappa,
                "Test Acc": test_acc,
                "Test F1": test_f1,
                "Best Fold": best_fold["fold_idx"],
                "Val QWK": checkpoint_meta.get("mean_qwk", 0.0),
                "Checkpoint": checkpoint_path
            })
            
            # Save outputs
            test_eval_dir = run_dir / config["directories"].get("test_evaluation_subdir", "test_evaluation")
            test_eval_dir.mkdir(exist_ok=True, parents=True)
            np.save(test_eval_dir / f"{model_name}_test_preds.npy", y_pred)
            np.save(test_eval_dir / f"{model_name}_test_labels.npy", y_true)
            pd.DataFrame(cm).to_csv(test_eval_dir / f"{model_name}_confusion_matrix.csv")
            with open(test_eval_dir / f"{model_name}_classification_report.txt", "w") as f:
                f.write(str(report))
            
            if verbose:
                logger.info(f"Test results for {model_name}: QWK={test_kappa:.4f}, Acc={test_acc:.4f}, F1={test_f1:.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name} on test set: {e}")
            continue
    
    df = pd.DataFrame(results)
    test_eval_dir = run_dir / config["directories"].get("test_evaluation_subdir", "test_evaluation")
    test_eval_dir.mkdir(exist_ok=True, parents=True)
    summary_path = test_eval_dir / "test_summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"Test summary saved to: {summary_path}")

    return df

def find_best_model(all_results: dict) -> Optional[dict]:
    """
    Find the best performing model across all results.
    Args:
        all_results: Dictionary of {model_name: [fold_results]}
    Returns:
        Dictionary with best model info:
        - model_name: Name of best model
        - mean_kappa: Mean QWK across folds
        - std_kappa: Standard deviation of QWK
        - fold_kappas: List of QWK values per fold
        Returns None if no successful models found.
    """
    valid_models = []
    for name, results in all_results.items():
        # Check if all folds succeeded
        if all(r.get("status") == "success" for r in results):
            kappas = [r["val_kappa"] for r in results]
            mean_kappa = np.mean(kappas)
            std_kappa = np.std(kappas, ddof=1) if len(kappas) > 1 else 0.0
            valid_models.append({
                "model_name": name,
                "mean_kappa": mean_kappa,
                "std_kappa": std_kappa,
                "fold_kappas": kappas
            })
    if not valid_models:
        return None
    # Return model with highest mean kappa
    return max(valid_models, key=lambda x: x["mean_kappa"])
