"""
Hyperparameter Optimization using Optuna
Supports model-specific HPO with extended search space and multi-fold validation.
"""

import logging
import sys
from pathlib import Path

import lightning as L
import numpy as np
import optuna
import yaml
from lightning.pytorch.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

# Ensure project root is importable when executed as a script.
# File is now at: src/hpo/hpo.py  -> project root is parents[2].
sys.path.append(str(Path(__file__).resolve().parents[2]))

from configs.utils import load_config
from src.data.inflammation_dataset import get_dataloaders
from src.models.model_factory import ModelFactory
from src.utils.seeds_logging import seed_everything


def objective(trial, model_name: str = None, n_folds: int = 3):
    """
    Objective function for Optuna HPO.

    Args:
        trial: Optuna trial object
        model_name: Specific model to optimize (e.g., "vit", "efficientnetv2").
                   If None, backbone will be sampled from search space.
        n_folds: Number of folds to average over (default: 3 for speed)

    Returns:
        Average validation QWK score across folds
    """
    # Load config (model-specific or base)
    if model_name:
        cfg = load_config(model_name)
        logger.info(f"[Trial {trial.number}] Optimizing {model_name}")
    else:
        cfg = load_config()
        logger.info(f"[Trial {trial.number}] Optimizing across models")

    # Get HPO search space from config (with fallback to defaults)
    search_space = cfg.get("hpo", {}).get("search_space", {})

    # ═══════════════════════════════════════════════════════════
    # EXTENDED SEARCH SPACE (using base.yaml ranges)
    # ═══════════════════════════════════════════════════════════

    # 1. Core Training Parameters
    lr_space = search_space.get("learning_rate", {"min": 1e-5, "max": 5e-4, "log": True})
    # Convert string values to float (e.g., '1e-5' -> 1e-5)
    lr_min = float(lr_space["min"]) if isinstance(lr_space["min"], str) else lr_space["min"]
    lr_max = float(lr_space["max"]) if isinstance(lr_space["max"], str) else lr_space["max"]
    cfg["training"]["learning_rate"] = trial.suggest_float(
        "learning_rate",
        lr_min,
        lr_max,
        log=lr_space.get("log", True),
    )

    wd_space = search_space.get("weight_decay", {"min": 1e-6, "max": 1e-3, "log": True})
    # Convert string values to float
    wd_min = float(wd_space["min"]) if isinstance(wd_space["min"], str) else wd_space["min"]
    wd_max = float(wd_space["max"]) if isinstance(wd_space["max"], str) else wd_space["max"]
    cfg["training"]["weight_decay"] = trial.suggest_float(
        "weight_decay",
        wd_min,
        wd_max,
        log=wd_space.get("log", True),
    )

    # 2. Optimizer Parameters
    if "optimizer" not in cfg["training"]:
        cfg["training"]["optimizer"] = {}

    beta1_space = search_space.get("beta1", {"min": 0.85, "max": 0.95})
    beta1 = trial.suggest_float("beta1", float(beta1_space["min"]), float(beta1_space["max"]))

    beta2_space = search_space.get("beta2", {"min": 0.99, "max": 0.999})
    beta2 = trial.suggest_float("beta2", float(beta2_space["min"]), float(beta2_space["max"]))
    cfg["training"]["optimizer"]["betas"] = [beta1, beta2]
    cfg["training"]["optimizer"]["type"] = "adamw"  # Fixed for consistency

    # 3. Learning Rate Scheduler
    if "scheduler" not in cfg["training"]:
        cfg["training"]["scheduler"] = {}

    sched_patience_space = search_space.get("scheduler_patience", {"min": 3, "max": 7})
    scheduler_patience = trial.suggest_int(
        "scheduler_patience",
        int(sched_patience_space["min"]),
        int(sched_patience_space["max"]),
    )
    cfg["training"]["scheduler"]["patience"] = scheduler_patience
    cfg["training"]["scheduler"]["type"] = "reduce_on_plateau"

    # 4. Data Parameters
    bs_space = search_space.get("batch_size", {"options": [16, 32, 64]})
    batch_size = trial.suggest_categorical("batch_size", bs_space.get("options", [16, 32, 64]))
    cfg["data"]["batch_size"] = batch_size

    # 5. Model-specific Parameters
    if "model" not in cfg:
        cfg["model"] = {}

    drop_space = search_space.get("drop_rate", {"min": 0.0, "max": 0.3})
    drop_rate = trial.suggest_float("drop_rate", float(drop_space["min"]), float(drop_space["max"]))
    cfg["model"]["drop_rate"] = drop_rate

    # 6. Backbone Selection (only if not model-specific)
    if not model_name:
        backbone_space = search_space.get(
            "backbones",
            [
                "efficientnetv2_rw_s",
                "regnety_032",
                "vit_small_patch16_224",
                "swin_tiny_patch4_window7_224",
            ],
        )
        backbone = trial.suggest_categorical("backbone", backbone_space)
        cfg["training"]["backbone"] = backbone
        # Create model config on the fly
        cfg["model"]["backbone"] = backbone
        cfg["model"]["pretrained"] = True

    # ═══════════════════════════════════════════════════════════
    # MULTI-FOLD VALIDATION
    # ═══════════════════════════════════════════════════════════

    # Cap n_folds by the available splits dynamically
    cv_strategy = cfg.get("data", {}).get("cv_strategy", "loao_balanced")
    max_folds = cfg.get("data", {}).get("cv_folds_config", {}).get(cv_strategy, 2)
    actual_n_folds = min(n_folds, max_folds)

    fold_scores = []

    for fold_idx in range(actual_n_folds):
        logger.info(f"Fold {fold_idx + 1}/{actual_n_folds}")

        # Set seed for reproducibility
        seed_everything(cfg["seed"] + fold_idx)

        # Get dataloaders
        train_loader, val_loader = get_dataloaders(cfg, fold_idx=fold_idx)

        # Create model
        if model_name:
            model = ModelFactory.create_model(model_name, cfg)
        else:
            # Generic supervised model
            from src.models.base_model import InflammationModel

            model = InflammationModel(cfg)

        model.suppress_overfitting_warnings = True

        # Fast training for HPO
        trainer = L.Trainer(
            max_epochs=cfg['hpo']['max_epochs_per_trial'],  # Use config value instead of hardcoded
            accelerator="auto",
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[EarlyStopping(monitor="val_loss", patience=cfg['training']['patience'], mode="min")],
        )

        try:
            trainer.fit(model, train_loader, val_loader)

            # Get validation QWK
            val_kappa = trainer.callback_metrics.get("val_kappa", 0.0)
            if isinstance(val_kappa, float):
                fold_scores.append(val_kappa)
            else:
                fold_scores.append(val_kappa.item())

        except Exception as e:
            logger.error(f"Fold {fold_idx} failed: {e}")
            fold_scores.append(0.0)  # Penalty for failed trials

    # Return average score across folds
    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    logger.info(f"Avg QWK: {avg_score:.4f} +/- {std_score:.4f}")

    return avg_score


def optimize_model(
    model_name: str = None,
    n_trials: int = 50,
    n_folds: int = 3,
    study_name: str = None,
    storage: str = "sqlite:///optuna_study.db",
    overwrite: bool = False,
    cv_strategy: str = None,
):
    """
    Run hyperparameter optimization for a specific model or generic search.

    Args:
        model_name: Model to optimize (e.g., "vit", "efficientnetv2").
                   If None, searches across multiple backbones.
        n_trials: Number of trials to run
        n_folds: Number of folds to average over per trial
        study_name: Name for the Optuna study (auto-generated if None)
        storage: Database URL for storing results
        overwrite: If True, deletes existing study and creates new one.
                  If False, loads existing study and continues.
        cv_strategy: CV strategy name (e.g., "random_stratified", "loao_balanced").
                    Included in study name and output filename for separation.

    Returns:
        study: Optuna study object with results
    """
    # Auto-generate study name (includes cv_strategy to prevent overwriting)
    if study_name is None:
        if model_name:
            if cv_strategy:
                study_name = f"hpo_{model_name}_{cv_strategy}_v2"
            else:
                study_name = f"hpo_{model_name}_v2"
        else:
            if cv_strategy:
                study_name = f"hpo_multimodel_{cv_strategy}_v2"
            else:
                study_name = "hpo_multimodel_v2"

    logger.info("=" * 80)
    logger.info(f"HYPERPARAMETER OPTIMIZATION: {study_name}")
    logger.info("=" * 80)
    if model_name:
        logger.info(f"Model: {model_name}")
    else:
        logger.info("Model: Multi-model search (includes backbone selection)")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Folds per trial: {n_folds}")
    logger.info(f"Storage: {storage}")
    logger.info(f"Overwrite mode: {overwrite}")
    logger.info("=" * 80)

    # Handle overwrite mode: Delete existing study if requested
    if overwrite:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            logger.info(f"Deleted existing study '{study_name}' (overwrite mode)")
        except KeyError:
            logger.info(f"No existing study '{study_name}' to delete (creating fresh)")

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Maximize QWK
        load_if_exists=not overwrite,  # Load if NOT overwriting
        sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible TPE
    )

    # Add callback for progress reporting
    def log_trial_callback(study, trial):
        """Log progress after each trial completes."""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Trial {trial.number} Complete")
        logger.info("=" * 80)
        logger.info(f"  QWK: {trial.value:.4f}")
        logger.info(f"  Parameters:")
        for key, value in trial.params.items():
            if isinstance(value, float):
                logger.info(f"    {key}: {value:.6f}")
            else:
                logger.info(f"    {key}: {value}")
        logger.info(f"  Best QWK so far: {study.best_value:.4f} (Trial {study.best_trial.number})")
        logger.info(f"  Progress: {len(study.trials)}/{n_trials} trials")
        logger.info("=" * 80)
        logger.info("")

    # Optimize
    study.optimize(
        lambda trial: objective(trial, model_name=model_name, n_folds=n_folds),
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[log_trial_callback],
    )

    # Log results
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best Trial ({study.best_trial.number}):")
    logger.info(f"Value (QWK): {study.best_trial.value:.4f}")
    logger.info("Best Hyperparameters:")
    for key, value in study.best_trial.params.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        else:
            logger.info(f"  {key}: {value}")

    # Save best config to file
    best_config_path = f"configs/hpo_best_{study_name}.yaml"
    with open(best_config_path, "w") as f:
        yaml.dump(study.best_trial.params, f, default_flow_style=False)
    logger.info(f"Best config saved to: {best_config_path}")

    logger.info("=" * 80)

    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g., 'vit', 'efficientnetv2'). If None, searches across models.",
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of folds to average over per trial",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Custom study name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing study and start fresh (default: continue existing study)",
    )

    args = parser.parse_args()

    # Run optimization
    study = optimize_model(
        model_name=args.model,
        n_trials=args.trials,
        n_folds=args.folds,
        study_name=args.study_name,
        overwrite=args.overwrite,
    )

    # Optionally: Generate optimization history plots
    try:
        import plotly  # noqa: F401

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"hpo_history_{study.study_name}.html")
        logger.info(f"Optimization history saved to: hpo_history_{study.study_name}.html")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"hpo_importance_{study.study_name}.html")
        logger.info(f"Parameter importance saved to: hpo_importance_{study.study_name}.html")
    except ImportError:
        logger.warning("Install plotly for visualization: pip install plotly")
