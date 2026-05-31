"""
Single Model Training Script
Automated training for one model across specified fold(s) with comprehensive tracking.

Usage:
    python src/train_runner.py  # Uses default model from base.yaml
    
    # Or from Python:
    from configs.utils import load_config
    cfg = load_config('densenet')  # Load specific model config
    # Then run main() with that config
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from configs.utils import load_config, merge_hpo_config, hpo_results_exist, load_hpo_config
from src.utils.seeds_logging import seed_everything, get_logger
from src.utils.experiment_tracker import ExperimentTracker
from src.data.inflammation_dataset import get_dataloaders
from src.data.split_validator import SplitValidator
from src.models.base_model import InflammationModel
from src.models.model_factory import ModelFactory
from src.utils.model_efficiency import compute_efficiency_metrics
from src.utils.calibration_metrics import evaluate_calibration
from src.hpo.hpo import optimize_model

logger = get_logger("TrainRunner")


# ---------------------------------------------------------------------------
# MLv13 artifact helpers — called automatically during train_single_model()
# ---------------------------------------------------------------------------

def _run_preflight_artifacts(cfg: dict, project_root: Path) -> None:
    """Generate pre-training MLv13 compliance artifacts (ML-5, ML-6, ML-16).

    Runs once per training session before the fold loop. Safe to re-run:
    each function skips gracefully when the artifact already exists.
    Failures are logged as warnings and never abort training.

    Args:
        cfg: Full configuration dictionary.
        project_root: Absolute path to the workspace root.
    """
    artifacts_dir = project_root / "docs" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ML-6: Data cleaning report
    try:
        from src.data.cleaning_report import generate_cleaning_report
        out = artifacts_dir / "cleaning_report.json"
        if not out.exists():
            logger.info("MLv13 ML-6: Generating data cleaning report...")
            generate_cleaning_report(cfg, out)
            logger.info(f"MLv13 ML-6: Cleaning report saved to {out}")
        else:
            logger.info("MLv13 ML-6: Cleaning report already exists, skipping.")
    except Exception as exc:
        logger.warning(f"MLv13 ML-6: Cleaning report generation failed (non-fatal): {exc}")

    # ML-16: Trivial baseline
    try:
        from src.analysis.trivial_baseline import run_trivial_baseline
        out = artifacts_dir / "trivial_baseline_results.json"
        if not out.exists():
            logger.info("MLv13 ML-16: Evaluating trivial baseline...")
            run_trivial_baseline(cfg, out)
            logger.info(f"MLv13 ML-16: Trivial baseline results saved to {out}")
        else:
            logger.info("MLv13 ML-16: Trivial baseline already exists, skipping.")
    except Exception as exc:
        logger.warning(f"MLv13 ML-16: Trivial baseline evaluation failed (non-fatal): {exc}")

    # ML-5: EDA export
    try:
        from src.analysis.eda_export import run_eda_export
        eda_dir = project_root / "figures" / "eda"
        if not (eda_dir / "eda_index.json").exists():
            logger.info("MLv13 ML-5: Running EDA export...")
            run_eda_export(cfg, eda_dir)
            logger.info(f"MLv13 ML-5: EDA artifacts saved to {eda_dir}")
        else:
            logger.info("MLv13 ML-5: EDA index already exists, skipping.")
    except Exception as exc:
        logger.warning(f"MLv13 ML-5: EDA export failed (non-fatal): {exc}")


def _run_posttraining_artifacts(
    cfg: dict,
    project_root: Path,
    run_id: str,
    model_name: str,
) -> None:
    """Generate post-training MLv13 compliance artifacts (ML-18, ML-19, ML-21).

    Called after all folds have completed successfully. Failures are logged
    as warnings and never affect the return value of train_single_model().

    Args:
        cfg: Full configuration dictionary.
        project_root: Absolute path to the workspace root.
        run_id: Unique run identifier string.
        model_name: Name of the model that was trained.
    """
    run_dir = project_root / cfg["directories"]["experiments_dir"] / run_id / model_name

    # ML-18: Training curves
    try:
        from src.analysis.training_curves import generate_training_curves
        curves_dir = run_dir / "training_curves"
        logger.info("MLv13 ML-18: Generating training curves...")
        generated = generate_training_curves(run_dir, output_dir=curves_dir)
        logger.info(f"MLv13 ML-18: {len(generated)} training curve plots saved to {curves_dir}")
    except Exception as exc:
        logger.warning(f"MLv13 ML-18: Training curve generation failed (non-fatal): {exc}")

    # ML-21: Bias analysis
    try:
        from src.analysis.bias_analysis import run_bias_analysis
        bias_out = run_dir / "bias_analysis.json"
        logger.info("MLv13 ML-21: Running bias analysis...")
        run_bias_analysis(run_dir, bias_out)
        logger.info(f"MLv13 ML-21: Bias analysis saved to {bias_out}")
    except Exception as exc:
        logger.warning(f"MLv13 ML-21: Bias analysis failed (non-fatal): {exc}")

    # ML-19: Figure QA
    try:
        from src.analysis.figure_qa import run_figure_qa
        figures_dir = run_dir / cfg["directories"].get("figures_subdir", "figures")
        qa_out = run_dir / "figure_qa_report.json"
        if figures_dir.exists():
            logger.info("MLv13 ML-19: Running figure quality assurance check...")
            run_figure_qa(figures_dir, qa_out)
            logger.info(f"MLv13 ML-19: Figure QA report saved to {qa_out}")
        else:
            logger.info("MLv13 ML-19: No figures directory found, skipping figure QA.")
    except Exception as exc:
        logger.warning(f"MLv13 ML-19: Figure QA failed (non-fatal): {exc}")

    # ML-24: XAI evaluation (only if GradCAM heatmaps already exist for this run)
    # xai_generator.py saves heatmaps to {model}/figures/xai/ under the run-level dir.
    # run_xai_evaluation() expects the run-level dir (parent of model_name), not model_dir.
    try:
        from src.analysis.xai_evaluation import run_xai_evaluation
        run_level_dir = run_dir.parent  # experiments/{run_id}/
        xai_dirs = list(run_level_dir.glob("*/figures/xai"))
        if xai_dirs:
            logger.info("MLv13 ML-24: Running XAI evaluation (heatmaps found)...")
            run_xai_evaluation(run_level_dir, run_level_dir)
            logger.info(f"MLv13 ML-24: XAI evaluation report saved to {run_level_dir}")
        else:
            logger.info(
                "MLv13 ML-24: No GradCAM heatmaps found yet — skipping XAI evaluation. "
                "Run the XAI notebook cells first, then re-run posttraining artifacts."
            )
    except Exception as exc:
        logger.warning(f"MLv13 ML-24: XAI evaluation failed (non-fatal): {exc}")


def check_system_requirements(cfg: dict) -> str:
    """
    Check system requirements and determine best accelerator.
    
    Returns:
        str: Accelerator type ('cuda', 'mps', or 'cpu')
    """
    logger.info("=" * 80)
    logger.info("SYSTEM CHECK")
    logger.info("=" * 80)
    
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        accelerator = "cuda"
    elif has_mps:
        logger.info("GPU: Apple Silicon (MPS)")
        accelerator = "mps"
    else:
        logger.warning("WARNING: No GPU detected - training will be SLOW!")
        logger.warning("Expected training time per epoch: 30-60 min (vs 2-5 min on GPU)")
        accelerator = "cpu"
    
    logger.info(f"Accelerator: {accelerator}")
    logger.info("=" * 80 + "\n")
    
    return accelerator


def validate_splits(cfg: dict):
    """
    Validate data splits before training (Fail-Fast principle).
    
    Performs comprehensive checks:
    - No image overlap between train/val
    - No animal_id overlap (for LOAO strategies)
    - Proper stratification maintained
    - All data used exactly once
    
    Aborts training if any check fails.
    """
    logger.info("=" * 80)
    logger.info("DATA SPLIT VALIDATION (Fail-Fast Check)")
    logger.info("=" * 80)
    logger.info(f"CV Strategy: {cfg['data']['cv_strategy']}")
    logger.info(f"Number of Folds: {cfg['data']['n_folds']}")
    
    try:
        validator = SplitValidator(cfg)
        summary_df = validator.validate_all_folds()
        
        # Check if any fold failed
        if any('FAILED' in str(val) for val in summary_df['Passed'].values):
            logger.error("CRITICAL: Data split validation failed!")
            logger.error("Training aborted to prevent invalid results")
            raise RuntimeError(
                "Data split validation failed!\n"
                "Fix data splits before training to ensure valid scientific results."
            )
        
        logger.info("Split validation PASSED - proceeding to training\n")
        
    except Exception as e:
        logger.exception("Failed to validate data splits")
        raise


def display_training_parameters(
    config: Dict[str, Any],
    model_name: str,
    hpo_applied: bool,
    logger_instance
) -> None:
    """
    Display training parameters before training starts.
    Shows clearly whether standard or HPO-optimized values are being used.
    
    Args:
        config: Current configuration (potentially with HPO values merged)
        model_name: Name of the model being trained
        hpo_applied: Whether HPO parameters were applied
        logger_instance: Logger instance for output
    """
    logger_instance.info("\n" + "=" * 80)
    logger_instance.info("TRAINING PARAMETERS - PRE-EXECUTION VERIFICATION")
    logger_instance.info("=" * 80)
    
    # Indicate source of parameters
    if hpo_applied:
        cv_strategy = config['data']['cv_strategy']
        hpo_params = load_hpo_config(model_name, cv_strategy=cv_strategy)
        source_indicator = "HPO-OPTIMIZED"
        logger_instance.info(f"Parameter Source: {source_indicator} (from HPO results file)")
        logger_instance.info(f"HPO File: configs/hpo_best_hpo_{model_name}_{cv_strategy}_*.yaml")
    else:
        source_indicator = "STANDARD"
        logger_instance.info(f"Parameter Source: {source_indicator} (from base.yaml + model config)")
    
    logger_instance.info("\n" + "-" * 80)
    logger_instance.info("HYPERPARAMETERS THAT WILL BE USED:")
    logger_instance.info("-" * 80)
    
    # Core training parameters
    lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']
    batch_size = config['data']['batch_size']
    max_epochs = config['training']['max_epochs']
    
    logger_instance.info(f"\n  Learning Rate:       {lr:.6e}  {'<-- HPO' if hpo_applied else '<-- Default'}")
    logger_instance.info(f"  Weight Decay:        {wd:.6e}  {'<-- HPO' if hpo_applied else '<-- Default'}")
    logger_instance.info(f"  Batch Size:          {batch_size}  {'<-- HPO' if hpo_applied else '<-- Default'}")
    logger_instance.info(f"  Max Epochs:          {max_epochs}")
    
    # Optimizer parameters
    optimizer_config = config['training'].get('optimizer', {})
    if 'betas' in optimizer_config:
        beta1, beta2 = optimizer_config['betas']
        logger_instance.info(f"  Adam Beta1:          {beta1:.6f}  {'<-- HPO' if hpo_applied else '<-- Default'}")
        logger_instance.info(f"  Adam Beta2:          {beta2:.6f}  {'<-- HPO' if hpo_applied else '<-- Default'}")
    
    # Scheduler parameters
    scheduler_config = config['training'].get('scheduler', {})
    if 'patience' in scheduler_config:
        sched_patience = scheduler_config['patience']
        logger_instance.info(f"  Scheduler Patience:  {sched_patience}  {'<-- HPO' if hpo_applied else '<-- Default'}")
    
    # Model parameters
    if 'model' in config:
        if 'drop_rate' in config['model']:
            drop_rate = config['model']['drop_rate']
            logger_instance.info(f"  Dropout Rate:        {drop_rate:.4f}  {'<-- HPO' if hpo_applied else '<-- Default'}")
        if 'backbone' in config['model']:
            backbone = config['model']['backbone']
            logger_instance.info(f"  Backbone:            {backbone}")
    
    # If HPO was applied, show comparison with base values
    if hpo_applied:
        logger_instance.info("\n" + "-" * 80)
        logger_instance.info("COMPARISON: HPO vs. BASE CONFIG")
        logger_instance.info("-" * 80)
        
        # Load base config for comparison
        base_config = load_config(model_name)
        base_lr = float(base_config['training']['learning_rate'])
        base_wd = float(base_config['training']['weight_decay'])
        base_bs = int(base_config['data']['batch_size'])
        
        lr_change = ((lr - base_lr) / base_lr) * 100 if base_lr != 0 else 0
        wd_change = ((wd - base_wd) / base_wd) * 100 if base_wd != 0 else 0
        bs_change = ((batch_size - base_bs) / base_bs) * 100 if base_bs != 0 else 0
        
        logger_instance.info(f"\n  Learning Rate:   Base={base_lr:.2e} -> HPO={lr:.2e} ({lr_change:+.1f}%)")
        logger_instance.info(f"  Weight Decay:    Base={base_wd:.2e} -> HPO={wd:.2e} ({wd_change:+.1f}%)")
        logger_instance.info(f"  Batch Size:      Base={base_bs} -> HPO={batch_size} ({bs_change:+.1f}%)")
    
    logger_instance.info("\n" + "=" * 80)
    logger_instance.info(f"READY TO START TRAINING WITH {source_indicator} PARAMETERS")
    logger_instance.info("=" * 80 + "\n")


def train_single_fold(
    model_name: str,
    cfg: dict,
    run_id: str,
    fold_idx: int,
    accelerator: str,
    tracker: ExperimentTracker,
    generate_eval_outputs: bool = False,
    experiment_subdir: Optional[str] = None,
):
    """
    Train model for a single fold.
    
    Args:
        model_name: Name of model to train
        cfg: Full configuration dict
        run_id: Unique run identifier
        fold_idx: Fold index
        accelerator: Device type ('cuda', 'mps', 'cpu')
        tracker: ExperimentTracker instance
        generate_eval_outputs: Generate visualizations after training (default: False)
                              Tier 1 mode: False (only checkpoints)
                              Legacy mode: True (checkpoints + visualizations)
        
    Returns:
        dict: Training results and metrics
    """
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
    
    # Custom callback for minimal, informative output
    class MinimalProgressCallback(Callback):
        """Shows only epoch-level progress without batch spam."""
        
        def on_train_epoch_start(self, trainer, pl_module):
            epoch = trainer.current_epoch
            max_epochs = trainer.max_epochs
            print(f"\n[Fold {fold_idx}] Epoch {epoch}/{max_epochs-1} - Training...", end="", flush=True)
        
        def on_validation_epoch_start(self, trainer, pl_module):
            print(" → Validating...", end="", flush=True)
        
        def on_validation_epoch_end(self, trainer, pl_module):
            # Get metrics
            metrics = trainer.callback_metrics
            val_kappa = metrics.get('val_kappa', 0.0)
            val_acc = metrics.get('val_acc', 0.0)
            train_loss = metrics.get('train_loss_epoch', 0.0)
            
            print(f" → val_kappa: {val_kappa:.4f}, val_acc: {val_acc:.3f}, train_loss: {train_loss:.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING: {model_name.upper()} - Fold {fold_idx}")
    logger.info("=" * 80 + "\n")
    
    # Load data for this fold
    logger.info(f"Loading fold {fold_idx} data...")
    train_loader, val_loader = get_dataloaders(cfg, fold_idx=fold_idx)
    logger.info(f"  Train samples: {len(train_loader.dataset)}")
    logger.info(f"  Val samples: {len(val_loader.dataset)}")
    
    # Log animal distribution
    train_animals = train_loader.dataset.df['animal_id'].unique()
    val_animals = val_loader.dataset.df['animal_id'].unique()
    logger.info(f"  Train animals: {sorted(train_animals)}")
    logger.info(f"  Val animals: {sorted(val_animals)}")
    logger.info("")
    
    # Start timing for this fold
    from datetime import datetime
    fold_start_time = datetime.now()
    logger.info(f"Fold {fold_idx} started at: {fold_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log hardware info
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.info("Hardware: CPU (no GPU detected)")
    logger.info("")
    
    # Create model
    logger.info(f"Building model: {model_name}")
    model_type = ModelFactory.get_model_type(model_name)
    logger.info(f"  Model type: {model_type}")

    # NOTE:
    # Current training/evaluation pipeline is classification-oriented and relies on
    # labeled (image, class) batches plus val_kappa monitoring.
    # True SSL training for SimCLR/DINO requires dedicated pair-view datasets and
    # different monitoring. Until that pipeline is integrated, keep compatibility
    # mode for self-supervised model names.
    if model_type == "self_supervised":
        logger.warning(
            "Self-supervised architecture requested, but current runner uses "
            "classification pipeline. Using InflammationModel compatibility mode."
        )
        model = InflammationModel(cfg)
    else:
        model = ModelFactory.create_model(model_name, cfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params/1e6:.2f}M")
    logger.info(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
    logger.info("")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=tracker.checkpoint_dir,
        filename=f"fold_{fold_idx}_best",
        monitor="val_kappa",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_kappa",
        patience=cfg['training']['patience'],
        min_delta=cfg['training'].get('min_delta', 0.002),
        mode="max",
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Add minimal progress callback
    minimal_progress = MinimalProgressCallback()
    
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor, minimal_progress]
    
    # Configure TensorBoard logger with fold-specific subdirectory.
    # Resolve project_root from config so this works correctly in Colab/Kaggle
    # where Path(__file__) would point to the wrong location.
    _project_root = Path(cfg["directories"].get("project_root", "."))
    if not _project_root.is_absolute():
        _project_root = Path.cwd() / _project_root
    _project_root = _project_root.resolve()
    _subdir_name = experiment_subdir or model_name
    run_dir = _project_root / cfg["directories"]["experiments_dir"] / run_id / _subdir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Explicitly create tensorboard dir so TensorBoardLogger never falls back
    # to a version-based subdirectory in an unexpected location.
    tensorboard_dir = run_dir / cfg["directories"]["tensorboard_subdir"]
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_logger = TensorBoardLogger(
        save_dir=str(tensorboard_dir),
        name=f"fold_{fold_idx}",
        version=""
    )
    
    # Configure CSV Logger alongside TensorBoard
    csv_logger = CSVLogger(
        save_dir=str(run_dir),
        name="csv_logs", 
        version=f"fold_{fold_idx}"
    )
    
    # Create trainer with minimal output (only epoch-level progress)
    trainer = L.Trainer(
        max_epochs=cfg['training']['max_epochs'],
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        log_every_n_steps=50,  # Reduced logging frequency
        enable_progress_bar=False,  # Disable batch-level progress bar
        enable_model_summary=True,
        deterministic=False,
        logger=[tensorboard_logger, csv_logger]  # Use custom loggers
    )
    
    # Log training config
    logger.info("Training Configuration:")
    logger.info(f"  Max epochs: {cfg['training']['max_epochs']}")
    logger.info(f"  Learning rate: {float(cfg['training']['learning_rate']):.1e}")
    logger.info(f"  Weight decay: {float(cfg['training']['weight_decay']):.1e}")
    logger.info(f"  Batch size: {cfg['data']['batch_size']}")
    logger.info(f"  Early stopping patience: {cfg['training']['patience']}")
    logger.info(f"  Optimizer: {cfg['training'].get('optimizer', {}).get('type', 'adamw')}")
    logger.info(f"  Scheduler: {cfg['training'].get('scheduler', {}).get('type', 'reduce_on_plateau')}")
    logger.info("")
    
    # Train
    logger.info("Starting training...\n")
    trainer.fit(model, train_loader, val_loader)
    
    # Load best checkpoint
    logger.info(f"\nLoading best checkpoint: {checkpoint_callback.best_model_path}")
    best_model = InflammationModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        cfg=cfg
    )
    
    # Final validation
    val_results = trainer.validate(best_model, val_loader, verbose=False)
    final_metrics = val_results[0] if val_results else {}
    
    # === COMPUTE ADDITIONAL METRICS ===
    logger.info("\nComputing additional metrics...")
    
    # 1. Efficiency Metrics
    device = torch.device(accelerator if accelerator != "mps" else "mps")
    if accelerator == "cuda":
        device = torch.device("cuda")
    elif accelerator == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    efficiency_metrics = compute_efficiency_metrics(
        model=best_model,
        input_size=(1, 3, cfg['data']['img_size'], cfg['data']['img_size']),
        batch_size=cfg['data']['batch_size'],
        device=device,
        verbose=True
    )
    
    # 2. Calibration Metrics
    logger.info("\nComputing calibration metrics...")
    calibration_metrics = evaluate_calibration(
        model=best_model,
        dataloader=val_loader,
        device=device,
        num_classes=4,
        ignore_index=cfg.get('ignore_class_index'),
        n_bins=10,
        verbose=True
    )
    
    # 3. Per-Class F1 (extract from logged metrics)
    per_class_f1 = []
    for i in range(4):
        f1_key = f"val_f1_class_{i}"
        if f1_key in final_metrics:
            per_class_f1.append(float(final_metrics[f1_key]))
    
    # Calculate fold duration
    fold_end_time = datetime.now()
    fold_duration = fold_end_time - fold_start_time
    fold_duration_seconds = fold_duration.total_seconds()
    fold_duration_str = f"{int(fold_duration_seconds // 3600)}h {int((fold_duration_seconds % 3600) // 60)}m {int(fold_duration_seconds % 60)}s"
    
    logger.info("")
    logger.info("="*80)
    logger.info(f"FOLD {fold_idx} COMPLETE")
    logger.info("="*80)
    logger.info(f"Started:  {fold_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Finished: {fold_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration: {fold_duration_str} ({fold_duration_seconds:.1f} seconds)")
    logger.info(f"Final QWK: {final_metrics.get('val_kappa', 0.0):.4f}")
    logger.info("="*80)
    logger.info("")
    
    # Compile results with ALL metrics
    results = {
        "fold": fold_idx,
        "status": "success",
        "model_name": model_name,
        "epochs_trained": trainer.current_epoch + 1,
        
        # Core Validation Metrics
        "val_loss": final_metrics.get("val_loss", 0.0),
        "val_acc": final_metrics.get("val_acc", 0.0),
        "val_kappa": final_metrics.get("val_kappa", 0.0),
        "val_macro_f1": final_metrics.get("val_macro_f1", 0.0),
        
        # Per-Class F1 Scores
        "per_class_f1": per_class_f1,
        "f1_class_0": per_class_f1[0] if len(per_class_f1) > 0 else 0.0,
        "f1_class_1": per_class_f1[1] if len(per_class_f1) > 1 else 0.0,
        "f1_class_2": per_class_f1[2] if len(per_class_f1) > 2 else 0.0,
        "f1_class_3": per_class_f1[3] if len(per_class_f1) > 3 else 0.0,
        
        # Efficiency Metrics
        "total_parameters": efficiency_metrics["parameters"]["total"],
        "trainable_parameters": efficiency_metrics["parameters"]["trainable"],
        "model_size_mb": efficiency_metrics["model_size"]["mb"],
        "inference_time_ms": efficiency_metrics["inference_time"]["single_image_ms"]["mean"],
        "inference_time_std_ms": efficiency_metrics["inference_time"]["single_image_ms"]["std"],
        "throughput_imgs_per_sec": efficiency_metrics["throughput"]["images_per_second"],
        
        # Calibration Metrics
        "mean_confidence": calibration_metrics["confidence"]["mean_confidence"],
        "confidence_correct": calibration_metrics["confidence"]["mean_confidence_correct"],
        "confidence_incorrect": calibration_metrics["confidence"]["mean_confidence_incorrect"],
        "confidence_gap": calibration_metrics["confidence"]["confidence_gap"],
        "ece": calibration_metrics["calibration"]["ece"],
        "brier_score": calibration_metrics["brier_score"],
        
        # Training Metadata
        "checkpoint_path": str(checkpoint_callback.best_model_path),
        "training_duration_seconds": fold_duration_seconds,
        "training_duration_str": fold_duration_str,
        "started_at": fold_start_time.isoformat(),
        "finished_at": fold_end_time.isoformat(),
        
        # Full metric objects for detailed analysis
        "_efficiency_full": efficiency_metrics,
        "_calibration_full": calibration_metrics,
    }
    
    # Log to tracker
    tracker.log_metrics(results, step=fold_idx, prefix=f"fold_{fold_idx}/")
    
    # Save detailed metric files
    logger.info("\nSaving detailed metric files...")
    tracker.save_efficiency_metrics(efficiency_metrics, fold_idx=fold_idx)
    tracker.save_calibration_metrics(calibration_metrics, fold_idx=fold_idx)
    
    # Generate all visualizations and save predictions (optional)
    if generate_eval_outputs:
        logger.info("\nGenerating evaluation outputs (optimized - single inference)...")
        from src.utils.visualization_optimized import generate_all_visualizations_optimized
        
        generate_all_visualizations_optimized(
            model=best_model,
            val_loader=val_loader,
            metrics_dir=tracker.metrics_dir,
            predictions_dir=tracker.predictions_dir,
            fold_idx=fold_idx,
            class_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3'],
            include_predictions_csv=False,  # Skip CSV during training for speed
            exclude_ignore_class=True,
            ignore_class_idx=4,
        )
        
        # Save final metrics JSON
        metrics_file = tracker.metrics_dir / f"fold_{fold_idx}_metrics.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved: {metrics_file}")
    else:
        logger.info("Skipping evaluation output generation (will be done in separate reporting phase)")
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING COMPLETE: {model_name.upper()} - Fold {fold_idx}")
    logger.info("=" * 80)
    logger.info("Final Metrics:")
    for key, value in results.items():
        if key in ["checkpoint_path", "model_name", "fold", "status"]:
            continue
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 80 + "\n")
    
    return results


def train_single_model(
    model_name: str,
    config: dict,
    run_id: str,
    generate_eval_outputs: bool = False,
    notes: str = "",
    experiment_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train a single model across all folds (simplified workflow for thesis).
    
    This function orchestrates training for ONE model with clear progression:
    1. System checks and split validation
    2. Train all CV folds
    3. Calculate cross-fold metrics
    4. Return results for Best Models Registry update
    
    3-Tier Storage System:
    - Tier 1 (Training): generate_eval_outputs=False → Only checkpoints + TensorBoard (default)
    - Tier 2 (Evaluation): Call generate_evaluation_outputs_for_run() after training
    - Tier 3 (Statistical): Use Model_Comparison_Analysis.ipynb for multi-model comparison
    
    Args:
        model_name: Name of model to train (e.g., "densenet", "vit")
        config: Configuration dictionary (loaded via load_config(model_name))
        run_id: Unique run identifier (format: YYYY-MM-DD_HH-MM-SS)
        generate_eval_outputs: Generate visualizations during training (default: False)
                              Tier 1 mode (recommended): False → Only checkpoints
                              Tier 2 mode: True → Checkpoints + visualizations
        notes: Optional notes for this training run (stored in Best Models Registry)
        
    Returns:
        dict: Training summary with keys:
            - model_name: str
            - run_id: str  
            - n_folds: int
            - fold_results: List[Dict] - metrics for each fold
            - mean_qwk: float - average QWK across folds
            - std_qwk: float - std of QWK across folds
            - mean_acc: float - average accuracy across folds
            - checkpoint_paths: List[str] - paths to fold checkpoints
            - notes: str
            - status: "success" or "failed"
        
    Example Usage (Master Notebook):
        ```python
        # Cell 1: Setup
        from configs.utils import load_config
        from src.training.runner import train_single_model
        
        model_name = "densenet"  # User sets this
        
        # Cell 2: Train (Tier 1 - Checkpoints only)
        cfg = load_config(model_name)
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        results = train_single_model(
            model_name=model_name,
            config=cfg,
            run_id=run_id,
            generate_eval_outputs=False,  # Tier 1: Fast training
            notes="Baseline training with dropout=0.3"
        )
        
        # Cell 3: Update Best Models Registry (manual)
        from src.utils.best_models_registry import load_registry
        
        registry = load_registry()
        status = registry.update_model(
            model_name=results['model_name'],
            run_id=results['run_id'],
            cv_strategy=cfg['data']['cv_strategy'],
            fold_metrics=results['fold_results'],
            checkpoint_paths=results['checkpoint_paths'],
            notes=results['notes']
        )
        
        if status['updated']:
            print(f"{status['reason']}")
            print(f"New QWK: {status['new_mean_qwk']:.4f}")
            if status['improvement']:
                print(f"Improvement: +{status['improvement']:.4f}")
        else:
            print(status['reason'])
        ```
    """
    import numpy as np
    
    # Load project_root from config
    project_root = Path(config['directories'].get('project_root', '.'))
    if not project_root.is_absolute():
        project_root = Path.cwd() / project_root
    project_root = project_root.resolve()
    
    # ========================================================================
    # HYPERPARAMETER OPTIMIZATION (HPO) - Runs BEFORE training if needed
    # ========================================================================
    hpo_mode = config.get('hpo', {}).get('mode', 'skip')
    hpo_applied = False
    hpo_params_used = None
    
    # Extract CV strategy for HPO file naming (prevents overwriting across strategies)
    cv_strategy = config['data']['cv_strategy']
    
    # Determine if HPO needs to run
    run_hpo = False
    if hpo_mode == 'skip':
        # Skip mode: No HPO, no loading of saved HPO parameters
        logger.info("\n" + "=" * 80)
        logger.info("HPO MODE: SKIP")
        logger.info("=" * 80)
        logger.info("HPO is disabled - using default parameters from config")
        logger.info("=" * 80 + "\n")
    elif hpo_mode == 'overwrite':
        # Always run HPO in overwrite mode
        run_hpo = True
        logger.info("\n" + "=" * 80)
        logger.info("HPO MODE: OVERWRITE")
        logger.info("=" * 80)
        logger.info("Will run HPO optimization (overwrites existing results)")
        logger.info("=" * 80 + "\n")
    elif hpo_mode == 'use_existing':
        # Only run if results don't exist
        if not hpo_results_exist(model_name, cv_strategy=cv_strategy):
            run_hpo = True
            logger.info("\n" + "=" * 80)
            logger.info("HPO MODE: USE_EXISTING")
            logger.info("=" * 80)
            logger.info(f"No HPO results found for {cv_strategy} - will run optimization first")
            logger.info(f"Results will be saved to: configs/hpo_best_hpo_{model_name}_{cv_strategy}_v2.yaml")
            logger.info("=" * 80 + "\n")
        else:
            logger.info("\n" + "=" * 80)
            logger.info("HPO MODE: USE_EXISTING")
            logger.info("=" * 80)
            logger.info("HPO results found - will use existing parameters")
            logger.info("=" * 80 + "\n")
    
    # Execute HPO if needed
    if run_hpo:
        logger.info("" + "=" * 80)
        logger.info("RUNNING HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Trials: {config['hpo']['n_trials']}")
        logger.info(f"Folds per trial: {config['hpo']['n_folds']}")
        logger.info(f"This will take approximately: {config['hpo']['n_trials'] * 5} minutes")
        logger.info("=" * 80 + "\n")
        
        try:
            # Run HPO optimization
            # Resolve hpo n_folds constraint against data n_folds to avoid IndexError
            hpo_n_folds = config['hpo']['n_folds']
            
            # Extract true n_folds from data config based on strategy
            if 'data' in config:
                cv_strategy_for_hpo = config['data']['cv_strategy']
                if 'cv_folds_config' in config['data']:
                    actual_dataset_folds = config['data']['cv_folds_config'].get(cv_strategy_for_hpo, 2)
                    # Limit HPO folds to the maximum available dataset splits
                    hpo_n_folds = min(hpo_n_folds, actual_dataset_folds)
            
            study = optimize_model(
                model_name=model_name,
                n_trials=config['hpo']['n_trials'],
                n_folds=hpo_n_folds,
                overwrite=(hpo_mode == 'overwrite'),
                cv_strategy=cv_strategy,
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("HPO OPTIMIZATION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Best QWK: {study.best_trial.value:.4f}")
            logger.info(f"Results saved to: configs/hpo_best_{study.study_name}.yaml")
            logger.info("=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"\nHPO OPTIMIZATION FAILED!")
            logger.error(f"Error: {e}")
            logger.error(f"\nTraining ABORTED - HPO is required but failed.")
            logger.error(f"Please fix the HPO configuration and try again.\n")
            # Return error result instead of continuing
            return {
                'status': 'failed',
                'error': f'HPO optimization failed: {str(e)}',
                'model_name': model_name,
                'run_id': run_id
            }
    
    # Apply HPO parameters ONLY if mode is 'use_existing' or 'overwrite'
    # IMPORTANT: In 'skip' mode, we do NOT load any HPO parameters!
    if hpo_mode in ['use_existing', 'overwrite']:
        # Check if HPO results exist (after potentially running HPO above)
        if hpo_results_exist(model_name, cv_strategy=cv_strategy):
            logger.info("\n" + "=" * 80)
            logger.info("HPO PARAMETERS DETECTED")
            logger.info("=" * 80)
            
            # Load and merge HPO parameters (strategy-specific)
            from configs.utils import load_hpo_config
            hpo_params_used = load_hpo_config(model_name, cv_strategy=cv_strategy)
            config = merge_hpo_config(config, model_name, cv_strategy=cv_strategy)
            hpo_applied = True
            
            logger.info("\nOptimized parameters loaded:")
            if 'learning_rate' in hpo_params_used:
                logger.info(f"  Learning Rate: {hpo_params_used['learning_rate']:.2e}")
            if 'weight_decay' in hpo_params_used:
                logger.info(f"  Weight Decay: {hpo_params_used['weight_decay']:.2e}")
            if 'batch_size' in hpo_params_used:
                logger.info(f"  Batch Size: {hpo_params_used['batch_size']}")
            if 'beta1' in hpo_params_used and 'beta2' in hpo_params_used:
                logger.info(f"  Adam Betas: [{hpo_params_used['beta1']:.4f}, {hpo_params_used['beta2']:.4f}]")
            if 'scheduler_patience' in hpo_params_used:
                logger.info(f"  Scheduler Patience: {hpo_params_used['scheduler_patience']}")
            if 'drop_rate' in hpo_params_used:
                logger.info(f"  Dropout Rate: {hpo_params_used['drop_rate']:.3f}")
            logger.info("=" * 80 + "\n")
        elif hpo_mode == 'use_existing' and not run_hpo:
            # This should never happen: use_existing mode without results and without running HPO
            logger.error("\n" + "!" * 80)
            logger.error("CRITICAL ERROR: HPO mode 'use_existing' but no results available!")
            logger.error("  This indicates a logic error in the HPO execution flow.")
            logger.error("!" * 80 + "\n")
            return {
                'status': 'failed',
                'error': 'HPO mode use_existing but no results available',
                'model_name': model_name,
                'run_id': run_id
            }
    
    # Get n_folds with fallback to cv_folds_config (corresponds to sklearn's n_splits parameter)
    if 'n_folds' in config['data']:
        n_folds = config['data']['n_folds']
    else:
        cv_strategy = config['data']['cv_strategy']
        cv_folds_config = config['data'].get('cv_folds_config', {
            'loao_balanced': 2,
            'random_stratified': 5
        })
        n_folds = cv_folds_config.get(cv_strategy, 2)
        config['data']['n_folds'] = n_folds  # Set for consistency
    
    # Record start time for timing statistics
    start_time = datetime.now()
    
    # Display all training parameters before starting (CRITICAL for HPO verification)
    display_training_parameters(
        config=config,
        model_name=model_name,
        hpo_applied=hpo_applied,
        logger_instance=logger
    )
    
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING: {model_name.upper()}")
    logger.info("=" * 80)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Folds: {n_folds}")
    logger.info(f"Max Epochs: {config['training']['max_epochs']}")
    logger.info(f"Learning Rate: {float(config['training']['learning_rate']):.1e}")
    logger.info(f"Batch Size: {config['data']['batch_size']}")
    logger.info(f"Image Size: {config['data']['img_size']}x{config['data']['img_size']}")
    logger.info(f"HPO Mode: {hpo_mode} {'(parameters applied)' if hpo_applied else '(using defaults)'}")
    logger.info(f"Tier Mode: {'2 (Checkpoints + Visualizations)' if generate_eval_outputs else '1 (Checkpoints Only)'}")
    if notes:
        logger.info(f"Notes: {notes}")
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80 + "\n")
    
    fold_results = []
    checkpoint_paths = []
    subdir_name = experiment_subdir or model_name

    # --- MLv13 pre-training artifacts (ML-5, ML-6, ML-16) ---
    _run_preflight_artifacts(config, project_root)

    try:
        # Train all folds
        for fold_idx in range(n_folds):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"FOLD {fold_idx + 1}/{n_folds}")
            logger.info("=" * 80 + "\n")
            
            # Use the passed config (preserves path corrections from notebook)
            # No need to reload - config already contains all necessary settings
            
            # Create ExperimentTracker — use subdir_name so the on-disk directory
            # matches the tensorboard logger path set in train_single_fold().
            experiment_dir = project_root / config["directories"]["experiments_dir"]
            tracker = ExperimentTracker(
                model_name=subdir_name,
                run_id=run_id,
                config=config,
                experiment_dir=str(experiment_dir),
                checkpoints_subdir=config["directories"]["checkpoints_subdir"],
                figures_subdir=config["directories"]["figures_subdir"]
            )
            
            # Train fold
            fold_metrics = train_single_fold(
                model_name=model_name,
                cfg=config,
                run_id=run_id,
                fold_idx=fold_idx,
                accelerator=config["accelerator"],
                tracker=tracker,
                generate_eval_outputs=generate_eval_outputs,
                experiment_subdir=experiment_subdir,
            )
            
            tracker.finish()
            
            # Store results with ALL metrics
            fold_results.append({
                'fold': fold_idx,
                # Core metrics
                'val_qwk': fold_metrics['val_kappa'],
                'val_acc': fold_metrics['val_acc'],
                'val_loss': fold_metrics['val_loss'],
                'val_macro_f1': fold_metrics.get('val_macro_f1', 0.0),
                'epochs_trained': fold_metrics['epochs_trained'],
                
                # Per-class F1
                'per_class_f1': fold_metrics.get('per_class_f1', []),
                'f1_class_0': fold_metrics.get('f1_class_0', 0.0),
                'f1_class_1': fold_metrics.get('f1_class_1', 0.0),
                'f1_class_2': fold_metrics.get('f1_class_2', 0.0),
                'f1_class_3': fold_metrics.get('f1_class_3', 0.0),
                
                # Efficiency
                'total_parameters': fold_metrics.get('total_parameters', 0),
                'model_size_mb': fold_metrics.get('model_size_mb', 0.0),
                'inference_time_ms': fold_metrics.get('inference_time_ms', 0.0),
                'throughput_imgs_per_sec': fold_metrics.get('throughput_imgs_per_sec', 0.0),
                
                # Calibration
                'mean_confidence': fold_metrics.get('mean_confidence', 0.0),
                'confidence_gap': fold_metrics.get('confidence_gap', 0.0),
                'ece': fold_metrics.get('ece', 0.0),
                'brier_score': fold_metrics.get('brier_score', 0.0),
                
                # Timing
                'training_duration_seconds': fold_metrics.get('training_duration_seconds', 0.0),
                'training_duration_str': fold_metrics.get('training_duration_str', ''),
            })
            checkpoint_paths.append(fold_metrics['checkpoint_path'])
        
        # Calculate summary statistics for ALL metrics
        mean_qwk = np.mean([f['val_qwk'] for f in fold_results])
        std_qwk = np.std([f['val_qwk'] for f in fold_results], ddof=1)
        mean_acc = np.mean([f['val_acc'] for f in fold_results])
        std_acc = np.std([f['val_acc'] for f in fold_results], ddof=1)
        mean_f1 = np.mean([f['val_macro_f1'] for f in fold_results])
        std_f1 = np.std([f['val_macro_f1'] for f in fold_results], ddof=1)
        
        # Per-class F1 statistics (if available)
        per_class_f1_stats = {}
        for class_idx in range(4):
            class_f1s = [f[f'f1_class_{class_idx}'] for f in fold_results if f'f1_class_{class_idx}' in f]
            if class_f1s:
                per_class_f1_stats[f'mean_f1_class_{class_idx}'] = float(np.mean(class_f1s))
                per_class_f1_stats[f'std_f1_class_{class_idx}'] = float(np.std(class_f1s, ddof=1))
        
        # Efficiency statistics (should be identical across folds, but calculate mean for consistency)
        mean_params = np.mean([f['total_parameters'] for f in fold_results])
        mean_model_size = np.mean([f['model_size_mb'] for f in fold_results])
        mean_inference_time = np.mean([f['inference_time_ms'] for f in fold_results])
        std_inference_time = np.std([f['inference_time_ms'] for f in fold_results], ddof=1)
        mean_throughput = np.mean([f['throughput_imgs_per_sec'] for f in fold_results])
        
        # Calibration statistics (variance shows robustness)
        mean_confidence = np.mean([f['mean_confidence'] for f in fold_results])
        std_confidence = np.std([f['mean_confidence'] for f in fold_results], ddof=1)
        mean_conf_gap = np.mean([f['confidence_gap'] for f in fold_results])
        std_conf_gap = np.std([f['confidence_gap'] for f in fold_results], ddof=1)
        mean_ece = np.mean([f['ece'] for f in fold_results])
        std_ece = np.std([f['ece'] for f in fold_results], ddof=1)
        mean_brier = np.mean([f['brier_score'] for f in fold_results])
        std_brier = np.std([f['brier_score'] for f in fold_results], ddof=1)
        
        end_time = datetime.now()
        duration = end_time - start_time
        duration_seconds = duration.total_seconds()
        duration_str = f"{int(duration_seconds // 3600)}h {int((duration_seconds % 3600) // 60)}m {int(duration_seconds % 60)}s"
        
        # Calculate per-fold timing statistics (safely handle missing timing data)
        fold_durations = [f.get('training_duration_seconds', 0) for f in fold_results if 'training_duration_seconds' in f]
        if fold_durations:
            mean_fold_duration = np.mean(fold_durations)
            min_fold_duration = np.min(fold_durations)
            max_fold_duration = np.max(fold_durations)
        else:
            # Fallback if no timing data available
            mean_fold_duration = duration_seconds / len(fold_results) if fold_results else 0
            min_fold_duration = mean_fold_duration
            max_fold_duration = mean_fold_duration
        
        # Save checkpoint metadata for evaluation phase
        experiment_dir = project_root / config["directories"]["experiments_dir"]
        model_dir = experiment_dir / run_id / subdir_name
        checkpoints_file = model_dir / "checkpoints.json"
        
        checkpoint_metadata = {
            "model_name": model_name,
            "run_id": run_id,
            "n_folds": n_folds,
            "mean_qwk": float(mean_qwk),
            "std_qwk": float(std_qwk),
            "mean_acc": float(mean_acc),
            "folds": [
                {
                    "fold_idx": result['fold'],
                    "val_qwk": result['val_qwk'],
                    "val_acc": result['val_acc'],
                    "val_loss": result['val_loss'],
                    "epochs_trained": result['epochs_trained'],
                    "checkpoint_path": checkpoint_paths[i]
                }
                for i, result in enumerate(fold_results)
            ],
            "timestamp": end_time.isoformat()
        }
        
        with open(checkpoints_file, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        logger.info(f"Checkpoint metadata saved: {checkpoints_file}")

        # --- MLv13 post-training artifacts (ML-18, ML-19, ML-21) ---
        _run_posttraining_artifacts(config, project_root, run_id, subdir_name)

        logger.info("\n" + "=" * 80)
        logger.info(f"TRAINING COMPLETE: {model_name.upper()}")
        logger.info("=" * 80)
        logger.info("CROSS-VALIDATION SUMMARY:")
        logger.info(f"  QWK:        {mean_qwk:.4f} ± {std_qwk:.4f}")
        logger.info(f"  Accuracy:   {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"  F1 Macro:   {mean_f1:.4f} ± {std_f1:.4f}")
        logger.info("")
        logger.info("EFFICIENCY METRICS:")
        logger.info(f"  Parameters:    {int(mean_params):,} ({mean_params/1e6:.2f}M)")
        logger.info(f"  Model Size:    {mean_model_size:.2f} MB")
        logger.info(f"  Inference:     {mean_inference_time:.3f} ± {std_inference_time:.3f} ms")
        logger.info(f"  Throughput:    {mean_throughput:.2f} images/sec")
        logger.info("")
        logger.info("CALIBRATION METRICS:")
        logger.info(f"  Confidence:    {mean_confidence:.4f} ± {std_confidence:.4f}")
        logger.info(f"  Conf. Gap:     {mean_conf_gap:.4f} ± {std_conf_gap:.4f}")
        logger.info(f"  ECE:           {mean_ece:.4f} ± {std_ece:.4f}")
        logger.info(f"  Brier Score:   {mean_brier:.4f} ± {std_brier:.4f}")
        logger.info("")
        logger.info("TIMING SUMMARY:")
        logger.info(f"  Training started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Training finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Total duration:    {duration_str} ({duration_seconds:.1f} seconds)")
        logger.info("")
        logger.info("PER-FOLD TIMING:")
        for result in fold_results:
            logger.info(f"  Fold {result['fold']}: {result['training_duration_str']} ({result['training_duration_seconds']:.1f}s) - QWK: {result['val_qwk']:.4f}")
        logger.info("")
        logger.info(f"  Mean fold duration:   {int(mean_fold_duration // 60)}m {int(mean_fold_duration % 60)}s ({mean_fold_duration:.1f}s)")
        logger.info(f"  Fastest fold:         {int(min_fold_duration // 60)}m {int(min_fold_duration % 60)}s ({min_fold_duration:.1f}s)")
        logger.info(f"  Slowest fold:         {int(max_fold_duration // 60)}m {int(max_fold_duration % 60)}s ({max_fold_duration:.1f}s)")
        logger.info("")
        logger.info("FOLD-WISE RESULTS:")
        for result in fold_results:
            logger.info(f"  Fold {result['fold']}: QWK={result['val_qwk']:.4f}, Acc={result['val_acc']:.4f}, F1={result['val_macro_f1']:.4f}")
        logger.info("=" * 80 + "\n")
        
        return {
            'model_name': model_name,
            'run_id': run_id,
            'n_folds': n_folds,
            'fold_results': fold_results,
            
            # Core metrics (with variance for stability analysis)
            'mean_qwk': float(mean_qwk),
            'std_qwk': float(std_qwk),
            'mean_acc': float(mean_acc),
            'std_acc': float(std_acc),
            'mean_f1': float(mean_f1),
            'std_f1': float(std_f1),
            
            # Per-class F1 statistics
            **per_class_f1_stats,
            
            # Efficiency metrics
            'total_parameters': int(mean_params),
            'model_size_mb': float(mean_model_size),
            'inference_time_ms': float(mean_inference_time),
            'inference_time_std_ms': float(std_inference_time),
            'throughput_imgs_per_sec': float(mean_throughput),
            
            # Calibration metrics
            'mean_confidence': float(mean_confidence),
            'std_confidence': float(std_confidence),
            'confidence_gap': float(mean_conf_gap),
            'std_confidence_gap': float(std_conf_gap),
            'ece': float(mean_ece),
            'std_ece': float(std_ece),
            'brier_score': float(mean_brier),
            'std_brier_score': float(std_brier),
            
            # Metadata
            'checkpoint_paths': checkpoint_paths,
            'notes': notes,
            'status': 'success',
            'duration': str(duration),
            'duration_seconds': float(duration_seconds),
            
            # HPO information for documentation
            'hpo_applied': hpo_applied,
            'hpo_mode': hpo_mode,
            'hpo_parameters': hpo_params_used if hpo_applied else None
        }
    
    except Exception as e:
        logger.exception(f"Training failed for {model_name}")
        return {
            'model_name': model_name,
            'run_id': run_id,
            'status': 'failed',
            'error': str(e)
        }


def train_both_cv_strategies(
    model_name: str,
    config: dict,
    run_id: str,
    generate_eval_outputs: bool = False,
    notes: str = "",
) -> Dict[str, Any]:
    """Train a model with both LOAO and stratified CV under the same run_id.

    Calls train_single_model() twice — once per strategy — and stores results
    in separate subdirectories under the same run:

        experiments/{run_id}/{model_name}_loao/
        experiments/{run_id}/{model_name}_stratified/

    The model architecture and weights are independent between strategies; each
    strategy gets its own folds, checkpoints, TensorBoard logs, and artifacts.

    Args:
        model_name: Model to train (e.g. "densenet", "vit").
        config: Base configuration dict (loaded via load_config(model_name)).
            cv_strategy inside this dict is overridden per strategy — the original
            dict is never mutated (deep copies are made internally).
        run_id: Shared run identifier for both strategies.
        generate_eval_outputs: Passed through to train_single_model() for both runs.
        notes: Optional notes stored in Best Models Registry for both runs.

    Returns:
        Dict with keys "loao" and "stratified", each containing the result dict
        returned by train_single_model() for that strategy.
    """
    import copy

    strategies = [
        ("loao_balanced", "loao"),
        ("random_stratified", "stratified"),
    ]

    all_results: Dict[str, Any] = {}

    for cv_strategy, suffix in strategies:
        logger.info("\n" + "=" * 80)
        logger.info(f"DUAL-CV RUN — Starting strategy: {cv_strategy.upper()}")
        logger.info(f"  Experiment subdir: {model_name}_{suffix}")
        logger.info("=" * 80 + "\n")

        strategy_cfg = copy.deepcopy(config)
        strategy_cfg["data"]["cv_strategy"] = cv_strategy

        # Set n_folds from cv_folds_config so the fold loop uses the right count.
        cv_folds_config = strategy_cfg["data"].get("cv_folds_config", {})
        strategy_cfg["data"]["n_folds"] = cv_folds_config.get(
            cv_strategy,
            2 if cv_strategy == "loao_balanced" else 5,
        )

        result = train_single_model(
            model_name=model_name,
            config=strategy_cfg,
            run_id=run_id,
            generate_eval_outputs=generate_eval_outputs,
            notes=notes,
            experiment_subdir=f"{model_name}_{suffix}",
        )
        all_results[suffix] = result

        status = result.get("status", "unknown")
        if status == "success":
            logger.info(
                f"DUAL-CV: {cv_strategy} complete — "
                f"mean QWK={result.get('mean_qwk', 0.0):.4f} "
                f"± {result.get('std_qwk', 0.0):.4f}"
            )
        else:
            logger.error(
                f"DUAL-CV: {cv_strategy} FAILED — {result.get('error', 'unknown error')}"
            )

    logger.info("\n" + "=" * 80)
    logger.info("DUAL-CV RUN COMPLETE")
    for suffix, result in all_results.items():
        status = result.get("status", "unknown")
        if status == "success":
            logger.info(
                f"  {suffix:>12}: QWK={result.get('mean_qwk', 0.0):.4f} "
                f"± {result.get('std_qwk', 0.0):.4f}  [{status}]"
            )
        else:
            logger.info(f"  {suffix:>12}: [{status}] {result.get('error', '')}")
    logger.info("=" * 80 + "\n")

    return all_results


def main():
    """
    Main training pipeline with CLI support.
    
    Usage:
        # Standard training with automatic reporting
        python src/training/runner.py
        
        # Training without report/summary
        python src/training/runner.py --skip-report
        python src/training/runner.py --skip-summary
        
        # Generate report only for existing run
        python src/training/runner.py --only-report --run-id 2026-01-06_14-30-00
        
        # Display summary only for existing run
        python src/training/runner.py --only-summary --run-id 2026-01-06_14-30-00
    """
    import argparse
    from .reporting import generate_comprehensive_report, display_results_summary
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train inflammation classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip automatic report generation after training"
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip automatic results summary display after training"
    )
    parser.add_argument(
        "--only-report",
        action="store_true",
        help="Only generate report for existing run (requires --run-id)"
    )
    parser.add_argument(
        "--only-summary",
        action="store_true",
        help="Only display summary for existing run (requires --run-id)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID for report/summary generation (format: YYYY-MM-DD_HH-MM-SS)"
    )
    
    args = parser.parse_args()
    
    # Load configuration early for directory info
    cfg = load_config()
    project_root = Path(__file__).resolve().parent.parent.parent
    experiments_dir = project_root / cfg['directories']['experiments_dir']
    
    # Handle report-only mode
    if args.only_report or args.only_summary:
        if not args.run_id:
            logger.error("--run-id is required when using --only-report or --only-summary")
            return
        
        run_dir = experiments_dir / args.run_id
        if not run_dir.exists():
            logger.error(f"Run directory not found: {run_dir}")
            return
        
        if args.only_report:
            logger.info("Generating report for run: %s", args.run_id)
            generate_comprehensive_report(run_dir, verbose=True)
        
        if args.only_summary:
            logger.info("Displaying summary for run: %s", args.run_id)
            # Load results from checkpoints.json
            import json
            checkpoint_file = run_dir / "checkpoints.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint_map = json.load(f)
                
                # Reconstruct all_results format from checkpoint_map
                all_results = {}
                for model_name, info in checkpoint_map.items():
                    # Minimal results dict for summary display
                    all_results[model_name] = [{
                        "status": "success",
                        "val_kappa": info["mean_kappa"]
                    }]
                
                models_to_train = list(checkpoint_map.keys())
                display_results_summary(all_results, run_dir, models_to_train, verbose=True)
            else:
                logger.error(f"Checkpoint metadata not found: {checkpoint_file}")
        
        return
    
    # Standard training mode
    logger.info("\n" + "=" * 80)
    logger.info("AUTOMATED TRAINING RUNNER")
    logger.info("=" * 80 + "\n")
    
    # 1. Load configuration (re-use from above to avoid duplicate loading)
    logger.info("Step 1: Loading configuration...")
    # cfg already loaded above for directory info
    if not cfg.get('models_to_train'):
        raise ValueError(
            "'models_to_train' is missing or empty in base.yaml. "
            "Set it to e.g. ['densenet'] before running."
        )
    model_name = cfg['models_to_train'][0]
    
    # Load model-specific config
    cfg = load_config(model_name)
    
    logger.info(f"  Model: {model_name}")
    logger.info(f"  CV Strategy: {cfg['data']['cv_strategy']}")
    logger.info(f"  Number of Folds: {cfg['data']['n_folds']}")
    logger.info(f"  Max Epochs: {cfg['training']['max_epochs']}")
    logger.info(f"  Learning Rate: {cfg['training']['learning_rate']:.1e}")
    logger.info(f"  HPO Mode: {cfg.get('hpo', {}).get('mode', 'skip')}")
    logger.info("")
    
    # 2. Set random seed
    logger.info("Step 2: Setting random seed...")
    seed_everything(cfg['seed'])
    logger.info(f"  Seed: {cfg['seed']}\n")
    
    # 3. Check system requirements
    logger.info("Step 3: Checking system requirements...")
    accelerator = check_system_requirements(cfg)
    cfg['accelerator'] = accelerator
    
    # 4. Validate data splits (Fail-Fast)
    logger.info("Step 4: Validating data splits...")
    validate_splits(cfg)
    
    # 5. Create experiment tracker
    logger.info("Step 5: Initializing experiment tracker...")
    cv_strategy = str(cfg['data']['cv_strategy']).lower()
    cv_label = 'loao' if cv_strategy.startswith('loao') else ('stratified' if 'stratified' in cv_strategy else cv_strategy)
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{cv_label}"
    
    experiments_dir = Path(cfg['directories']['experiments_dir'])
    tracker = ExperimentTracker(
        model_name=model_name,
        run_id=run_id,
        config=cfg,
        experiment_dir=str(experiments_dir),
        checkpoints_subdir=cfg['directories']['checkpoints_subdir'],
        figures_subdir=cfg['directories']['figures_subdir']
    )
    
    logger.info(f"  Run ID: {run_id}")
    logger.info(f"  Experiment directory: {tracker.exp_dir}")
    logger.info(f"  Checkpoints: {tracker.checkpoint_dir}")
    logger.info(f"  Metrics: {tracker.metrics_dir}")
    logger.info("")
    
    # 6. Train model (fold 0 by default for single run)
    logger.info("Step 6: Starting training...")
    fold_idx = 0
    
    try:
        results = train_single_fold(
            model_name=model_name,
            cfg=cfg,
            run_id=run_id,
            fold_idx=fold_idx,
            accelerator=accelerator,
            tracker=tracker
        )
        
        # Close tracker
        tracker.finish()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RUN COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Fold: {fold_idx}")
        logger.info(f"Status: {results['status']}")
        logger.info(f"Final Kappa: {results['val_kappa']:.4f}")
        logger.info(f"Checkpoint: {results['checkpoint_path']}")
        logger.info("=" * 80 + "\n")
        
        # Post-training reporting (controlled by CLI flags)
        if not args.skip_report:
            logger.info("Generating comprehensive report...")
            generate_comprehensive_report(tracker.exp_dir, verbose=True)
        
        if not args.skip_summary:
            logger.info("Displaying results summary...")
            # Construct minimal all_results dict for summary
            all_results = {
                model_name: [{
                    "status": results["status"],
                    "val_kappa": results["val_kappa"]
                }]
            }
            display_results_summary(all_results, tracker.exp_dir, [model_name], verbose=True)
        
    except Exception as e:
        logger.exception("Training failed!")
        tracker.finish()
        raise


def generate_evaluation_outputs_for_run(run_id: str, config: Dict[str, Any], models_to_process: Optional[List[str]] = None):
    """
    Generate evaluation outputs (visualizations, predictions) for a completed training run.
    
    This function uses OPTIMIZED single-inference approach:
    - Old: 3 separate inference passes per fold (~15 min)
    - New: 1 inference pass per fold (~5 min)  
    - Speedup: 3x faster
    
    This function loads the best checkpoints for each model and fold, then generates:
    - Confusion matrix plots
    - Per-class metrics JSON files
    - Predictions CSV files (optional)
    
    Args:
        run_id: Unique run identifier
        config: Full configuration dict
        models_to_process: List of model names to process. If None, processes all models in run.
    """
    from pathlib import Path
    from src.models.base_model import InflammationModel
    from src.data.inflammation_dataset import get_dataloaders
    from src.utils.experiment_tracker import ExperimentTracker
    from src.utils.visualization_optimized import generate_all_visualizations_optimized
    
    experiments_subdir = config["directories"]["experiments_dir"]

    def _resolve_run_dir() -> Optional[Path]:
        """Resolve the actual run directory across notebook/root execution contexts."""
        configured_root = Path(config.get("directories", {}).get("project_root", "."))
        candidate_roots = []

        if configured_root.is_absolute():
            candidate_roots.append(configured_root)
        else:
            candidate_roots.append((Path.cwd() / configured_root).resolve())

        # Repository root inferred from this module location.
        candidate_roots.append(Path(__file__).parent.parent.parent.resolve())
        candidate_roots.append(Path.cwd().resolve())

        checked_candidates = []
        seen = set()
        for root in candidate_roots:
            if str(root) in seen:
                continue
            seen.add(str(root))
            candidate = (root / experiments_subdir / run_id).resolve()
            checked_candidates.append(candidate)
            if candidate.exists():
                return candidate

        logger.error("Run directory not found for run_id '%s'", run_id)
        for candidate in checked_candidates:
            logger.error("  checked: %s", candidate)
        return None

    run_dir = _resolve_run_dir()
    if run_dir is None:
        return

    def _resolve_checkpoint_path(checkpoint_path_str: str, model_name: str) -> Optional[Path]:
        """Resolve checkpoint paths saved from different execution contexts."""
        raw_path = Path(checkpoint_path_str)
        candidates = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
            # If absolute path is stale (e.g., another machine/session), recover from experiments suffix.
            if not raw_path.exists() and "experiments" in raw_path.parts:
                exp_idx = raw_path.parts.index("experiments")
                relative_exp_path = Path(*raw_path.parts[exp_idx:])
                candidates.append((Path.cwd() / relative_exp_path).resolve())
                candidates.append((Path(__file__).parent.parent.parent.resolve() / relative_exp_path).resolve())
        else:
            candidates.append((run_dir / raw_path).resolve())
            candidates.append((Path.cwd() / raw_path).resolve())
            candidates.append((Path(__file__).parent.parent.parent.resolve() / raw_path).resolve())
            # Common case: only filename stored in metadata
            candidates.append((run_dir / model_name / config["directories"]["checkpoints_subdir"] / raw_path.name).resolve())

        seen = set()
        for candidate in candidates:
            if str(candidate) in seen:
                continue
            seen.add(str(candidate))
            if candidate.exists():
                return candidate

        logger.warning("Checkpoint not found after resolution attempts: %s", checkpoint_path_str)
        return None
    
    # Find all trained models if not specified
    if models_to_process is None:
        models_to_process = []
        for model_dir in run_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "checkpoints.json").exists():
                models_to_process.append(model_dir.name)
    
    if not models_to_process:
        logger.warning("No trained models found to process")
        return
    
    logger.info(f"Generating evaluation outputs for run {run_id}")
    logger.info(f"Models to process: {models_to_process}")
    
    for model_name in models_to_process:
        model_dir = run_dir / model_name
        checkpoint_file = model_dir / "checkpoints.json"
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint metadata not found for {model_name}: {checkpoint_file}")
            continue
        
        # Load checkpoint metadata
        with open(checkpoint_file, "r") as f:
            checkpoint_map = json.load(f)
        
        logger.info(f"Processing model: {model_name}")
        
        # Get n_folds and fold list from checkpoint metadata (stored during training)
        n_folds = checkpoint_map.get("n_folds", 0)
        folds_list = checkpoint_map.get("folds", [])
        folds_by_idx = {f["fold_idx"]: f for f in folds_list}
        
        # Process each fold
        for fold_idx in range(n_folds):
            if fold_idx not in folds_by_idx:
                logger.warning(f"No checkpoint for {model_name} fold {fold_idx}")
                continue
            
            fold_info = folds_by_idx[fold_idx]
            checkpoint_path = _resolve_checkpoint_path(fold_info["checkpoint_path"], model_name)

            if checkpoint_path is None:
                continue
            
            try:
                # Create tracker for this model
                tracker = ExperimentTracker(
                    model_name=model_name,
                    config=config,
                    run_id=run_id,
                    experiment_dir=str(run_dir.parent),
                    checkpoints_subdir=config["directories"]["checkpoints_subdir"],
                    figures_subdir=config["directories"]["figures_subdir"]
                )
                
                # Load model
                model = InflammationModel.load_from_checkpoint(str(checkpoint_path), cfg=config)
                model.eval()
                
                # Get validation dataloader for this fold
                _, val_loader = get_dataloaders(config, fold_idx=fold_idx)
                
                # Generate evaluation outputs (OPTIMIZED - single inference)
                logger.info(f"Generating evaluation outputs for {model_name} fold {fold_idx}...")
                generate_all_visualizations_optimized(
                    model=model,
                    val_loader=val_loader,
                    metrics_dir=tracker.metrics_dir,
                    predictions_dir=tracker.predictions_dir,
                    fold_idx=fold_idx,
                    class_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3'],
                    include_predictions_csv=True,  # Include CSV for post-training analysis
                    exclude_ignore_class=True,
                    ignore_class_idx=4,
                )
                
                # Save final metrics JSON (copy from checkpoint metadata)
                metrics_file = tracker.metrics_dir / f"fold_{fold_idx}_metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(fold_info, f, indent=2)
                logger.info(f"Metrics saved: {metrics_file}")
                
            except Exception as e:
                logger.error(f"Failed to generate evaluation outputs for {model_name} fold {fold_idx}: {e}")
                continue
    
    logger.info(f"Evaluation output generation complete for run {run_id}")


if __name__ == "__main__":
    main()
