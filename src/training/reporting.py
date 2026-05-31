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
from src.training.evaluation import find_best_model

logger = get_logger(__name__)


def generate_comprehensive_report(run_dir: Path, verbose: bool = True) -> bool:
    """
    Generate comprehensive statistical report using ReportGenerator.
    
    Wrapper around ReportGenerator.generate_all() with error handling
    and optional logging.
    
    Args:
        run_dir: Path to experiment run directory
        verbose: Whether to log progress (default: True)
        
    Returns:
        True if report generation succeeded, False otherwise
    """
    from src.utils.report_generator import ReportGenerator
    
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("=" * 80 + "\n")
    
    try:
        report_gen = ReportGenerator(run_dir)
        report_gen.generate_all()
        
        if verbose:
            logger.info("Report generation complete.")
            logger.info("  - PDF:      %s/report.pdf", run_dir)
            logger.info("  - Markdown: %s/report.md", run_dir)
            logger.info("  - HTML:     %s/report.html", run_dir)
        
        return True
        
    except Exception as e:
        logger.exception("Report generation failed: %s", e)
        return False


def display_results_summary(
    all_results: dict,
    run_dir: Path,
    models_to_train: List[str],
    verbose: bool = True
) -> Optional[dict]:
    """
    Display comprehensive results summary with best model identification.

    Shows:
    - Run information
    - Report locations
    - TensorBoard instructions
    - Experiment directory structure
    - Best performing model

    Args:
        all_results: Dictionary of {model_name: [fold_results]}
        run_dir: Path to experiment run directory
        models_to_train: List of model names that were trained
        verbose: Whether to display full summary (default: True)

    Returns:
        Dictionary with best model info, or None if no successful models found.
    """
    logger.info("   tensorboard --logdir %s", run_dir)

    # Experiment structure
    logger.info("\nEXPERIMENT STRUCTURE:")
    logger.info("   %s/", run_dir)
    logger.info("   ├── report.pdf              # Comprehensive comparison report")
    logger.info("   ├── report.md               # Markdown version")
    logger.info("   ├── report.html             # Interactive HTML version")
    logger.info("   ├── figures/                # All visualizations")
    logger.info("   ├── summary.csv             # Quick results table")

    for model_name in models_to_train:
        logger.info("   ├── %s/", model_name)
        logger.info("   │   ├── config.yaml")
        logger.info("   │   ├── checkpoints/        # fold_0_best.pth, fold_1_best.pth, ...")
        logger.info("   │   ├── metrics/")
        logger.info("   │   │   ├── fold_0_metrics.json")
        logger.info("   │   │   ├── fold_1_metrics.json")
        logger.info("   │   │   ├── ...")
        logger.info("   │   │   └── model_complexity.json")
        logger.info("   │   └── tensorboard/")

    # Local checkpoints
    logger.info("\nLOCAL CHECKPOINTS:")
    logger.info("   All model checkpoints saved in experiment directories")
    logger.info("   See ../docs/HOW_TO_FIND_MODELS.md for loading instructions")

    # Winner identification
    logger.info("\nWINNER:")
    best_model_info = find_best_model(all_results)

    if best_model_info:
        logger.info("   Model: %s", best_model_info["model_name"])
        logger.info(
            "   QWK:   %.4f ± %.4f",
            best_model_info["mean_kappa"],
            best_model_info["std_kappa"]
        )
    else:
        logger.info("   No complete successful models to compare.")

    logger.info("=" * 80 + "\n")

    return best_model_info
