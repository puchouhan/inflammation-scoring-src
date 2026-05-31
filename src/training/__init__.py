"""
Training module for inflammation classification models.

This module provides utilities for training deep learning models on
histopathology images for automated inflammation scoring.
"""

from .runner import (
    main, 
    train_single_fold, 
    train_single_model,
    train_both_cv_strategies,
    generate_evaluation_outputs_for_run
)
from .setup import (
    check_system_requirements,
    create_run_id,
    setup_run_directory,
    list_available_models
)
from .reporting import (
    find_best_model,
    generate_comprehensive_report,
    display_results_summary
)

__all__ = [
    "main",
    "train_single_fold",
    "train_single_model",
    "train_both_cv_strategies",
    "check_system_requirements",
    "create_run_id",
    "setup_run_directory",
    "list_available_models",
    "find_best_model",
    "generate_comprehensive_report",
    "display_results_summary",
    "generate_evaluation_outputs_for_run",
]
