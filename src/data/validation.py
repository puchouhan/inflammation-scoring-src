"""
Data validation utilities for inflammation classification.

Handles dataset preparation, normalization checks, and split integrity validation.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def check_and_prepare_normalized_dataset(
    config: dict,
    project_root: Path,
    show_plots: bool = False
) -> None:
    """
    Check if normalized dataset exists and create if necessary.
    
    Uses Macenko stain normalization for consistent results.
    Only needs to run once - subsequent runs skip if dataset exists.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory
        show_plots: Whether to show visualization plots
        
    Raises:
        FileNotFoundError: If raw dataset doesn't exist
    """
    from src.data.data_exploration import DataExplorer
    
    raw_dir = project_root / config["data"]["raw_dir"]
    norm_dir = project_root / config["data"]["norm_dir"]
    
    logger.info("Project root: %s", project_root)
    logger.info("Raw dataset: %s", raw_dir)
    logger.info("Normalized dataset: %s", norm_dir)
    
    # Verify raw dataset exists
    if not raw_dir.exists():
        logger.error("Raw dataset not found at: %s", raw_dir)
        logger.error("Please ensure dataset/ directory exists in project root")
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")
    
    explorer = DataExplorer()
    explorer.check_and_prepare_normalized_dataset(
        norm_dir=norm_dir,
        raw_dir=raw_dir,
        show_plots=show_plots
    )


def validate_data_splits(config: dict, project_root: Path) -> bool:
    """
    Validate data split integrity before training (CRITICAL).
    
    Performs comprehensive validation across all folds:
    - No image overlap between train/val sets
    - No animal_id overlap (Leave-One-Animal-Out)
    - Proper stratification maintained
    - All data used exactly once per fold
    
    Uses deterministic K-Fold splits (fixed random_state=42) to ensure
    reproducibility and scientific validity.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory
        
    Returns:
        bool: True if all validations pass
        
    Raises:
        RuntimeError: If any validation check fails
    """
    from src.data.split_validator import SplitValidator
    
    # Convert relative paths to absolute
    config_with_abs_paths = config.copy()
    config_with_abs_paths['data'] = config['data'].copy()
    config_with_abs_paths['data']['norm_dir'] = str(project_root / config['data']['norm_dir'])
    config_with_abs_paths['data']['raw_dir'] = str(project_root / config['data']['raw_dir'])
    
    # Ensure n_folds is set (auto-configure from cv_folds_config if needed)
    if 'n_folds' not in config_with_abs_paths['data']:
        cv_strategy = config_with_abs_paths['data'].get('cv_strategy', 'random_stratified')
        cv_folds_config = config_with_abs_paths['data'].get('cv_folds_config', {
            'loao_balanced': 2,
            'random_stratified': 5
        })
        config_with_abs_paths['data']['n_folds'] = cv_folds_config.get(cv_strategy, 2)
    
    try:
        # Validate all folds - aborts if any check fails
        validator = SplitValidator(config_with_abs_paths)
        summary_df = validator.validate_all_folds(verbose=True)  # Enable verbose for debugging
        
        # Check if any fold failed
        if any('FAILED' in str(val) for val in summary_df['Passed'].values):
            logger.error("CRITICAL: Data split validation failed!")
            logger.error("Training aborted to prevent invalid results")
            raise RuntimeError("Data split validation failed - fix splits before training!")
        
        logger.info("Split validation completed successfully - proceeding to training")
        return True
        
    except Exception as e:
        logger.error("Failed to validate data splits: %s", e)
        raise
