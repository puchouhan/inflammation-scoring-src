"""Data split validation utilities for inflammation dataset.

This module provides critical validation checks to ensure data split integrity
and prevent data leakage in cross-validation experiments. All checks must pass
before training to guarantee scientifically valid results.

Critical for:
- Animal-level Leave-One-Animal-Out (LOAO) cross-validation
- Preventing spatial data leakage (patches from same animal in train/val)
- Ensuring proper stratification across inflammation grades
- Validating completeness of data splits
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from configs.utils import load_config
from src.data.inflammation_dataset import create_dataframe, get_dataloaders

logger = logging.getLogger(__name__)


class SplitValidator:
    """Validates train/val splits for data leakage and correctness.
    
    This class performs comprehensive validation of data splits to ensure:
    1. No image overlap between train and val sets
    2. No animal_id overlap (critical for LOAO cross-validation)
    3. Proper stratification maintained across classes
    4. All data used exactly once per fold
    
    Example:
        validator = SplitValidator(cfg)
        summary_df = validator.validate_all_folds()
        
        # Check if all validations passed
        if 'FAILED' in summary_df['Passed'].values:
            raise RuntimeError("Data split validation failed!")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize split validator.
        
        Args:
            config: Configuration dict. If None, loads default base config.
        """
        self.config = config if config is not None else load_config()
        
    def validate_fold(self, fold_idx: int = 0, verbose: bool = True) -> Dict:
        """Validate a single fold for data leakage and correctness.
        
        Performs 4 critical checks:
        1. No image overlap between train/val
        2. No animal_id overlap (prevents spatial leakage)
        3. Class distribution maintained (stratification quality)
        4. All data used exactly once
        
        Args:
            fold_idx: Index of fold to validate (0 to n_folds-1)
            verbose: If True, print detailed validation report
            
        Returns:
            dict: Validation results containing:
                - fold_idx: Fold index validated
                - all_checks_passed: Boolean indicating if all checks passed
                - checks: Dict with individual check results
                - details: Detailed statistics and metrics
                
        Raises:
            ValueError: If fold_idx is out of range
        """
        if fold_idx < 0 or fold_idx >= self.config['data']['n_folds']:
            raise ValueError(
                f"fold_idx {fold_idx} out of range. "
                f"Valid range: 0 to {self.config['data']['n_folds']-1}"
            )
        
        logger.info(f"Validating train/val split for fold {fold_idx}...")
        
        # Load data for this fold
        train_loader, val_loader = get_dataloaders(self.config, fold_idx=fold_idx)
        
        # Get datasets
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        # Extract information from datasets
        train_paths = set(train_dataset.df['filepath'].values)
        val_paths = set(val_dataset.df['filepath'].values)
        
        train_animals = set(train_dataset.df['animal_id'].values)
        val_animals = set(val_dataset.df['animal_id'].values)
        
        train_labels = train_dataset.df['label'].values
        val_labels = val_dataset.df['label'].values
        
        # CHECK 1: No image path overlap
        path_overlap = train_paths.intersection(val_paths)
        path_check = len(path_overlap) == 0
        
        # CHECK 2: No animal_id overlap (critical for LOAO, expected for heinemann2018)
        animal_overlap = train_animals.intersection(val_animals)
        cv_strategy = self.config['data'].get('cv_strategy', 'loao_balanced')
        
        if cv_strategy == 'random_stratified':
            # Random stratified allows animal overlap (random splits without grouping)
            animal_check = True  # Skip this check for random_stratified
            logger.info(f"  Skipping animal overlap check (cv_strategy={cv_strategy})")
        else:
            # LOAO strategies MUST have zero animal overlap
            animal_check = len(animal_overlap) == 0
        
        # CHECK 3: Class distribution (stratification quality)
        train_dist = pd.Series(train_labels).value_counts(normalize=True).sort_index()
        val_dist = pd.Series(val_labels).value_counts(normalize=True).sort_index()
        
        # Stratification threshold depends on CV strategy
        # LOAO: Each animal has its own distribution → relax threshold (10%)
        # Random: StratifiedGroupKFold controls balance → strict threshold (5%)
        max_dist_diff = abs(train_dist - val_dist).max()
        if cv_strategy in ['loao_balanced', 'loao_all']:
            stratification_threshold = 0.10  # 10% for LOAO (animal separation priority)
        else:
            stratification_threshold = 0.05  # 5% for random_stratified
        stratification_check = max_dist_diff < stratification_threshold
        
        # CHECK 4: All data used (train + val = total)
        training_dir = Path(self.config['data']['norm_dir']) / 'training'
        if training_dir.exists():
            full_df = create_dataframe(str(training_dir))
        else:
            full_df = create_dataframe(self.config['data']['norm_dir'])
        
        # IMPORTANT: Apply same exclusions as in get_dataloaders()
        # For loao_balanced, exclude Animal 15_304 (only 84 images)
        cv_strategy = self.config['data'].get('cv_strategy', 'loao_balanced')
        if cv_strategy == 'loao_balanced':
            exclude_animals = self.config['data'].get('exclude_animals', ['15_304'])
            original_size = len(full_df)
            full_df = full_df[~full_df['animal_id'].isin(exclude_animals)]
            if len(full_df) < original_size:
                logger.info(f"  Excluded animals {exclude_animals} for completeness check: {original_size} → {len(full_df)} images")
        
        total_samples = len(full_df)
        split_samples = len(train_dataset) + len(val_dataset)
        completeness_check = (total_samples == split_samples)
        
        # Compile results
        results = {
            'fold_idx': fold_idx,
            'all_checks_passed': all([path_check, animal_check, stratification_check, completeness_check]),
            'checks': {
                'no_image_overlap': path_check,
                'no_animal_overlap': animal_check,
                'stratification_ok': stratification_check,
                'all_data_used': completeness_check
            },
            'details': {
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'total_samples': total_samples,
                'train_animals': len(train_animals),
                'val_animals': len(val_animals),
                'image_overlap_count': len(path_overlap),
                'animal_overlap_count': len(animal_overlap),
                'max_class_dist_diff': max_dist_diff,
                'stratification_threshold': stratification_threshold,
                'train_class_distribution': train_dist.to_dict(),
                'val_class_distribution': val_dist.to_dict()
            }
        }
        
        if verbose:
            self._print_fold_report(results, path_overlap, animal_overlap, train_dist, val_dist)
        
        return results
    
    def validate_all_folds(self, verbose: bool = False) -> pd.DataFrame:
        """Validate all folds and return summary statistics.
        
        This method validates every fold in the cross-validation setup and
        compiles a summary report. Use this before training to ensure all
        folds are correctly configured.
        
        Args:
            verbose: If True, print detailed report for each fold
            
        Returns:
            pd.DataFrame: Summary table with validation results for all folds.
                Columns: Fold, Passed, Train Samples, Val Samples, Train Animals,
                         Val Animals, Animal Overlap, Max Class Diff, CV Strategy
                         
        Example:
            validator = SplitValidator(cfg)
            summary = validator.validate_all_folds()
            
            # Check if any fold failed
            if 'FAILED' in summary['Passed'].values:
                print("CRITICAL: Some folds failed validation!")
                print(summary)
        """
        # Get n_folds with fallback to cv_folds_config (corresponds to sklearn's n_splits)
        if 'n_folds' in self.config['data']:
            n_folds = self.config['data']['n_folds']
        else:
            cv_strategy = self.config['data'].get('cv_strategy', 'random_stratified')
            cv_folds_config = self.config['data'].get('cv_folds_config', {
                'loao_balanced': 2,
                'random_stratified': 5
            })
            n_folds = cv_folds_config.get(cv_strategy, 2)
            self.config['data']['n_folds'] = n_folds  # Set for future use
        
        cv_strategy = self.config['data'].get('cv_strategy', 'loao_balanced')
        logger.info(f"Validating all {n_folds} folds using CV strategy: {cv_strategy}...")
        
        all_results = []
        for fold_idx in range(n_folds):
            results = self.validate_fold(fold_idx=fold_idx, verbose=verbose)
            all_results.append({
                'Fold': fold_idx,
                'Passed': 'PASSED' if results['all_checks_passed'] else 'FAILED',
                'Train Samples': results['details']['train_samples'],
                'Val Samples': results['details']['val_samples'],
                'Train Animals': results['details']['train_animals'],
                'Val Animals': results['details']['val_animals'],
                'Animal Overlap': results['details']['animal_overlap_count'],
                'Max Class Diff': f"{results['details']['max_class_dist_diff']:.3f}",
                'CV Strategy': cv_strategy
            })
        
        summary_df = pd.DataFrame(all_results)
        
        # Print summary (use print for immediate output)
        print("\n" + "=" * 80)
        print(f"VALIDATION SUMMARY - ALL FOLDS (Strategy: {cv_strategy})")
        print("=" * 80)
        print(f"\n{summary_df.to_string(index=False)}\n")
        logger.info("\n" + "=" * 80)
        logger.info(f"VALIDATION SUMMARY - ALL FOLDS (Strategy: {cv_strategy})")
        logger.info("=" * 80)
        logger.info(f"\n{summary_df.to_string(index=False)}\n")
        
        # Check if all passed
        all_passed = all(r['all_checks_passed'] for r in 
                        [self.validate_fold(i, verbose=False) for i in range(n_folds)])
        
        if all_passed:
            logger.info("SUCCESS: ALL FOLDS VALIDATED SUCCESSFULLY - No data leakage detected!")
        else:
            logger.error("FAILED: SOME FOLDS FAILED VALIDATION - Please review results above!")
        logger.info("=" * 80 + "\n")
        
        return summary_df
    
    def _print_fold_report(
        self, 
        results: Dict, 
        path_overlap: set, 
        animal_overlap: set,
        train_dist: pd.Series,
        val_dist: pd.Series
    ) -> None:
        """Print detailed validation report for a single fold.
        
        Args:
            results: Validation results dict
            path_overlap: Set of overlapping image paths
            animal_overlap: Set of overlapping animal IDs
            train_dist: Training set class distribution
            val_dist: Validation set class distribution
        """
        fold_idx = results['fold_idx']
        checks = results['checks']
        details = results['details']
        
        logger.info("=" * 80)
        logger.info(f"TRAIN/VAL SPLIT VALIDATION - FOLD {fold_idx}")
        logger.info("=" * 80)
        
        # Display check results
        logger.info("\nVALIDATION CHECKS:")
        logger.info(f"  [{'PASS' if checks['no_image_overlap'] else 'FAIL'}] No image overlap: {checks['no_image_overlap']}")
        if not checks['no_image_overlap']:
            logger.error(f"    Found {len(path_overlap)} overlapping images!")
            logger.error(f"    Examples: {list(path_overlap)[:3]}")
        
        logger.info(f"  [{'PASS' if checks['no_animal_overlap'] else 'FAIL'}] No animal_id overlap: {checks['no_animal_overlap']}")
        cv_strategy = self.config['data'].get('cv_strategy', 'loao_balanced')
        if cv_strategy == 'random_stratified':
            logger.info(f"    Strategy: {cv_strategy} (animal overlap expected)")
            if len(animal_overlap) > 0:
                logger.info(f"    Animals in both sets: {animal_overlap} [OK for {cv_strategy}]")
        elif not checks['no_animal_overlap']:
            logger.error(f"    Found {len(animal_overlap)} overlapping animals!")
            logger.error(f"    Animals in both sets: {animal_overlap}")
        
        logger.info(f"  [{'PASS' if checks['stratification_ok'] else 'FAIL'}] Stratification quality: {checks['stratification_ok']}")
        logger.info(f"    Max class distribution difference: {details['max_class_dist_diff']:.3f} (threshold: {details['stratification_threshold']:.3f})")
        
        logger.info(f"  [{'PASS' if checks['all_data_used'] else 'FAIL'}] All data used: {checks['all_data_used']}")
        logger.info(f"    Train + Val = {details['train_samples'] + details['val_samples']}, Total = {details['total_samples']}")
        
        # Display split statistics
        logger.info("\nSPLIT STATISTICS:")
        logger.info(f"  Training:   {details['train_samples']:4d} samples, {details['train_animals']:3d} animals")
        logger.info(f"  Validation: {details['val_samples']:4d} samples, {details['val_animals']:3d} animals")
        logger.info(f"  Total:      {details['total_samples']:4d} samples")
        
        # Display class distributions
        logger.info("\nCLASS DISTRIBUTION (stratification):")
        logger.info("  Class  Train%  Val%    Diff")
        for cls in sorted(train_dist.index):
            train_pct = train_dist.get(cls, 0) * 100
            val_pct = val_dist.get(cls, 0) * 100
            diff = abs(train_pct - val_pct)
            logger.info(f"    {cls}    {train_pct:5.1f}%  {val_pct:5.1f}%  {diff:5.1f}%")
        
        logger.info("\n" + "=" * 80)
        if results['all_checks_passed']:
            logger.info("SUCCESS: ALL VALIDATION CHECKS PASSED - Split is correct!")
        else:
            logger.error("FAILED: VALIDATION FAILED - Data leakage detected!")
        logger.info("=" * 80 + "\n")


def validate_splits_before_training(config: Optional[Dict] = None) -> None:
    """Convenience function to validate splits and abort if validation fails.
    
    Use this at the start of training scripts to ensure data integrity.
    Raises RuntimeError if any validation check fails.
    
    Args:
        config: Configuration dict. If None, loads default base config.
        
    Raises:
        RuntimeError: If any fold fails validation
        
    Example:
        # At start of train_runner.py
        from src.data.split_validator import validate_splits_before_training
        
        cfg = load_config()
        validate_splits_before_training(cfg)  # Aborts if invalid
        # ... continue with training ...
    """
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING DATA SPLIT INTEGRITY")
    logger.info("=" * 80)
    logger.info("Checking for data leakage and split correctness...")
    
    validator = SplitValidator(config)
    summary_df = validator.validate_all_folds(verbose=False)
    
    # Check if all validations passed
    if 'FAILED' in summary_df['Passed'].values:
        logger.error("\n" + "=" * 80)
        logger.error("CRITICAL ERROR: DATA SPLIT VALIDATION FAILED")
        logger.error("=" * 80)
        logger.error("Data leakage detected! Training cannot proceed.")
        logger.error("Please review the validation results above.")
        logger.error("=" * 80 + "\n")
        raise RuntimeError(
            "Data split validation failed. Detected data leakage or incorrect splits. "
            "Training aborted to prevent invalid results."
        )
    
    logger.info("\n" + "=" * 80)
    logger.info("DATA SPLIT VALIDATION PASSED")
    logger.info("=" * 80)
    logger.info("All folds validated successfully - no data leakage detected.")
    logger.info("Proceeding with training...")
    logger.info("=" * 80 + "\n")
