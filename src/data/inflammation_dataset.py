import os
import re
import inspect
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.seeds_logging import get_logger

logger = get_logger("Dataset")

class InflammationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, root_dir: str, transform: Optional[A.Compose] = None):
        """
        Args:
            dataframe: DataFrame containing 'filepath', 'label', 'slide_id', 'animal_id'.
            root_dir: Root directory of the images (e.g., 'dataset_norm').
            transform: Albumentations transforms.
        """
        self.df = dataframe
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row['filepath']
        label = row['label']
        
        img_path = self.root_dir / rel_path
        
        # Read Image
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default to Tensor conversion if no transform provided
            image = ToTensorV2()(image=image)['image']

        return image, torch.tensor(label, dtype=torch.long)

def parse_filename(filename: str) -> Tuple[str, str, int, int]:
    """
    Parses filename to extract metadata.
    Supports two formats:
    - Format 1 (5 parts): Study_Animal_Slide_X_Y.png
      Example: 17_305_01_25_17.png → slide_id = "17_305_01"
    - Format 2 (6 parts): Study_Animal_Slide_Section_X_Y.png
      Example: 15_304_39_1_18_22.png → slide_id = "15_304_39_1"
    
    NOTE: For 6-part format, Section is included in slide_id to ensure that
    patches from different sections are not treated as spatially adjacent in GNN.
    
    Returns:
        animal_id (str): "17_305" or "15_304"
        slide_id (str): "17_305_01" or "15_304_39_1" (includes section if present)
        x (int): X coordinate
        y (int): Y coordinate
    """
    # Remove extension
    name = Path(filename).stem
    parts = name.split('_')
    
    if len(parts) >= 5:
        study = parts[0]
        animal = parts[1]
        slide = parts[2]
        # X and Y are always the last two parts (regardless of format)
        x = int(parts[-2])
        y = int(parts[-1])
        
        animal_id = f"{study}_{animal}"
        
        # For 6-part format, include section in slide_id to maintain spatial integrity
        if len(parts) == 6:
            section = parts[3]
            slide_id = f"{study}_{animal}_{slide}_{section}"
        else:
            slide_id = f"{study}_{animal}_{slide}"
        
        return animal_id, slide_id, x, y
    raise ValueError(
        f"Unexpected filename format: {filename}. "
        "Expected format Study_Animal_Slide_X_Y.png or Study_Animal_Slide_Section_X_Y.png"
    )

def create_dataframe(root_dir: str) -> pd.DataFrame:
    """
    Recursively scans root_dir for images and creates a DataFrame.
    Assumes structure: root_dir/class_name/image.png or root_dir/split/class_name/image.png
    """
    logger.info(f"DEBUG: create_dataframe called with root_dir = {root_dir}")
    root = Path(root_dir)
    logger.info(f"DEBUG: resolved root path = {root.resolve()}")
    data = []
    
    # Map class names to integers
    # 0, 1, 2, 3 are standard. 'ignore' -> 4
    class_map = {'0': 0, '1': 1, '2': 2, '3': 3, 'ignore': 4}
    
    # Walk through all files
    all_paths = list(root.rglob('*.png'))
    logger.info(f"DEBUG: Found {len(all_paths)} .png files via rglob")
    if all_paths:
        logger.info(f"DEBUG: First 5 paths: {[str(p.relative_to(root)) for p in all_paths[:5]]}")
    
    for path in all_paths:
        # Infer label from parent folder name
        # It might be nested like training/0/img.png or just 0/img.png
        # We look for the part of the path that matches a class name
        label = -1
        for part in path.parts:
            if part in class_map:
                label = class_map[part]
                break
        
        if label == -1:
            continue # Skip if no valid class found
            
        animal_id, slide_id, x, y = parse_filename(path.name)
        
        # Store relative path to keep it clean
        rel_path = path.relative_to(root)
        
        data.append({
            'filepath': str(rel_path),
            'label': label,
            'animal_id': animal_id,
            'slide_id': slide_id,
            'x': x,
            'y': y
        })
        
    df = pd.DataFrame(data)
    logger.info(f"Created DataFrame with {len(df)} images.")
    if len(df) == 0:
        raise ValueError(f"No images found in {root_dir}. Check if dataset_norm exists and contains .png files in subfolders like 0/, 1/, etc.")
    if 'label' not in df.columns:
        raise ValueError(f"No 'label' column found in DataFrame. Check folder structure - images should be in subfolders named 0, 1, 2, 3, ignore.")
    logger.info(f"Class distribution:\n{df['label'].value_counts().sort_index()}")
    logger.info(f"Unique Animals: {df['animal_id'].nunique()}")
    return df

def get_transforms(cfg: dict, split: str = 'train'):
    """
    Returns Albumentations transforms.
    """
    img_size = cfg['data']['img_size']

    def _build_coarse_dropout_transform(image_size: int, probability: float = 0.4) -> A.CoarseDropout:
        """Build CoarseDropout with API-compatible arguments across Albumentations versions.

        Newer Albumentations releases use range-based argument names, while older
        releases use max_* arguments. This helper picks valid arguments dynamically
        so augmentation settings are applied as intended.

        Args:
            image_size: Input image size used to derive hole size.
            probability: Probability of applying CoarseDropout.

        Returns:
            Configured CoarseDropout transform.
        """
        parameter_names = set(inspect.signature(A.CoarseDropout.__init__).parameters)

        # Albumentations 2.x style arguments (ranges).
        if {
            'num_holes_range',
            'hole_height_range',
            'hole_width_range'
        }.issubset(parameter_names):
            return A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.05, 0.10),
                hole_width_range=(0.05, 0.10),
                fill=0,
                p=probability,
            )

        # Albumentations 1.x style arguments.
        if {'max_holes', 'max_height', 'max_width'}.issubset(parameter_names):
            return A.CoarseDropout(
                max_holes=8,
                max_height=image_size // 10,
                max_width=image_size // 10,
                p=probability,
            )

        logger.warning(
            "CoarseDropout signature changed; falling back to probability-only initialization."
        )
        return A.CoarseDropout(p=probability)
    
    if split == 'train':
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=2, sigma=50),
                A.GridDistortion(),
            ], p=0.4),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=0.5),
            _build_coarse_dropout_transform(image_size=img_size, probability=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

def get_dataloaders(cfg: dict, fold_idx: int = 0):
    """
    Creates DataLoaders for a specific fold based on cv_strategy config.
    
    Supports two cross-validation strategies:
    1. "loao_balanced": Leave-One-Animal-Out excluding Animal 15_304 (2-fold, balanced)
    2. "random_stratified": Random stratified split without animal grouping (5-fold standard)
    
    NOTE: Uses only training/ folder for K-Fold CV, ignoring pre-split val/ folder.
    
    Args:
        cfg: Configuration dict with cv_strategy, n_folds, exclude_animals
        fold_idx: Fold index for cross-validation
        
    Returns:
        train_loader, val_loader: DataLoaders for training and validation
    """
    from sklearn.model_selection import StratifiedKFold
    
    logger.info(f"DEBUG: get_dataloaders called with cfg['data']['norm_dir'] = {cfg['data']['norm_dir']}")
    # Load only from training/ folder for proper K-Fold CV
    training_dir = Path(cfg['data']['norm_dir']) / 'training'
    logger.info(f"DEBUG: training_dir = {training_dir}, exists = {training_dir.exists()}")
    dataset_root = cfg['data']['norm_dir']
    
    if not training_dir.exists():
        raise FileNotFoundError(
            f"Training subfolder not found at {training_dir}. "
            "Refusing to fall back to full dataset root to prevent split contamination."
        )

    logger.info(f"Loading data from {training_dir} for K-Fold CV (ignoring val/ folder)")
    df = create_dataframe(str(training_dir))
    dataset_root = str(training_dir)
    
    # Get CV strategy from config (default to loao_balanced)
    cv_strategy = cfg['data'].get('cv_strategy', 'loao_balanced')
    logger.info(f"Using CV strategy: {cv_strategy}")
    
    # Automatically set n_folds from cv_folds_config (corresponds to sklearn's n_splits parameter)
    cv_folds_config = cfg['data'].get('cv_folds_config', {
        'loao_balanced': 2,
        'random_stratified': 5
    })
    
    if cv_strategy not in cv_folds_config:
        raise ValueError(
            f"Unknown cv_strategy '{cv_strategy}'. "
            f"Valid options: {list(cv_folds_config.keys())}"
        )
    
    n_folds = cv_folds_config[cv_strategy]
    cfg['data']['n_folds'] = n_folds
    logger.info(f"Auto-configured: n_folds={n_folds} for strategy '{cv_strategy}'")
    
    # Filter out excluded animals if using LOAO strategy
    if cv_strategy.startswith('loao'):
        exclude_animals = cfg['data'].get('exclude_animals', [])
        if exclude_animals:
            original_size = len(df)
            df = df[~df['animal_id'].isin(exclude_animals)].reset_index(drop=True)
            logger.info(f"Excluded animals {exclude_animals}: {original_size} → {len(df)} images ({len(df)/original_size*100:.1f}%)")
            logger.info(f"Remaining animals: {sorted(df['animal_id'].unique())}")
    
    # Prepare data for sklearn splitters
    X = df['filepath'].values  # Features placeholder (not used for splitting, just required by sklearn)
    y = df['label'].values      # Labels for stratification
    animal_ids = df['animal_id'].values  # Groups for StratifiedGroupKFold
    
    # Select splitter based on strategy
    if cv_strategy == 'loao_balanced':
        # Leave-One-Animal-Out: StratifiedGroupKFold with animal_id grouping
        splitter = StratifiedGroupKFold(
            n_splits=cfg['data']['n_folds'], 
            shuffle=True, 
            random_state=cfg['seed']
        )
        splits = list(splitter.split(X, y, groups=animal_ids))
        logger.info(f"LOAO Cross-Validation: {cfg['data']['n_folds']} folds with animal-level separation")
        
    elif cv_strategy == 'random_stratified':
        # Random stratified split: StratifiedKFold without animal grouping
        splitter = StratifiedKFold(
            n_splits=cfg['data']['n_folds'], 
            shuffle=True, 
            random_state=cfg['seed']
        )
        splits = list(splitter.split(X, y))
        logger.info(f"Random Stratified Cross-Validation: {cfg['data']['n_folds']} folds without animal grouping")
        logger.warning(
            "WARNING: cv_strategy='random_stratified' allows same animal in train+val. "
            "Risk of spatial autocorrelation/data leakage. Use only for comparison with Heinemann 2018 paper."
        )
    else:
        raise ValueError(
            f"Unknown cv_strategy '{cv_strategy}'. "
            f"Valid options: {list(cv_folds_config.keys())}"
        )
    
    # Get train/val indices for this fold (with bounds check)
    if fold_idx >= len(splits):
        raise IndexError(
            f"fold_idx={fold_idx} out of range: only {len(splits)} splits available "
            f"(cv_strategy='{cv_strategy}', n_folds={n_folds}). "
            f"Ensure HPO n_folds does not exceed available folds."
        )
    train_idx, val_idx = splits[fold_idx]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    logger.info(f"Fold {fold_idx}: Train size {len(train_df)}, Val size {len(val_df)}")
    
    # Log animal distribution for transparency
    if 'animal_id' in train_df.columns:
        train_animals = train_df['animal_id'].unique()
        val_animals = val_df['animal_id'].unique()
        logger.info(f"  Train animals: {sorted(train_animals)}")
        logger.info(f"  Val animals: {sorted(val_animals)}")
        
        # Check for animal overlap (should be empty for LOAO, may have overlap for random_stratified)
        overlap = set(train_animals) & set(val_animals)
        if overlap:
            if cv_strategy == 'random_stratified':
                logger.info(f"  Animal overlap (expected for random_stratified): {sorted(overlap)}")
            else:
                logger.error(f"  CRITICAL: Animal overlap detected in LOAO strategy: {sorted(overlap)}")
                raise ValueError("Animal overlap detected in LOAO strategy - this should not happen!")
    
    train_ds = InflammationDataset(
        train_df, 
        dataset_root, 
        transform=get_transforms(cfg, 'train')
    )
    
    val_ds = InflammationDataset(
        val_df, 
        dataset_root, 
        transform=get_transforms(cfg, 'val')
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['data']['num_workers'] > 0
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=False, 
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['data']['num_workers'] > 0
    )
    
    return train_loader, val_loader

def get_test_dataloader(cfg: dict):
    """
    Creates DataLoader for the final test set from the val/ folder.
    This is the held-out test set that was never used during K-Fold CV.
    
    Returns:
        test_loader: DataLoader for the test set
    """
    test_dir = Path(cfg['data']['norm_dir']) / 'val'
    
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found at {test_dir}. "
            f"Make sure the dataset has a 'val/' subfolder for final testing."
        )
    
    logger.info(f"Loading test data from {test_dir}")
    df = create_dataframe(str(test_dir))
    
    logger.info(f"Test set size: {len(df)}")
    logger.info(f"Test set class distribution:\n{df['label'].value_counts().sort_index()}")
    
    test_ds = InflammationDataset(
        df, 
        str(test_dir), 
        transform=get_transforms(cfg, 'val')  # Use validation transforms (no augmentation)
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=False,  # Never shuffle test data
        num_workers=cfg['data']['num_workers'],
        pin_memory=True,
        persistent_workers=cfg['data']['num_workers'] > 0
    )
    
    return test_loader