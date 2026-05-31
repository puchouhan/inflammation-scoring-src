"""
Training setup utilities for inflammation classification.

Provides system checks, run ID generation, and model registry access.
"""

import os
import torch
import shutil
import logging
from pathlib import Path
from datetime import datetime
import importlib.util
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def check_system_requirements(config: dict, project_root: Path) -> Dict[str, any]:
    """
    Verify all system prerequisites before training.
    
    Checks:
    - GPU/MPS availability
    - Required Python packages
    - Dataset paths
    - Disk space
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory path
        
    Returns:
        dict: System info with keys 'accelerator', 'device_name', 'free_disk_gb'
        
    Raises:
        ImportError: If required packages are missing
        FileNotFoundError: If dataset directories don't exist
    """
    logger.info("\n" + "=" * 80)
    logger.info("SYSTEM REQUIREMENTS CHECK")
    logger.info("=" * 80 + "\n")
    
    # ========================================================================
    # 1. GPU/Accelerator Check - Priority: CUDA > MPS > CPU
    # ========================================================================
    logger.info("1. Checking GPU/Accelerator availability...")
    logger.info("   Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU")
    
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("   SUCCESS [CUDA] NVIDIA GPU detected: %s", gpu_name)
        logger.info("   SUCCESS [CUDA] GPU Memory: %.1f GB", gpu_memory)
        accelerator_type = "cuda"
        device_name = f"CUDA ({gpu_name})"
    elif has_mps:
        logger.info("   SUCCESS [MPS] Apple Silicon GPU detected")
        logger.info("   SUCCESS [MPS] Using Metal Performance Shaders acceleration")
        accelerator_type = "mps"
        device_name = "MPS (Apple Silicon)"
    else:
        logger.warning("   WARNING [WARNING] No GPU detected!")
        logger.warning("   WARNING [WARNING] Training will run on CPU (very slow)")
        logger.warning("   WARNING Expected training time per epoch: 30-60 minutes (vs 2-5 min on GPU)")
        accelerator_type = "cpu"
        device_name = "CPU"
    
    logger.info("   Selected Accelerator: %s\n", device_name)
    
    # ========================================================================
    # 2. Required Packages Check
    # ========================================================================
    logger.info("2. Checking required packages...")
    
    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "lightning": "PyTorch Lightning",
        "albumentations": "Albumentations (data augmentation)",
        "timm": "TIMM (model architectures)",
        "sklearn": "Scikit-learn (metrics)",
        "pandas": "Pandas (data handling)",
        "numpy": "NumPy (numerical operations)",
        "PIL": "Pillow (image processing)",
        "yaml": "PyYAML (config loading)",
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        if importlib.util.find_spec(package) is None:
            logger.error("   [MISSING] %s - %s", package, description)
            missing_packages.append(package)
        else:
            logger.info("   [OK] %s", description)
    
    if missing_packages:
        logger.error("\nMissing packages: %s", ", ".join(missing_packages))
        logger.error("Install with: pip install %s", " ".join(missing_packages))
        raise ImportError(f"Missing required packages: {missing_packages}")
    else:
        logger.info("   SUCCESS All required packages installed\n")
    
    # ========================================================================
    # 3. Dataset Check
    # ========================================================================
    logger.info("3. Checking dataset directories...")
    
    dataset_raw = project_root / config["data"]["raw_dir"]
    dataset_norm = project_root / config["data"]["norm_dir"]
    
    paths_to_check = {
        "Raw Dataset": dataset_raw,
        "Normalized Dataset": dataset_norm,
    }
    
    for name, path in paths_to_check.items():
        if path.exists():
            train_images = len(list((path / "training").rglob("*.png"))) if (path / "training").exists() else 0
            val_images = len(list((path / "val").rglob("*.png"))) if (path / "val").exists() else 0
            logger.info("   [OK] %s: %s", name, path)
            logger.info("        Training images: %d", train_images)
            logger.info("        Validation images: %d", val_images)
        else:
            logger.error("   [MISSING] %s: %s", name, path)
            logger.error("   Run preprocessing: python src/data/preprocess_stains.py")
    
    logger.info("")
    
    # ========================================================================
    # 4. Disk Space Check
    # ========================================================================
    logger.info("4. Checking available disk space...")
    
    runs_dir = project_root / config["directories"]["runs_dir"]
    runs_dir.mkdir(exist_ok=True)
    
    disk_usage = shutil.disk_usage(runs_dir)
    free_gb = disk_usage.free / (1024**3)
    
    if free_gb < 5:
        logger.warning("   [WARNING] Low disk space: %.1f GB free", free_gb)
        logger.warning("   [WARNING] Training may fail if disk fills up")
        logger.warning("   [WARNING] Recommended: At least 10 GB free")
    elif free_gb < 10:
        logger.warning("   [CAUTION] Disk space: %.1f GB free", free_gb)
        logger.warning("   [CAUTION] Recommended: At least 10 GB free for comfort")
    else:
        logger.info("   [OK] Disk space: %.1f GB free", free_gb)
    
    logger.info("")
    
    # ========================================================================
    # 5. Summary
    # ========================================================================
    logger.info("=" * 80)
    logger.info("SYSTEM CHECK COMPLETE")
    logger.info("=" * 80)
    logger.info("SUCCESS Accelerator: %s", device_name)
    logger.info("SUCCESS Packages: All installed")
    logger.info("SUCCESS Disk Space: %.1f GB", free_gb)
    logger.info("=" * 80 + "\n")
    
    return {
        "accelerator": accelerator_type,
        "device_name": device_name,
        "free_disk_gb": free_gb,
    }


def create_run_id(suffix: str = None) -> str:
    """
    Create unique timestamp-based run ID with an optional suffix (e.g. model name).
    Reads from environment variable 'INFLAMMATION_RUN_ID' if set,
    otherwise generates a new timestamp.
    
    Args:
        suffix: Optional string to append to the run_id (e.g. 'vit')
        
    Returns:
        str: Run ID in format YYYY-MM-DD_HH-MM-SS or YYYY-MM-DD_HH-MM-SS_suffix
    """
    env_id = os.environ.get("INFLAMMATION_RUN_ID")
    if env_id:
        return env_id
        
    base_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if suffix:
        return f"{base_id}_{suffix}"
    return base_id


def setup_run_directory(config: dict, project_root: Path, run_id: str) -> Path:
    """
    Create and return run directory path.
    
    Args:
        config: Configuration dictionary
        project_root: Project root directory
        run_id: Unique run identifier
        
    Returns:
        Path: Run directory path
    """
    experiments_dir = project_root / config["directories"]["experiments_dir"]
    run_dir = experiments_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Run ID: %s", run_id)
    logger.info("Run Directory: %s", run_dir)
    logger.info("All models will be saved under: %s/{model_name}/\n", run_dir)
    
    return run_dir


def list_available_models(config: dict) -> Dict[str, str]:
    """
    List all available models from ModelFactory.
    
    Args:
        config: Configuration dict (to check which models are selected for training)
        
    Returns:
        dict: {model_name: description}
    """
    from src.models.model_factory import ModelFactory
    
    logger.info("=" * 80)
    logger.info("AVAILABLE MODELS")
    logger.info("=" * 80)
    
    available_models = ModelFactory.list_models()
    models_to_train = config.get("models_to_train", [])
    
    for name, description in available_models.items():
        model_type = ModelFactory.get_model_type(name)
        status = "*" if name in models_to_train else " "
        logger.info("[%s] %-20s - %-40s (%s)", status, name, description, model_type)
    
    logger.info("=" * 80 + "\n")
    
    return available_models
