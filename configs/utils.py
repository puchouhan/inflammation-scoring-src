"""
Configuration Utilities
Utilities for loading and merging base and model-specific configurations.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import copy

logger = logging.getLogger(__name__)


def display_config(config: Dict[str, Any], logger_instance: logging.Logger = None) -> None:
    """
    Display configuration in a structured, human-readable format.
    
    Args:
        config: Configuration dictionary to display
        logger_instance: Logger instance to use for output. If None, uses module logger.
    """
    log = logger_instance or logger
    
    log.info("=" * 80)
    log.info("CONFIGURATION OVERVIEW")
    log.info("=" * 80)

    # General Settings
    log.info("\n[GENERAL]")
    log.info(f"  Seed: {config['seed']}")
    log.info(f"  Accelerator: {config.get('accelerator', 'not set')}")

    # Data Settings
    log.info("\n[DATA]")
    log.info(f"  Normalized Data Dir: {config['data']['norm_dir']}")
    if 'raw_dir' in config['data']:
        log.info(f"  Raw Data Dir: {config['data']['raw_dir']}")
    log.info(f"  Batch Size: {config['data']['batch_size']}")
    log.info(f"  Image Size: {config['data']['img_size']}")
    
    # CV Strategy with automatic n_folds (corresponds to sklearn's n_splits parameter)
    cv_strategy = config['data'].get('cv_strategy', 'loao_balanced')
    log.info(f"  CV Strategy: {cv_strategy}")
    
    # Get n_folds (either already set or derive from cv_folds_config)
    if 'n_folds' in config['data']:
        n_folds = config['data']['n_folds']
    else:
        cv_folds_config = config['data'].get('cv_folds_config', {
            'loao_balanced': 2,
            'random_stratified': 5
        })
        n_folds = cv_folds_config.get(cv_strategy, 'auto')
    log.info(f"  Number of Folds: {n_folds}")

    # Model Settings
    if 'model' in config:
        log.info("\n[MODEL]")
        log.info(f"  Architecture: {config['model'].get('name', 'N/A')}")
        log.info(f"  Backbone: {config['model'].get('backbone', 'N/A')}")
        if 'dropout' in config['model']:
            log.info(f"  Dropout: {config['model']['dropout']}")
        if 'pretrained' in config['model']:
            log.info(f"  Pretrained: {config['model']['pretrained']}")

    # Training Settings
    log.info("\n[TRAINING]")
    log.info(f"  Max Epochs: {config['training']['max_epochs']}")
    lr = float(config['training']['learning_rate'])
    wd = float(config['training']['weight_decay'])
    log.info(f"  Learning Rate: {lr:.1e}")
    log.info(f"  Weight Decay: {wd:.1e}")
    log.info(f"  Optimizer: {config['training'].get('optimizer', {}).get('type', 'adamw')}")
    log.info(f"  Scheduler: {config['training'].get('scheduler', {}).get('type', 'reduce_on_plateau')}")
    log.info(f"  Early Stopping Patience: {config['training']['patience']}")

    # Directories
    log.info("\n[DIRECTORIES]")
    log.info(f"  Experiments: {config['directories']['experiments_dir']}")
    log.info(f"  Checkpoints: {config['directories']['checkpoints_subdir']}")
    log.info(f"  Figures: {config['directories']['figures_subdir']}")

    # HPO Settings (if present)
    if 'hpo' in config:
        log.info("\n[HYPERPARAMETER OPTIMIZATION]")
        log.info(f"  Mode: {config['hpo']['mode']}")
        log.info(f"  Trials: {config['hpo']['n_trials']}")

    log.info("\n" + "=" * 80)


def load_config(model_name: str = None, config_dir: Path = None) -> Dict[str, Any]:
    """
    Load configuration by merging base.yaml with model-specific config.
    
    Args:
        model_name: Name of the model (e.g., "vit", "efficientnetv2")
                   If None, only base config is loaded.
        config_dir: Path to configs directory. If None, auto-detects.
        
    Returns:
        Merged configuration dictionary
        
    Example:
        >>> config = load_config("vit")
        >>> print(config['model']['backbone'])
        'vit_base_patch16_224'
        >>> print(config['training']['learning_rate'])
        0.0001
    """
    # Auto-detect config directory if not provided
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    # Load base configuration
    base_config_path = config_dir / "base.yaml"
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # If no model specified, return base config only
    if model_name is None:
        return config
    
    # Load model-specific configuration
    model_config_path = config_dir / "models" / f"{model_name}.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {model_config_path}. "
            f"Refusing to use base-only fallback for model '{model_name}'."
        )
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Merge configurations (model config overrides base config)
    merged_config = deep_merge(config, model_config)
    
    return merged_config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries. Override values take precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_baseline_config(config_dir: Path = None) -> Dict[str, Any]:
    """
    Load baseline configuration from baseline.yaml.
    
    Args:
        config_dir: Path to configs directory. If None, auto-detects.
        
    Returns:
        Baseline configuration dictionary
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    baseline_path = config_dir / "baseline.yaml"
    if not baseline_path.exists():
        return None
    
    with open(baseline_path, 'r') as f:
        return yaml.safe_load(f)


def list_available_models(config_dir: Path = None) -> list:
    """
    List all available model configurations.
    
    Args:
        config_dir: Path to configs directory. If None, auto-detects.
        
    Returns:
        List of model names
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    models_dir = config_dir / "models"
    if not models_dir.exists():
        return []
    
    return [f.stem for f in models_dir.glob("*.yaml")]


def load_hpo_config(model_name: str, config_dir: Path = None, cv_strategy: str = None) -> Dict[str, Any]:
    """
    Load HPO-optimized configuration if available.
    
    Args:
        model_name: Name of the model (e.g., "vit", "efficientnetv2")
        config_dir: Path to configs directory. If None, auto-detects.
        cv_strategy: CV strategy name (e.g., "random_stratified", "loao_balanced").
                    Required for strategy-specific HPO results lookup.
        
    Returns:
        Dictionary with HPO parameters, or None if not found
        
    Example:
        >>> hpo_params = load_hpo_config("vit", cv_strategy="loao_balanced")
        >>> if hpo_params:
        ...     print(f"Optimized LR: {hpo_params.get('learning_rate')}")
    """
    if config_dir is None:
        config_dir = Path(__file__).parent
    
    # Build list of possible HPO config names
    possible_names: list[str] = []
    if cv_strategy:
        # Strategy-specific file only -- no fallback to avoid cross-strategy contamination
        possible_names.append(f"hpo_best_hpo_{model_name}_{cv_strategy}_v2.yaml")
    else:
        # Legacy names only when no strategy is specified (e.g., CLI usage)
        possible_names.extend([
            f"hpo_best_hpo_{model_name}_v2.yaml",
            f"hpo_best_hpo_{model_name}_master_run.yaml",
            f"hpo_best_{model_name}.yaml",
        ])
    
    for name in possible_names:
        hpo_path = config_dir / name
        if hpo_path.exists():
            with open(hpo_path, 'r') as f:
                return yaml.safe_load(f)
    
    return None


def hpo_results_exist(model_name: str, config_dir: Path = None, cv_strategy: str = None) -> bool:
    """
    Check if HPO results exist for a given model.
    
    Args:
        model_name: Name of the model
        config_dir: Path to configs directory. If None, auto-detects.
        cv_strategy: CV strategy name. If provided, checks for strategy-specific results.
        
    Returns:
        True if HPO results file exists, False otherwise
    """
    return load_hpo_config(model_name, config_dir, cv_strategy) is not None


def merge_hpo_config(config: Dict[str, Any], model_name: str, config_dir: Path = None, cv_strategy: str = None) -> Dict[str, Any]:
    """
    Merge HPO-optimized parameters into config if available.
    
    Args:
        config: Base configuration dictionary
        model_name: Name of the model
        config_dir: Path to configs directory. If None, auto-detects.
        cv_strategy: CV strategy name. If provided, loads strategy-specific HPO results.
        
    Returns:
        Configuration with HPO parameters merged (if available)
        
    Example:
        >>> config = load_config("vit")
        >>> config = merge_hpo_config(config, "vit", cv_strategy="loao_balanced")
        >>> # Now uses optimized hyperparameters if HPO was run
    """
    hpo_params = load_hpo_config(model_name, config_dir, cv_strategy)
    
    if hpo_params is None:
        return config
    
    # Merge HPO parameters into config
    result = copy.deepcopy(config)
    
    # Update training parameters
    if 'learning_rate' in hpo_params:
        result['training']['learning_rate'] = hpo_params['learning_rate']
    if 'weight_decay' in hpo_params:
        result['training']['weight_decay'] = hpo_params['weight_decay']
    if 'batch_size' in hpo_params:
        result['data']['batch_size'] = hpo_params['batch_size']
    
    # Update optimizer parameters
    if 'beta1' in hpo_params or 'beta2' in hpo_params:
        if 'optimizer' not in result['training']:
            result['training']['optimizer'] = {}
        if 'betas' not in result['training']['optimizer']:
            result['training']['optimizer']['betas'] = [0.9, 0.999]
        if 'beta1' in hpo_params:
            result['training']['optimizer']['betas'][0] = hpo_params['beta1']
        if 'beta2' in hpo_params:
            result['training']['optimizer']['betas'][1] = hpo_params['beta2']
    
    # Update scheduler parameters
    if 'scheduler_patience' in hpo_params:
        if 'scheduler' not in result['training']:
            result['training']['scheduler'] = {}
        result['training']['scheduler']['patience'] = hpo_params['scheduler_patience']
    
    # Update model parameters
    if 'drop_rate' in hpo_params:
        if 'model' not in result:
            result['model'] = {}
        result['model']['drop_rate'] = hpo_params['drop_rate']
    
    # Update backbone if specified (multi-model HPO)
    if 'backbone' in hpo_params:
        if 'model' not in result:
            result['model'] = {}
        result['model']['backbone'] = hpo_params['backbone']
        
    # Mark explicitly that HPO was applied
    result['hpo'] = result.get('hpo', {})
    result['hpo']['hpo_applied'] = True
    result['hpo']['hpo_source_file'] = "Loaded from HPO results"
    
    print(f"✓ Loaded HPO-optimized parameters for {model_name}")
    print(f"  LR: {hpo_params.get('learning_rate', 'N/A'):.2e}" if 'learning_rate' in hpo_params else "")
    print(f"  Batch size: {hpo_params.get('batch_size', 'N/A')}")
    
    return result


# Example usage
if __name__ == "__main__":
    print("Available models:")
    for model in list_available_models():
        print(f"  - {model}")
    
    print("\nLoading ViT config:")
    config = load_config("vit")
    print(f"  Backbone: {config['model']['backbone']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Patch size: {config['model'].get('patch_size', 'N/A')}")
