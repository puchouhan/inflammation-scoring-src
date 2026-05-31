"""
Ensemble Inference for Cross-Validation Fold Models.

Implements robust test set evaluation by averaging predictions (Soft Voting)
from all K LOAO-CV fold models of the same architecture (Intra-Model Ensemble).

See docs/ENSEMBLE_GUIDE.md for full scientific rationale, code examples,
and statistical validation methods.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from src.models.base_model import InflammationModel
from src.models.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class EnsembleInference:
    """
    Ensemble inference manager for CV fold models.
    
    Loads all fold checkpoints and averages predictions (soft voting).
    """
    
    def __init__(
        self,
        checkpoint_paths: List[str],
        cfg: Dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ensemble with fold checkpoints.
        
        Args:
            checkpoint_paths: List of paths to fold checkpoints
            cfg: Model configuration dict (must contain 'model' key)
            device: Torch device (auto-detected if None)
        """
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.cfg = cfg
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        
        logger.info(f"Initializing ensemble with {len(checkpoint_paths)} fold models")
        logger.info(f"Device: {self.device}")
        
        self._load_models()
    
    def _load_models(self):
        """Load all fold models from checkpoints."""
        for i, checkpoint_path in enumerate(self.checkpoint_paths):
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Extract model name from config
            model_name = self.cfg.get('model', {}).get('name') or self.cfg.get('model_name')
            if not model_name:
                raise ValueError("Config must contain 'model.name' or 'model_name' field")
            
            # Create model (default path via ModelFactory)
            model = ModelFactory.create_model(model_name, self.cfg)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Handle Lightning checkpoint format
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'model.' prefix if present (from Lightning)
            cleaned_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    cleaned_state_dict[key[6:]] = value
                else:
                    cleaned_state_dict[key] = value
            
            model = self._load_checkpoint_with_compatibility(
                model=model,
                model_name=model_name,
                state_dict=cleaned_state_dict,
                checkpoint_path=checkpoint_path,
            )
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
            logger.info(f"Loaded fold {i} model from {checkpoint_path.name}")
        
        logger.info(f"All {len(self.models)} models loaded successfully")

    def _load_checkpoint_with_compatibility(
        self,
        model: torch.nn.Module,
        model_name: str,
        state_dict: Dict[str, torch.Tensor],
        checkpoint_path: Path,
    ) -> torch.nn.Module:
        """Load checkpoint into model with targeted architecture compatibility fallbacks.

        The current training runner instantiates `InflammationModel` for all models,
        including the `simclr` label. In that case, checkpoint keys are prefixed with
        `backbone.*` (supervised wrapper), while `ModelFactory.create_model("simclr", ...)
        returns `ssl_model.SimCLR` expecting `encoder.*` + `projection_head.*`.

        This helper keeps strict loading by default and only applies a controlled
        fallback for known key-space mismatches (simclr, dino) where the runner
        trains via InflammationModel but ModelFactory returns the SSL class.
        """
        try:
            model.load_state_dict(state_dict, strict=True)
            return model
        except RuntimeError as load_error:
            if model_name not in ("simclr", "dino"):
                raise

            has_backbone_keys = any(key.startswith("backbone.") for key in state_dict)

            if has_backbone_keys:
                logger.warning(
                    f"{model_name.upper()} checkpoint appears to be saved from "
                    f"InflammationModel (keys start with 'backbone.'): "
                    f"{checkpoint_path.name}. Using InflammationModel for "
                    "ensemble loading compatibility."
                )
                fallback_model = InflammationModel(self.cfg)
                fallback_model.load_state_dict(state_dict, strict=True)
                return fallback_model

            raise RuntimeError(
                f"Failed to load checkpoint {checkpoint_path} for model '{model_name}'. "
                "No compatible fallback was applicable."
            ) from load_error
    
    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict on a batch using ensemble.
        
        Args:
            images: Batch of images [B, C, H, W]
            
        Returns:
            probs: Ensemble probabilities [B, num_classes]
            preds: Ensemble predictions [B]
        """
        images = images.to(self.device)
        
        # Collect predictions from all models
        all_probs = []
        
        for model in self.models:
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        
        # Average probabilities (soft voting)
        ensemble_probs = torch.stack(all_probs).mean(dim=0)  # [B, num_classes]
        ensemble_preds = ensemble_probs.argmax(dim=1)  # [B]
        
        return ensemble_probs, ensemble_preds
    
    @torch.no_grad()
    def predict_dataloader(
        self, 
        dataloader: DataLoader,
        return_targets: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Predict on entire dataloader using ensemble.
        
        Args:
            dataloader: DataLoader with test/validation data
            return_targets: Whether to return ground truth labels
            
        Returns:
            Dictionary with keys:
                - probs: Ensemble probabilities [N, num_classes]
                - preds: Ensemble predictions [N]
                - targets: Ground truth labels [N] (if return_targets=True)
        """
        all_probs = []
        all_preds = []
        all_targets = []
        
        logger.info(f"Running ensemble inference on {len(dataloader)} batches")
        
        for batch in tqdm(dataloader, desc="Ensemble Inference"):
            images = batch[0]
            
            # Get ensemble predictions
            probs, preds = self.predict_batch(images)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            
            if return_targets:
                targets = batch[1]
                all_targets.append(targets.numpy())
        
        # Concatenate all batches
        results = {
            'probs': np.concatenate(all_probs, axis=0),
            'preds': np.concatenate(all_preds, axis=0)
        }
        
        if return_targets:
            results['targets'] = np.concatenate(all_targets, axis=0)
        
        logger.info(f"Inference complete: {len(results['preds'])} samples")
        
        return results
    
    def get_continuous_scores(
        self,
        probs: np.ndarray,
        include_ignore: bool = False
    ) -> np.ndarray:
        """
        Calculate continuous inflammation scores from probabilities.
        
        Uses weighted sum of class probabilities (0-3 scale).
        
        Args:
            probs: Probability matrix [N, num_classes]
            include_ignore: Whether to include "ignore" class in calculation
                           (default: False - renormalizes without ignore class)
            
        Returns:
            Continuous scores [N] in range [0, 3]
        """
        if not include_ignore and probs.shape[1] == 5:
            # Remove ignore class (index 4) and renormalize
            probs_no_ignore = probs[:, :4]
            probs_normalized = probs_no_ignore / probs_no_ignore.sum(axis=1, keepdims=True)
        else:
            probs_normalized = probs
        
        # Weighted sum: Score = sum(class_idx * prob_class)
        class_weights = np.arange(probs_normalized.shape[1])
        continuous_scores = (probs_normalized * class_weights).sum(axis=1)
        
        return continuous_scores


def load_ensemble_from_registry(
    model_name: str,
    cv_strategy: str,
    registry_path: Optional[Path] = None,
    cfg: Optional[Dict] = None
) -> EnsembleInference:
    """
    Convenience function to load ensemble from Best Models Registry.
    
    Args:
        model_name: Architecture name (e.g., "densenet")
        cv_strategy: CV strategy key (loao_balanced or random_stratified)
        registry_path: Optional path to registry JSON
        cfg: Model config (loaded automatically if None)
        
    Returns:
        EnsembleInference instance ready for inference
        
    Example:
        >>> ensemble = load_ensemble_from_registry("densenet", cv_strategy="loao_balanced")
        >>> results = ensemble.predict_dataloader(test_loader)
        >>> print(f"Test Accuracy: {(results['preds'] == results['targets']).mean():.4f}")
    """
    from src.utils.best_models_registry import load_registry
    from configs.utils import load_config
    
    # Load registry
    registry = load_registry(registry_path)
    
    # Get checkpoint paths for ensemble
    checkpoint_paths = registry.get_checkpoint_paths(
        model_name=model_name,
        cv_strategy=cv_strategy,
        strategy='ensemble'
    )
    
    # Load config if not provided
    if cfg is None:
        cfg = load_config(model_name)
    
    # Add model_name to config for ModelFactory
    cfg['model_name'] = model_name
    
    # Create ensemble
    ensemble = EnsembleInference(checkpoint_paths, cfg)
    
    logger.info(f"Loaded ensemble for {model_name} [{cv_strategy}] from registry")
    
    return ensemble


# Example usage (for documentation)
if __name__ == "__main__":
    """
    Example: Ensemble inference on test set
    
    # Option 1: Load from registry
    from src.utils.ensemble_inference import load_ensemble_from_registry
    from src.data.inflammation_dataset import get_dataloaders
    
    # Load test data
    cfg = load_config("densenet")
    _, _, test_loader = get_dataloaders(cfg, fold_idx=0)  # fold_idx irrelevant for test
    
    # Load ensemble from registry
    ensemble = load_ensemble_from_registry("densenet", cv_strategy="loao_balanced", cfg=cfg)
    
    # Run inference
    results = ensemble.predict_dataloader(test_loader)
    
    # Calculate metrics
    from sklearn.metrics import cohen_kappa_score, accuracy_score
    qwk = cohen_kappa_score(results['targets'], results['preds'], weights='quadratic')
    acc = accuracy_score(results['targets'], results['preds'])
    
    print(f"Test QWK: {qwk:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # Get continuous scores
    continuous_scores = ensemble.get_continuous_scores(results['probs'])
    print(f"Mean Continuous Score: {continuous_scores.mean():.4f}")
    
    
    # Option 2: Manual checkpoint loading
    from src.utils.ensemble_inference import EnsembleInference
    
    checkpoint_paths = [
        "experiments/2026-01-08_densenet/fold_0/checkpoints/best_model.pth",
        "experiments/2026-01-08_densenet/fold_1/checkpoints/best_model.pth",
        "experiments/2026-01-08_densenet/fold_2/checkpoints/best_model.pth"
    ]
    
    ensemble = EnsembleInference(checkpoint_paths, cfg)
    results = ensemble.predict_dataloader(test_loader)
    """
    pass
