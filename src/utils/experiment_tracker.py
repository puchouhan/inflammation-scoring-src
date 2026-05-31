"""
Experiment Tracking System
Combines TensorBoard, JSON logging, and optional Weights & Biases.
"""
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Manages experiment tracking with TensorBoard, JSON logs, and optional W&B.
    
    Directory structure created:
    experiments/
    └── {timestamp}_{model_name}/
        ├── config.yaml              # Exact config used
        ├── run_info.json            # Metadata (git hash, timestamps, etc.)
        ├── checkpoints/
        │   ├── best_model.pth
        │   └── last_model.pth
        ├── metrics/
        │   ├── final_metrics.json
        │   ├── confusion_matrix.png
        │   └── per_fold_results.csv
        ├── tensorboard/             # TensorBoard logs
        └── predictions/
            └── val_predictions.csv
    """
    
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        experiment_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        checkpoints_subdir: Optional[str] = None,
        figures_subdir: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            model_name: Name of the model being trained
            config: Full configuration dict
            experiment_dir: Root directory for experiments
            run_id: Optional run ID (timestamp). If None, creates a new one.
                   Multiple models in the same run should share the same run_id.
            checkpoints_subdir: Subdirectory name for checkpoints
            figures_subdir: Subdirectory name for figures/metrics
        """
        self.model_name = model_name
        self.config = config
        
        # Get experiment_dir from config if not provided
        if experiment_dir is None:
            experiment_dir = config.get('directories', {}).get('experiments_dir', 'experiments')
        
        # Get subdirectory names from config if not provided
        if checkpoints_subdir is None:
            checkpoints_subdir = config.get('directories', {}).get('checkpoints_subdir', 'checkpoints')
        if figures_subdir is None:
            figures_subdir = config.get('directories', {}).get('figures_subdir', 'figures')
        
        # Create experiment directory structure: experiments/{run_id}/{model_name}/
        if run_id is None:
            run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.run_id = run_id
        self.run_dir = Path(experiment_dir) / run_id
        self.exp_name = f"{run_id}_{model_name}"
        self.exp_dir = self.run_dir / model_name
        
        # Create subdirectories - use parameters for configurability
        self.checkpoint_dir = self.exp_dir / checkpoints_subdir
        self.metrics_dir = self.exp_dir / figures_subdir
        self.tensorboard_dir = self.exp_dir / "tensorboard"
        self.predictions_dir = self.exp_dir / "predictions"
        
        for dir_path in [self.checkpoint_dir, self.metrics_dir, 
                         self.tensorboard_dir, self.predictions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard
        self.tb_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # Save config and metadata
        self._save_config()
        self._save_run_info()
        
        logger.info(f"Experiment initialized: {self.exp_dir}")
        logger.info(f"Run ID: {self.run_id}")
    
    def _save_config(self):
        """Save the exact config used for this run."""
        config_path = self.exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def _save_run_info(self):
        """Save metadata about this run (git hash, timestamps, system info)."""
        run_info = {
            "experiment_name": self.exp_name,
            "model_name": self.model_name,
            "start_time": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
            "python_version": f"{torch.__version__}",
            "device": self.config.get("device", "unknown"),
            # Cross-validation strategy metadata
            "cv_strategy": self.config.get("data", {}).get("cv_strategy", "loao_balanced"),
            "n_folds": self.config.get("data", {}).get("n_folds", 2),
            "exclude_animals": self.config.get("data", {}).get("exclude_animals", []),
        }
        
        info_path = self.exp_dir / "run_info.json"
        with open(info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to TensorBoard and W&B.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Current step/epoch
            prefix: Optional prefix for metric names (e.g., "train/", "val/")
        """
        for name, value in metrics.items():
            # Only log numeric values to TensorBoard (exclude strings and other non-numeric types)
            if isinstance(value, (int, float, np.number)) or (hasattr(value, 'item') and callable(getattr(value, 'item', None))):
                # Handle PyTorch tensors and numpy scalars
                if hasattr(value, 'item') and callable(getattr(value, 'item', None)):
                    value = value.item()
                # Ensure it's numeric after conversion
                try:
                    float_value = float(value)
                    full_name = f"{prefix}{name}" if prefix else name
                    # TensorBoard
                    self.tb_writer.add_scalar(full_name, float_value, step)
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Current metrics dict
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        
        # Always save as last
        last_path = self.checkpoint_dir / "last_model.pth"
        torch.save(checkpoint, last_path)
        
        # Save as best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            shutil.copy(last_path, best_path)
            print(f"Saved best model (epoch {epoch})")
    
    def save_final_metrics(self, metrics: Dict[str, Any], fold_idx: Optional[int] = None):
        """
        Save final metrics after training completion.
        
        Args:
            metrics: Dictionary containing final evaluation results
            fold_idx: Optional fold index for cross-validation
        """
        if fold_idx is not None:
            # Save per-fold metrics
            metrics_path = self.metrics_dir / f"fold_{fold_idx}_metrics.json"
        else:
            metrics_path = self.metrics_dir / "final_metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")
    
    def save_model_complexity(self, model: torch.nn.Module, input_size=(1, 3, 256, 256)):
        """
        Save model complexity metrics (parameters, size, inference time).
        
        Args:
            model: PyTorch model
            input_size: Input tensor size for inference timing
        """
        import time
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model size on disk
        temp_path = self.checkpoint_dir / "temp_model.pth"
        torch.save(model.state_dict(), temp_path)
        model_size_mb = temp_path.stat().st_size / (1024 * 1024)
        temp_path.unlink()
        
        # Inference time (average over 100 runs)
        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        complexity = {
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "model_size_mb": float(model_size_mb),
            "inference_time_ms_mean": float(np.mean(times)),
            "inference_time_ms_std": float(np.std(times)),
            "input_size": list(input_size),
        }
        
        complexity_path = self.metrics_dir / "model_complexity.json"
        with open(complexity_path, 'w') as f:
            json.dump(complexity, f, indent=2)
        
        print(f"Model complexity saved: {total_params:,} params, {model_size_mb:.2f} MB")
    
    def save_confusion_matrix(self, model, dataloader, fold_idx: Optional[int] = None):
        """
        Generate and save confusion matrix visualization.
        
        Args:
            model: Trained model
            dataloader: DataLoader for evaluation
            fold_idx: Optional fold index for cross-validation
        """
        from src.utils.visualization_helpers import save_confusion_matrix
        
        if fold_idx is not None:
            save_path = self.metrics_dir / f"fold_{fold_idx}_confusion_matrix.png"
        else:
            save_path = self.metrics_dir / "confusion_matrix.png"
        
        save_confusion_matrix(model, dataloader, save_path, fold_idx=fold_idx)
    
    def save_per_class_metrics(self, model, dataloader, fold_idx: Optional[int] = None):
        """
        Extract and save per-class metrics from trained model.
        
        Args:
            model: Trained model
            dataloader: DataLoader for evaluation
            fold_idx: Optional fold index for cross-validation
        """
        from src.utils.visualization_helpers import save_per_class_metrics
        
        if fold_idx is not None:
            save_path = self.metrics_dir / f"fold_{fold_idx}_per_class.json"
        else:
            save_path = self.metrics_dir / "per_class_metrics.json"
        
        save_per_class_metrics(model, dataloader, save_path, fold_idx=fold_idx)
    
    def save_efficiency_metrics(self, efficiency_metrics: Dict[str, Any], fold_idx: Optional[int] = None):
        """
        Save model efficiency metrics.
        
        Args:
            efficiency_metrics: Dictionary from compute_efficiency_metrics
            fold_idx: Optional fold index
        """
        if fold_idx is not None:
            save_path = self.metrics_dir / f"fold_{fold_idx}_efficiency.json"
        else:
            save_path = self.metrics_dir / "efficiency_metrics.json"
        
        with open(save_path, 'w') as f:
            json.dump(efficiency_metrics, f, indent=2)
        
        print(f"Efficiency metrics saved to {save_path}")
    
    def save_calibration_metrics(self, calibration_metrics: Dict[str, Any], fold_idx: Optional[int] = None):
        """
        Save calibration and confidence metrics.
        
        Args:
            calibration_metrics: Dictionary from evaluate_calibration
            fold_idx: Optional fold index
        """
        if fold_idx is not None:
            save_path = self.metrics_dir / f"fold_{fold_idx}_calibration.json"
        else:
            save_path = self.metrics_dir / "calibration_metrics.json"
        
        with open(save_path, 'w') as f:
            json.dump(calibration_metrics, f, indent=2)
        
        print(f"Calibration metrics saved to {save_path}")
    
    def save_figure(self, fig, filename: str):
        """
        Save a matplotlib figure to metrics directory.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename (e.g., "confusion_matrix.png")
        """
        save_path = self.metrics_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    def finish(self):
        """Clean up and finalize the experiment."""
        # Update run_info with end time
        info_path = self.exp_dir / "run_info.json"
        with open(info_path, 'r') as f:
            run_info = json.load(f)
        
        run_info["end_time"] = datetime.now().isoformat()
        
        with open(info_path, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        # Close TensorBoard
        self.tb_writer.close()
        
        print(f"Experiment finished: {self.exp_dir}")
    
    def get_checkpoint_path(self, best: bool = True) -> Path:
        """Get path to best or last checkpoint."""
        filename = "best_model.pth" if best else "last_model.pth"
        return self.checkpoint_dir / filename
