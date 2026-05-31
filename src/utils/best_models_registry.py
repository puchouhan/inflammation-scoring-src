"""
Best Models Registry System

Tracks the best-performing model for each architecture across all training runs.
Enables easy comparison and inference without retraining.

Scientific Rationale:
- Maintains reproducibility (all runs preserved)
- Tracks performance improvements over time
- Enables fair model comparison (best version of each architecture)
- Supports ensemble inference with all CV folds
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np
import subprocess
import logging


class BestModelsRegistry:
    """
    Manages registry of best-performing models per architecture.
    
        Registry Structure:
        {
            "densenet": {
                "loao_balanced": { ... best entry for LOAO ... },
                "random_stratified": { ... best entry for stratified ... }
            },
            "vit": {
                "loao_balanced": { ... },
                "random_stratified": { ... }
            }
        }
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize registry manager.
        
        Args:
            registry_path: Path to registry JSON file. 
                          Default: src/experiments/best_models_registry.json
        """
        # Project root is always 3 levels up from this file (src/utils/ -> src/ -> root)
        self.project_root = Path(__file__).parent.parent.parent

        if registry_path is None:
            # Load registry_dir from config (base.yaml)
            try:
                from configs.utils import load_config
                config = load_config()
                registry_dir = self.project_root / config['directories'].get(
                    'registry_dir', 'src/experiments'
                )
            except Exception:
                # Fallback to hardcoded path if config loading fails
                registry_dir = self.project_root / "src" / "experiments"
            
            registry_dir.mkdir(parents=True, exist_ok=True)
            registry_path = registry_dir / "best_models_registry_new.json"
        
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()

    @staticmethod
    def _normalize_cv_strategy(cv_strategy: str) -> str:
        """Normalize CV strategy names and validate supported values."""
        normalized = str(cv_strategy).strip().lower()
        alias_map = {
            "random_stratifies": "random_stratified",
            "stratified": "random_stratified",
            "loao": "loao_balanced",
        }
        normalized = alias_map.get(normalized, normalized)

        supported = {"loao_balanced", "random_stratified"}
        if normalized not in supported:
            raise ValueError(
                f"Unsupported cv_strategy '{cv_strategy}'. "
                f"Supported values: {sorted(supported)}"
            )
        return normalized
    
    def _load_registry(self) -> Dict:
        """Load registry from disk or create empty if not exists."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    content = f.read()
                    if content.strip():  # Check if file has content
                        return json.loads(content)
            except (json.JSONDecodeError, IOError):
                # File exists but is empty or invalid - return empty dict
                pass
        return {}
    
    def _save_registry(self):
        """Save registry to disk with pretty formatting."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
        print(f"Registry saved to {self.registry_path}")
    
    def update_model(
        self,
        model_name: str,
        run_id: str,
        cv_strategy: str,
        fold_metrics: List[Dict],  # [{fold:0, val_qwk:0.83, val_acc:0.86}, ...]
        checkpoint_paths: List[str],  # Paths to all fold checkpoints
        test_qwk: Optional[float] = None,
        with_hpo: Optional[bool] = None,
        hpo_mode: Optional[str] = None,
        notes: str = "",
        auto_commit: bool = True
    ) -> Dict[str, Any]:
        """
        Update registry with new model results if it's better than current best.
        
        Args:
            model_name: Architecture name (e.g., "densenet", "vit")
            run_id: Training run identifier
            fold_metrics: Metrics for each fold
            checkpoint_paths: Paths to fold checkpoints
            test_qwk: Optional test set QWK
            with_hpo: Whether this model was trained with HPO enabled
            hpo_mode: HPO mode used (e.g., skip/use_existing/overwrite)
            notes: Optional notes about this run
            auto_commit: Whether to auto-commit to git
            
        Returns:
            dict: Update status with keys:
                - updated: bool (True if registry was updated)
                - reason: str (explanation)
                - old_mean_qwk: float or None
                - new_mean_qwk: float
                - improvement: float or None
        """
        cv_key = self._normalize_cv_strategy(cv_strategy)

        # Calculate mean metrics
        qwk_values = [m['val_qwk'] for m in fold_metrics]
        acc_values = [m['val_acc'] for m in fold_metrics]
        
        new_mean_qwk = np.mean(qwk_values)
        new_std_qwk = np.std(qwk_values, ddof=1) if len(qwk_values) > 1 else 0.0
        new_mean_acc = np.mean(acc_values)
        
        # Find best single fold
        best_fold_idx = np.argmax(qwk_values)
        
        # Check if update needed for this model + strategy pair
        model_entry = self.registry.get(model_name, {})
        strategy_entry = model_entry.get(cv_key)
        is_new = strategy_entry is None
        
        if not is_new:
            old_mean_qwk = strategy_entry['mean_qwk']
            
            # Only update if new model is better
            if new_mean_qwk <= old_mean_qwk:
                return {
                    'updated': False,
                    'reason': (
                        f"Current best kept for {model_name} [{cv_key}] "
                        f"(QWK {old_mean_qwk:.4f} >= {new_mean_qwk:.4f})"
                    ),
                    'old_mean_qwk': old_mean_qwk,
                    'new_mean_qwk': new_mean_qwk,
                    'improvement': None
                }
            
            # Prepare to replace
            replaced_run_id = strategy_entry['run_id']
            improvement = new_mean_qwk - old_mean_qwk
            
            # Add to history
            history = strategy_entry.get('history', [])
            history.append({
                'run_id': replaced_run_id,
                'mean_qwk': old_mean_qwk,
                'replaced_at': datetime.now().isoformat()
            })
        else:
            old_mean_qwk = None
            improvement = None
            replaced_run_id = None
            history = []
        
        # Build fold_models dict (store relative paths for portability)
        fold_models = {}
        for i, (metrics, checkpoint) in enumerate(zip(fold_metrics, checkpoint_paths)):
            checkpoint_path = Path(checkpoint)
            try:
                # Store path relative to project root so it works across machines/sessions
                relative_checkpoint = checkpoint_path.relative_to(self.project_root)
                stored_checkpoint = str(relative_checkpoint)
            except ValueError:
                # Path is not under project root - store absolute as-is
                stored_checkpoint = str(checkpoint_path)
            fold_models[f"fold_{i}"] = {
                'val_qwk': metrics['val_qwk'],
                'val_acc': metrics['val_acc'],
                'checkpoint': stored_checkpoint
            }
        
        # Update registry (nested by architecture -> cv_strategy)
        if model_name not in self.registry:
            self.registry[model_name] = {}

        self.registry[model_name][cv_key] = {
            'cv_strategy': cv_key,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'mean_qwk': float(new_mean_qwk),
            'std_qwk': float(new_std_qwk),
            'mean_acc': float(new_mean_acc),
            'test_qwk': float(test_qwk) if test_qwk is not None else None,
            'with_hpo': bool(with_hpo) if with_hpo is not None else False,
            'hpo_mode': hpo_mode,
            'inference_strategy': 'ensemble',
            'best_single_fold': int(best_fold_idx),
            'fold_models': fold_models,
            'replaced_run_id': replaced_run_id,
            'notes': notes,
            'history': history,
        }
        
        # Save to disk
        self._save_registry()
        
        # Auto-commit to git if requested
        if auto_commit:
            self._git_commit(model_name, cv_key, new_mean_qwk, run_id, is_new)
        
        # Return status
        status = {
            'updated': True,
            'reason': (
                f"New best {model_name} [{cv_key}]!"
                if is_new
                else f"Improved {model_name} [{cv_key}]!"
            ),
            'old_mean_qwk': old_mean_qwk,
            'new_mean_qwk': new_mean_qwk,
            'improvement': improvement
        }
        
        return status

    def update_test_qwk(
        self,
        model_name: str,
        cv_strategy: str,
        test_qwk: float,
        run_id: Optional[str] = None,
        test_acc: Optional[float] = None,
    ) -> bool:
        """Patch test_qwk (and optionally test_acc) into an existing registry entry.

        Args:
            model_name: Architecture name (e.g. "densenet")
            cv_strategy: CV strategy key (loao_balanced or random_stratified)
            test_qwk: Holdout test-set QWK to store
            run_id: If provided, only patch when the stored run_id matches
            test_acc: Holdout test-set accuracy (Classes 0-3, Ignore excluded) to store

        Returns:
            True if the entry was found and patched, False otherwise.
        """
        cv_key = self._normalize_cv_strategy(cv_strategy)
        entry = self.registry.get(model_name, {}).get(cv_key)
        if entry is None:
            logging.getLogger(__name__).warning(
                f"update_test_qwk: no registry entry for {model_name} [{cv_key}]"
            )
            return False

        if run_id is not None and entry.get('run_id') != run_id:
            logging.getLogger(__name__).warning(
                f"update_test_qwk: run_id mismatch for {model_name} [{cv_key}] "
                f"(registry={entry.get('run_id')}, requested={run_id})"
            )
            return False

        entry['test_qwk'] = float(test_qwk)
        if test_acc is not None:
            entry['test_acc'] = float(test_acc)
        self._save_registry()
        logging.getLogger(__name__).info(
            f"Registry patched: {model_name} [{cv_key}] test_qwk={test_qwk:.4f}"
            + (f" test_acc={test_acc:.4f}" if test_acc is not None else "")
        )
        return True

    def get_model(self, model_name: str, cv_strategy: str) -> Optional[Dict[str, Any]]:
        """
        Get registry entry for a specific model.
        
        Args:
            model_name: Architecture name
            cv_strategy: CV strategy key (loao_balanced or random_stratified)
            
        Returns:
            Registry entry dict or None if not found
        """
        cv_key = self._normalize_cv_strategy(cv_strategy)
        return self.registry.get(model_name, {}).get(cv_key)
    
    def get_all_models(self) -> Dict[str, Any]:
        """Get all models in registry."""
        return self.registry.copy()

    def get_model_strategy_comparison(self, model_name: str) -> Dict[str, Optional[Dict[str, Any]]]:
        """Return best-entry comparison for LOAO vs random stratified for one architecture."""
        model_entry = self.registry.get(model_name, {})
        return {
            'loao_balanced': model_entry.get('loao_balanced'),
            'random_stratified': model_entry.get('random_stratified'),
        }

    def print_model_strategy_comparison(self, model_name: str) -> None:
        """Print side-by-side summary of best entries for both CV strategies of one model."""
        comparison = self.get_model_strategy_comparison(model_name)
        print("\n" + "=" * 80)
        print(f"BEST MODEL COMPARISON: {model_name}")
        print("=" * 80)

        for cv_key in ['loao_balanced', 'random_stratified']:
            entry = comparison.get(cv_key)
            print(f"CV Strategy: {cv_key}")
            if entry is None:
                print("  No entry")
                print()
                continue

            print(f"  Run ID: {entry['run_id']}")
            print(f"  Mean QWK: {entry['mean_qwk']:.4f} +- {entry['std_qwk']:.4f}")
            print(f"  Mean Acc: {entry['mean_acc']:.4f}")
            print(f"  Updated: {entry['timestamp'][:10]}")
            print()

        print("=" * 80)
    
    def get_model_ranking(self, cv_strategy: str, metric: str = 'mean_qwk') -> List[Tuple[str, float]]:
        """
        Get models ranked by performance.
        
        Args:
            cv_strategy: CV strategy key (loao_balanced or random_stratified)
            metric: Metric to rank by (default: 'mean_qwk')
            
        Returns:
            List of (model_name, metric_value) tuples, sorted descending
        """
        cv_key = self._normalize_cv_strategy(cv_strategy)
        ranking = []
        for name, strategy_map in self.registry.items():
            entry = strategy_map.get(cv_key, {})
            if metric in entry:
                ranking.append((name, entry[metric]))
        return sorted(ranking, key=lambda x: x[1], reverse=True)
    
    def _resolve_checkpoint_path(self, stored_path: str) -> str:
        """
        Resolve a stored checkpoint path to an existing absolute path.

        Handles three formats:
        1. Relative path (e.g. "experiments/run_id/model/checkpoints/fold_0_best.ckpt")
           stored by new code - resolved against self.project_root.
        2. Absolute path that still exists on this machine - used as-is.
        3. Stale absolute path from a different machine/session (e.g. old /content/...
           or another user's home directory) - the "experiments/..." suffix is extracted
           and resolved against self.project_root as a fallback.

        Args:
            stored_path: Checkpoint path as stored in the registry JSON.

        Returns:
            Resolved absolute path string.

        Raises:
            FileNotFoundError: If the checkpoint cannot be located.
        """
        p = Path(stored_path)

        # Normalise: convert absolute path to relative "experiments/..." suffix
        # so all cases share a single resolution + cross-session fallback path.
        if p.is_absolute():
            if p.exists():
                return str(p)
            # Try to extract the portable "experiments/..." suffix from any absolute path
            parts = p.parts
            try:
                exp_idx = next(i for i, part in enumerate(parts) if part == "experiments")
                p = Path(*parts[exp_idx:])
            except StopIteration:
                raise FileNotFoundError(
                    f"Checkpoint not found: {stored_path}\n"
                    f"  Absolute path does not exist and contains no 'experiments/' segment.\n"
                    f"  project_root: {self.project_root}"
                )

        # p is now relative (either originally relative, or normalised above)
        resolved = self.project_root / p
        if resolved.exists():
            return str(resolved)

        raise FileNotFoundError(
            f"Checkpoint not found (relative path): {resolved}\n"
            f"  stored as: {stored_path}\n"
            f"  project_root: {self.project_root}\n"
            f"  Cross-run fallback is disabled to prevent accidental run mixing."
        )

    def get_checkpoint_paths(
        self, 
        model_name: str, 
        cv_strategy: str,
        strategy: str = 'ensemble'
    ) -> List[str]:
        """
        Get checkpoint paths for inference.
        
        Args:
            model_name: Architecture name
            cv_strategy: CV strategy key (loao_balanced or random_stratified)
            strategy: 'ensemble' (all folds) or 'best_fold' (single best)
            
        Returns:
            List of resolved absolute checkpoint paths
            
        Raises:
            KeyError: If model not in registry
            FileNotFoundError: If any checkpoint file cannot be located on disk
        """
        cv_key = self._normalize_cv_strategy(cv_strategy)
        if model_name not in self.registry or cv_key not in self.registry[model_name]:
            raise KeyError(f"Model '{model_name}' with cv_strategy '{cv_key}' not found in registry")

        entry = self.registry[model_name][cv_key]
        fold_models = entry['fold_models']
        
        if strategy == 'ensemble':
            raw_paths = [
                fold_models[f"fold_{i}"]['checkpoint']
                for i in range(len(fold_models))
            ]
        elif strategy == 'best_fold':
            best_fold = entry['best_single_fold']
            raw_paths = [fold_models[f"fold_{best_fold}"]['checkpoint']]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return [self._resolve_checkpoint_path(p) for p in raw_paths]
    
    def print_summary(self):
        """Print human-readable registry summary."""
        if not self.registry:
            print("Registry is empty. Train some models first!")
            return
        
        print("\n" + "=" * 80)
        print("BEST MODELS REGISTRY SUMMARY")
        print("=" * 80)
        print(f"Total architectures: {len(self.registry)}")
        print()

        for cv_key in ["loao_balanced", "random_stratified"]:
            ranking = self.get_model_ranking(cv_strategy=cv_key, metric='mean_qwk')
            print(f"CV Strategy: {cv_key}")
            if not ranking:
                print("  No entries")
                print()
                continue

            for rank, (name, qwk) in enumerate(ranking, 1):
                entry = self.registry[name][cv_key]
                print(f"  {rank}. {name.upper()}")
                print(f"     Mean QWK: {entry['mean_qwk']:.4f} ± {entry['std_qwk']:.4f}")
                print(f"     Mean Acc: {entry['mean_acc']:.4f}")
                if entry.get('test_qwk'):
                    print(f"     Test QWK: {entry['test_qwk']:.4f}")
                print(f"     Run ID: {entry['run_id']}")
                print(f"     Date: {entry['timestamp'][:10]}")
                if entry.get('notes'):
                    print(f"     Notes: {entry['notes']}")

                if entry.get('history'):
                    last_replaced = entry['history'][-1]
                    improvement = entry['mean_qwk'] - last_replaced['mean_qwk']
                    print(f"     Improvement: +{improvement:.4f} over previous best")
                print()
        
        print("=" * 80)
    
    def _git_commit(self, model_name: str, cv_strategy: str, qwk: float, run_id: str, is_new: bool):
        """Auto-commit registry update to git."""
        try:
            # Stage registry file
            subprocess.run(
                ['git', 'add', str(self.registry_path)],
                cwd=self.registry_path.parent,
                check=True,
                capture_output=True
            )
            
            # Commit with descriptive message
            action = "Added" if is_new else "Updated"
            message = (
                f"{action} best {model_name} [{cv_strategy}]: "
                f"QWK {qwk:.4f} (run {run_id})"
            )
            
            subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.registry_path.parent,
                check=True,
                capture_output=True
            )
            
            print(f"Git commit: {message}")
            
        except subprocess.CalledProcessError:
            # Git commit failed (maybe no changes or not a git repo)
            # Don't fail, just skip
            pass
        except FileNotFoundError:
            # Git not installed
            pass


def load_registry(registry_path: Optional[Path] = None) -> BestModelsRegistry:
    """
    Convenience function to load registry.
    
    Args:
        registry_path: Optional path to registry JSON
        
    Returns:
        BestModelsRegistry instance
    """
    return BestModelsRegistry(registry_path)
