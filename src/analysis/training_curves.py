"""
Training Curves Generation Module.

Extracts train/validation metrics from TensorBoard event files (or Lightning
CSV logs) and saves per-fold PNG curve plots for loss, accuracy, kappa, and
macro-F1 (MLv13 compliance: ML-18).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/notebook use
import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.tensorboard_extractor import extract_model_training_history
from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)

# Metrics to request from TensorBoard / CSV logs.
# Extractor falls back gracefully for missing tags.
_CURVE_METRICS: List[str] = [
    "train_loss_epoch",
    "val_loss",
    "train_acc",
    "val_acc",
    "val_kappa",
    "val_qwk",
    "val_macro_f1",
]


def _get_metric_series(df: pd.DataFrame, metric: str) -> Optional[pd.Series]:
    """Extract a single metric as an epoch-indexed Series from a fold DataFrame.

    Args:
        df: DataFrame with columns: epoch, metric, value.
        metric: Tag name to filter on.

    Returns:
        Epoch-indexed Series of values, or None if the metric is absent.
    """
    sub = df[df["metric"] == metric]
    if sub.empty:
        return None
    return sub.set_index("epoch")["value"].sort_index()


def _detect_n_folds(tb_dir: Path, fallback: int = 2) -> int:
    """Detect fold count from a TensorBoard directory structure.

    Checks for fold_N/ subdirectories first; falls back to counting event
    files in a flat layout; returns the provided fallback if neither applies.

    Args:
        tb_dir: Path to the tensorboard subdirectory.
        fallback: Number of folds to return if detection fails.

    Returns:
        Detected or fallback fold count.
    """
    fold_dirs = [d for d in tb_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")]
    if fold_dirs:
        return len(fold_dirs)
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if event_files:
        return len(event_files)
    return fallback


def _find_model_dirs(run_dir: Path) -> List[Path]:
    """Find model experiment directories within a run directory.

    Looks for immediate subdirectories that contain a tensorboard/ subdir.
    Falls back to treating run_dir itself as a model directory if it contains
    tensorboard/ directly.

    Args:
        run_dir: Root run directory or single model experiment directory.

    Returns:
        Sorted list of model experiment directory Paths.
    """
    if not run_dir.exists():
        logger.error(f"Run directory does not exist: {run_dir}")
        return []

    model_dirs = [
        d for d in sorted(run_dir.iterdir())
        if d.is_dir() and (d / "tensorboard").exists()
    ]
    if model_dirs:
        return model_dirs

    if (run_dir / "tensorboard").exists():
        return [run_dir]

    logger.warning(f"No model directories with tensorboard/ found in {run_dir}")
    return []


def _plot_fold_curves(
    fold_data: Dict[int, pd.DataFrame],
    model_name: str,
    output_dir: Path,
) -> List[Path]:
    """Generate per-fold curve PNG files for a single model.

    Creates a 2x2 subplot figure per fold showing loss, accuracy, QWK kappa,
    and macro-F1 curves. Saves each figure as a PNG.

    Args:
        fold_data: Mapping of fold_idx to DataFrame (columns: epoch, metric, value).
        model_name: Model name string used for file naming and figure titles.
        output_dir: Directory where PNG files will be saved.

    Returns:
        List of Paths to successfully saved PNG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for fold_idx, df in fold_data.items():
        if df.empty:
            logger.warning(f"Empty DataFrame for {model_name} fold {fold_idx}, skipping")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{model_name} - Fold {fold_idx} Training Curves", fontsize=12)

        # Subplot (0, 0): Loss
        ax_loss = axes[0, 0]
        train_loss = _get_metric_series(df, "train_loss_epoch")
        val_loss = _get_metric_series(df, "val_loss")
        if train_loss is not None:
            ax_loss.plot(train_loss.index, train_loss.values, label="Train Loss")
        if val_loss is not None:
            ax_loss.plot(val_loss.index, val_loss.values, label="Val Loss")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        if ax_loss.get_lines():
            ax_loss.legend()

        # Subplot (0, 1): Accuracy
        ax_acc = axes[0, 1]
        train_acc = _get_metric_series(df, "train_acc")
        val_acc = _get_metric_series(df, "val_acc")
        if train_acc is not None:
            ax_acc.plot(train_acc.index, train_acc.values, label="Train Acc")
        if val_acc is not None:
            ax_acc.plot(val_acc.index, val_acc.values, label="Val Acc")
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        if ax_acc.get_lines():
            ax_acc.legend()

        # Subplot (1, 0): Validation QWK / Kappa
        ax_kappa = axes[1, 0]
        val_kappa = _get_metric_series(df, "val_kappa")
        if val_kappa is None:
            val_kappa = _get_metric_series(df, "val_qwk")
        if val_kappa is not None:
            ax_kappa.plot(val_kappa.index, val_kappa.values, label="Val QWK", color="green")
        ax_kappa.set_title("Validation QWK (Kappa)")
        ax_kappa.set_xlabel("Epoch")
        ax_kappa.set_ylabel("QWK")
        if ax_kappa.get_lines():
            ax_kappa.legend()

        # Subplot (1, 1): Validation Macro-F1
        ax_f1 = axes[1, 1]
        val_f1 = _get_metric_series(df, "val_macro_f1")
        if val_f1 is not None:
            ax_f1.plot(val_f1.index, val_f1.values, label="Val Macro-F1", color="orange")
        ax_f1.set_title("Validation Macro-F1")
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("Macro-F1")
        if ax_f1.get_lines():
            ax_f1.legend()

        plt.tight_layout()
        out_path = output_dir / f"{model_name}_fold{fold_idx}_curves.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(out_path)
        logger.info(f"Saved: {out_path}")

    return saved_paths


def generate_training_curves(
    run_dir: Path,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Extract TensorBoard metrics and generate per-fold training curve plots.

    Scans run_dir for model experiment directories (subdirectories containing
    tensorboard/), extracts training history for each, and saves PNG figures.
    Skips models gracefully if no TensorBoard data is found.

    Args:
        run_dir: Path to a run directory containing model subdirs, or a single
            model experiment directory that contains tensorboard/ directly.
        output_dir: Directory for PNG output files. Defaults to
            run_dir/training_curves/.

    Returns:
        List of Paths to all generated PNG files.
    """
    if output_dir is None:
        output_dir = run_dir / "training_curves"

    model_dirs = _find_model_dirs(run_dir)
    if not model_dirs:
        logger.warning(f"No model directories found in {run_dir}")
        return []

    all_paths: List[Path] = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        tb_dir = model_dir / "tensorboard"
        n_folds = _detect_n_folds(tb_dir) if tb_dir.exists() else 2

        logger.info(f"Processing model '{model_name}' ({n_folds} folds)")

        try:
            fold_data = extract_model_training_history(
                experiment_dir=model_dir,
                model_name=model_name,
                n_folds=n_folds,
                metric_names=_CURVE_METRICS,
            )
        except Exception as exc:
            logger.warning(f"Failed to extract TensorBoard data for '{model_name}': {exc}")
            continue

        if not fold_data:
            logger.warning(f"No TensorBoard data found for '{model_name}', skipping")
            continue

        model_output_dir = output_dir / model_name
        paths = _plot_fold_curves(fold_data, model_name, model_output_dir)
        all_paths.extend(paths)

    logger.info(f"Generated {len(all_paths)} training curve plots total")
    return all_paths


def main() -> None:
    """Entry point for training curve generation.

    Parses --run-dir and optional --output-dir CLI arguments, then calls
    generate_training_curves.
    """
    parser = argparse.ArgumentParser(
        description="Generate train-vs-val curve plots from TensorBoard logs"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help=(
            "Path to a run directory (with model subdirs) or a single "
            "model experiment directory (containing tensorboard/)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PNG files (default: run_dir/training_curves/)",
    )
    args = parser.parse_args()
    generate_training_curves(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
