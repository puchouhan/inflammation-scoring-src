"""
Trivial Baseline Evaluation Module.

Evaluates majority-class, stratified-random, and uniform-random DummyClassifiers
over LOAO cross-validation splits. Establishes the performance floor that DL
models must exceed (MLv13 compliance: ML-16).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.utils import load_config
from src.data.inflammation_dataset import create_dataframe
from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)

_STRATEGIES: List[str] = ["most_frequent", "stratified", "uniform"]


def _compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute QWK, accuracy, and macro-F1 for a single fold prediction.

    Args:
        y_true: Ground truth labels (integers 0-3).
        y_pred: Predicted labels from a DummyClassifier.

    Returns:
        Dictionary with keys: qwk, acc, f1.
    """
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"qwk": float(qwk), "acc": float(acc), "f1": float(f1)}


def _print_markdown_table(summary: Dict[str, Dict]) -> None:
    """Print a markdown comparison table of strategy metrics to stdout.

    Args:
        summary: Mapping of strategy name to aggregated metrics dict.
    """
    print("\n## Trivial Baseline Results\n")
    print("| Strategy | Mean QWK | Std QWK | Mean Acc | Mean F1 |")
    print("|---|---|---|---|---|")
    for strategy, metrics in summary.items():
        print(
            f"| {strategy} "
            f"| {metrics['mean_qwk']:.4f} "
            f"| {metrics['std_qwk']:.4f} "
            f"| {metrics['mean_acc']:.4f} "
            f"| {metrics['mean_f1']:.4f} |"
        )


def _build_split_arrays(
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load dataset and return filtered label/animal arrays plus fold count.

    Loads the normalized dataset, filters out the ignore class, and returns
    label and animal ID arrays ready for StratifiedGroupKFold.

    Args:
        cfg: Loaded configuration dictionary.

    Returns:
        Tuple of (labels, animal_ids, n_folds).
    """
    ignore_class: int = cfg.get("ignore_class_index", 4)
    data_root: str = cfg["data"]["norm_dir"]

    logger.info(f"Loading dataset from: {data_root}")
    df = create_dataframe(data_root)

    df_filtered = df[df["label"] != ignore_class].reset_index(drop=True)
    logger.info(
        f"Dataset after filtering ignore class (label={ignore_class}): "
        f"{len(df_filtered)} samples "
        f"(removed {len(df) - len(df_filtered)} ignore-class samples)"
    )

    cv_strategy: str = cfg["data"].get("cv_strategy", "loao_balanced")
    n_folds: int = cfg["data"]["cv_folds_config"].get(cv_strategy, 2)

    labels: np.ndarray = df_filtered["label"].values
    animal_ids: np.ndarray = df_filtered["animal_id"].values

    return labels, animal_ids, n_folds


def run_trivial_baseline(cfg: dict, output_path: Path) -> dict:
    """Evaluate DummyClassifier baselines over LOAO cross-validation splits.

    Loads the full normalized dataset from config, runs StratifiedGroupKFold
    splits by animal ID, and evaluates three DummyClassifier strategies per
    fold. Ignore-class samples are excluded from fitting and scoring.

    Args:
        cfg: Loaded configuration dictionary (from load_config()).
        output_path: Path where JSON results will be saved.

    Returns:
        Dictionary with keys 'strategies', 'n_folds', 'dataset_root'.
    """
    labels, animal_ids, n_folds = _build_split_arrays(cfg)
    cv_strategy: str = cfg["data"].get("cv_strategy", "loao_balanced")
    data_root: str = cfg["data"]["norm_dir"]

    logger.info(f"Running {n_folds}-fold CV (strategy: {cv_strategy})")

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(labels, labels, groups=animal_ids))

    fold_results: Dict[str, Dict[str, List[float]]] = {
        s: {"fold_qwk": [], "fold_acc": [], "fold_f1": []} for s in _STRATEGIES
    }

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_labels: np.ndarray = labels[train_idx]
        val_labels: np.ndarray = labels[val_idx]
        x_train = np.zeros((len(train_labels), 1))
        x_val = np.zeros((len(val_labels), 1))

        logger.info(
            f"Fold {fold_idx}: train={len(train_labels)} val={len(val_labels)}"
        )

        for strategy in _STRATEGIES:
            random_state = 42 if strategy in ("stratified", "uniform") else None
            clf = DummyClassifier(strategy=strategy, random_state=random_state)
            clf.fit(x_train, train_labels)
            preds: np.ndarray = clf.predict(x_val)

            metrics = _compute_fold_metrics(val_labels, preds)
            fold_results[strategy]["fold_qwk"].append(metrics["qwk"])
            fold_results[strategy]["fold_acc"].append(metrics["acc"])
            fold_results[strategy]["fold_f1"].append(metrics["f1"])

            logger.info(
                f"  [{strategy}] QWK={metrics['qwk']:.4f} "
                f"Acc={metrics['acc']:.4f} F1={metrics['f1']:.4f}"
            )

    summary: Dict[str, Dict] = {}
    for strategy, data in fold_results.items():
        qwk_arr = np.array(data["fold_qwk"])
        acc_arr = np.array(data["fold_acc"])
        f1_arr = np.array(data["fold_f1"])
        summary[strategy] = {
            "fold_qwk": data["fold_qwk"],
            "mean_qwk": float(np.mean(qwk_arr)),
            "std_qwk": float(np.std(qwk_arr)),
            "mean_acc": float(np.mean(acc_arr)),
            "mean_f1": float(np.mean(f1_arr)),
        }

    output_data: dict = {
        "strategies": summary,
        "n_folds": n_folds,
        "dataset_root": str(data_root),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(output_data, fh, indent=2)
    logger.info(f"Results saved to {output_path}")

    _print_markdown_table(summary)
    return output_data


def main() -> None:
    """Entry point for trivial baseline evaluation.

    Loads base config, runs baseline evaluation, and saves results to
    docs/artifacts/trivial_baseline_results.json.
    """
    cfg = load_config()
    output_path = Path("docs/artifacts/trivial_baseline_results.json")
    run_trivial_baseline(cfg, output_path)


if __name__ == "__main__":
    main()
