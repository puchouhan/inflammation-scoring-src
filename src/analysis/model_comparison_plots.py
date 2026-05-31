"""
Cross-Model Comparison Plots Module.

Generates publication-quality comparison plots for all 11 models,
separated by CV strategy (LOAO vs Stratified). Designed for thesis
inclusion with consistent styling and no bar plots.

Plot types:
    1. Box Plots -- QWK and Accuracy per model (with CLD letters)
    2. ROC Curves -- One-vs-Rest, grouped by architecture family (square)
    3. Precision-Recall Curves -- grouped by architecture family
    4. Learning Curves -- Train/Val loss over epochs
    5. Calibration Plot (Reliability Diagram, square)
    6. Critical Difference Diagram (Nemenyi post-hoc)
    7. Per-Class F1 Boxplots
    8. Radar/Spider Chart -- multi-metric comparison
    9. Pairwise Significance Table (Wilcoxon + BH-FDR heatmap)
   10. Cohen's d Effect Size Matrix (heatmap)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-use constants from boxplot_analysis (single source of truth)
# ---------------------------------------------------------------------------
ARCHITECTURE_FAMILIES: Dict[str, str] = {
    "densenet": "CNN",
    "efficientnetv2": "CNN",
    "regnety": "CNN",
    "convnext": "CNN",
    "vit": "Transformer",
    "swin": "Transformer",
    "maxvit": "Hybrid",
    "convit": "Hybrid",
    "tnt": "Hybrid",
    "simclr": "Self-Supervised",
    "dino": "Self-Supervised",
}

FAMILY_ORDER: List[str] = ["CNN", "Transformer", "Hybrid", "Self-Supervised"]

DISPLAY_NAMES: Dict[str, str] = {
    "densenet": "DenseNet",
    "efficientnetv2": "EfficientNetV2",
    "regnety": "RegNetY",
    "convnext": "ConvNeXt",
    "vit": "ViT",
    "swin": "Swin",
    "maxvit": "MaxViT",
    "convit": "ConViT",
    "tnt": "TNT",
    "simclr": "SimCLR",
    "dino": "DINO",
}

FAMILY_COLORS: Dict[str, str] = {
    "CNN": "#4ECDC4",
    "Transformer": "#F38181",
    "Hybrid": "#95E1D3",
    "Self-Supervised": "#FFD93D",
}

# ---------------------------------------------------------------------------
# Reference baseline: Heinemann et al. (2018) PLOS ONE
# InceptionV3, 90/10 random split, Masson trichrome stain
# ---------------------------------------------------------------------------
REFERENCE_PAPER_LABEL: str = "Heinemann et al. 2018 (InceptionV3)"
REFERENCE_ACCURACY: float = 0.80
# Row-normalized confusion matrix (classes 0-3 + ignore), from Fig 8
REFERENCE_CONFUSION_MATRIX: List[List[float]] = [
    [0.79, 0.14, 0.00, 0.01, 0.05],  # GT 0
    [0.10, 0.73, 0.15, 0.01, 0.01],  # GT 1
    [0.01, 0.11, 0.69, 0.19, 0.00],  # GT 2
    [0.00, 0.01, 0.07, 0.90, 0.03],  # GT 3
    [0.07, 0.00, 0.02, 0.05, 0.87],  # GT Ignore
]
# Only classes 0-3 (exclude Ignore row/col for fair comparison)
REFERENCE_CM_NO_IGNORE: List[List[float]] = [
    [0.79, 0.14, 0.00, 0.01],
    [0.10, 0.73, 0.15, 0.01],
    [0.01, 0.11, 0.69, 0.19],
    [0.00, 0.01, 0.07, 0.90],
]

# Per-model colors (distinct within families, consistent across all plots)
MODEL_COLORS: Dict[str, str] = {
    # CNN family
    "densenet": "#2CA58D",
    "efficientnetv2": "#4ECDC4",
    "regnety": "#84DCC6",
    "convnext": "#1B9AAA",
    # Transformer family
    "vit": "#F38181",
    "swin": "#C0392B",
    # Hybrid family
    "maxvit": "#95E1D3",
    "convit": "#45B7A0",
    "tnt": "#6EC6A0",
    # Self-Supervised family
    "simclr": "#FFD93D",
    "dino": "#F4A261",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_registry(project_root: Path) -> Dict:
    """Load best_models_registry_new.json."""
    for candidate in [
        project_root / "src" / "experiments" / "best_models_registry_new.json",
        project_root / "experiments" / "best_models_registry_new.json",
    ]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return json.load(f)
    raise FileNotFoundError("best_models_registry_new.json not found")


def _extract_model_name(registry_key: str) -> str:
    """Extract base model name from registry key."""
    for suffix in ("_stratified", "_loao"):
        if registry_key.endswith(suffix):
            return registry_key[: -len(suffix)]
    return registry_key


def _cv_label(cv_strategy: str) -> str:
    """Human-readable CV strategy label."""
    if cv_strategy == "random_stratified":
        return "Stratified 5-Fold CV"
    return "LOAO 2-Fold CV"


def _cv_suffix(cv_strategy: str) -> str:
    """Registry key suffix for a CV strategy."""
    return "_stratified" if cv_strategy == "random_stratified" else "_loao"


def _get_fold_values(
    cv_data: Dict, field: str,
) -> List[float]:
    """Extract per-fold values for a given field from cv_data dict."""
    fold_models = cv_data.get("fold_models", {})
    return [
        fm[field]
        for fm in fold_models.values()
        if fm.get(field, 0) > 0
    ]


def _filter_registry(
    registry: Dict, cv_strategy: str,
) -> List[Tuple[str, Dict]]:
    """Return (base_model_name, cv_data) pairs for a CV strategy."""
    suffix = _cv_suffix(cv_strategy)
    results: List[Tuple[str, Dict]] = []
    for key, entry in registry.items():
        if not key.endswith(suffix):
            continue
        cv_data = list(entry.values())[0]
        if cv_data.get("cv_strategy") != cv_strategy:
            continue
        base_name = _extract_model_name(key)
        results.append((base_name, cv_data))
    return results


def _group_by_family(
    model_names: List[str],
) -> Dict[str, List[str]]:
    """Group model names by architecture family."""
    groups: Dict[str, List[str]] = {f: [] for f in FAMILY_ORDER}
    for name in model_names:
        family = ARCHITECTURE_FAMILIES.get(name, "Other")
        if family in groups:
            groups[family].append(name)
    return groups


def _setup_style() -> None:
    """Apply consistent plot style with thesis-ready font sizes."""
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    })


def _add_boxplot_legend(ax: plt.Axes) -> None:
    """Add standard mean/median/fold-point legend."""
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
               markersize=6, label="Mean"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=5, alpha=0.6, label="Fold values"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)


def _load_predictions_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    """Load a predictions CSV file.

    Expected columns: ground_truth, prediction, confidence_0 .. confidence_N.

    Args:
        csv_path: Path to predictions CSV.

    Returns:
        DataFrame or None if file does not exist.
    """
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _find_predictions_csvs(
    project_root: Path,
    run_id: str,
    model_name: str,
    n_folds: int,
    experiments_dir: Optional[Path] = None,
) -> Dict[int, Path]:
    """Locate predictions CSV files for a model's folds.

    Args:
        project_root: Project root directory.
        run_id: Experiment run ID.
        model_name: Base model name.
        n_folds: Number of folds.
        experiments_dir: Override directory containing experiment runs.
            Defaults to project_root / 'experiments'.

    Returns:
        Dict mapping fold_idx to CSV Path (only existing files).
    """
    found: Dict[int, Path] = {}
    base = experiments_dir or (project_root / "experiments")
    exp_dir = base / run_id / model_name
    for fold_idx in range(n_folds):
        csv_path = exp_dir / "predictions" / f"fold_{fold_idx}_predictions.csv"
        if csv_path.exists():
            found[fold_idx] = csv_path
    return found


def _collect_predictions(
    project_root: Path,
    registry: Dict,
    cv_strategy: str,
    experiments_dir: Optional[Path] = None,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Collect all available predictions CSVs for a CV strategy.

    Args:
        project_root: Project root directory.
        registry: Loaded registry dict.
        cv_strategy: CV strategy filter.
        experiments_dir: Override directory containing experiment runs.

    Returns:
        {model_name: {fold_idx: DataFrame}}.
    """
    all_preds: Dict[str, Dict[int, pd.DataFrame]] = {}
    for base_name, cv_data in _filter_registry(registry, cv_strategy):
        run_id = cv_data.get("run_id", "")
        fold_models = cv_data.get("fold_models", {})
        if not run_id:
            continue
        csv_paths = _find_predictions_csvs(
            project_root, run_id, base_name, len(fold_models),
            experiments_dir=experiments_dir,
        )
        if csv_paths:
            fold_dfs: Dict[int, pd.DataFrame] = {}
            for fold_idx, csv_path in csv_paths.items():
                df = _load_predictions_csv(csv_path)
                if df is not None:
                    fold_dfs[fold_idx] = df
            if fold_dfs:
                all_preds[base_name] = fold_dfs
    return all_preds


# ===========================================================================
# Statistical helpers for significance annotation
# ===========================================================================


def _pairwise_wilcoxon_bh(
    model_names: List[str],
    fold_values: List[List[float]],
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Wilcoxon signed-rank tests with BH correction.

    Args:
        model_names: List of model names (sorted by performance).
        fold_values: List of per-fold metric values for each model.
        alpha: Significance level.

    Returns:
        Tuple of (raw_p_matrix, adjusted_p_matrix) as DataFrames.
    """
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    n = len(model_names)
    pairs: List[Tuple[int, int]] = []
    p_vals: List[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            min_len = min(len(fold_values[i]), len(fold_values[j]))
            if min_len < 3:
                p_vals.append(1.0)
            else:
                try:
                    _, p = wilcoxon(
                        fold_values[i][:min_len],
                        fold_values[j][:min_len],
                    )
                    p_vals.append(p)
                except Exception:
                    p_vals.append(1.0)
            pairs.append((i, j))

    # BH correction (only on valid p-values)
    adj_p_vals = np.ones(len(p_vals))
    valid_mask = [p < 1.0 for p in p_vals]
    if any(valid_mask):
        valid_p = [p for p, v in zip(p_vals, valid_mask) if v]
        _, corrected, _, _ = multipletests(
            valid_p, alpha=alpha, method="fdr_bh",
        )
        ci = 0
        for idx, v in enumerate(valid_mask):
            if v:
                adj_p_vals[idx] = corrected[ci]
                ci += 1

    raw_matrix = np.ones((n, n))
    adj_matrix = np.ones((n, n))
    for idx, (i, j) in enumerate(pairs):
        raw_matrix[i, j] = p_vals[idx]
        raw_matrix[j, i] = p_vals[idx]
        adj_matrix[i, j] = adj_p_vals[idx]
        adj_matrix[j, i] = adj_p_vals[idx]

    np.fill_diagonal(raw_matrix, 1.0)
    np.fill_diagonal(adj_matrix, 1.0)

    display = [DISPLAY_NAMES.get(m, m) for m in model_names]
    raw_df = pd.DataFrame(raw_matrix, index=display, columns=display)
    adj_df = pd.DataFrame(adj_matrix, index=display, columns=display)

    return raw_df, adj_df


def _compute_cld(
    model_names: List[str],
    fold_values: List[List[float]],
    alpha: float = 0.05,
) -> Dict[str, str]:
    """Compact Letter Display via insert-absorb algorithm (Piepho 2004).

    Models sharing a letter are NOT significantly different.

    Args:
        model_names: Model names sorted by mean performance (best first).
        fold_values: Per-fold metric values for each model.
        alpha: Significance level.

    Returns:
        Dict mapping model name to letter string.
    """
    n = len(model_names)
    if n < 2:
        return {model_names[i]: "a" for i in range(n)}

    _, adj_df = _pairwise_wilcoxon_bh(model_names, fold_values, alpha)
    adj_matrix = adj_df.values

    sig = adj_matrix < alpha
    np.fill_diagonal(sig, False)

    # Pass 1: Insert-absorb
    letters: List[set] = [set() for _ in range(n)]
    next_letter = 0

    letters[0].add(chr(ord("a") + next_letter))

    for i in range(1, n):
        absorbed = False
        for l_idx in range(next_letter + 1):
            letter = chr(ord("a") + l_idx)
            group = [j for j in range(n) if letter in letters[j]]
            if all(not sig[i][j] for j in group):
                letters[i].add(letter)
                absorbed = True
        if not absorbed:
            next_letter += 1
            letters[i].add(chr(ord("a") + next_letter))

    # Pass 2: Additional absorb passes
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for l_idx in range(next_letter + 1):
                letter = chr(ord("a") + l_idx)
                if letter in letters[i]:
                    continue
                group = [j for j in range(n) if letter in letters[j]]
                if group and all(not sig[i][j] for j in group):
                    letters[i].add(letter)
                    changed = True

    return {
        model_names[i]: "".join(sorted(letters[i]))
        for i in range(n)
    }


def _compute_cohens_d_matrix(
    model_names: List[str],
    fold_values: List[List[float]],
) -> pd.DataFrame:
    """Compute pairwise Cohen's d effect size matrix.

    Positive value means row model is better than column model.

    Args:
        model_names: List of model names.
        fold_values: Per-fold metric values for each model.

    Returns:
        DataFrame with Cohen's d values.
    """
    n = len(model_names)
    d_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            vi = np.array(fold_values[i])
            vj = np.array(fold_values[j])

            pooled_std = np.sqrt(
                (vi.std(ddof=1) ** 2 + vj.std(ddof=1) ** 2) / 2
            )

            if pooled_std > 0:
                d = (vi.mean() - vj.mean()) / pooled_std
            else:
                d = 0.0

            d_matrix[i, j] = d
            d_matrix[j, i] = -d

    display = [DISPLAY_NAMES.get(m, m) for m in model_names]
    return pd.DataFrame(d_matrix, index=display, columns=display)


# ===========================================================================
# 1. Box Plots -- QWK and Accuracy per Model
# ===========================================================================

def plot_metric_boxplots(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
) -> Path:
    """Box plots of per-fold QWK and Accuracy for each model.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG.
    """
    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names: List[str] = []
    qwk_data: List[List[float]] = []
    acc_data: List[List[float]] = []

    for base_name, cv_data in _filter_registry(registry, cv_strategy):
        qwks = _get_fold_values(cv_data, "val_qwk")
        accs = _get_fold_values(cv_data, "val_acc")
        if not qwks:
            continue
        model_names.append(base_name)
        qwk_data.append(qwks)
        acc_data.append(accs if accs else [0.0])

    if not model_names:
        logger.warning("No models found for metric boxplots")
        return output_dir / f"comparison_boxplots_{cv_strategy}.png"

    # Sort by median QWK descending
    medians = [np.median(q) for q in qwk_data]
    sorted_idx = np.argsort(medians)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    qwk_data = [qwk_data[i] for i in sorted_idx]
    acc_data = [acc_data[i] for i in sorted_idx]

    n = len(model_names)
    x = np.arange(n)
    width = 0.35

    # Compute CLD for significance annotation
    cld = _compute_cld(model_names, qwk_data)

    fig, ax = plt.subplots(figsize=(max(12, n * 1.4), 7))

    # QWK boxes
    bp_qwk = ax.boxplot(
        qwk_data,
        positions=x - width / 2,
        widths=width * 0.8,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for box in bp_qwk["boxes"]:
        box.set_facecolor("#4ECDC4")
        box.set_alpha(0.7)

    # Accuracy boxes
    bp_acc = ax.boxplot(
        acc_data,
        positions=x + width / 2,
        widths=width * 0.8,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for box in bp_acc["boxes"]:
        box.set_facecolor("#F38181")
        box.set_alpha(0.7)

    # Overlay fold points
    rng = np.random.default_rng(42)
    for i, (qwks, accs) in enumerate(zip(qwk_data, acc_data)):
        jitter_q = rng.uniform(-0.06, 0.06, len(qwks))
        ax.scatter(
            [x[i] - width / 2 + j for j in jitter_q], qwks,
            color="black", alpha=0.6, s=30, zorder=5,
        )
        jitter_a = rng.uniform(-0.06, 0.06, len(accs))
        ax.scatter(
            [x[i] + width / 2 + j for j in jitter_a], accs,
            color="black", alpha=0.6, s=30, zorder=5,
        )

    # X-labels with mean QWK value
    display_labels = [
        f"{DISPLAY_NAMES.get(m, m)}\n(mean={np.mean(q):.3f})"
        for m, q in zip(model_names, qwk_data)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(
        f"QWK and Accuracy Distribution per Model -- {_cv_label(cv_strategy)}",
        fontweight="bold",
    )
    ax.set_ylim([-0.05, 1.1])
    ax.grid(axis="y", alpha=0.3)

    # CLD letters above each model
    for i, name in enumerate(model_names):
        letters = cld.get(name, "")
        if letters:
            model_max = max(
                max(qwk_data[i]),
                max(acc_data[i]) if acc_data[i] else 0,
            )
            ax.text(
                x[i], model_max + 0.03, letters,
                ha="center", va="bottom", fontsize=13,
                fontweight="bold", color="#333333",
            )

    # Heinemann et al. 2018 baseline (Accuracy only)
    ax.axhline(
        y=REFERENCE_ACCURACY, color="#E67E22", linewidth=1.5,
        linestyle="--", zorder=2,
    )
    ax.text(
        len(model_names) - 0.5, REFERENCE_ACCURACY + 0.012,
        f"{REFERENCE_PAPER_LABEL}\n(Acc={REFERENCE_ACCURACY:.1%})",
        fontsize=8, color="#E67E22", ha="right", va="bottom",
        fontstyle="italic",
    )

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4ECDC4",
               markersize=10, label="QWK"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F38181",
               markersize=10, label="Accuracy"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
               markersize=6, label="Mean"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=5, alpha=0.6, label="Fold values"),
        Line2D([0], [0], color="#E67E22", linewidth=1.5, linestyle="--",
               label=REFERENCE_PAPER_LABEL),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    # CLD method annotation
    ax.text(
        0.98, 0.02,
        "CLD: Wilcoxon signed-rank + BH-FDR correction",
        transform=ax.transAxes, fontsize=8,
        ha="right", va="bottom", style="italic", color="0.4",
    )

    plt.tight_layout()
    path = output_dir / f"comparison_boxplots_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 2. ROC Curves -- One-vs-Rest, Grouped by Architecture Family
# ===========================================================================

def plot_roc_curves(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Multi-class ROC curves (macro-average), grouped by architecture family.

    Requires predictions CSVs with per-class confidence columns.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if no predictions data available.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = _collect_predictions(
        project_root, registry, cv_strategy,
        experiments_dir=experiments_dir,
    )
    if not all_preds:
        logger.warning(
            f"No predictions CSVs found for ROC curves ({cv_strategy}). "
            "Run training with include_predictions_csv=True first."
        )
        return None

    n_classes = 4  # Classes 0-3 (exclude ignore class 4)
    families = _group_by_family(list(all_preds.keys()))
    active_families = [f for f in FAMILY_ORDER if families.get(f)]

    if not active_families:
        logger.warning("No models with predictions found for ROC curves")
        return None

    n_cols = len(active_families)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    for ax_idx, family in enumerate(active_families):
        ax = axes[ax_idx]
        family_models = families[family]

        for model_name in family_models:
            fold_dfs = all_preds.get(model_name, {})
            if not fold_dfs:
                continue

            # Concatenate all folds
            combined = pd.concat(fold_dfs.values(), ignore_index=True)
            y_true = combined["ground_truth"].values
            # Filter to classes 0-3
            mask = y_true < n_classes
            y_true = y_true[mask]
            prob_cols = [f"confidence_{i}" for i in range(n_classes)]
            y_score = combined.loc[mask, prob_cols].values

            # Re-normalize probabilities to sum to 1 over classes 0-3
            row_sums = y_score.sum(axis=1, keepdims=True) + 1e-8
            y_score = y_score / row_sums

            y_bin = label_binarize(y_true, classes=list(range(n_classes)))

            # Macro-average ROC
            fpr_grid = np.linspace(0, 1, 200)
            mean_tpr = np.zeros_like(fpr_grid)
            for cls in range(n_classes):
                fpr_cls, tpr_cls, _ = roc_curve(y_bin[:, cls], y_score[:, cls])
                mean_tpr += np.interp(fpr_grid, fpr_cls, tpr_cls)
            mean_tpr /= n_classes
            macro_auc = auc(fpr_grid, mean_tpr)

            color = MODEL_COLORS.get(model_name, "#999999")
            display = DISPLAY_NAMES.get(model_name, model_name)
            ax.plot(
                fpr_grid, mean_tpr,
                color=color, linewidth=2,
                label=f"{display} (AUC={macro_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(family, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Macro-Average ROC Curves (One-vs-Rest) -- {_cv_label(cv_strategy)}",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = output_dir / f"comparison_roc_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 3. Precision-Recall Curves -- Grouped by Architecture Family
# ===========================================================================

def plot_pr_curves(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Multi-class Precision-Recall curves, grouped by architecture family.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if no predictions data available.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = _collect_predictions(
        project_root, registry, cv_strategy,
        experiments_dir=experiments_dir,
    )
    if not all_preds:
        logger.warning(
            f"No predictions CSVs found for PR curves ({cv_strategy}). "
            "Run training with include_predictions_csv=True first."
        )
        return None

    n_classes = 4
    families = _group_by_family(list(all_preds.keys()))
    active_families = [f for f in FAMILY_ORDER if families.get(f)]

    if not active_families:
        return None

    n_cols = len(active_families)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5.5))
    if n_cols == 1:
        axes = [axes]

    for ax_idx, family in enumerate(active_families):
        ax = axes[ax_idx]
        family_models = families[family]

        for model_name in family_models:
            fold_dfs = all_preds.get(model_name, {})
            if not fold_dfs:
                continue

            combined = pd.concat(fold_dfs.values(), ignore_index=True)
            y_true = combined["ground_truth"].values
            mask = y_true < n_classes
            y_true = y_true[mask]
            prob_cols = [f"confidence_{i}" for i in range(n_classes)]
            y_score = combined.loc[mask, prob_cols].values

            row_sums = y_score.sum(axis=1, keepdims=True) + 1e-8
            y_score = y_score / row_sums

            y_bin = label_binarize(y_true, classes=list(range(n_classes)))

            # Macro-average PR
            recall_grid = np.linspace(0, 1, 200)
            mean_precision = np.zeros_like(recall_grid)
            for cls in range(n_classes):
                prec_cls, rec_cls, _ = precision_recall_curve(
                    y_bin[:, cls], y_score[:, cls],
                )
                # Interpolate (precision is monotonically decreasing for PR)
                mean_precision += np.interp(
                    recall_grid, rec_cls[::-1], prec_cls[::-1],
                )
            mean_precision /= n_classes
            macro_ap = average_precision_score(
                y_bin, y_score, average="macro",
            )

            color = MODEL_COLORS.get(model_name, "#999999")
            display = DISPLAY_NAMES.get(model_name, model_name)
            ax.plot(
                recall_grid, mean_precision,
                color=color, linewidth=2,
                label=f"{display} (AP={macro_ap:.3f})",
            )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(family, fontweight="bold")
        ax.legend(fontsize=9, loc="lower left")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Macro-Average Precision-Recall Curves -- {_cv_label(cv_strategy)}",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = output_dir / f"comparison_pr_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 4. Learning Curves -- Train/Val Loss over Epochs
# ===========================================================================

def plot_learning_curves(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Training and validation loss curves, grouped by architecture family.

    Requires TensorBoard event files in experiment directories.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if no TensorBoard data available.
    """
    from src.analysis.tensorboard_extractor import (
        extract_all_models_training_history,
    )

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_history = extract_all_models_training_history(
        project_root, registry, cv_strategy,
        metric_names=["train_loss_epoch", "val_loss"],
        experiments_dir=experiments_dir,
    )

    if not all_history:
        raise FileNotFoundError(
            f"No training history data found for learning curves ({cv_strategy}). "
            "Requires TensorBoard logs (tensorboard/fold_N/) or "
            "Lightning CSV logs (csv_logs/fold_N/metrics.csv) in experiment directories."
        )

    families = _group_by_family(list(all_history.keys()))
    active_families = [f for f in FAMILY_ORDER if families.get(f)]

    if not active_families:
        return None

    n_cols = len(active_families)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for ax_idx, family in enumerate(active_families):
        ax = axes[ax_idx]
        family_models = families[family]

        for model_name in family_models:
            fold_data = all_history.get(model_name, {})
            if not fold_data:
                continue

            color = MODEL_COLORS.get(model_name, "#999999")
            display = DISPLAY_NAMES.get(model_name, model_name)

            # Aggregate train_loss across folds
            _plot_metric_mean_band(
                ax, fold_data, "train_loss_epoch", color,
                linestyle="--", alpha_line=0.8,
            )
            # Aggregate val_loss across folds
            _plot_metric_mean_band(
                ax, fold_data, "val_loss", color,
                linestyle="-", alpha_line=1.0, label=display,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(family, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)

    # Add global legend for line styles
    fig.legend(
        [
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5),
            Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5),
        ],
        ["Train Loss", "Val Loss"],
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        f"Learning Curves -- {_cv_label(cv_strategy)}",
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = output_dir / f"comparison_learning_curves_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def _plot_metric_mean_band(
    ax: plt.Axes,
    fold_data: Dict[int, pd.DataFrame],
    metric: str,
    color: str,
    linestyle: str = "-",
    alpha_line: float = 1.0,
    label: Optional[str] = None,
) -> None:
    """Plot mean +/- std band for a metric across folds.

    Args:
        ax: Matplotlib axis.
        fold_data: {fold_idx: DataFrame with columns epoch, metric, value}.
        metric: Metric name to plot.
        color: Line color.
        linestyle: Line style.
        alpha_line: Line alpha.
        label: Legend label (only applied to the line, not the band).
    """
    fold_series: List[pd.Series] = []
    for _fold_idx, df in fold_data.items():
        metric_df = df[df["metric"] == metric].sort_values("epoch")
        if not metric_df.empty:
            fold_series.append(
                metric_df.set_index("epoch")["value"]
            )

    if not fold_series:
        return

    combined = pd.concat(fold_series, axis=1)
    mean_vals = combined.mean(axis=1)
    std_vals = combined.std(axis=1).fillna(0)

    epochs = mean_vals.index.values
    ax.plot(
        epochs, mean_vals.values,
        color=color, linestyle=linestyle, linewidth=1.8,
        alpha=alpha_line, label=label,
    )
    ax.fill_between(
        epochs,
        (mean_vals - std_vals).values,
        (mean_vals + std_vals).values,
        color=color, alpha=0.15,
    )


# ===========================================================================
# 5. Calibration Plot (Reliability Diagram)
# ===========================================================================

def plot_calibration_curves(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    n_bins: int = 10,
    experiments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Calibration reliability diagrams, grouped by architecture family.

    Shows how well predicted probabilities match actual frequencies.
    Diagonal = perfectly calibrated.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.
        n_bins: Number of bins for calibration curve.

    Returns:
        Path to saved PNG, or None if no predictions data available.
    """
    from sklearn.calibration import calibration_curve

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = _collect_predictions(
        project_root, registry, cv_strategy,
        experiments_dir=experiments_dir,
    )
    if not all_preds:
        logger.warning(
            f"No predictions CSVs found for calibration curves ({cv_strategy})."
        )
        return None

    families = _group_by_family(list(all_preds.keys()))
    active_families = [f for f in FAMILY_ORDER if families.get(f)]

    if not active_families:
        return None

    n_cols = len(active_families)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    for ax_idx, family in enumerate(active_families):
        ax = axes[ax_idx]
        family_models = families[family]

        for model_name in family_models:
            fold_dfs = all_preds.get(model_name, {})
            if not fold_dfs:
                continue

            combined = pd.concat(fold_dfs.values(), ignore_index=True)
            y_true = combined["ground_truth"].values
            y_pred = combined["prediction"].values

            # Use max confidence as the predicted probability
            n_classes = 4
            prob_cols = [f"confidence_{i}" for i in range(n_classes)]
            available_cols = [c for c in prob_cols if c in combined.columns]
            if not available_cols:
                continue

            probs = combined[available_cols].values
            mask = y_true < n_classes
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            probs_filtered = probs[mask]

            # Row-normalize
            row_sums = probs_filtered.sum(axis=1, keepdims=True) + 1e-8
            probs_norm = probs_filtered / row_sums

            # Calibration: predicted confidence vs actual correctness
            max_probs = probs_norm.max(axis=1)
            correct = (y_true_filtered == y_pred_filtered).astype(int)

            try:
                prob_true, prob_pred = calibration_curve(
                    correct, max_probs, n_bins=n_bins, strategy="uniform",
                )
            except ValueError:
                continue

            # ECE (Expected Calibration Error)
            ece = np.mean(np.abs(prob_true - prob_pred))

            color = MODEL_COLORS.get(model_name, "#999999")
            display = DISPLAY_NAMES.get(model_name, model_name)
            ax.plot(
                prob_pred, prob_true,
                color=color, linewidth=2, marker="o", markersize=4,
                label=f"{display} (ECE={ece:.3f})",
            )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1,
                label="Perfectly calibrated")
        ax.set_xlabel("Mean Predicted Confidence")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(family, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Calibration Reliability Diagrams -- {_cv_label(cv_strategy)}",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = output_dir / f"comparison_calibration_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 6. Critical Difference Diagram (Nemenyi Post-Hoc)
# ===========================================================================


def _find_cliques(
    sorted_ranks: np.ndarray, cd: float,
) -> List[Tuple[int, int]]:
    """Find maximal cliques for CD diagram (non-dominated intervals).

    A clique is a maximal interval [i, j] such that
    rank[j] - rank[i] < cd and no wider interval has the same property.

    Args:
        sorted_ranks: Average ranks sorted ascending.
        cd: Critical difference value.

    Returns:
        List of (start_idx, end_idx) tuples for each clique.
    """
    n = len(sorted_ranks)
    intervals: List[Tuple[int, int]] = []

    for i in range(n):
        j = i
        while j < n - 1 and sorted_ranks[j + 1] - sorted_ranks[i] < cd:
            j += 1
        if j > i:
            intervals.append((i, j))

    if not intervals:
        return []

    unique: List[Tuple[int, int]] = []
    for a, b in intervals:
        dominated = any(
            c <= a and b <= d and (c, d) != (a, b)
            for c, d in intervals
        )
        if not dominated:
            unique.append((a, b))

    return unique


def _draw_cd_diagram(
    sorted_names: List[str],
    sorted_ranks: np.ndarray,
    cd: float,
    n_models: int,
    p_value: float,
    cv_strategy: str,
) -> Tuple:
    """Draw a clean Demsar-style Critical Difference diagram.

    Args:
        sorted_names: Model names sorted by rank (best first).
        sorted_ranks: Corresponding average ranks.
        cd: Critical difference threshold.
        n_models: Number of models compared.
        p_value: Friedman test p-value.
        cv_strategy: CV strategy label string.

    Returns:
        Tuple of (fig, ax).
    """
    fig_w = max(10, n_models * 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5 + n_models * 0.25))

    rank_lo, rank_hi = 1, n_models
    margin = 0.8
    ax.set_xlim(rank_lo - margin, rank_hi + margin)
    ax.invert_xaxis()

    # Rank axis
    axis_y = 0.0
    ax.hlines(axis_y, rank_lo, rank_hi, colors="black", linewidth=1.5)
    for r in range(1, n_models + 1):
        ax.vlines(r, axis_y - 0.08, axis_y + 0.08, colors="black",
                  linewidth=1.0)
        ax.text(r, axis_y - 0.18, str(r), ha="center", va="top",
                fontsize=9, color="0.3")
    ax.set_xlabel("Average Rank")

    # CD reference bar
    cd_y = axis_y + 0.35
    mid = (rank_lo + rank_hi) / 2
    ax.hlines(cd_y, mid - cd / 2, mid + cd / 2, colors="black",
              linewidth=2.5)
    ax.vlines(mid - cd / 2, cd_y - 0.05, cd_y + 0.05,
              colors="black", linewidth=2)
    ax.vlines(mid + cd / 2, cd_y - 0.05, cd_y + 0.05,
              colors="black", linewidth=2)
    ax.text(mid, cd_y + 0.08, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=11, fontweight="bold")

    # Place model labels (alternating top/bottom halves)
    half = (n_models + 1) // 2
    top_names = sorted_names[:half]
    top_ranks = sorted_ranks[:half]
    bot_names = sorted_names[half:]
    bot_ranks = sorted_ranks[half:]

    top_base = axis_y + 0.7
    top_spacing = 0.28
    for i, (name, rank) in enumerate(zip(top_names, top_ranks)):
        y_label = top_base + i * top_spacing
        display = DISPLAY_NAMES.get(name, name)
        ax.vlines(rank, axis_y, y_label, colors="0.4", linewidth=0.7)
        ax.plot(rank, axis_y, "ko", markersize=4, zorder=5)
        ax.text(
            rank, y_label + 0.04,
            f"{display} ({rank:.2f})",
            ha="center", va="bottom", fontsize=8.5, fontweight="medium",
        )

    bot_base = axis_y - 0.55
    bot_spacing = 0.28
    for i, (name, rank) in enumerate(zip(bot_names, bot_ranks)):
        y_label = bot_base - i * bot_spacing
        display = DISPLAY_NAMES.get(name, name)
        ax.vlines(rank, y_label, axis_y, colors="0.4", linewidth=0.7)
        ax.plot(rank, axis_y, "ko", markersize=4, zorder=5)
        ax.text(
            rank, y_label - 0.04,
            f"{display} ({rank:.2f})",
            ha="center", va="top", fontsize=8.5, fontweight="medium",
        )

    # Clique bars (models not significantly different)
    cliques = _find_cliques(sorted_ranks, cd)
    clique_base = top_base + len(top_names) * top_spacing + 0.15
    clique_spacing = 0.18
    for ci, (start, end) in enumerate(cliques):
        y_bar = clique_base + ci * clique_spacing
        ax.hlines(
            y_bar, sorted_ranks[start], sorted_ranks[end],
            colors="#4ECDC4", linewidth=4, alpha=0.75,
        )

    # Finalize
    y_top = clique_base + len(cliques) * clique_spacing + 0.3
    y_bot = bot_base - len(bot_names) * bot_spacing - 0.1
    ax.set_ylim(y_bot, y_top)

    ax.set_title(
        f"Critical Difference Diagram -- {_cv_label(cv_strategy)}\n"
        f"Friedman p={p_value:.4f}",
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    plt.tight_layout()
    return fig, ax


def plot_critical_difference(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Critical Difference diagram comparing model rankings.

    Uses Friedman test followed by Nemenyi post-hoc test.
    Models connected by a horizontal bar are not significantly different.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if insufficient data.
    """
    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-fold QWK for all models
    model_names: List[str] = []
    fold_qwks: List[List[float]] = []

    for base_name, cv_data in _filter_registry(registry, cv_strategy):
        qwks = _get_fold_values(cv_data, "val_qwk")
        if len(qwks) < 2:
            continue
        model_names.append(base_name)
        fold_qwks.append(qwks)

    if len(model_names) < 3:
        logger.warning(
            f"Need at least 3 models for CD diagram, found {len(model_names)}"
        )
        return None

    # Align fold counts (pad shorter with NaN, use min overlap)
    min_folds = min(len(q) for q in fold_qwks)
    data_matrix = np.array([q[:min_folds] for q in fold_qwks]).T  # (folds, models)

    # Friedman test
    from scipy.stats import friedmanchisquare
    try:
        stat, p_value = friedmanchisquare(
            *[data_matrix[:, i] for i in range(data_matrix.shape[1])]
        )
    except Exception as e:
        logger.error(f"Friedman test failed: {e}")
        return None

    logger.info(f"Friedman test: chi2={stat:.4f}, p={p_value:.6f}")

    # Compute average ranks (lower rank = better)
    from scipy.stats import rankdata
    n_folds, n_models = data_matrix.shape
    ranks = np.zeros_like(data_matrix)
    for fold_idx in range(n_folds):
        # Rank descending: highest QWK gets rank 1
        ranks[fold_idx] = rankdata(-data_matrix[fold_idx])
    avg_ranks = ranks.mean(axis=0)

    # Sort models by average rank
    sort_idx = np.argsort(avg_ranks)
    sorted_names = [model_names[i] for i in sort_idx]
    sorted_ranks = avg_ranks[sort_idx]

    # Critical difference (Nemenyi)
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(0.95, n_models, np.inf)
    cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (12 * n_folds))

    # --- Draw CD diagram (Demsar 2006 style) ---
    fig, ax = _draw_cd_diagram(
        sorted_names, sorted_ranks, cd, n_models, p_value, cv_strategy,
    )

    path = output_dir / f"comparison_critical_difference_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 7. Per-Class F1 Boxplots
# ===========================================================================

def plot_per_class_f1_boxplots(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Per-class F1 boxplots showing which classes are hardest per model.

    Requires predictions CSVs. Computes F1 per fold per class.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if no predictions data available.
    """
    from sklearn.metrics import f1_score

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = _collect_predictions(
        project_root, registry, cv_strategy,
        experiments_dir=experiments_dir,
    )
    if not all_preds:
        logger.warning(
            f"No predictions CSVs found for per-class F1 ({cv_strategy})."
        )
        return None

    n_classes = 4
    class_names = ["Grade 0", "Grade 1", "Grade 2", "Grade 3"]

    # Compute per-fold, per-class F1 for each model
    # Structure: {model: {class_idx: [f1_fold_0, f1_fold_1, ...]}}
    model_class_f1: Dict[str, Dict[int, List[float]]] = {}

    for model_name, fold_dfs in all_preds.items():
        model_class_f1[model_name] = {c: [] for c in range(n_classes)}
        for _fold_idx, df in fold_dfs.items():
            y_true = df["ground_truth"].values
            y_pred = df["prediction"].values
            mask = y_true < n_classes
            y_true_f = y_true[mask]
            y_pred_f = y_pred[mask]

            per_class = f1_score(
                y_true_f, y_pred_f,
                labels=list(range(n_classes)),
                average=None, zero_division=0,
            )
            for c in range(n_classes):
                if c < len(per_class):
                    model_class_f1[model_name][c].append(per_class[c])

    if not model_class_f1:
        return None

    # Sort models by overall mean F1
    model_order = sorted(
        model_class_f1.keys(),
        key=lambda m: np.mean([
            np.mean(v) for v in model_class_f1[m].values() if v
        ]),
        reverse=True,
    )

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6),
                             sharey=True)

    for cls_idx, ax in enumerate(axes):
        data = []
        labels = []
        colors = []
        for model_name in model_order:
            f1_vals = model_class_f1[model_name].get(cls_idx, [])
            if f1_vals:
                data.append(f1_vals)
                labels.append(DISPLAY_NAMES.get(model_name, model_name))
                colors.append(MODEL_COLORS.get(model_name, "#999999"))

        if not data:
            ax.set_title(class_names[cls_idx])
            continue

        bp = ax.boxplot(
            data,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
            medianprops=dict(color="black", linewidth=1.5),
            widths=0.6,
        )

        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(colors[i])
            box.set_alpha(0.7)

        # Overlay fold points
        rng = np.random.default_rng(42)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(
                [i + 1 + j for j in jitter], vals,
                color="black", alpha=0.6, s=25, zorder=5,
            )

        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_title(class_names[cls_idx], fontweight="bold")
        ax.set_ylabel("F1 Score" if cls_idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([-0.05, 1.05])

    # Add boxplot legend to last subplot
    _add_boxplot_legend(axes[-1])

    fig.suptitle(
        f"Per-Class F1 Score Distribution -- {_cv_label(cv_strategy)}",
        fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / f"comparison_per_class_f1_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 8. Radar / Spider Chart -- Multi-Metric Comparison
# ===========================================================================

def plot_radar_chart(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    top_n: Optional[int] = None,
) -> Path:
    """Radar chart comparing models across multiple metrics.

    Axes: QWK, Accuracy, Test QWK, Consistency (1 - std_qwk), Generalization.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.
        top_n: If set, show only the top N models by mean QWK.

    Returns:
        Path to saved PNG.
    """
    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect multi-metric data
    records: List[Dict[str, Any]] = []
    for base_name, cv_data in _filter_registry(registry, cv_strategy):
        mean_qwk = cv_data.get("mean_qwk", 0)
        if mean_qwk == 0:
            continue
        std_qwk = cv_data.get("std_qwk", 0)
        mean_acc = cv_data.get("mean_acc", 0)
        test_qwk = cv_data.get("test_qwk", 0)

        # Generalization: 1 - abs(mean_qwk - test_qwk) -- higher = better
        gen_score = 1.0 - abs(mean_qwk - test_qwk) if test_qwk > 0 else 0.5
        # Consistency: 1 - normalized std -- higher = more consistent
        consistency = 1.0 - min(std_qwk * 10, 1.0)  # scale std to [0, 1]

        records.append({
            "model": base_name,
            "QWK": mean_qwk,
            "Accuracy": mean_acc,
            "Test QWK": test_qwk if test_qwk > 0 else mean_qwk,
            "Consistency": consistency,
            "Generalization": gen_score,
        })

    if not records:
        logger.warning("No models found for radar chart")
        return output_dir / f"comparison_radar_{cv_strategy}.png"

    # Sort by QWK and optionally limit
    records.sort(key=lambda r: r["QWK"], reverse=True)
    if top_n and len(records) > top_n:
        records = records[:top_n]

    metrics = ["QWK", "Accuracy", "Test QWK", "Consistency", "Generalization"]
    n_metrics = len(metrics)

    # Normalize each metric to [0, 1] across all models
    for metric in metrics:
        vals = [r[metric] for r in records]
        vmin, vmax = min(vals), max(vals)
        rng_val = vmax - vmin if vmax > vmin else 1.0
        for r in records:
            r[f"{metric}_norm"] = (r[metric] - vmin) / rng_val

    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for record in records:
        model_name = record["model"]
        values = [record[f"{m}_norm"] for m in metrics]
        values += values[:1]  # Close polygon

        color = MODEL_COLORS.get(model_name, "#999999")
        display = DISPLAY_NAMES.get(model_name, model_name)
        ax.plot(angles, values, "o-", linewidth=2, color=color,
                label=display, markersize=5)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(alpha=0.3)

    title_suffix = f" (Top {top_n})" if top_n else ""
    ax.set_title(
        f"Multi-Metric Radar Chart{title_suffix} -- {_cv_label(cv_strategy)}",
        fontweight="bold", y=1.08,
    )
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.3, 1.1),
        framealpha=0.9,
    )

    plt.tight_layout()
    path = output_dir / f"comparison_radar_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 9. Pairwise Significance Table (Wilcoxon + BH)
# ===========================================================================

def plot_pairwise_significance_table(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Heatmap of pairwise Wilcoxon signed-rank BH-corrected p-values.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if insufficient data.
    """
    import seaborn as sns

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names: List[str] = []
    fold_qwks: List[List[float]] = []

    for base_name, cv_data in _filter_registry(registry, cv_strategy):
        qwks = _get_fold_values(cv_data, "val_qwk")
        if len(qwks) >= 2:
            model_names.append(base_name)
            fold_qwks.append(qwks)

    if len(model_names) < 2:
        logger.warning("Need >= 2 models for significance table")
        return None

    # Sort by mean QWK descending
    means = [np.mean(q) for q in fold_qwks]
    sort_idx = np.argsort(means)[::-1]
    model_names = [model_names[i] for i in sort_idx]
    fold_qwks = [fold_qwks[i] for i in sort_idx]

    _, adj_df = _pairwise_wilcoxon_bh(model_names, fold_qwks)

    n = len(model_names)
    fig, ax = plt.subplots(
        figsize=(max(8, n * 0.9), max(7, n * 0.8)),
    )

    mask = np.eye(n, dtype=bool)

    sns.heatmap(
        adj_df, mask=mask, annot=True, fmt=".4f",
        cmap="RdYlGn_r", center=0.05,
        vmin=0, vmax=1, ax=ax,
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "BH-adjusted p-value"},
    )

    ax.set_title(
        f"Pairwise Wilcoxon Signed-Rank Tests (BH-FDR corrected)\n"
        f"{_cv_label(cv_strategy)}",
        fontweight="bold",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    path = output_dir / f"comparison_pairwise_significance_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 10. Cohen's d Effect Size Matrix
# ===========================================================================

def plot_cohens_d_matrix(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Heatmap of pairwise Cohen's d effect sizes for QWK.

    Positive value means row model outperforms column model.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if insufficient data.
    """
    import seaborn as sns

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names: List[str] = []
    fold_qwks: List[List[float]] = []

    for base_name, cv_data in _filter_registry(registry, cv_strategy):
        qwks = _get_fold_values(cv_data, "val_qwk")
        if len(qwks) >= 2:
            model_names.append(base_name)
            fold_qwks.append(qwks)

    if len(model_names) < 2:
        logger.warning("Need >= 2 models for Cohen's d matrix")
        return None

    # Sort by mean QWK descending
    means = [np.mean(q) for q in fold_qwks]
    sort_idx = np.argsort(means)[::-1]
    model_names = [model_names[i] for i in sort_idx]
    fold_qwks = [fold_qwks[i] for i in sort_idx]

    d_df = _compute_cohens_d_matrix(model_names, fold_qwks)

    n = len(model_names)
    fig, ax = plt.subplots(
        figsize=(max(8, n * 0.9), max(7, n * 0.8)),
    )

    mask = np.eye(n, dtype=bool)
    max_abs = max(abs(d_df.values.min()), abs(d_df.values.max()), 0.1)

    sns.heatmap(
        d_df, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu", center=0,
        vmin=-max_abs, vmax=max_abs, ax=ax,
        linewidths=0.5, linecolor="white",
        annot_kws={"fontsize": 9},
        cbar_kws={"label": "Cohen's d (positive = row better)"},
    )

    ax.set_title(
        f"Pairwise Cohen's d Effect Sizes (QWK)\n"
        f"{_cv_label(cv_strategy)}",
        fontweight="bold",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    path = output_dir / f"comparison_cohens_d_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 11. Confusion Matrix Comparison
# ===========================================================================

def plot_confusion_matrix_comparison(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Normalized confusion matrices for all models side-by-side.

    Each subplot shows one model's confusion matrix (row-normalized to show
    per-class recall). Models are grouped by architecture family.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.
        experiments_dir: Override directory for experiment data.

    Returns:
        Path to saved PNG, or None if no data.
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = _collect_predictions(
        project_root, registry, cv_strategy,
        experiments_dir=experiments_dir,
    )

    if not all_preds:
        logger.warning(
            f"No predictions CSVs found for confusion matrices ({cv_strategy})."
        )
        return None

    class_names = ["Grade 0", "Grade 1", "Grade 2", "Grade 3"]
    n_classes = 4
    model_names = sorted(all_preds.keys())
    # +1 for the reference paper baseline panel
    n_total = len(model_names) + 1
    n_models = len(model_names)

    n_cols = min(4, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
    )
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, model_name in enumerate(model_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        fold_dfs = all_preds[model_name]

        y_true_all = []
        y_pred_all = []
        for df in fold_dfs.values():
            gt = df["ground_truth"].values
            pred = df["prediction"].values
            mask = (gt < n_classes) & (pred < n_classes)
            y_true_all.extend(gt[mask])
            y_pred_all.extend(pred[mask])

        if not y_true_all:
            ax.set_visible(False)
            continue

        cm = sk_confusion_matrix(
            y_true_all, y_pred_all, labels=list(range(n_classes)),
        )
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, vmin=0, vmax=1, cbar=False,
            annot_kws={"fontsize": 9},
        )
        display = DISPLAY_NAMES.get(model_name, model_name)
        ax.set_title(display, fontweight="bold", fontsize=11)
        ax.set_ylabel("True" if col == 0 else "")
        ax.set_xlabel("Predicted")

    # Reference paper baseline panel (last model panel)
    ref_idx = n_models
    ref_row, ref_col = divmod(ref_idx, n_cols)
    ref_ax = axes[ref_row, ref_col]
    ref_cm = np.array(REFERENCE_CM_NO_IGNORE)
    sns.heatmap(
        ref_cm, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=class_names, yticklabels=class_names,
        ax=ref_ax, vmin=0, vmax=1, cbar=False,
        annot_kws={"fontsize": 9},
    )
    ref_ax.set_title(
        REFERENCE_PAPER_LABEL, fontweight="bold", fontsize=10,
        color="#E67E22",
    )
    ref_ax.set_ylabel("True" if ref_col == 0 else "")
    ref_ax.set_xlabel("Predicted")

    for idx in range(n_total, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Normalized Confusion Matrices -- {_cv_label(cv_strategy)}",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / f"comparison_confusion_matrix_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 12. Ensemble vs Best-Single-Fold Comparison
# ===========================================================================

def plot_ensemble_vs_best_single(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Grouped bar chart: ensemble test QWK vs best single-fold test QWK.

    Shows the performance gain (or loss) from ensembling all folds
    compared to using only the best individual fold.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if insufficient data.
    """
    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names: List[str] = []
    ensemble_qwks: List[float] = []
    best_fold_qwks: List[float] = []

    for base_name, cv_data in _filter_registry(registry, cv_strategy):
        test_qwk = cv_data.get("test_qwk")
        fold_models = cv_data.get("fold_models", {})
        if test_qwk is None or not fold_models:
            continue

        fold_qwks = [
            fm.get("val_qwk", 0.0) for fm in fold_models.values()
        ]
        best_val = max(fold_qwks) if fold_qwks else 0.0

        model_names.append(base_name)
        ensemble_qwks.append(test_qwk)
        best_fold_qwks.append(best_val)

    if len(model_names) < 2:
        logger.warning("Need >= 2 models for ensemble comparison")
        return None

    sort_idx = np.argsort(ensemble_qwks)[::-1]
    model_names = [model_names[i] for i in sort_idx]
    ensemble_qwks = [ensemble_qwks[i] for i in sort_idx]
    best_fold_qwks = [best_fold_qwks[i] for i in sort_idx]

    display_names = [DISPLAY_NAMES.get(m, m) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.2), 6))
    bars_ens = ax.bar(
        x - width / 2, ensemble_qwks, width,
        label="Ensemble (all folds)", color="#2CA58D", edgecolor="white",
    )
    bars_single = ax.bar(
        x + width / 2, best_fold_qwks, width,
        label="Best Single Fold (val QWK)", color="#F38181",
        edgecolor="white",
    )

    for bar in bars_ens:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
        )
    for bar in bars_single:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Test QWK")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    y_min = min(min(ensemble_qwks), min(best_fold_qwks))
    ax.set_ylim(max(0, y_min - 0.1), 1.0)

    fig.suptitle(
        f"Ensemble vs Best Single Fold -- {_cv_label(cv_strategy)}",
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / f"comparison_ensemble_vs_single_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 13. Stratified vs LOAO Scatter Plot
# ===========================================================================

def plot_stratified_vs_loao_scatter(
    project_root: Path,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Scatter plot comparing stratified vs LOAO test QWK per model.

    Each point is one model. The diagonal line represents equal
    performance. Points below the diagonal indicate LOAO
    underperformance relative to stratified CV.

    Args:
        project_root: Project root directory.
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if insufficient data.
    """
    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    stratified_map: Dict[str, float] = {}
    loao_map: Dict[str, float] = {}

    for base_name, cv_data in _filter_registry(registry, "random_stratified"):
        test_qwk = cv_data.get("test_qwk")
        if test_qwk is not None:
            stratified_map[base_name] = test_qwk

    for base_name, cv_data in _filter_registry(registry, "loao_balanced"):
        test_qwk = cv_data.get("test_qwk")
        if test_qwk is not None:
            loao_map[base_name] = test_qwk

    common = sorted(set(stratified_map) & set(loao_map))
    if len(common) < 2:
        logger.warning("Need >= 2 models with both strategies for scatter")
        return None

    fig, ax = plt.subplots(figsize=(8, 8))

    min_val = 1.0
    max_val = 0.0

    for model_name in common:
        s_qwk = stratified_map[model_name]
        l_qwk = loao_map[model_name]
        color = MODEL_COLORS.get(model_name, "#999999")
        display = DISPLAY_NAMES.get(model_name, model_name)
        family = ARCHITECTURE_FAMILIES.get(model_name, "Other")
        marker = {"CNN": "o", "Transformer": "s", "Hybrid": "D",
                  "Self-Supervised": "^"}.get(family, "o")

        ax.scatter(
            s_qwk, l_qwk, c=color, s=120, marker=marker,
            edgecolors="black", linewidths=0.8, zorder=5,
        )
        ax.annotate(
            display, (s_qwk, l_qwk),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9,
        )
        min_val = min(min_val, s_qwk, l_qwk)
        max_val = max(max_val, s_qwk, l_qwk)

    pad = 0.03
    lim_min = max(0, min_val - pad)
    lim_max = min(1.0, max_val + pad)
    ax.plot(
        [lim_min, lim_max], [lim_min, lim_max],
        "k--", alpha=0.4, label="Equal performance",
    )
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect("equal")
    ax.set_xlabel("Stratified 5-Fold Test QWK")
    ax.set_ylabel("LOAO 2-Fold Test QWK")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    family_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="grey",
               markersize=8, label=f)
        for f, m in [("CNN", "o"), ("Transformer", "s"),
                     ("Hybrid", "D"), ("Self-Supervised", "^")]
    ]
    ax.legend(handles=family_handles + [
        Line2D([0], [0], color="black", linestyle="--", alpha=0.4,
               label="Equal performance"),
    ], loc="lower right", fontsize=9)

    fig.suptitle(
        "Stratified vs LOAO -- Test QWK Comparison",
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "comparison_stratified_vs_loao.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 14. Continuous Inflammation Score Distribution
# ===========================================================================

def plot_continuous_score_distribution(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Histogram / KDE of continuous inflammation scores per model.

    Continuous score = softmax-weighted sum over classes 0-3 (ignoring
    the artifact class). Shows how each model distributes its predictions
    on the 0-3 continuous scale, overlaid with ground truth distribution.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.
        experiments_dir: Override directory for experiment data.

    Returns:
        Path to saved PNG, or None if no data.
    """
    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = _collect_predictions(
        project_root, registry, cv_strategy,
        experiments_dir=experiments_dir,
    )

    if not all_preds:
        logger.warning(
            f"No predictions CSVs for continuous score dist ({cv_strategy})."
        )
        return None

    model_names = sorted(all_preds.keys())
    n_models = len(model_names)
    n_cols = min(4, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
    )
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    class_indices = [0, 1, 2, 3]

    for idx, model_name in enumerate(model_names):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        fold_dfs = all_preds[model_name]

        all_scores = []
        all_gt = []
        for df in fold_dfs.values():
            conf_cols = [c for c in df.columns if c.startswith("confidence_")]
            if len(conf_cols) < 4:
                continue
            confs = df[[f"confidence_{i}" for i in class_indices]].values
            confs_norm = confs / confs.sum(axis=1, keepdims=True)
            scores = confs_norm @ np.array(class_indices, dtype=float)
            all_scores.extend(scores)

            gt = df["ground_truth"].values
            mask = gt < 4
            all_gt.extend(gt[mask].astype(float))

        if not all_scores:
            ax.set_visible(False)
            continue

        color = MODEL_COLORS.get(model_name, "#999999")
        display = DISPLAY_NAMES.get(model_name, model_name)

        ax.hist(
            all_scores, bins=30, range=(0, 3), alpha=0.7,
            color=color, edgecolor="white", density=True,
            label="Predicted",
        )
        if all_gt:
            ax.hist(
                all_gt, bins=30, range=(0, 3), alpha=0.3,
                color="grey", edgecolor="white", density=True,
                label="Ground Truth",
            )
        ax.set_title(display, fontweight="bold", fontsize=11)
        ax.set_xlabel("Continuous Score")
        ax.set_ylabel("Density" if col == 0 else "")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 3)

    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Continuous Inflammation Score Distribution -- "
        f"{_cv_label(cv_strategy)}",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / f"comparison_continuous_score_dist_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# Orchestrator
# ===========================================================================

# ===========================================================================
# 15. QWK Bar Chart with Error Bars (Supervisor Mandatory: req. #4)
# ===========================================================================

def plot_qwk_bar_with_errorbars(
    project_root: Path,
    cv_strategy: str = "loao_balanced",
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Bar chart of mean QWK per model with +/-1 std error bars across folds.

    Primary comparative visualization for the thesis Results chapter.
    Satisfies supervisor mandatory requirement #4 (bar plots with std for
    LOAO folds). Models are sorted by mean QWK descending and colored by
    architecture family.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy to plot ('loao_balanced' or
            'random_stratified').
        output_dir: Directory to save figure.

    Returns:
        Path to saved PNG, or None if no registry data available.
    """
    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    colors: List[str] = []

    for base_name, cv_data in _filter_registry(registry, cv_strategy):
        qwks = _get_fold_values(cv_data, "val_qwk")
        if not qwks:
            continue
        family = ARCHITECTURE_FAMILIES.get(base_name, "Other")
        model_names.append(base_name)
        means.append(float(np.mean(qwks)))
        stds.append(float(np.std(qwks)))
        colors.append(FAMILY_COLORS.get(family, "#999999"))

    if not model_names:
        logger.warning(f"No models found for QWK bar chart ({cv_strategy})")
        return None

    sorted_idx = np.argsort(means)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    display_labels = [DISPLAY_NAMES.get(m, m) for m in model_names]

    n = len(model_names)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), 6))
    bars = ax.bar(
        x, means,
        color=colors, alpha=0.85,
        edgecolor="black", linewidth=0.6,
        yerr=stds, capsize=5,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "black"},
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean + std + 0.015,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean QWK")
    y_min = max(0.0, min(m - s for m, s in zip(means, stds)) - 0.05)
    y_min = round(y_min * 10) / 10
    ax.set_ylim([y_min, 1.05])
    ax.grid(axis="y", alpha=0.3)

    family_handles = [
        mpatches.Patch(
            facecolor=FAMILY_COLORS[f], edgecolor="black",
            linewidth=0.5, alpha=0.85, label=f,
        )
        for f in FAMILY_ORDER
        if any(ARCHITECTURE_FAMILIES.get(m) == f for m in model_names)
    ]
    ax.legend(
        handles=family_handles, title="Architecture Family",
        loc="upper right", fontsize=10,
    )

    fig.suptitle(
        f"QWK per Model -- Mean +/- Std ({_cv_label(cv_strategy)})",
        fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / f"comparison_qwk_bar_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ===========================================================================
# 16. Per-Class F1 Heatmap (Supervisor Mandatory: MLv13 ML-17)
# ===========================================================================

def plot_per_class_f1_heatmap(
    project_root: Path,
    cv_strategy: str = "loao_balanced",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Heatmap of mean per-class F1 score for all models.

    Rows: models sorted by overall mean F1 descending.
    Columns: Grade 0, Grade 1, Grade 2, Grade 3 (Ignore excluded).
    Cell value: mean F1 across LOAO folds.

    Satisfies MLv13 ML-17 (per-class F1 must be reported) and reveals
    which models fail at specific inflammation grades.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy filter.
        output_dir: Directory to save figure.
        experiments_dir: Override path for experiment run directories.

    Returns:
        Path to saved PNG, or None if no predictions data available.
    """
    from sklearn.metrics import f1_score as sk_f1_score

    _setup_style()
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = _collect_predictions(
        project_root, registry, cv_strategy,
        experiments_dir=experiments_dir,
    )
    if not all_preds:
        logger.warning(
            f"No predictions CSVs for per-class F1 heatmap ({cv_strategy})."
        )
        return None

    n_classes = 4
    class_labels = ["Grade 0", "Grade 1", "Grade 2", "Grade 3"]

    model_order: List[str] = []
    heatmap_rows: List[List[float]] = []

    for model_name, fold_dfs in all_preds.items():
        per_class_accumulator: List[List[float]] = [[] for _ in range(n_classes)]
        for df in fold_dfs.values():
            y_true = df["ground_truth"].values
            y_pred = df["prediction"].values
            mask = y_true < n_classes
            f1_vals = sk_f1_score(
                y_true[mask], y_pred[mask],
                labels=list(range(n_classes)),
                average=None, zero_division=0,
            )
            for c in range(n_classes):
                if c < len(f1_vals):
                    per_class_accumulator[c].append(float(f1_vals[c]))

        mean_row = [
            float(np.mean(vals)) if vals else 0.0
            for vals in per_class_accumulator
        ]
        model_order.append(model_name)
        heatmap_rows.append(mean_row)

    if not model_order:
        return None

    overall_means = [float(np.mean(row)) for row in heatmap_rows]
    sorted_idx = list(np.argsort(overall_means)[::-1])
    model_order = [model_order[i] for i in sorted_idx]
    heatmap_rows = [heatmap_rows[i] for i in sorted_idx]
    display_labels = [DISPLAY_NAMES.get(m, m) for m in model_order]

    matrix = np.array(heatmap_rows)
    n_models = len(model_order)

    fig, ax = plt.subplots(figsize=(7, max(4, n_models * 0.55 + 1.5)))

    try:
        import seaborn as sns
        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=class_labels,
            yticklabels=display_labels,
            vmin=0.0, vmax=1.0,
            cmap="YlOrRd",
            annot=True, fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Mean F1 Score"},
        )
    except ImportError:
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_labels)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(display_labels)
        for i in range(n_models):
            for j in range(n_classes):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                )
        plt.colorbar(im, ax=ax, label="Mean F1 Score")

    ax.set_title(
        f"Per-Class Mean F1 Score -- {_cv_label(cv_strategy)}",
        fontweight="bold",
    )
    plt.tight_layout()
    path = output_dir / f"comparison_per_class_f1_heatmap_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


def generate_all_comparison_plots(
    project_root: Path,
    output_dir: Optional[Path] = None,
    cv_strategies: Optional[List[str]] = None,
    experiments_dir: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    """Generate all comparison plots for all CV strategies.

    Args:
        project_root: Project root directory.
        output_dir: Directory to save all figures.
        cv_strategies: List of CV strategies to generate plots for.
            Defaults to both 'random_stratified' and 'loao_balanced'.
        experiments_dir: Override directory containing experiment runs.
            Defaults to project_root / 'experiments'.

    Returns:
        Dict mapping plot type to list of saved file paths.
    """
    if cv_strategies is None:
        cv_strategies = ["random_stratified", "loao_balanced"]

    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Preflight: check TensorBoard data availability (non-blocking) ------
    from src.analysis.tensorboard_extractor import (
        extract_all_models_training_history,
    )

    registry = _load_registry(project_root)
    tb_available: bool = True
    for cv_strat in cv_strategies:
        tb_history = extract_all_models_training_history(
            project_root, registry, cv_strat,
            metric_names=["train_loss_epoch", "val_loss"],
            experiments_dir=experiments_dir,
        )
        if not tb_history:
            logger.warning(
                f"No training history data found for learning curves "
                f"({cv_strat}). Learning curves will be skipped. "
                f"Requires TensorBoard logs (tensorboard/fold_N/) or "
                f"Lightning CSV logs (csv_logs/fold_N/metrics.csv)."
            )
            tb_available = False
        else:
            logger.info(
                f"Preflight OK: TensorBoard data found for {cv_strat} "
                f"({len(tb_history)} models)"
            )

    results: Dict[str, List[Path]] = {}

    plot_functions = [
        ("boxplots", plot_metric_boxplots),
        ("roc", plot_roc_curves),
        ("pr", plot_pr_curves),
        ("learning_curves", plot_learning_curves),
        ("calibration", plot_calibration_curves),
        ("critical_difference", plot_critical_difference),
        ("per_class_f1", plot_per_class_f1_boxplots),
        ("radar", plot_radar_chart),
        ("pairwise_significance", plot_pairwise_significance_table),
        ("cohens_d", plot_cohens_d_matrix),
        ("confusion_matrix", plot_confusion_matrix_comparison),
        ("ensemble_vs_single", plot_ensemble_vs_best_single),
        ("continuous_score_dist", plot_continuous_score_distribution),
        ("qwk_bar", plot_qwk_bar_with_errorbars),
        ("per_class_f1_heatmap", plot_per_class_f1_heatmap),
    ]

    # Plot types that read experiment files (predictions CSVs / TensorBoard)
    needs_experiments_dir = {
        "roc", "pr", "learning_curves", "calibration", "per_class_f1",
        "confusion_matrix", "continuous_score_dist", "per_class_f1_heatmap",
    }

    # Plots that require >= 3 folds (skip for LOAO 2-fold)
    needs_min_3_folds: set = {
        "critical_difference", "pairwise_significance", "cohens_d",
    }

    for plot_name, plot_fn in plot_functions:
        paths: List[Path] = []
        if not tb_available and plot_name == "learning_curves":
            logger.warning(
                "Skipping learning_curves (no TensorBoard data available)"
            )
            results[plot_name] = []
            continue
        for cv_strategy in cv_strategies:
            if (
                cv_strategy == "loao_balanced"
                and plot_name in needs_min_3_folds
            ):
                logger.info(
                    f"Skipping {plot_name} for {cv_strategy} "
                    "(requires >= 3 folds)"
                )
                continue
            logger.info(f"Generating {plot_name} for {cv_strategy}...")
            try:
                kwargs: Dict[str, Any] = {
                    "project_root": project_root,
                    "cv_strategy": cv_strategy,
                    "output_dir": output_dir,
                }
                if plot_name in needs_experiments_dir:
                    kwargs["experiments_dir"] = experiments_dir
                path = plot_fn(**kwargs)
                if path is not None:
                    paths.append(path)
                    logger.info(f"  -> {path.name}")
                else:
                    logger.info(
                        f"  -> Skipped (missing data for {plot_name})"
                    )
            except Exception as e:
                logger.error(f"Failed to generate {plot_name} ({cv_strategy}): {e}")
        results[plot_name] = paths

    # Stratified vs LOAO scatter (cross-strategy, only once)
    logger.info("Generating stratified_vs_loao scatter...")
    try:
        path = plot_stratified_vs_loao_scatter(
            project_root=project_root,
            output_dir=output_dir,
        )
        results["stratified_vs_loao"] = [path] if path else []
        if path:
            logger.info(f"  -> {path.name}")
    except Exception as e:
        logger.error(f"Failed to generate stratified_vs_loao: {e}")
        results["stratified_vs_loao"] = []

    # Spatial inflammation maps (per-animal, best model per family)
    from src.analysis.spatial_inflammation_map import (
        plot_spatial_inflammation_maps,
        plot_continuous_inflammation_heatmaps,
    )

    for cv_strategy in cv_strategies:
        logger.info(
            f"Generating spatial inflammation maps for {cv_strategy}..."
        )
        try:
            spatial_paths = plot_spatial_inflammation_maps(
                project_root=project_root,
                cv_strategy=cv_strategy,
                output_dir=output_dir,
                experiments_dir=experiments_dir,
            )
            results.setdefault("spatial_inflammation_map", []).extend(
                spatial_paths
            )
            for p in spatial_paths:
                logger.info(f"  -> {p.name}")
        except Exception as e:
            logger.error(
                f"Failed to generate spatial maps ({cv_strategy}): {e}"
            )

        # Continuous inflammation score heatmaps (Fig 9-style)
        logger.info(
            f"Generating continuous heatmaps for {cv_strategy}..."
        )
        try:
            cont_paths = plot_continuous_inflammation_heatmaps(
                project_root=project_root,
                cv_strategy=cv_strategy,
                output_dir=output_dir,
                experiments_dir=experiments_dir,
            )
            results.setdefault("continuous_inflammation", []).extend(
                cont_paths
            )
            for p in cont_paths:
                logger.info(f"  -> {p.name}")
        except Exception as e:
            logger.error(
                f"Failed to generate continuous heatmaps ({cv_strategy}): {e}"
            )

    # Summary
    total = sum(len(p) for p in results.values())
    logger.info(f"\nGenerated {total} comparison plots in {output_dir}")
    for plot_name, paths in results.items():
        status = ", ".join(p.name for p in paths) if paths else "SKIPPED"
        logger.info(f"  {plot_name}: {status}")

    return results
