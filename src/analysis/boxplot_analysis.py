"""
Boxplot Analysis Module for Cross-Model Comparison.

Generates publication-quality boxplots from the Best Models Registry.
All plots read directly from best_models_registry.json -- no experiment
run directories or fold metric files required.

Boxplots:
    1. QWK per model (Stratified 5-fold)
    2. Architecture family QWK comparison
    3. LOAO vs. Stratified CV strategy comparison
    4. Per-class difficulty (F1 per inflammation class)
    5. Generalization gap per model (CV-QWK vs. Test-QWK)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# -- Architecture family mapping -----------------------------------------------
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

# -- Pretty display names -------------------------------------------------------
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


def _load_registry(project_root: Path) -> Dict:
    """Load best_models_registry_new.json from the project."""
    for registry_path in [
        project_root / "src" / "experiments" / "best_models_registry_new.json",
        project_root / "experiments" / "best_models_registry_new.json",
    ]:
        if registry_path.exists():
            with open(registry_path, "r") as f:
                return json.load(f)
    raise FileNotFoundError("best_models_registry_new.json not found")


def _extract_model_name(registry_key: str) -> str:
    """Extract base model name from registry key like 'densenet_stratified'."""
    for suffix in ("_stratified", "_loao"):
        if registry_key.endswith(suffix):
            return registry_key[: -len(suffix)]
    return registry_key


def _get_fold_qwks(entry: Dict) -> List[float]:
    """Extract per-fold val_qwk values from a registry entry."""
    cv_data = list(entry.values())[0]
    fold_models = cv_data.get("fold_models", {})
    return [fm["val_qwk"] for fm in fold_models.values() if fm.get("val_qwk", 0) > 0]


def _get_fold_accs(entry: Dict) -> List[float]:
    """Extract per-fold val_acc values from a registry entry."""
    cv_data = list(entry.values())[0]
    fold_models = cv_data.get("fold_models", {})
    return [fm["val_acc"] for fm in fold_models.values() if fm.get("val_acc", 0) > 0]


def _add_boxplot_legend(ax: plt.Axes) -> None:
    """Add standard mean/median/fold-point legend to a boxplot axis."""
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
               markersize=6, label="Mean"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=5, alpha=0.6, label="Fold values"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)


# ==============================================================================
# 1. QWK per Model -- Stratified (5 Folds)
# ==============================================================================

def plot_qwk_per_model_stratified(
    project_root: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Boxplot of per-fold QWK for each model under Stratified 5-fold CV.

    Args:
        project_root: Project root directory.
        output_dir: Directory to save the figure (default: project_root/figures).

    Returns:
        Path to the saved PNG file.
    """
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names: List[str] = []
    qwk_data: List[List[float]] = []

    for key, entry in registry.items():
        if not key.endswith("_stratified"):
            continue
        cv_data = list(entry.values())[0]
        if cv_data.get("cv_strategy") != "random_stratified":
            continue
        qwks = _get_fold_qwks(entry)
        if not qwks:
            continue
        base_name = _extract_model_name(key)
        model_names.append(DISPLAY_NAMES.get(base_name, base_name))
        qwk_data.append(qwks)

    if not model_names:
        logger.warning("No stratified models found in registry")
        return output_dir / "boxplot_qwk_per_model_stratified.png"

    # Sort by median descending
    medians = [np.median(q) for q in qwk_data]
    sorted_idx = np.argsort(medians)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    qwk_data = [qwk_data[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(model_names) * 1.2), 6))

    bp = ax.boxplot(
        qwk_data,
        labels=model_names,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=7),
        medianprops=dict(color="black", linewidth=1.5),
        widths=0.55,
    )

    # Gradient colouring (green=best -> red=worst)
    cmap = plt.cm.RdYlGn
    n = len(model_names)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(cmap(1.0 - i / max(n - 1, 1)))
        box.set_alpha(0.7)

    # Overlay individual fold points
    rng = np.random.default_rng(42)
    for i, scores in enumerate(qwk_data):
        jitter = rng.uniform(-0.15, 0.15, len(scores))
        ax.scatter(
            [i + 1 + j for j in jitter], scores,
            color="black", alpha=0.6, s=35, zorder=5,
        )

    ax.set_ylabel("Quadratic Weighted Kappa (QWK)", fontsize=12)
    ax.set_title(
        "QWK Distribution per Model -- Stratified 5-Fold CV",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.03))
    _add_boxplot_legend(ax)

    plt.tight_layout()
    path = output_dir / "boxplot_qwk_per_model_stratified.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ==============================================================================
# 2. Architecture Family QWK Comparison
# ==============================================================================

def plot_qwk_per_family(
    project_root: Path,
    output_dir: Optional[Path] = None,
    cv_strategy: str = "random_stratified",
) -> Path:
    """Boxplot of mean QWK grouped by architecture family.

    Each data point is one model's mean_qwk within the chosen CV strategy.

    Args:
        project_root: Project root directory.
        output_dir: Directory to save the figure.
        cv_strategy: Which CV strategy to use ('random_stratified' or 'loao_balanced').

    Returns:
        Path to the saved PNG file.
    """
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_stratified" if cv_strategy == "random_stratified" else "_loao"
    family_qwks: Dict[str, List[float]] = {f: [] for f in FAMILY_ORDER}
    family_models: Dict[str, List[str]] = {f: [] for f in FAMILY_ORDER}

    for key, entry in registry.items():
        if not key.endswith(suffix):
            continue
        cv_data = list(entry.values())[0]
        if cv_data.get("cv_strategy") != cv_strategy:
            continue
        mean_qwk = cv_data.get("mean_qwk", 0)
        if mean_qwk == 0:
            continue

        base_name = _extract_model_name(key)
        family = ARCHITECTURE_FAMILIES.get(base_name, "Other")
        if family not in family_qwks:
            continue
        family_qwks[family].append(mean_qwk)
        family_models[family].append(DISPLAY_NAMES.get(base_name, base_name))

    families_with_data = [f for f in FAMILY_ORDER if family_qwks[f]]
    if not families_with_data:
        logger.warning("No family data found")
        return output_dir / "boxplot_qwk_per_family.png"

    box_data = [family_qwks[f] for f in families_with_data]
    labels = [f"{f}\n(n={len(family_qwks[f])})" for f in families_with_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        box_data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
        medianprops=dict(color="black", linewidth=2),
        widths=0.5,
    )

    for i, fam in enumerate(families_with_data):
        bp["boxes"][i].set_facecolor(FAMILY_COLORS.get(fam, "#CCCCCC"))
        bp["boxes"][i].set_alpha(0.7)

    # Individual model points with labels
    rng = np.random.default_rng(42)
    for i, fam in enumerate(families_with_data):
        scores = family_qwks[fam]
        names = family_models[fam]
        jitter = rng.uniform(-0.15, 0.15, len(scores))
        ax.scatter(
            [i + 1 + j for j in jitter], scores,
            color="black", alpha=0.7, s=50, zorder=5,
        )
        for j, (score, name) in enumerate(zip(scores, names)):
            ax.annotate(
                name,
                xy=(i + 1 + jitter[j], score),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7.5, alpha=0.85,
            )

    cv_label = "Stratified 5-Fold" if cv_strategy == "random_stratified" else "LOAO 2-Fold"
    ax.set_ylabel("Mean QWK", fontsize=12)
    ax.set_title(
        f"QWK by Architecture Family -- {cv_label} CV",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.05))

    # Legend
    family_legend = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=FAMILY_COLORS[f], markersize=10, label=f)
        for f in families_with_data
    ]
    family_legend.append(
        Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
               markersize=7, label="Mean")
    )
    ax.legend(handles=family_legend, loc="lower left", fontsize=9)

    plt.tight_layout()
    path = output_dir / f"boxplot_qwk_per_family_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ==============================================================================
# 3. LOAO vs. Stratified CV Strategy Comparison
# ==============================================================================

def plot_cv_strategy_comparison(
    project_root: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Boxplot comparing mean QWK distributions between LOAO and Stratified CV.

    Each data point is one model's mean_qwk under the respective strategy.

    Args:
        project_root: Project root directory.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG file.
    """
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies: Dict[str, List[float]] = {"Stratified\n(5-Fold)": [], "LOAO\n(2-Fold)": []}
    model_labels: Dict[str, List[str]] = {"Stratified\n(5-Fold)": [], "LOAO\n(2-Fold)": []}

    for key, entry in registry.items():
        cv_data = list(entry.values())[0]
        mean_qwk = cv_data.get("mean_qwk", 0)
        if mean_qwk == 0:
            continue
        base_name = _extract_model_name(key)
        display = DISPLAY_NAMES.get(base_name, base_name)

        cv_strat = cv_data.get("cv_strategy", "")
        if cv_strat == "random_stratified":
            strategies["Stratified\n(5-Fold)"].append(mean_qwk)
            model_labels["Stratified\n(5-Fold)"].append(display)
        elif cv_strat == "loao_balanced":
            strategies["LOAO\n(2-Fold)"].append(mean_qwk)
            model_labels["LOAO\n(2-Fold)"].append(display)

    strat_keys = [k for k in strategies if strategies[k]]
    if len(strat_keys) < 2:
        logger.warning("Need both CV strategies for comparison")
        return output_dir / "boxplot_cv_strategy_comparison.png"

    box_data = [strategies[k] for k in strat_keys]
    labels = [f"{k}\n(n={len(strategies[k])})" for k in strat_keys]

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(
        box_data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
        medianprops=dict(color="black", linewidth=2),
        widths=0.45,
    )

    colors = ["#4ECDC4", "#F38181"]
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)

    # Individual model points with labels
    rng = np.random.default_rng(42)
    for i, k in enumerate(strat_keys):
        scores = strategies[k]
        names = model_labels[k]
        jitter = rng.uniform(-0.12, 0.12, len(scores))
        ax.scatter(
            [i + 1 + j for j in jitter], scores,
            color="black", alpha=0.7, s=40, zorder=5,
        )
        for j, (score, name) in enumerate(zip(scores, names)):
            ax.annotate(
                name,
                xy=(i + 1 + jitter[j], score),
                xytext=(8, 0), textcoords="offset points",
                fontsize=7.5, alpha=0.85,
            )

    # Mean difference annotation
    mean_strat = np.mean(strategies["Stratified\n(5-Fold)"])
    mean_loao = np.mean(strategies["LOAO\n(2-Fold)"])
    delta = mean_strat - mean_loao
    ax.text(
        0.98, 0.02,
        f"Mean difference: {delta:.4f}\n"
        f"Stratified: {mean_strat:.4f}  |  LOAO: {mean_loao:.4f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, style="italic", alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_ylabel("Mean QWK", fontsize=12)
    ax.set_title(
        "CV Strategy Comparison: Stratified vs. LOAO",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.05))

    plt.tight_layout()
    path = output_dir / "boxplot_cv_strategy_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ==============================================================================
# 4. Per-Class F1 Difficulty Analysis
# ==============================================================================

def plot_per_class_f1_from_registry(
    project_root: Path,
    output_dir: Optional[Path] = None,
    cv_strategy: str = "random_stratified",
) -> Path:
    """Boxplot of per-class F1 from fold-level accuracy proxy.

    Since the registry stores only val_qwk and val_acc (not per-class F1),
    this plot uses per-fold val_acc across models as a distributional proxy.
    For true per-class F1, use the report_generator with experiment run dirs.

    As a practical alternative, this generates a paired boxplot comparing
    val_qwk and val_acc per model, highlighting the accuracy-kappa gap.

    Args:
        project_root: Project root directory.
        output_dir: Directory to save the figure.
        cv_strategy: CV strategy to filter ('random_stratified' or 'loao_balanced').

    Returns:
        Path to the saved PNG file.
    """
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_stratified" if cv_strategy == "random_stratified" else "_loao"

    model_names: List[str] = []
    qwk_data: List[List[float]] = []
    acc_data: List[List[float]] = []

    for key, entry in registry.items():
        if not key.endswith(suffix):
            continue
        cv_data = list(entry.values())[0]
        if cv_data.get("cv_strategy") != cv_strategy:
            continue
        qwks = _get_fold_qwks(entry)
        accs = _get_fold_accs(entry)
        if not qwks:
            continue
        base_name = _extract_model_name(key)
        model_names.append(DISPLAY_NAMES.get(base_name, base_name))
        qwk_data.append(qwks)
        acc_data.append(accs)

    if not model_names:
        logger.warning("No models found for per-class analysis")
        return output_dir / "boxplot_qwk_vs_acc.png"

    # Sort by mean QWK descending
    mean_qwks = [np.mean(q) for q in qwk_data]
    sorted_idx = np.argsort(mean_qwks)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    qwk_data = [qwk_data[i] for i in sorted_idx]
    acc_data = [acc_data[i] for i in sorted_idx]

    n = len(model_names)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 6))

    # QWK boxes
    bp_qwk = ax.boxplot(
        qwk_data,
        positions=x - width / 2,
        widths=width * 0.8,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
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
        meanprops=dict(marker="D", markerfacecolor="red", markersize=5),
        medianprops=dict(color="black", linewidth=1.5),
    )
    for box in bp_acc["boxes"]:
        box.set_facecolor("#F38181")
        box.set_alpha(0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)

    cv_label = "Stratified 5-Fold" if cv_strategy == "random_stratified" else "LOAO 2-Fold"
    ax.set_title(
        f"QWK vs. Accuracy per Model -- {cv_label} CV",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4ECDC4",
               markersize=10, label="QWK"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F38181",
               markersize=10, label="Accuracy"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="red",
               markersize=6, label="Mean"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9)

    plt.tight_layout()
    path = output_dir / f"boxplot_qwk_vs_acc_{cv_strategy}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ==============================================================================
# 5. Generalization Gap (CV-QWK minus Test-QWK)
# ==============================================================================

def plot_generalization_gap(
    project_root: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Boxplot of generalization gap grouped by CV strategy.

    Gap = mean_qwk (CV validation) - test_qwk (holdout test set).
    Positive gap = overfitting indicator; negative gap = conservative CV.
    Also shows a per-model bar chart for detailed comparison.

    Args:
        project_root: Project root directory.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved PNG file.
    """
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-model gaps
    records: List[Dict] = []
    for key, entry in registry.items():
        cv_data = list(entry.values())[0]
        mean_qwk = cv_data.get("mean_qwk", 0)
        test_qwk = cv_data.get("test_qwk", 0)
        if mean_qwk == 0 or test_qwk == 0:
            continue
        base_name = _extract_model_name(key)
        cv_strat = cv_data.get("cv_strategy", "")
        gap = mean_qwk - test_qwk
        records.append({
            "model": DISPLAY_NAMES.get(base_name, base_name),
            "cv_strategy": cv_strat,
            "mean_qwk": mean_qwk,
            "test_qwk": test_qwk,
            "gap": gap,
        })

    if not records:
        logger.warning("No models with both CV and test QWK found")
        return output_dir / "boxplot_generalization_gap.png"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 2]})

    # -- Left panel: Boxplot by CV strategy --
    strat_groups: Dict[str, List[float]] = {}
    for r in records:
        label = "Stratified" if r["cv_strategy"] == "random_stratified" else "LOAO"
        strat_groups.setdefault(label, []).append(r["gap"])

    strat_keys = sorted(strat_groups.keys())
    box_data = [strat_groups[k] for k in strat_keys]
    labels = [f"{k}\n(n={len(strat_groups[k])})" for k in strat_keys]

    bp = axes[0].boxplot(
        box_data,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
        medianprops=dict(color="black", linewidth=2),
        widths=0.45,
    )

    colors = {"LOAO": "#F38181", "Stratified": "#4ECDC4"}
    for i, k in enumerate(strat_keys):
        bp["boxes"][i].set_facecolor(colors.get(k, "#CCCCCC"))
        bp["boxes"][i].set_alpha(0.7)

    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axes[0].set_ylabel("Gap (CV-QWK - Test-QWK)", fontsize=11)
    axes[0].set_title("Gap by CV Strategy", fontsize=12, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # Annotation
    axes[0].text(
        0.5, 0.02, "< 0: CV underestimates | > 0: CV overestimates",
        transform=axes[0].transAxes, ha="center", va="bottom",
        fontsize=8, style="italic", alpha=0.7,
    )

    # -- Right panel: Per-model bar chart --
    # Sort records: stratified first, then LOAO, each by gap descending
    strat_records = sorted(
        [r for r in records if r["cv_strategy"] == "random_stratified"],
        key=lambda r: r["gap"], reverse=True,
    )
    loao_records = sorted(
        [r for r in records if r["cv_strategy"] == "loao_balanced"],
        key=lambda r: r["gap"], reverse=True,
    )
    ordered = strat_records + loao_records

    bar_labels = [
        f"{r['model']} ({'S' if r['cv_strategy'] == 'random_stratified' else 'L'})"
        for r in ordered
    ]
    gaps = [r["gap"] for r in ordered]
    bar_colors = [
        "#4ECDC4" if r["cv_strategy"] == "random_stratified" else "#F38181"
        for r in ordered
    ]

    y_pos = np.arange(len(bar_labels))
    axes[1].barh(y_pos, gaps, color=bar_colors, alpha=0.8, height=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(bar_labels, fontsize=9)
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].set_xlabel("Gap (CV-QWK - Test-QWK)", fontsize=11)
    axes[1].set_title("Generalization Gap per Model", fontsize=12, fontweight="bold")
    axes[1].grid(axis="x", alpha=0.3)
    axes[1].invert_yaxis()

    # Value labels on bars
    for i, (gap, label) in enumerate(zip(gaps, bar_labels)):
        ha = "left" if gap >= 0 else "right"
        offset = 0.002 if gap >= 0 else -0.002
        axes[1].text(gap + offset, i, f"{gap:+.4f}", va="center", ha=ha, fontsize=8)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#4ECDC4",
               markersize=10, label="Stratified (S)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#F38181",
               markersize=10, label="LOAO (L)"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.suptitle(
        "Generalization Gap Analysis: CV Validation vs. Holdout Test",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    path = output_dir / "boxplot_generalization_gap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ==============================================================================
# Generate All Boxplots
# ==============================================================================

def generate_all_boxplots(
    project_root: Path,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Generate all 5 boxplot analyses and return paths to saved figures.

    Args:
        project_root: Project root directory.
        output_dir: Directory to save figures (default: project_root/figures).

    Returns:
        List of paths to all generated PNG files.
    """
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating all boxplots to {output_dir}")

    paths: List[Path] = []

    # 1. QWK per model (Stratified)
    logger.info("[1/5] QWK per model (Stratified 5-Fold)")
    paths.append(plot_qwk_per_model_stratified(project_root, output_dir))

    # 2. Architecture family comparison (both strategies)
    logger.info("[2/5] Architecture family comparison")
    paths.append(plot_qwk_per_family(project_root, output_dir, "random_stratified"))
    paths.append(plot_qwk_per_family(project_root, output_dir, "loao_balanced"))

    # 3. CV strategy comparison
    logger.info("[3/5] CV strategy comparison (LOAO vs Stratified)")
    paths.append(plot_cv_strategy_comparison(project_root, output_dir))

    # 4. QWK vs Accuracy comparison
    logger.info("[4/5] QWK vs Accuracy per model")
    paths.append(plot_per_class_f1_from_registry(project_root, output_dir, "random_stratified"))

    # 5. Generalization gap
    logger.info("[5/5] Generalization gap analysis")
    paths.append(plot_generalization_gap(project_root, output_dir))

    logger.info(f"All {len(paths)} boxplots generated successfully")
    return paths


# ==============================================================================
# CLI entry point
# ==============================================================================

if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / "figures"

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    paths = generate_all_boxplots(project_root, output_dir)
    print(f"\nGenerated {len(paths)} boxplots:")
    for p in paths:
        print(f"  - {p}")
