"""
Regenerate Figures 4 and 5 with y-axis and x-axis spanning [0, 1].

Punkt 9 supervisor feedback: scatter plots must use the full 0-1 QWK range
so the reader can see the absolute performance level, not only relative
differences.

Figures regenerated:
- figures/generalization_gap.png
    CV-QWK (validation) vs Test-QWK scatter for all 11 architectures under
    both cross-validation strategies.
- figures/comparison_stratified_vs_loao.png
    Stratified test QWK (x) vs LOAO test QWK (y) scatter.

Usage (from project root):
    python scripts/generate_figures4_5_axis_fix.py
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data from Table tab:results:overview in the thesis
# (model_key, display, loao_mean, strat_mean, test_loao, test_strat)
# ---------------------------------------------------------------------------
MODEL_DATA: List[Tuple[str, str, float, float, float, float]] = [
    ("maxvit",       "MaxViT",             0.845, 0.898, 0.859, 0.909),
    ("convnext",     "ConvNeXt",           0.829, 0.895, 0.868, 0.908),
    ("swin",         "Swin-T",             0.806, 0.891, 0.866, 0.910),
    ("efficientnet", "EfficientNetV2",     0.780, 0.862, 0.863, 0.873),
    ("convit",       "ConViT",             0.777, 0.856, 0.849, 0.872),
    ("densenet",     "DenseNet-121",       0.775, 0.872, 0.865, 0.894),
    ("regnety",      "RegNetY",            0.770, 0.891, 0.853, 0.895),
    ("simclr",       "SimCLR (ResNet-50)", 0.738, 0.874, 0.750, 0.877),
    ("tnt",          "TNT",                0.733, 0.882, 0.798, 0.901),
    ("vit",          "ViT-B/16",           0.689, 0.849, 0.825, 0.865),
    ("dino",         "DINO",               0.650, 0.848, 0.795, 0.871),
]

FAMILY: Dict[str, str] = {
    "maxvit":       "Hybrid",
    "convnext":     "CNN",
    "swin":         "Transformer",
    "efficientnet": "CNN",
    "convit":       "Hybrid",
    "densenet":     "CNN",
    "regnety":      "CNN",
    "simclr":       "SSL",
    "tnt":          "Hybrid",
    "vit":          "Transformer",
    "dino":         "SSL",
}

FAMILY_COLORS: Dict[str, str] = {
    "CNN":         "#4878cf",
    "Transformer": "#6acc65",
    "Hybrid":      "#d65f5f",
    "SSL":         "#b47cc7",
}

FAMILY_ORDER = ["CNN", "Transformer", "Hybrid", "SSL"]

PROJECT_ROOT = Path(
    "/Users/pulkit/Library/Mobile Documents/"
    "com~apple~CloudDocs/master_thesis/master_thesis_inflammation"
)
FIGURES_DIR = PROJECT_ROOT / "figures"


def _arch_color(key: str) -> str:
    return FAMILY_COLORS.get(FAMILY.get(key, "CNN"), "#999999")


def _arch_marker(key: str) -> str:
    markers = {"CNN": "o", "Transformer": "s", "Hybrid": "D", "SSL": "^"}
    return markers.get(FAMILY.get(key, "CNN"), "o")


def _family_legend_handles() -> List:
    return [
        mpatches.Patch(
            facecolor=FAMILY_COLORS[f], edgecolor="black",
            linewidth=0.5, alpha=0.85, label=f,
        )
        for f in FAMILY_ORDER
    ]


def generate_generalization_gap() -> Path:
    """Scatter: CV-QWK (validation) vs Test-QWK, axes [0, 1].

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    strat_xs, strat_ys, loao_xs, loao_ys = [], [], [], []

    for key, label, loao_mean, strat_mean, test_loao, test_strat in MODEL_DATA:
        color = _arch_color(key)
        marker = _arch_marker(key)

        # Stratified: CV-QWK = strat_mean, Test-QWK = test_strat
        ax.scatter(
            strat_mean, test_strat,
            color=color, marker=marker, s=90, alpha=0.9, zorder=3,
        )
        ax.annotate(
            label, (strat_mean, test_strat),
            fontsize=7, xytext=(4, 2), textcoords="offset points",
        )
        strat_xs.append(strat_mean)
        strat_ys.append(test_strat)

        # LOAO: CV-QWK = loao_mean, Test-QWK = test_loao
        ax.scatter(
            loao_mean, test_loao,
            color=color, marker=marker, s=90, alpha=0.9, zorder=3,
            facecolors="none", edgecolors=color, linewidths=1.5,
        )
        loao_xs.append(loao_mean)
        loao_ys.append(test_loao)

    # Perfect-generalisation diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1.2,
            label="Perfect Generalisation (CV = Test)")

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xlabel("CV-QWK (Validation)", fontsize=11)
    ax.set_ylabel("Test-QWK (Hold-out)", fontsize=11)
    ax.set_title("Generalisation Gap: CV-QWK vs Test-QWK", fontweight="bold")
    ax.grid(alpha=0.3, linewidth=0.5)

    # Legend: filled = stratified, hollow = LOAO
    fill_h = mpatches.Patch(
        facecolor="grey", edgecolor="black", linewidth=0.5, alpha=0.85,
        label="Stratified 5-fold (filled)",
    )
    hollow_h = mpatches.Patch(
        facecolor="white", edgecolor="grey", linewidth=1.5,
        label="LOAO 2-fold (hollow)",
    )
    diag_h = plt.Line2D([0], [0], color="black", linestyle="--",
                         linewidth=1.2, alpha=0.3, label="Perfect Generalisation")
    family_handles = _family_legend_handles()
    ax.legend(
        handles=family_handles + [fill_h, hollow_h, diag_h],
        loc="upper left", fontsize=8, title="Legend", title_fontsize=8,
    )

    plt.tight_layout()
    path = FIGURES_DIR / "generalization_gap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def generate_stratified_vs_loao() -> Path:
    """Scatter: Stratified test QWK (x) vs LOAO test QWK (y), axes [0, 1].

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    for key, label, loao_mean, strat_mean, test_loao, test_strat in MODEL_DATA:
        color = _arch_color(key)
        marker = _arch_marker(key)
        ax.scatter(
            test_strat, test_loao,
            color=color, marker=marker, s=100, alpha=0.9, zorder=3,
        )
        ax.annotate(
            label, (test_strat, test_loao),
            fontsize=8, xytext=(4, 3), textcoords="offset points",
        )

    # Equal-performance diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, linewidth=1.2,
            label="Equal performance (LOAO = Stratified)")

    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks(np.arange(0.0, 1.1, 0.1))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax.set_xlabel("Stratified 5-Fold Test QWK", fontsize=11)
    ax.set_ylabel("LOAO 2-Fold Test QWK", fontsize=11)
    ax.set_title("Stratified vs LOAO -- Test QWK Comparison", fontweight="bold")
    ax.grid(alpha=0.3, linewidth=0.5)

    marker_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="grey",
            markersize=9, label="CNN",
        ),
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="grey",
            markersize=9, label="Transformer",
        ),
        plt.Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="grey",
            markersize=9, label="Hybrid",
        ),
        plt.Line2D(
            [0], [0], marker="^", color="w", markerfacecolor="grey",
            markersize=9, label="Self-Supervised",
        ),
        plt.Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.2,
            alpha=0.35, label="Equal performance",
        ),
    ]
    family_handles = _family_legend_handles()
    ax.legend(
        handles=family_handles + marker_handles,
        loc="upper left", fontsize=8, title="Legend", title_fontsize=8,
    )

    plt.tight_layout()
    path = FIGURES_DIR / "comparison_stratified_vs_loao.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def main() -> None:
    """Regenerate both scatter figures with axes spanning [0, 1]."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_generalization_gap()
    generate_stratified_vs_loao()


if __name__ == "__main__":
    main()
