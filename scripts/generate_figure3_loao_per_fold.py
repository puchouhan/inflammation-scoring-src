"""
Generate redesigned Figure 3 (LOAO QWK bar chart) for Punkt 7.

New design:
- Two bars per architecture (Fold 0 and Fold 1) side by side
- Horizontal mean line across each pair
- Y-axis from 0.0 to 1.0

Usage (from project root):
    python scripts/generate_figure3_loao_per_fold.py

Output: figures/comparison_qwk_bar_loao_balanced.png  (overwrites old figure)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# LOAO fold results (from Table tab:results:loao, sorted by mean QWK desc)
# ---------------------------------------------------------------------------
LOAO_RESULTS: List[Tuple[str, str, float, float]] = [
    # (model_key, display_label, fold_0_qwk, fold_1_qwk)
    ("maxvit",      "MaxViT",              0.877, 0.812),
    ("convnext",    "ConvNeXt",            0.900, 0.758),
    ("swin",        "Swin-T",              0.871, 0.742),
    ("efficientnet","EfficientNetV2",      0.832, 0.728),
    ("convit",      "ConViT",              0.852, 0.701),
    ("densenet",    "DenseNet-121",        0.863, 0.688),
    ("regnety",     "RegNetY",             0.847, 0.693),
    ("simclr",      "SimCLR (ResNet-50)",  0.836, 0.639),
    ("tnt",         "TNT",                 0.833, 0.634),
    ("vit",         "ViT-B/16",            0.773, 0.605),
    ("dino",        "DINO",                0.769, 0.532),
]

# Architecture family colouring (consistent with thesis palette)
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

OUTPUT_PATH = Path(
    "/Users/pulkit/Library/Mobile Documents/"
    "com~apple~CloudDocs/master_thesis/master_thesis_inflammation/"
    "figures/comparison_qwk_bar_loao_balanced.png"
)

FONT_SIZE_LABEL = 11
FONT_SIZE_TICK  = 10
FONT_SIZE_VALUE = 8
FOLD_BAR_WIDTH  = 0.35


def _arch_color(key: str) -> str:
    return FAMILY_COLORS.get(FAMILY.get(key, "CNN"), "#999999")


def main() -> None:
    """Generate and save the per-fold LOAO QWK grouped bar chart."""
    # Sort by descending mean QWK
    data = sorted(
        LOAO_RESULTS,
        key=lambda r: (r[2] + r[3]) / 2.0,
        reverse=True,
    )

    labels     = [r[1] for r in data]
    fold0_qwks = [r[2] for r in data]
    fold1_qwks = [r[3] for r in data]
    means      = [(r[2] + r[3]) / 2.0 for r in data]
    keys       = [r[0] for r in data]
    n          = len(data)

    x = np.arange(n)
    w = FOLD_BAR_WIDTH

    plt.rcParams.update({
        "font.size":        FONT_SIZE_LABEL,
        "axes.titlesize":   FONT_SIZE_LABEL + 1,
        "axes.labelsize":   FONT_SIZE_LABEL,
        "xtick.labelsize":  FONT_SIZE_TICK,
        "ytick.labelsize":  FONT_SIZE_TICK,
        "figure.dpi":       150,
    })

    fig, ax = plt.subplots(figsize=(max(12, n * 1.3), 6))

    for i, (key, f0, f1, mean) in enumerate(
        zip(keys, fold0_qwks, fold1_qwks, means)
    ):
        color = _arch_color(key)
        light = _lighten(color, 0.4)

        # Fold 0 bar (darker shade)
        ax.bar(
            x[i] - w / 2, f0,
            width=w,
            color=color, alpha=0.88,
            edgecolor="black", linewidth=0.6,
            label="Fold 0" if i == 0 else "_nolegend_",
        )
        # Fold 1 bar (lighter shade)
        ax.bar(
            x[i] + w / 2, f1,
            width=w,
            color=light, alpha=0.88,
            edgecolor="black", linewidth=0.6,
            label="Fold 1" if i == 0 else "_nolegend_",
        )
        # Mean horizontal line spanning both bars
        ax.hlines(
            mean,
            xmin=x[i] - w / 2 - w * 0.45,
            xmax=x[i] + w / 2 + w * 0.45,
            colors="black", linewidths=1.4, linestyles="--", zorder=5,
        )
        # Annotate mean value above the line
        ax.text(
            x[i],
            mean + 0.015,
            f"{mean:.3f}",
            ha="center", va="bottom",
            fontsize=FONT_SIZE_VALUE,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Validation QWK", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim([0.0, 1.05])
    ax.set_yticks(np.arange(0.0, 1.05, 0.1))
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_title(
        "Validation QWK per Architecture -- LOAO Protocol (Fold 0 and Fold 1)",
        fontweight="bold",
    )

    # Legend: fold shade + architecture families
    family_handles = [
        mpatches.Patch(
            facecolor=FAMILY_COLORS[f], edgecolor="black",
            linewidth=0.5, alpha=0.85, label=f,
        )
        for f in FAMILY_ORDER
    ]
    fold_handles = [
        mpatches.Patch(
            facecolor="grey", edgecolor="black",
            linewidth=0.5, alpha=0.88, label="Fold 0 (darker)",
        ),
        mpatches.Patch(
            facecolor="#cccccc", edgecolor="black",
            linewidth=0.5, alpha=0.88, label="Fold 1 (lighter)",
        ),
        plt.Line2D(
            [0], [0], color="black", linewidth=1.4,
            linestyle="--", label="Mean",
        ),
    ]
    ax.legend(
        handles=family_handles + fold_handles,
        title="Legend",
        loc="lower right",
        fontsize=9,
        title_fontsize=9,
    )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


def _lighten(hex_color: str, factor: float) -> str:
    """Lighten a hex colour by blending towards white.

    Args:
        hex_color: Hex colour string e.g. '#4878cf'.
        factor: Blend factor 0.0 (no change) to 1.0 (white).

    Returns:
        Lightened hex colour string.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r_new = int(r + (255 - r) * factor)
    g_new = int(g + (255 - g) * factor)
    b_new = int(b + (255 - b) * factor)
    return f"#{r_new:02x}{g_new:02x}{b_new:02x}"


if __name__ == "__main__":
    main()
