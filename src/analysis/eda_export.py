"""Systematic EDA export for the inflammation dataset (ML-5 compliance).

Generates all EDA visualizations via DataExplorer and writes a JSON index
to figures/eda/.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from configs.utils import load_config
from src.data.data_exploration import DataExplorer
from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_explorer(cfg: dict, save_dir: Path) -> DataExplorer:
    """Instantiate a DataExplorer with the given save directory.

    Args:
        cfg: Loaded configuration dictionary (unused directly, explorer reloads
             its own config, but kept for consistency with the project pattern).
        save_dir: Directory where plots will be saved.

    Returns:
        Configured DataExplorer instance with data loaded for fold 0.
    """
    explorer = DataExplorer(save_dir=save_dir)
    explorer.load_data(fold_idx=0)
    return explorer


def _generate_plots(explorer: DataExplorer, cfg: dict, save_dir: Path) -> List[Dict[str, str]]:
    """Run each DataExplorer plot method and record generated artefacts.

    Args:
        explorer: Initialised DataExplorer with data already loaded.
        cfg: Configuration dictionary with 'data.raw_dir' and 'data.norm_dir'.
        save_dir: Directory where plots are saved.

    Returns:
        List of dicts with keys 'filename' and 'description' for each saved plot.
    """
    generated: List[Dict[str, str]] = []

    project_root = Path(cfg.get("directories", {}).get("project_root", "."))

    # 1. Class distribution
    logger.info("Generating class distribution plot...")
    explorer.show_class_distribution()
    if (save_dir / "class_distribution.png").exists():
        generated.append({
            "filename": "class_distribution.png",
            "description": "Class (inflammation grade) distribution in training and validation splits.",
        })

    # 2. Sample images
    logger.info("Generating sample images plot...")
    explorer.show_sample_images(num_samples=8)
    if (save_dir / "sample_images.png").exists():
        generated.append({
            "filename": "sample_images.png",
            "description": "Grid of sample patches with inflammation grade labels.",
        })

    # 3. Normalization comparison (requires both raw and normalised dirs)
    raw_dir = project_root / cfg["data"]["raw_dir"]
    norm_dir = project_root / cfg["data"]["norm_dir"]
    if raw_dir.exists() and norm_dir.exists():
        logger.info("Generating stain normalization comparison plot...")
        explorer.show_normalization_comparison(raw_dir=raw_dir, norm_dir=norm_dir, n_samples=3)
        if (save_dir / "normalization_comparison.png").exists():
            generated.append({
                "filename": "normalization_comparison.png",
                "description": (
                    "Side-by-side comparison of original, Macenko-normalised, "
                    "and difference images for 3 random patches."
                ),
            })

        # 4. RGB histograms per channel
        logger.info("Generating RGB histogram plots...")
        explorer.show_rgb_histograms(raw_dir=raw_dir, norm_dir=norm_dir, n_samples=3)
        for channel in ("R", "G", "B"):
            fname = f"histogram_{channel}_channel.png"
            if (save_dir / fname).exists():
                generated.append({
                    "filename": fname,
                    "description": f"{channel}-channel pixel intensity histograms before/after normalisation.",
                })
    else:
        logger.warning(
            "Skipping normalization comparison: raw_dir=%s or norm_dir=%s not found.",
            raw_dir,
            norm_dir,
        )

    return generated


def _write_index(save_dir: Path, entries: List[Dict[str, str]]) -> Path:
    """Write eda_index.json listing all generated artefacts.

    Args:
        save_dir: Directory containing the generated figures.
        entries: List of dicts with 'filename' and 'description' keys.

    Returns:
        Path to the written index file.
    """
    index_path = save_dir / "eda_index.json"
    index_data = {
        "description": "EDA visualizations for the lung inflammation histopathology dataset.",
        "figures": entries,
    }
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index_data, fh, indent=2)
    logger.info("EDA index written to: %s", index_path)
    return index_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eda_export(cfg: dict, save_dir: Path) -> Dict[str, object]:
    """Generate all EDA visualisations and write a JSON index.

    Args:
        cfg: Loaded configuration dictionary.
        save_dir: Output directory for figures and the JSON index.

    Returns:
        Dictionary with keys 'save_dir', 'generated_figures', and 'index_path'.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("EDA export starting. Output directory: %s", save_dir)

    explorer = _build_explorer(cfg, save_dir)
    entries = _generate_plots(explorer, cfg, save_dir)
    index_path = _write_index(save_dir, entries)

    logger.info("EDA export complete. %d figure(s) generated.", len(entries))
    return {
        "save_dir": str(save_dir),
        "generated_figures": entries,
        "index_path": str(index_path),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run systematic EDA export from the command line."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()

    project_root = Path(cfg.get("directories", {}).get("project_root", "."))
    save_dir = project_root / "figures" / "eda"

    result = run_eda_export(cfg, save_dir)
    print(f"EDA export complete. {len(result['generated_figures'])} figure(s) saved to {result['save_dir']}")
    print(f"Index: {result['index_path']}")


if __name__ == "__main__":
    main()
