"""Duplicate and outlier detection for the inflammation image dataset (ML-6 compliance).

Produces a JSON cleaning report artefact that documents dataset integrity checks
before any model training.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

from configs.utils import load_config
from src.data.inflammation_dataset import create_dataframe
from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)

# Expected filename pattern:  Study_Animal_Slide_X_Y.png  or
#                             Study_Animal_Slide_Section_X_Y.png
_FILENAME_PATTERN = re.compile(
    r"^\d+_\d+_\d+(?:_\d+)?_\d+_\d+\.png$"
)


# ---------------------------------------------------------------------------
# Internal check helpers
# ---------------------------------------------------------------------------

def _check_duplicate_filenames(df: pd.DataFrame) -> List[str]:
    """Return list of filenames that appear more than once in the dataset.

    Args:
        df: Full dataset DataFrame with a 'filepath' column.

    Returns:
        List of duplicate filename strings (basename only).
    """
    basenames = df["filepath"].apply(lambda p: Path(p).name)
    counts = basenames.value_counts()
    return counts[counts > 1].index.tolist()


def _check_spatial_duplicates(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return records where (animal_id, x, y) coordinate triplets repeat.

    Spatial duplicates indicate that two patches from the same animal at the same
    tile position appear in different class folders, which would be an annotation
    error or dataset construction error.

    Args:
        df: Full dataset DataFrame with 'animal_id', 'x', 'y' columns.

    Returns:
        List of dicts, each with keys 'animal_id', 'x', 'y', 'count', 'labels'.
    """
    if not {"animal_id", "x", "y"}.issubset(df.columns):
        logger.warning("DataFrame missing spatial columns; skipping spatial duplicate check.")
        return []

    grouped = df.groupby(["animal_id", "x", "y"])
    duplicates: List[Dict[str, Any]] = []
    for (animal_id, x, y), group in grouped:
        if len(group) > 1:
            duplicates.append({
                "animal_id": str(animal_id),
                "x": int(x),
                "y": int(y),
                "count": int(len(group)),
                "labels": [int(v) for v in group["label"].tolist()],
            })
    return duplicates


def _check_invalid_labels(df: pd.DataFrame) -> int:
    """Count rows whose label is -1 (invalid / not matched to any class folder).

    Note: create_dataframe already skips label=-1 rows, so this count is
    expected to be 0 when called on its output.  The check is kept to
    document and verify that assumption.

    Args:
        df: Full dataset DataFrame with a 'label' column.

    Returns:
        Number of rows with label == -1.
    """
    if "label" not in df.columns:
        return 0
    return int((df["label"] == -1).sum())


def _per_class_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Compute per-class image counts.

    Args:
        df: Full dataset DataFrame with a 'label' column.

    Returns:
        Dict mapping string class index to integer count.
    """
    if "label" not in df.columns:
        return {}
    counts = df["label"].value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items()}


def _check_outlier_filenames(df: pd.DataFrame) -> List[str]:
    """Return filenames that do not match the expected naming pattern.

    Args:
        df: Full dataset DataFrame with a 'filepath' column.

    Returns:
        List of non-conforming filename strings.
    """
    outliers: List[str] = []
    for filepath in df["filepath"]:
        name = Path(filepath).name
        if not _FILENAME_PATTERN.match(name):
            outliers.append(name)
    return outliers


def _build_summary_text(
    total: int,
    dup_filenames: List[str],
    spatial_dups: List[Dict[str, Any]],
    invalid_labels: int,
    outlier_filenames: List[str],
) -> str:
    """Compose a human-readable cleaning summary string.

    Args:
        total: Total number of images found.
        dup_filenames: List of duplicate filename strings.
        spatial_dups: List of spatial duplicate dicts.
        invalid_labels: Count of invalid-label rows removed.
        outlier_filenames: List of non-conforming filenames.

    Returns:
        Plain-text summary string.
    """
    lines = [
        f"Total images scanned: {total}",
        f"Duplicate filenames: {len(dup_filenames)}",
        f"Spatial duplicates (animal_id, x, y): {len(spatial_dups)}",
        f"Invalid labels removed: {invalid_labels}",
        f"Outlier filenames (pattern mismatch): {len(outlier_filenames)}",
    ]
    if not dup_filenames and not spatial_dups and not outlier_filenames and invalid_labels == 0:
        lines.append("Dataset integrity check PASSED: no issues detected.")
    else:
        lines.append("Dataset integrity check COMPLETED WITH FINDINGS: review above counts.")
    return "  ".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_cleaning_report(cfg: dict, output_path: Path) -> dict:
    """Run all dataset integrity checks and save a JSON cleaning report.

    Args:
        cfg: Loaded configuration dictionary.  Must contain
             ``cfg['data']['norm_dir']`` pointing to the normalised image root.
        output_path: Destination path for the JSON report file.

    Returns:
        Report dictionary (same content as the saved JSON file).
    """
    norm_dir = Path(cfg["data"]["norm_dir"])
    logger.info("Loading dataset from: %s", norm_dir)

    df = create_dataframe(str(norm_dir))
    total = len(df)
    logger.info("Total images loaded: %d", total)

    # Run checks
    dup_filenames = _check_duplicate_filenames(df)
    spatial_dups = _check_spatial_duplicates(df)
    invalid_labels = _check_invalid_labels(df)
    per_class = _per_class_counts(df)
    outlier_filenames = _check_outlier_filenames(df)

    summary = _build_summary_text(
        total, dup_filenames, spatial_dups, invalid_labels, outlier_filenames
    )

    report: dict = {
        "total_images_found": total,
        "duplicate_filenames": dup_filenames,
        "spatial_duplicates": spatial_dups,
        "invalid_labels_removed": invalid_labels,
        "per_class_counts": per_class,
        "outlier_filenames": outlier_filenames,
        "cleaning_summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Cleaning report saved to: %s", output_path)

    # Log summary to console for quick inspection
    logger.info("Cleaning report summary: %s", summary)

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the cleaning report from the command line."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()

    output_path = Path("docs/artifacts/cleaning_report.json")
    report = generate_cleaning_report(cfg, output_path)

    print(f"Cleaning report saved to: {output_path}")
    print(f"Total images: {report['total_images_found']}")
    print(f"Duplicate filenames: {len(report['duplicate_filenames'])}")
    print(f"Spatial duplicates: {len(report['spatial_duplicates'])}")
    print(f"Invalid labels removed: {report['invalid_labels_removed']}")
    print(f"Outlier filenames: {len(report['outlier_filenames'])}")
    print(f"Per-class counts: {report['per_class_counts']}")


if __name__ == "__main__":
    main()
