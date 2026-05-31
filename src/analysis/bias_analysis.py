"""Bias and fairness analysis for the inflammation scoring system (ML-21 compliance).

Produces a JSON artefact and a Markdown summary documenting per-animal
performance, class error asymmetry, underrepresented class risk, and
deployment limitations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)

# Threshold below which a class is considered underrepresented (fraction of total).
_UNDERREPRESENTED_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Helpers – loading predictions/metrics from experiment runs
# ---------------------------------------------------------------------------

def _discover_fold_metric_files(run_dir: Path) -> List[Path]:
    """Recursively find fold metrics JSON files under run_dir.

    Looks for files matching common naming patterns produced by the training
    pipeline (``fold_*_metrics.json``).

    Args:
        run_dir: Root experiment directory, e.g. ``experiments/20260101_densenet``.

    Returns:
        Sorted list of matching Path objects (may be empty).
    """
    patterns = [
        "**/fold_*_metrics.json",
        "**/fold_*.json",
    ]
    found: List[Path] = []
    for pattern in patterns:
        found.extend(run_dir.glob(pattern))
    # Deduplicate and sort for deterministic processing order
    return sorted(set(found))


def _load_fold_data(metric_files: List[Path]) -> List[Dict[str, Any]]:
    """Load and parse fold metric JSON files.

    Invalid or unreadable files are logged as warnings and skipped.

    Args:
        metric_files: List of paths to JSON metric files.

    Returns:
        List of parsed metric dictionaries.
    """
    records: List[Dict[str, Any]] = []
    for path in metric_files:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["_source_file"] = str(path)
            records.append(data)
        except Exception as exc:
            logger.warning("Could not load metrics file %s: %s", path, exc)
    return records


# ---------------------------------------------------------------------------
# Helpers – per-animal performance
# ---------------------------------------------------------------------------

def _extract_per_animal_performance(
    fold_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate performance metrics grouped by validation animal.

    Each fold in a Leave-One-Animal-Out setup has a single animal in the
    validation set.  This function collects ``val_qwk``, ``val_f1``, and
    ``val_acc`` (when present) per animal.

    Args:
        fold_records: List of fold metric dicts, each expected to contain
                      at least a ``val_animal_id`` (or derivable) key and
                      metric values.

    Returns:
        Dict mapping animal_id string to performance summary dict.
    """
    per_animal: Dict[str, Dict[str, Any]] = {}
    for record in fold_records:
        animal_id: Optional[str] = (
            record.get("val_animal_id")
            or record.get("animal_id")
            or record.get("validation_animal")
        )
        if animal_id is None:
            # Try to infer from source file path (e.g. fold_0 → first animal)
            animal_id = f"unknown_fold_{record.get('fold', 'N')}"

        metrics: Dict[str, Any] = {}
        for key in ("val_qwk", "val_f1", "val_acc", "qwk", "f1", "accuracy"):
            if key in record:
                metrics[key] = record[key]

        per_animal[str(animal_id)] = metrics
    return per_animal


# ---------------------------------------------------------------------------
# Helpers – class error asymmetry (FNR / FPR)
# ---------------------------------------------------------------------------

def _compute_fnr_fpr(
    confusion: List[List[int]],
) -> Tuple[List[float], List[float]]:
    """Compute per-class false-negative and false-positive rates from a confusion matrix.

    Args:
        confusion: Square confusion matrix as list-of-lists,
                   shape (n_classes, n_classes), rows = true, cols = predicted.

    Returns:
        Tuple of (fnr_per_class, fpr_per_class), each a list of floats.
    """
    n = len(confusion)
    fnr: List[float] = []
    fpr: List[float] = []

    for cls in range(n):
        tp = confusion[cls][cls]
        fn = sum(confusion[cls][j] for j in range(n) if j != cls)
        fp = sum(confusion[i][cls] for i in range(n) if i != cls)
        tn = sum(
            confusion[i][j]
            for i in range(n)
            for j in range(n)
            if i != cls and j != cls
        )

        fnr_cls = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_cls = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fnr.append(round(fnr_cls, 4))
        fpr.append(round(fpr_cls, 4))

    return fnr, fpr


def _extract_class_error_asymmetry(
    fold_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Average FNR and FPR across all folds that contain a confusion matrix.

    Args:
        fold_records: List of fold metric dicts.

    Returns:
        Dict with keys 'FNR_per_class' and 'FPR_per_class' (averaged across folds),
        or empty lists if no confusion matrices were found.
    """
    all_fnr: List[List[float]] = []
    all_fpr: List[List[float]] = []

    for record in fold_records:
        cm = record.get("confusion_matrix") or record.get("val_confusion_matrix")
        if cm is None:
            continue
        try:
            fnr, fpr = _compute_fnr_fpr(cm)
            all_fnr.append(fnr)
            all_fpr.append(fpr)
        except Exception as exc:
            logger.warning("Could not compute FNR/FPR from confusion matrix: %s", exc)

    if not all_fnr:
        return {"FNR_per_class": [], "FPR_per_class": [], "note": "No confusion matrices found."}

    n_cls = len(all_fnr[0])
    avg_fnr = [round(sum(row[i] for row in all_fnr) / len(all_fnr), 4) for i in range(n_cls)]
    avg_fpr = [round(sum(row[i] for row in all_fpr) / len(all_fpr), 4) for i in range(n_cls)]

    return {"FNR_per_class": avg_fnr, "FPR_per_class": avg_fpr}


# ---------------------------------------------------------------------------
# Helpers – underrepresented classes
# ---------------------------------------------------------------------------

def _identify_underrepresented_classes(
    fold_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flag classes whose share of total validation samples falls below the threshold.

    Args:
        fold_records: List of fold metric dicts, each may contain
                      'class_counts' or 'val_class_counts' dict.

    Returns:
        List of dicts with keys 'class_index' and 'fraction'.
    """
    aggregated: Dict[int, int] = {}
    for record in fold_records:
        counts = record.get("class_counts") or record.get("val_class_counts") or {}
        for cls_key, cnt in counts.items():
            idx = int(cls_key)
            aggregated[idx] = aggregated.get(idx, 0) + int(cnt)

    if not aggregated:
        return []

    total = sum(aggregated.values())
    if total == 0:
        return []

    flagged: List[Dict[str, Any]] = []
    for cls_idx in sorted(aggregated):
        fraction = aggregated[cls_idx] / total
        if fraction < _UNDERREPRESENTED_THRESHOLD:
            flagged.append({
                "class_index": cls_idx,
                "fraction": round(fraction, 4),
            })
    return flagged


# ---------------------------------------------------------------------------
# Skeleton / template when no experiment data is available
# ---------------------------------------------------------------------------

_DATA_LIMITATIONS_TEXT = (
    "The dataset consists of H&E-stained lung histopathology tiles from a small "
    "number of laboratory animals (n=3 unique animals after exclusion). "
    "All images originate from a single study and staining protocol at one "
    "institution, which limits generalizability to different staining batches, "
    "tissue preparation methods, or scanner devices. "
    "Class imbalance is present across inflammation grades (grades 0-3), and the "
    "'Ignore' class (index 4) captures preparation artefacts that are excluded "
    "from the ordinal scoring metric (QWK). "
    "The small cohort size means that inter-animal variance is conflated with "
    "true biological variance; performance estimates from Leave-One-Animal-Out "
    "cross-validation should be interpreted as lower-bound generalization estimates. "
    "No demographic or treatment metadata beyond animal identity is included."
)

_DEPLOYMENT_RISKS = [
    "Domain shift: model trained on normalised tiles from one staining protocol may "
    "degrade on slides prepared under different conditions.",
    "Class imbalance: under-represented high-grade inflammation classes (grade 3) "
    "may have elevated false-negative rates, which carries clinical risk when used "
    "for triage.",
    "Out-of-distribution artefacts: tiles containing tissue folds, bubbles, or "
    "out-of-focus regions are handled by the 'Ignore' class during training but "
    "could be misclassified in deployment if the artefact distribution shifts.",
    "Animal-level overfitting: with only three animals, feature patterns specific "
    "to individual animals may be memorised rather than generalised.",
    "No human expert calibration loop: the model has not been validated in an "
    "interactive clinical workflow and should not be used as a sole decision tool.",
]


def _build_skeleton_report() -> Dict[str, Any]:
    """Return a template bias report when no experimental data is available.

    Returns:
        Dict conforming to the bias analysis schema with methodology notes.
    """
    return {
        "per_animal_performance": {
            "note": (
                "No experiment run directory was provided or no fold metrics files "
                "were found.  Run training and re-execute with the correct run_dir "
                "to populate per-animal QWK and F1 scores."
            )
        },
        "class_error_asymmetry": {
            "FNR_per_class": [],
            "FPR_per_class": [],
            "note": "No confusion matrices found; re-run after training.",
        },
        "underrepresented_classes": [],
        "deployment_risks": _DEPLOYMENT_RISKS,
        "data_limitations": _DATA_LIMITATIONS_TEXT,
    }


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _write_markdown_summary(report: Dict[str, Any], output_path: Path) -> None:
    """Write a human-readable Markdown summary of the bias analysis.

    Args:
        report: Bias analysis report dictionary.
        output_path: Destination Markdown file path.
    """
    lines: List[str] = [
        "# Bias and Ethics Analysis Report",
        "",
        "## 1. Per-Animal Validation Performance",
        "",
    ]

    per_animal = report.get("per_animal_performance", {})
    if "note" in per_animal:
        lines.append(f"> {per_animal['note']}")
    else:
        lines.append("| Animal ID | val_qwk | val_f1 | val_acc |")
        lines.append("|-----------|---------|--------|---------|")
        for animal_id, metrics in per_animal.items():
            qwk = metrics.get("val_qwk", metrics.get("qwk", "N/A"))
            f1 = metrics.get("val_f1", metrics.get("f1", "N/A"))
            acc = metrics.get("val_acc", metrics.get("accuracy", "N/A"))
            lines.append(f"| {animal_id} | {qwk} | {f1} | {acc} |")

    lines += [
        "",
        "## 2. Class Error Asymmetry",
        "",
    ]

    asymmetry = report.get("class_error_asymmetry", {})
    fnr = asymmetry.get("FNR_per_class", [])
    fpr = asymmetry.get("FPR_per_class", [])
    if fnr and fpr:
        lines.append("| Class | FNR (avg across folds) | FPR (avg across folds) |")
        lines.append("|-------|------------------------|------------------------|")
        for i, (fn_val, fp_val) in enumerate(zip(fnr, fpr)):
            lines.append(f"| {i} | {fn_val:.4f} | {fp_val:.4f} |")
    else:
        note = asymmetry.get("note", "No data available.")
        lines.append(f"> {note}")

    lines += [
        "",
        "## 3. Underrepresented Classes",
        "",
    ]
    under = report.get("underrepresented_classes", [])
    if under:
        lines.append("| Class Index | Fraction of Total Samples |")
        lines.append("|-------------|---------------------------|")
        for entry in under:
            lines.append(f"| {entry['class_index']} | {entry['fraction']:.4f} |")
    else:
        lines.append("No classes flagged as underrepresented (threshold < 10%).")

    lines += [
        "",
        "## 4. Deployment Risks",
        "",
    ]
    for risk in report.get("deployment_risks", []):
        lines.append(f"- {risk}")

    lines += [
        "",
        "## 5. Data Limitations",
        "",
        report.get("data_limitations", ""),
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Bias analysis Markdown report written to: %s", output_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_bias_analysis(run_dir: Path, output_path: Path) -> Dict[str, Any]:
    """Run bias and fairness analysis and save JSON and Markdown artefacts.

    Args:
        run_dir: Directory containing experiment results (subdirectories per model
                 with fold metric JSON files).  Pass a non-existent path to generate
                 a skeleton/template report.
        output_path: Destination path for the JSON report file.

    Returns:
        Bias analysis report dictionary.
    """
    markdown_path = output_path.with_suffix(".md").parent / "bias_analysis_report.md"

    if not run_dir.exists():
        logger.warning(
            "run_dir does not exist (%s). Generating template bias report.", run_dir
        )
        report = _build_skeleton_report()
    else:
        metric_files = _discover_fold_metric_files(run_dir)
        logger.info("Found %d fold metric file(s) under %s.", len(metric_files), run_dir)

        if not metric_files:
            logger.warning("No fold metric files found. Generating template bias report.")
            report = _build_skeleton_report()
        else:
            fold_records = _load_fold_data(metric_files)
            per_animal = _extract_per_animal_performance(fold_records)
            asymmetry = _extract_class_error_asymmetry(fold_records)
            underrepresented = _identify_underrepresented_classes(fold_records)

            report = {
                "per_animal_performance": per_animal,
                "class_error_asymmetry": asymmetry,
                "underrepresented_classes": underrepresented,
                "deployment_risks": _DEPLOYMENT_RISKS,
                "data_limitations": _DATA_LIMITATIONS_TEXT,
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Bias analysis JSON report saved to: %s", output_path)

    _write_markdown_summary(report, Path("docs/artifacts/bias_analysis_report.md"))

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run bias analysis from the command line.

    Searches for the most recent experiment run directory or generates a
    skeleton report if no runs exist.
    """
    logging.basicConfig(level=logging.INFO)

    experiments_dir = Path("experiments")
    run_dirs = sorted(experiments_dir.glob("*")) if experiments_dir.exists() else []
    run_dir = run_dirs[-1] if run_dirs else experiments_dir

    output_path = Path("docs/artifacts/bias_analysis.json")

    report = run_bias_analysis(run_dir=run_dir, output_path=output_path)

    per_animal_count = (
        len(report["per_animal_performance"])
        if isinstance(report["per_animal_performance"], dict)
        and "note" not in report["per_animal_performance"]
        else 0
    )
    print(f"Bias analysis complete.")
    print(f"  JSON report:      {output_path}")
    print(f"  Markdown report:  docs/artifacts/bias_analysis_report.md")
    print(f"  Animals analysed: {per_animal_count}")
    print(f"  Deployment risks: {len(report['deployment_risks'])}")


if __name__ == "__main__":
    main()
