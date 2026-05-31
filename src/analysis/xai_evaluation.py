# src/analysis/xai_evaluation.py
"""Consolidated xAI evaluation report generator (ML-24).

Ties per-model GradCAM attribution statistics to a biological plausibility
checklist and produces JSON + Markdown artifacts for the experiment run.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Local imports path setup
_project_root: str = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.append(_project_root)

from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)

_BIOLOGICAL_PLAUSIBILITY_CHECKLIST: List[str] = [
    "Heatmaps focus on tissue areas, not background",
    "Inflammatory infiltrates visible in high-activation regions",
    "Low-score (class 0) images show diffuse/low activation",
    "High-score (class 3) images show concentrated focal activation",
]

_LIMITATIONS_TEXT: str = (
    "Heatmaps require visual inspection by domain expert for full biological validation."
)

_NOT_YET_GENERATED: str = "NOT YET GENERATED"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_heatmap_dirs(run_dir: Path) -> Dict[str, Path]:
    """Discover per-model xAI heatmap directories under run_dir.

    Searches for the pattern run_dir/{model_name}/figures/xai/.

    Args:
        run_dir: Root experiment run directory.

    Returns:
        Mapping of model_name -> heatmap directory Path.
    """
    heatmap_dirs: Dict[str, Path] = {}
    for figures_dir in run_dir.glob("*/figures"):
        model_name: str = figures_dir.parent.name
        xai_dir: Path = figures_dir / "xai"
        if xai_dir.is_dir():
            heatmap_dirs[model_name] = xai_dir
    return heatmap_dirs


def _fmt_value(value: Optional[float], fmt: str = ".4f") -> str:
    """Format a numeric value or return placeholder if absent.

    Args:
        value: Numeric value or None.
        fmt: Python format spec string.

    Returns:
        Formatted string or _NOT_YET_GENERATED placeholder.
    """
    if value is None:
        return _NOT_YET_GENERATED
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def _format_statistics_table(stats_by_model: Dict[str, dict]) -> str:
    """Render per-model attribution statistics as a Markdown table.

    Args:
        stats_by_model: Mapping of model_name -> statistics dict.

    Returns:
        Markdown table string.
    """
    col_names: List[str] = [
        "Model", "N", "Mean Peak", "Std Peak",
        "Mean Entropy", "Std Entropy", "Ctr/Bdr Ratio", "Top Region",
    ]
    separator: str = "|" + "|".join("---" for _ in col_names) + "|"
    header: str = "| " + " | ".join(col_names) + " |"
    rows: List[str] = [header, separator]

    for model_name in sorted(stats_by_model.keys()):
        stats: dict = stats_by_model[model_name]
        if stats.get("error"):
            cells = [model_name, "0"] + [_NOT_YET_GENERATED] * 6
        else:
            cells = [
                model_name,
                str(stats.get("n_heatmaps", 0)),
                _fmt_value(stats.get("mean_peak_activation")),
                _fmt_value(stats.get("std_peak_activation")),
                _fmt_value(stats.get("mean_entropy")),
                _fmt_value(stats.get("std_entropy")),
                _fmt_value(stats.get("center_vs_border_ratio")),
                str(stats.get("top_activated_region", _NOT_YET_GENERATED)),
            ]
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows) + "\n"


def _format_biological_plausibility_section(model_name: str) -> str:
    """Generate biological plausibility checklist markdown for one model.

    Args:
        model_name: Name of the model.

    Returns:
        Markdown string for the checklist section.
    """
    checklist: str = "\n".join(
        f"- [ ] {item}" for item in _BIOLOGICAL_PLAUSIBILITY_CHECKLIST
    )
    return f"### {model_name}\n\n{checklist}\n"


def _format_cross_model_comparison(stats_by_model: Dict[str, dict]) -> str:
    """Generate a cross-model comparison section ranked by center/border ratio.

    Args:
        stats_by_model: Mapping of model_name -> statistics dict.

    Returns:
        Markdown string for the comparison section.
    """
    valid: Dict[str, dict] = {
        name: s for name, s in stats_by_model.items()
        if not s.get("error") and s.get("center_vs_border_ratio") is not None
    }
    if not valid:
        return "No valid xAI outputs available for cross-model comparison.\n"

    sorted_models: List[Tuple[str, dict]] = sorted(
        valid.items(),
        key=lambda x: x[1].get("center_vs_border_ratio", 0.0),
        reverse=True,
    )

    col_names: List[str] = ["Model", "Center/Border Ratio", "Top Region"]
    separator: str = "|" + "|".join("---" for _ in col_names) + "|"
    header: str = "| " + " | ".join(col_names) + " |"
    rows: List[str] = [header, separator]
    for name, stats in sorted_models:
        cells = [
            name,
            _fmt_value(stats.get("center_vs_border_ratio")),
            str(stats.get("top_activated_region", _NOT_YET_GENERATED)),
        ]
        rows.append("| " + " | ".join(cells) + " |")

    most_focused: str = sorted_models[0][0]
    least_focused: str = sorted_models[-1][0]
    summary: str = (
        f"\nModels ranked by center/border activation ratio (higher = more focal activation).\n"
        f"Most focused: **{most_focused}**. Least focused: **{least_focused}**.\n"
    )
    return "\n".join(rows) + "\n" + summary


def _build_markdown_report(
    stats_by_model: Dict[str, dict],
    run_dir: Path,
    has_real_data: bool,
) -> str:
    """Assemble the full consolidated xAI evaluation Markdown report.

    Args:
        stats_by_model: Mapping of model_name -> statistics dict.
        run_dir: Experiment run directory used for reference.
        has_real_data: Whether any real heatmap data was found.

    Returns:
        Full Markdown report string.
    """
    sections: List[str] = [
        "# xAI Evaluation Report\n",
        f"\nRun directory: `{run_dir}`\n",
        "\n---\n",
        "\n## 1. Per-Model Attribution Statistics\n\n",
        _format_statistics_table(stats_by_model),
        "\n---\n",
        "\n## 2. Biological Plausibility Checklist\n\n",
        "> Note: Items below require manual visual inspection by a domain expert.\n\n",
    ]

    if has_real_data:
        for model_name in sorted(stats_by_model.keys()):
            sections.append(_format_biological_plausibility_section(model_name))
    else:
        sections.append("No xAI outputs found yet. Checklist template:\n\n")
        template_checklist: str = "\n".join(
            f"- [ ] {item}" for item in _BIOLOGICAL_PLAUSIBILITY_CHECKLIST
        )
        sections.append(template_checklist + "\n")

    sections += [
        "\n---\n",
        "\n## 3. Limitations\n\n",
        _LIMITATIONS_TEXT + "\n",
        "\n---\n",
        "\n## 4. Cross-Model Comparison\n\n",
        _format_cross_model_comparison(stats_by_model),
    ]

    return "".join(sections)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_xai_evaluation(run_dir: Path, output_dir: Path) -> dict:
    """Run consolidated xAI evaluation over all models in a run directory.

    For each model found under run_dir/{model_name}/figures/xai/, calls
    compute_attribution_statistics and aggregates results into a JSON and
    Markdown report saved under output_dir.

    Falls back gracefully if no xAI outputs exist: saves a template Markdown
    report with _NOT_YET_GENERATED placeholder values for all statistics.

    Args:
        run_dir: Root experiment run directory.
        output_dir: Directory to save evaluation artifacts.

    Returns:
        Consolidated result dict with keys: stats_by_model, json_report,
        markdown_report.
    """
    from src.analysis.xai_generator import compute_attribution_statistics

    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_dirs: Dict[str, Path] = _find_heatmap_dirs(run_dir)
    if not heatmap_dirs:
        logger.warning(
            f"No xAI heatmap directories found under {run_dir}. Generating template report."
        )

    stats_by_model: Dict[str, dict] = {}
    for model_name, heatmap_dir in sorted(heatmap_dirs.items()):
        logger.info(f"Computing attribution statistics for model: {model_name}")
        stats_output: Path = output_dir / f"{model_name}_attribution_stats.json"
        try:
            stats: dict = compute_attribution_statistics(heatmap_dir, stats_output)
        except Exception as e:
            logger.error(
                f"Failed to compute attribution statistics for {model_name}: {e}"
            )
            stats = {"error": str(e), "n_heatmaps": 0}
        stats_by_model[model_name] = stats

    has_real_data: bool = bool(heatmap_dirs)
    if not stats_by_model:
        stats_by_model = {
            _NOT_YET_GENERATED: {"error": "No xAI outputs found", "n_heatmaps": 0}
        }

    json_output: Path = output_dir / "xai_evaluation_report.json"
    with open(json_output, "w", encoding="utf-8") as fh:
        json.dump(stats_by_model, fh, indent=2)
    logger.info(f"Saved xAI evaluation JSON: {json_output}")

    md_report: str = _build_markdown_report(stats_by_model, run_dir, has_real_data)
    md_output: Path = output_dir / "xai_evaluation_report.md"
    with open(md_output, "w", encoding="utf-8") as fh:
        fh.write(md_report)
    logger.info(f"Saved xAI evaluation report: {md_output}")

    return {
        "stats_by_model": stats_by_model,
        "json_report": str(json_output),
        "markdown_report": str(md_output),
    }


def main() -> None:
    """CLI entry point for xAI evaluation."""
    parser = argparse.ArgumentParser(
        description="Run consolidated xAI evaluation over a training run directory."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Root experiment run directory (e.g. experiments/20260101_120000_convnext).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save xAI evaluation artifacts.",
    )
    args = parser.parse_args()
    run_xai_evaluation(args.run_dir, args.output_dir)
    logger.info(f"xAI evaluation complete. Reports saved to {args.output_dir}")


if __name__ == "__main__":
    main()
