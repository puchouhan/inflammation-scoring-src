"""
Figure Quality Assurance Module.

Automated quality checks for matplotlib figures and PNG files: axis label
presence, legend completeness, font size compliance, ROC square geometry,
and file-level integrity (MLv13 compliance: ML-19).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)

_MIN_FONT_SIZE: float = 8.0
_MIN_DPI: int = 100


def configure_figure_defaults() -> None:
    """Configure matplotlib rcParams for consistent project-wide figure quality.

    Sets DPI, font sizes, and layout defaults. Call this once at the start of
    a script or notebook to enforce uniform figure standards across all plots.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.autolayout": True,
        }
    )
    logger.debug("Matplotlib figure defaults configured")


class FigureQA:
    """Quality assurance checker for matplotlib Figure objects and PNG files.

    Provides three independent checking methods:
    - check_png_file: file-level checks (existence, size, DPI metadata)
    - check_figure_quality: in-memory Figure object checks
    - scan_figures_directory: batch PNG checks over a directory
    """

    def check_png_file(self, file_path: Path) -> dict:
        """Check basic quality properties of a PNG file on disk.

        Verifies that the file exists, is non-empty, and (if DPI metadata is
        present) meets the minimum DPI requirement. DPI check is best-effort:
        many PNGs lack embedded DPI metadata, which generates no error.

        Args:
            file_path: Absolute path to the PNG file.

        Returns:
            Dictionary with keys:
                passed (bool): True if no errors were found.
                warnings (list[str]): Non-fatal quality issues.
                errors (list[str]): Fatal quality failures.
                file_size_bytes (int): File size in bytes.
                dpi (tuple or None): Embedded DPI from metadata, or None.
        """
        warnings: List[str] = []
        errors: List[str] = []

        if not file_path.exists():
            errors.append(f"File not found: {file_path}")
            return {
                "passed": False,
                "warnings": warnings,
                "errors": errors,
                "file_size_bytes": 0,
                "dpi": None,
            }

        file_size: int = file_path.stat().st_size
        if file_size == 0:
            errors.append(f"File is empty (0 bytes): {file_path}")
            return {
                "passed": False,
                "warnings": warnings,
                "errors": errors,
                "file_size_bytes": 0,
                "dpi": None,
            }

        dpi_value: Optional[tuple] = None
        try:
            from PIL import Image

            with Image.open(str(file_path)) as img:
                dpi_meta = img.info.get("dpi")
                if dpi_meta is not None:
                    dpi_value = tuple(dpi_meta)
                    dpi_min = min(dpi_value[0], dpi_value[1])
                    if dpi_min < _MIN_DPI:
                        warnings.append(
                            f"Embedded DPI is {dpi_min:.0f}, "
                            f"below recommended minimum of {_MIN_DPI}"
                        )
        except ImportError:
            warnings.append("Pillow not installed; DPI metadata check skipped")
        except Exception as exc:
            warnings.append(f"Could not read image metadata: {exc}")

        return {
            "passed": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
            "file_size_bytes": file_size,
            "dpi": dpi_value,
        }

    def check_figure_quality(self, fig: plt.Figure) -> dict:
        """Check quality properties of a live matplotlib Figure object.

        Inspects each Axes in the figure for:
        - Non-empty title or axis labels
        - Legend presence when multiple Line2D objects are plotted
        - Minimum font size (>= 8pt) on titles, labels, and tick labels
        - Square aspect ratio for axes whose title contains "roc" (case-insensitive)

        Args:
            fig: A matplotlib Figure instance to inspect.

        Returns:
            Dictionary with keys:
                passed (bool): True if no errors were found.
                warnings (list[str]): Non-fatal quality issues.
                errors (list[str]): Fatal quality failures.
        """
        warnings: List[str] = []
        errors: List[str] = []

        for ax_idx, ax in enumerate(fig.get_axes()):
            label_prefix = f"Axes[{ax_idx}]"
            title = ax.get_title().strip()
            xlabel = ax.get_xlabel().strip()
            ylabel = ax.get_ylabel().strip()

            if not title and not xlabel and not ylabel:
                warnings.append(
                    f"{label_prefix}: no title, x-label, or y-label is set"
                )

            lines = ax.get_lines()
            if len(lines) > 1 and ax.get_legend() is None:
                warnings.append(
                    f"{label_prefix}: {len(lines)} lines plotted but no legend present"
                )

            for text_obj, text_name in [
                (ax.title, "title"),
                (ax.xaxis.label, "x-label"),
                (ax.yaxis.label, "y-label"),
            ]:
                fs = text_obj.get_fontsize()
                if fs is not None and fs < _MIN_FONT_SIZE:
                    warnings.append(
                        f"{label_prefix}: {text_name} font size {fs:.1f}pt "
                        f"is below minimum {_MIN_FONT_SIZE}pt"
                    )

            for tick_axis_name, tick_axis in [
                ("x-tick", ax.xaxis),
                ("y-tick", ax.yaxis),
            ]:
                tick_labels = tick_axis.get_ticklabels()
                if tick_labels:
                    fs = tick_labels[0].get_fontsize()
                    if fs is not None and fs < _MIN_FONT_SIZE:
                        warnings.append(
                            f"{label_prefix}: {tick_axis_name} font size {fs:.1f}pt "
                            f"is below minimum {_MIN_FONT_SIZE}pt"
                        )

            if "roc" in title.lower():
                aspect = ax.get_aspect()
                if aspect != "equal" and aspect != 1.0:
                    errors.append(
                        f"{label_prefix}: ROC curve axes are not square "
                        f"(aspect={aspect!r}); call ax.set_aspect('equal') "
                        "for geometrically correct AUC visualization"
                    )

        return {
            "passed": len(errors) == 0,
            "warnings": warnings,
            "errors": errors,
        }

    def scan_figures_directory(self, fig_dir: Path) -> dict:
        """Scan all PNG files in a directory and return a QA summary report.

        Runs check_png_file on every *.png file found directly in fig_dir
        (non-recursive). Results are aggregated into pass/fail counts.

        Args:
            fig_dir: Path to the directory containing PNG figure files.

        Returns:
            Dictionary with keys:
                directory (str): Scanned directory path.
                total_files (int): Number of PNG files found.
                passed (int): Files that passed all checks.
                failed (int): Files with at least one error.
                file_results (list[dict]): Per-file result dicts.
        """
        if not fig_dir.exists():
            logger.warning(f"Figures directory does not exist: {fig_dir}")
            return {
                "directory": str(fig_dir),
                "total_files": 0,
                "passed": 0,
                "failed": 0,
                "file_results": [],
            }

        png_files = sorted(fig_dir.glob("*.png"))
        file_results: List[dict] = []
        passed_count = 0
        failed_count = 0

        for png_file in png_files:
            result = self.check_png_file(png_file)
            result["filename"] = png_file.name
            file_results.append(result)
            if result["passed"]:
                passed_count += 1
            else:
                failed_count += 1

        logger.info(
            f"Scanned {len(png_files)} PNG files in {fig_dir}: "
            f"{passed_count} passed, {failed_count} failed"
        )

        return {
            "directory": str(fig_dir),
            "total_files": len(png_files),
            "passed": passed_count,
            "failed": failed_count,
            "file_results": file_results,
        }


def run_figure_qa(experiment_dir: Path, output_path: Path) -> dict:
    """Run PNG-level figure QA across all model figure directories.

    Recursively scans experiment_dir for paths matching **/figures/*.png and
    applies check_png_file to each file found. Aggregates results into a
    single report and saves it as JSON.

    Args:
        experiment_dir: Root experiment directory containing model subdirs.
        output_path: Path where the JSON QA report will be saved.

    Returns:
        Aggregate QA report dictionary.
    """
    if not experiment_dir.exists():
        logger.error(f"Experiment directory does not exist: {experiment_dir}")
        return {"error": f"Directory not found: {experiment_dir}"}

    qa = FigureQA()
    png_files = sorted(experiment_dir.rglob("figures/*.png"))
    logger.info(f"Found {len(png_files)} figure files in {experiment_dir}")

    all_results: List[dict] = []
    for png_file in png_files:
        result = qa.check_png_file(png_file)
        result["path"] = str(png_file.relative_to(experiment_dir))
        all_results.append(result)

    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    failed = total - passed

    report: dict = {
        "experiment_dir": str(experiment_dir),
        "total_figures": total,
        "passed": passed,
        "failed": failed,
        "results": all_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)

    logger.info(
        f"QA report saved to {output_path} ({passed}/{total} figures passed)"
    )
    return report


def main() -> None:
    """Entry point for figure quality assurance.

    Parses --experiment-dir and --output CLI arguments, then calls
    run_figure_qa.
    """
    parser = argparse.ArgumentParser(
        description="Run figure quality assurance checks on experiment PNG files"
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        type=Path,
        help="Root experiment directory to scan for figure files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for the JSON QA report",
    )
    args = parser.parse_args()
    run_figure_qa(args.experiment_dir, args.output)


if __name__ == "__main__":
    main()
