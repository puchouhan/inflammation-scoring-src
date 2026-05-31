"""Spatial Inflammation Map Module.

Generates whole-slide inflammation heatmaps from tile-level predictions,
similar to Fig 6 in Heinemann et al. (2018). For each animal's largest
slide, shows the original tile mosaic alongside color-coded inflammation
predictions from the best model per architecture family.

Layout per figure (2 rows x 3 cols):
    [Original H&E]    [Ground Truth]      [Best CNN]
    [Best Transformer] [Best Hybrid]      [Best Self-Supervised]
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from src.data.inflammation_dataset import parse_filename

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (shared with model_comparison_plots where applicable)
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

# Discrete inflammation colormap (0-3 + ignore)
INFLAMMATION_COLORS: List[str] = [
    "#2166AC",  # Grade 0 (none)   - dark blue
    "#67A9CF",  # Grade 1 (mild)   - light blue
    "#FDDBC7",  # Grade 2 (moderate) - light orange
    "#B2182B",  # Grade 3 (severe) - dark red
]

INFLAMMATION_LABELS: List[str] = [
    "Grade 0 (none)",
    "Grade 1 (mild)",
    "Grade 2 (moderate)",
    "Grade 3 (severe)",
]

IGNORE_COLOR: str = "#808080"
BACKGROUND_VALUE: float = -1.0
THUMBNAIL_SIZE: int = 16


# ---------------------------------------------------------------------------
# Registry / data helpers
# ---------------------------------------------------------------------------


def _load_registry(project_root: Path) -> Dict[str, Any]:
    """Load best_models_registry_new.json."""
    for candidate in [
        project_root / "src" / "experiments" / "best_models_registry_new.json",
        project_root / "experiments" / "best_models_registry_new.json",
    ]:
        if candidate.exists():
            with open(candidate, "r") as f:
                return json.load(f)
    raise FileNotFoundError("best_models_registry_new.json not found")


def _get_best_model_per_family(
    registry: Dict[str, Any],
    cv_strategy: str,
) -> Dict[str, str]:
    """Pick the best model (by mean_qwk) from each architecture family.

    Args:
        registry: Loaded best_models_registry.json.
        cv_strategy: CV strategy filter.

    Returns:
        {family_name: model_name}, e.g., {"CNN": "convnext", ...}
    """
    suffix = (
        "_stratified" if cv_strategy == "random_stratified" else "_loao"
    )
    family_best: Dict[str, Tuple[str, float]] = {}

    for key, entry in registry.items():
        if not key.endswith(suffix):
            continue
        base_name = key[: -len(suffix)]
        family = ARCHITECTURE_FAMILIES.get(base_name)
        if not family:
            continue
        # Registry nests data: {"convnext_stratified": {"random_stratified": {...}}}
        cv_data = entry.get(cv_strategy, {})
        if not cv_data:
            cv_data = list(entry.values())[0] if entry else {}
        mean_qwk = cv_data.get("mean_qwk", 0.0)
        if family not in family_best or mean_qwk > family_best[family][1]:
            family_best[family] = (base_name, mean_qwk)

    return {fam: name for fam, (name, _) in family_best.items()}


def _find_experiment_dir(
    exp_root: Path,
    run_id: str,
) -> Optional[Path]:
    """Locate experiment directory by run_id prefix.

    Args:
        exp_root: Root experiments directory.
        run_id: Run identifier (timestamp prefix).

    Returns:
        Path to experiment directory, or None.
    """
    if not exp_root.exists():
        return None
    for d in sorted(exp_root.iterdir()):
        if d.is_dir() and run_id in d.name:
            return d
    return None


def _collect_predictions_for_model(
    project_root: Path,
    registry: Dict[str, Any],
    model_name: str,
    cv_strategy: str,
    experiments_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Load and combine predictions from all folds for one model.

    Combines validation-set predictions across folds so that every tile
    appears exactly once (each tile is validated in exactly one fold).

    Args:
        project_root: Project root.
        registry: Loaded registry.
        model_name: Model key (e.g., "convnext").
        cv_strategy: CV strategy filter.
        experiments_dir: Override experiments dir.

    Returns:
        Combined DataFrame with added columns: animal_id, slide_id, x, y.
        Empty DataFrame if no predictions found.
    """
    suffix = (
        "_stratified" if cv_strategy == "random_stratified" else "_loao"
    )
    reg_key = f"{model_name}{suffix}"
    entry = registry.get(reg_key, {})
    # Registry nests data: {"convnext_stratified": {"random_stratified": {...}}}
    cv_data = entry.get(cv_strategy, {})
    if not cv_data:
        cv_data = list(entry.values())[0] if entry else {}
    run_id = cv_data.get("run_id", "")

    if not run_id:
        logger.warning(f"No run_id for {reg_key}")
        return pd.DataFrame()

    exp_root = experiments_dir or project_root / "src" / "experiments"
    exp_dir = _find_experiment_dir(exp_root, run_id)

    if exp_dir is None:
        logger.warning(f"Experiment dir not found for run_id {run_id}")
        return pd.DataFrame()

    preds_dir = exp_dir / model_name / "predictions"
    if not preds_dir.exists():
        logger.warning(f"No predictions dir: {preds_dir}")
        return pd.DataFrame()

    all_dfs: List[pd.DataFrame] = []
    for csv_file in sorted(preds_dir.glob("fold_*_predictions.csv")):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {csv_file}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    animals: List[str] = []
    slides: List[str] = []
    xs: List[int] = []
    ys: List[int] = []
    for fp in combined["filepath"]:
        filename = Path(fp).name
        animal_id, slide_id, x, y = parse_filename(filename)
        animals.append(animal_id)
        slides.append(slide_id)
        xs.append(x)
        ys.append(y)

    combined["animal_id"] = animals
    combined["slide_id"] = slides
    combined["x"] = xs
    combined["y"] = ys

    return combined


def _get_largest_slide_per_animal(
    df: pd.DataFrame,
) -> Dict[str, str]:
    """Find the slide with most tiles for each animal.

    Args:
        df: Predictions DataFrame with animal_id and slide_id columns.

    Returns:
        {animal_id: slide_id}
    """
    counts = (
        df.groupby(["animal_id", "slide_id"])
        .size()
        .reset_index(name="n")
    )
    idx = counts.groupby("animal_id")["n"].idxmax()
    selected = counts.loc[idx]
    return dict(zip(selected["animal_id"], selected["slide_id"]))


def _get_most_inflamed_slide(
    df: pd.DataFrame,
    min_tiles: int = 20,
) -> Tuple[str, str]:
    """Find the single (animal, slide) with the highest proportion of Grade 2/3 tiles.

    Selects the slide with the most high-grade inflammation (grades 2 and 3)
    relative to total non-ignore tiles. Only considers slides with at least
    `min_tiles` tiles to avoid trivial single-tile slides.

    Args:
        df: Predictions DataFrame with animal_id, slide_id, ground_truth columns.
        min_tiles: Minimum number of tiles required for a slide to be considered.

    Returns:
        (animal_id, slide_id) tuple for the most inflamed slide.
    """
    gt_col = pd.to_numeric(df["ground_truth"], errors="coerce")
    valid_mask = gt_col < 4
    valid = df[valid_mask].copy()
    valid["_gt_num"] = gt_col[valid_mask].astype(int)

    counts = valid.groupby(["animal_id", "slide_id"]).size().reset_index(name="total")
    counts = counts[counts["total"] >= min_tiles]

    high_grade = (
        valid[valid["_gt_num"] >= 2]
        .groupby(["animal_id", "slide_id"])
        .size()
        .reset_index(name="high_grade_n")
    )
    merged = counts.merge(high_grade, on=["animal_id", "slide_id"], how="left")
    merged["high_grade_n"] = merged["high_grade_n"].fillna(0)
    merged["high_grade_ratio"] = merged["high_grade_n"] / merged["total"]

    logger.info(
        "Grade distribution in reference_df: %s",
        dict(pd.to_numeric(df["ground_truth"], errors="coerce").value_counts().sort_index()),
    )
    logger.info(
        "Top 5 slides by high_grade_ratio:\n%s",
        merged.nlargest(5, "high_grade_ratio")[["animal_id", "slide_id", "high_grade_n", "total", "high_grade_ratio"]].to_string(),
    )

    best_row = merged.loc[merged["high_grade_ratio"].idxmax()]
    animal_id = str(best_row["animal_id"])
    slide_id = str(best_row["slide_id"])
    logger.info(
        "Most inflamed slide: animal=%s slide=%s high_grade_ratio=%.3f (%d/%d tiles)",
        animal_id, slide_id,
        best_row["high_grade_ratio"],
        int(best_row["high_grade_n"]),
        int(best_row["total"]),
    )
    return animal_id, slide_id


def _get_most_inflamed_slide_from_dataset(
    data_roots: List[Path],
    allowed_animals: Optional[List[str]] = None,
) -> Optional[Tuple[str, str]]:
    """Find the slide with the most Grade 2/3 tiles by scanning dataset directories.

    More reliable than CSV-based selection because it uses the ground-truth
    label directory structure directly.

    Args:
        data_roots: List of dataset root directories to scan (e.g. dataset_norm/training).
        allowed_animals: If given, restrict to these animal_ids only.

    Returns:
        (animal_id, slide_id) or None if no data found.
    """
    from src.data.inflammation_dataset import parse_filename

    from src.data.inflammation_dataset import parse_filename as _parse

    rows: List[Dict[str, Any]] = []
    for root in data_roots:
        for grade_dir in root.iterdir():
            if not grade_dir.is_dir():
                continue
            try:
                grade = int(grade_dir.name)
            except ValueError:
                continue  # skip 'ignore'
            for img in grade_dir.glob("*.png"):
                try:
                    a_id, s_id, _, _ = _parse(img.name)
                    rows.append({"animal_id": a_id, "slide_id": s_id, "grade": grade})
                except (ValueError, IndexError):
                    continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if allowed_animals:
        df = df[df["animal_id"].isin(allowed_animals)]
    if df.empty:
        return None

    counts = df.groupby(["animal_id", "slide_id"]).size().reset_index(name="total")
    high_grade = (
        df[df["grade"] >= 2]
        .groupby(["animal_id", "slide_id"])
        .size()
        .reset_index(name="high_grade_n")
    )
    merged = counts.merge(high_grade, on=["animal_id", "slide_id"], how="left")
    merged["high_grade_n"] = merged["high_grade_n"].fillna(0)
    merged["high_grade_ratio"] = merged["high_grade_n"] / merged["total"]

    logger.info(
        "Top 5 slides from dataset scan:\n%s",
        merged.nlargest(5, "high_grade_n")[["animal_id", "slide_id", "high_grade_n", "total", "high_grade_ratio"]].to_string(),
    )
    best_row = merged.loc[merged["high_grade_n"].idxmax()]
    animal_id = str(best_row["animal_id"])
    slide_id = str(best_row["slide_id"])
    logger.info(
        "Dataset-based selection: animal=%s slide=%s high_grade_n=%d total=%d ratio=%.3f",
        animal_id, slide_id,
        int(best_row["high_grade_n"]),
        int(best_row["total"]),
        best_row["high_grade_ratio"],
    )
    return animal_id, slide_id


def _get_least_inflamed_slide_from_dataset(
    data_roots: List[Path],
    allowed_animals: Optional[List[str]] = None,
    min_tiles: int = 20,
    exclude: Optional[Tuple[str, str]] = None,
) -> Optional[Tuple[str, str]]:
    """Find the slide with the fewest Grade 2/3 tiles (mostly Grade 0/1).

    Provides a low-inflammation contrast example to complement the most-inflamed
    slide. Only considers slides with at least `min_tiles` tiles.

    Args:
        data_roots: List of dataset root directories to scan.
        allowed_animals: If given, restrict to these animal_ids only.
        min_tiles: Minimum tile count to consider a slide.
        exclude: (animal_id, slide_id) to exclude (e.g. the most-inflamed slide).

    Returns:
        (animal_id, slide_id) or None if no data found.
    """
    from src.data.inflammation_dataset import parse_filename as _parse

    rows: List[Dict[str, Any]] = []
    for root in data_roots:
        for grade_dir in root.iterdir():
            if not grade_dir.is_dir():
                continue
            try:
                grade = int(grade_dir.name)
            except ValueError:
                continue
            for img in grade_dir.glob("*.png"):
                try:
                    a_id, s_id, _, _ = _parse(img.name)
                    rows.append({"animal_id": a_id, "slide_id": s_id, "grade": grade})
                except (ValueError, IndexError):
                    continue

    if not rows:
        return None

    df = pd.DataFrame(rows)
    if allowed_animals:
        df = df[df["animal_id"].isin(allowed_animals)]
    if df.empty:
        return None

    counts = df.groupby(["animal_id", "slide_id"]).size().reset_index(name="total")
    counts = counts[counts["total"] >= min_tiles]

    high_grade = (
        df[df["grade"] >= 2]
        .groupby(["animal_id", "slide_id"])
        .size()
        .reset_index(name="high_grade_n")
    )
    merged = counts.merge(high_grade, on=["animal_id", "slide_id"], how="left")
    merged["high_grade_n"] = merged["high_grade_n"].fillna(0)
    merged["high_grade_ratio"] = merged["high_grade_n"] / merged["total"]

    if exclude is not None:
        merged = merged[
            ~((merged["animal_id"] == exclude[0]) & (merged["slide_id"] == exclude[1]))
        ]
    if merged.empty:
        return None

    best_row = merged.loc[merged["high_grade_n"].idxmin()]
    animal_id = str(best_row["animal_id"])
    slide_id = str(best_row["slide_id"])
    logger.info(
        "Least-inflamed slide: animal=%s slide=%s high_grade_n=%d total=%d ratio=%.3f",
        animal_id, slide_id,
        int(best_row["high_grade_n"]),
        int(best_row["total"]),
        best_row["high_grade_ratio"],
    )
    return animal_id, slide_id


# ---------------------------------------------------------------------------
# Image / grid construction
# ---------------------------------------------------------------------------


def _find_tile_path(
    filepath: str,
    data_roots: List[Path],
) -> Optional[Path]:
    """Locate a tile image on disk using the predictions CSV filepath.

    Args:
        filepath: Relative path from predictions CSV (e.g. "2/17_305_01_25_17.png").
        data_roots: Root directories to search.

    Returns:
        Absolute path to image, or None.
    """
    for root in data_roots:
        candidate = root / filepath
        if candidate.exists():
            return candidate
    return None


def _build_tissue_mosaic(
    slide_df: pd.DataFrame,
    data_roots: List[Path],
    thumbnail_size: int = THUMBNAIL_SIZE,
) -> np.ndarray:
    """Build a downsampled tissue mosaic from individual tiles.

    Each tile is resized to (thumbnail_size x thumbnail_size) and placed
    at its (x, y) grid position. Empty grid cells are white.

    Args:
        slide_df: DataFrame for one slide with x, y, filepath columns.
        data_roots: Root directories to find tile images.
        thumbnail_size: Downsample each tile to this size (pixels).

    Returns:
        RGB numpy array of shape (grid_h * ts, grid_w * ts, 3).
    """
    x_min = slide_df["x"].min()
    x_max = slide_df["x"].max()
    y_min = slide_df["y"].min()
    y_max = slide_df["y"].max()
    grid_w = x_max - x_min + 1
    grid_h = y_max - y_min + 1
    ts = thumbnail_size

    mosaic = np.ones((grid_h * ts, grid_w * ts, 3), dtype=np.uint8) * 255

    for _, row in slide_df.iterrows():
        tile_path = _find_tile_path(row["filepath"], data_roots)
        if tile_path is None:
            continue

        img = cv2.imread(str(tile_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        thumb = cv2.resize(img, (ts, ts), interpolation=cv2.INTER_AREA)

        gx = row["x"] - x_min
        gy = row["y"] - y_min
        mosaic[gy * ts : (gy + 1) * ts, gx * ts : (gx + 1) * ts] = thumb

    return mosaic


def _build_class_grid(
    slide_df: pd.DataFrame,
    value_column: str,
) -> np.ndarray:
    """Build a 2D grid of class values for heatmap display.

    Args:
        slide_df: DataFrame for one slide with x, y columns.
        value_column: Column with class values ("prediction" or "ground_truth").

    Returns:
        Float array of shape (grid_h, grid_w) with NaN for empty cells.
    """
    x_min = slide_df["x"].min()
    x_max = slide_df["x"].max()
    y_min = slide_df["y"].min()
    y_max = slide_df["y"].max()
    grid_w = x_max - x_min + 1
    grid_h = y_max - y_min + 1

    grid = np.full((grid_h, grid_w), np.nan)

    for _, row in slide_df.iterrows():
        gx = row["x"] - x_min
        gy = row["y"] - y_min
        grid[gy, gx] = float(row[value_column])

    return grid


# ---------------------------------------------------------------------------
# Colormap helpers
# ---------------------------------------------------------------------------


def _make_inflammation_cmap() -> Tuple[ListedColormap, BoundaryNorm]:
    """Create discrete colormap for inflammation classes 0-4.

    Returns:
        (cmap, norm) tuple for use with imshow.
    """
    colors = INFLAMMATION_COLORS + [IGNORE_COLOR]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _make_legend_handles() -> List[Line2D]:
    """Create legend handles for inflammation classes."""
    handles = [
        Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=c, markersize=10, label=lbl,
        )
        for c, lbl in zip(INFLAMMATION_COLORS, INFLAMMATION_LABELS)
    ]
    handles.append(
        Line2D(
            [0], [0], marker="s", color="w",
            markerfacecolor=IGNORE_COLOR, markersize=10,
            label="Ignore (artifact)",
        ),
    )
    return handles


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------


def plot_spatial_inflammation_maps(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
    thumbnail_size: int = THUMBNAIL_SIZE,
) -> List[Path]:
    """Generate spatial inflammation maps for each animal.

    Creates one figure per animal showing the original tissue mosaic,
    ground truth inflammation map, and predictions from the best model
    per architecture family (CNN, Transformer, Hybrid, Self-Supervised).

    Layout per figure (2 rows x 3 cols):
        [Original H&E]     [Ground Truth]      [Best CNN]
        [Best Transformer]  [Best Hybrid]       [Best Self-Supervised]

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy for predictions.
        output_dir: Directory to save figures.
        experiments_dir: Override experiments directory.
        thumbnail_size: Pixels per tile in tissue mosaic.

    Returns:
        List of paths to saved PNGs.
    """
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    family_models = _get_best_model_per_family(registry, cv_strategy)
    if not family_models:
        logger.warning("No models found for spatial inflammation maps")
        return []

    logger.info(
        "Best model per family (%s): %s",
        cv_strategy,
        ", ".join(f"{f}: {m}" for f, m in family_models.items()),
    )

    model_predictions = _load_all_model_predictions(
        project_root, registry, family_models, cv_strategy,
        experiments_dir,
    )
    if not model_predictions:
        logger.warning("No predictions available for spatial maps")
        return []

    reference_df = next(iter(model_predictions.values()))
    data_roots = _get_data_roots(project_root)
    cmap, norm = _make_inflammation_cmap()

    # Prefer dataset-directory scan (uses ground-truth labels directly).
    # Fall back to predictions-CSV scan if dataset is not mounted.
    known_animals = sorted(reference_df["animal_id"].unique().tolist())

    most_sel = _get_most_inflamed_slide_from_dataset(
        data_roots, allowed_animals=known_animals
    )
    if most_sel is None:
        logger.warning(
            "Dataset scan failed (no data roots found); "
            "falling back to predictions-CSV slide selection"
        )
        most_sel = _get_most_inflamed_slide(reference_df)

    least_sel = _get_least_inflamed_slide_from_dataset(
        data_roots, allowed_animals=known_animals, exclude=most_sel
    )

    selections: List[Tuple[str, str, str]] = []
    if most_sel is not None:
        selections.append((most_sel[0], most_sel[1], "high_inflammation"))
    if least_sel is not None:
        selections.append((least_sel[0], least_sel[1], "low_inflammation"))

    saved_paths: List[Path] = []
    for animal_id, slide_id, label in selections:
        path = _generate_single_animal_figure(
            animal_id=animal_id,
            slide_id=slide_id,
            family_models=family_models,
            model_predictions=model_predictions,
            reference_df=reference_df,
            data_roots=data_roots,
            cmap=cmap,
            norm=norm,
            output_dir=output_dir,
            cv_strategy=cv_strategy,
            thumbnail_size=thumbnail_size,
            label=label,
        )
        if path is not None:
            saved_paths.append(path)

    logger.info(
        "Generated %d spatial inflammation maps in %s",
        len(saved_paths), output_dir,
    )
    return saved_paths


# ---------------------------------------------------------------------------
# Internal orchestration helpers
# ---------------------------------------------------------------------------


def _load_all_model_predictions(
    project_root: Path,
    registry: Dict[str, Any],
    family_models: Dict[str, str],
    cv_strategy: str,
    experiments_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Load predictions for every selected model.

    Args:
        project_root: Project root.
        registry: Loaded registry.
        family_models: {family: model_name} mapping.
        cv_strategy: CV strategy filter.
        experiments_dir: Override experiments dir.

    Returns:
        {model_name: DataFrame} for models with available predictions.
    """
    result: Dict[str, pd.DataFrame] = {}
    for family in FAMILY_ORDER:
        model_name = family_models.get(family)
        if not model_name:
            continue
        df = _collect_predictions_for_model(
            project_root, registry, model_name, cv_strategy,
            experiments_dir=experiments_dir,
        )
        if not df.empty:
            result[model_name] = df
    return result


def _get_data_roots(project_root: Path) -> List[Path]:
    """Return existing dataset root directories for tile lookup.

    Args:
        project_root: Project root.

    Returns:
        List of existing Paths.
    """
    candidates = [
        project_root / "dataset_norm" / "training",
        project_root / "dataset_norm" / "val",
        project_root / "dataset" / "training",
        project_root / "dataset" / "val",
    ]
    return [r for r in candidates if r.exists()]


def _generate_single_animal_figure(
    *,
    animal_id: str,
    slide_id: str,
    family_models: Dict[str, str],
    model_predictions: Dict[str, pd.DataFrame],
    reference_df: pd.DataFrame,
    data_roots: List[Path],
    cmap: ListedColormap,
    norm: BoundaryNorm,
    output_dir: Path,
    cv_strategy: str,
    thumbnail_size: int,
    label: str = "",
) -> Optional[Path]:
    """Render one figure for a single animal/slide.

    Args:
        animal_id: Animal identifier.
        slide_id: Slide identifier.
        family_models: {family: model_name}.
        model_predictions: {model_name: DataFrame}.
        reference_df: A reference DataFrame for ground truth / mosaic.
        data_roots: Tile image root directories.
        cmap: Discrete inflammation colormap.
        norm: BoundaryNorm for cmap.
        output_dir: Save directory.
        cv_strategy: CV strategy label.
        thumbnail_size: Tile thumbnail size in pixels.
        label: Optional suffix for filename (e.g. "high_inflammation").
        cmap: Discrete inflammation colormap.
        norm: BoundaryNorm for cmap.
        output_dir: Save directory.
        cv_strategy: CV strategy label.
        thumbnail_size: Tile thumbnail size in pixels.

    Returns:
        Path to saved PNG, or None on failure.
    """
    logger.info(
        "Generating spatial map for animal %s, slide %s",
        animal_id, slide_id,
    )

    ref_slide_df = reference_df[
        (reference_df["animal_id"] == animal_id)
        & (reference_df["slide_id"] == slide_id)
    ]
    if ref_slide_df.empty:
        logger.warning("No data for %s / %s", animal_id, slide_id)
        return None

    panels = _build_panels(
        ref_slide_df=ref_slide_df,
        animal_id=animal_id,
        slide_id=slide_id,
        family_models=family_models,
        model_predictions=model_predictions,
        data_roots=data_roots,
        thumbnail_size=thumbnail_size,
    )
    if len(panels) < 3:
        logger.warning("Not enough panels for %s", animal_id)
        return None

    return _render_figure(
        panels=panels,
        animal_id=animal_id,
        slide_id=slide_id,
        cmap=cmap,
        norm=norm,
        output_dir=output_dir,
        cv_strategy=cv_strategy,
        label=label,
    )


def _build_panels(
    *,
    ref_slide_df: pd.DataFrame,
    animal_id: str,
    slide_id: str,
    family_models: Dict[str, str],
    model_predictions: Dict[str, pd.DataFrame],
    data_roots: List[Path],
    thumbnail_size: int,
) -> List[Tuple[str, np.ndarray]]:
    """Assemble all panels for one animal figure.

    Returns:
        List of (title, data) tuples. data is either an RGB array
        (for the mosaic) or a 2D float array (for heatmaps).
    """
    panels: List[Tuple[str, np.ndarray]] = []

    mosaic = _build_tissue_mosaic(
        ref_slide_df, data_roots, thumbnail_size,
    )
    panels.append(("Original H&E", mosaic))

    gt_grid = _build_class_grid(ref_slide_df, "ground_truth")
    panels.append(("Ground Truth", gt_grid))

    for family in FAMILY_ORDER:
        model_name = family_models.get(family)
        if not model_name or model_name not in model_predictions:
            continue
        model_df = model_predictions[model_name]
        model_slide_df = model_df[
            (model_df["animal_id"] == animal_id)
            & (model_df["slide_id"] == slide_id)
        ]
        if model_slide_df.empty:
            continue
        pred_grid = _build_class_grid(model_slide_df, "prediction")
        display = DISPLAY_NAMES.get(model_name, model_name)
        panels.append((f"{display} ({family})", pred_grid))

    return panels


def _render_figure(
    *,
    panels: List[Tuple[str, np.ndarray]],
    animal_id: str,
    slide_id: str,
    cmap: ListedColormap,
    norm: BoundaryNorm,
    output_dir: Path,
    cv_strategy: str,
    label: str = "",
) -> Path:
    """Render panels into a single figure and save to disk.

    Args:
        panels: List of (title, data) tuples.
        animal_id: Animal identifier for title/filename.
        slide_id: Slide identifier for title.
        cmap: Discrete inflammation colormap.
        norm: BoundaryNorm for cmap.
        output_dir: Save directory.
        cv_strategy: CV strategy label.

    Returns:
        Path to saved PNG.
    """
    n_panels = len(panels)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (title, data) in enumerate(panels):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        if data.ndim == 3:
            ax.imshow(data)
        else:
            ax.imshow(
                data, cmap=cmap, norm=norm, interpolation="nearest",
            )
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    legend_handles = _make_legend_handles()
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=5, fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    cv_label = (
        "Stratified" if "stratified" in cv_strategy else "LOAO"
    )
    title_suffix = " -- High Inflammation" if "high" in label else " -- Low Inflammation" if "low" in label else ""
    fig.suptitle(
        f"Spatial Inflammation Map{title_suffix} -- Animal {animal_id} "
        f"(Slide {slide_id}, {cv_label})",
        fontweight="bold", fontsize=13,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    label_suffix = f"_{label}" if label else ""
    path = (
        output_dir
        / f"spatial_inflammation_map_{animal_id}_{cv_strategy}{label_suffix}.png"
    )
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Continuous inflammation score heatmap (Fig 9-style)
# ---------------------------------------------------------------------------


def _get_best_overall_model(
    registry: Dict[str, Any],
    cv_strategy: str,
) -> Optional[str]:
    """Find the single best model by mean_qwk across all families.

    Args:
        registry: Loaded best_models_registry.json.
        cv_strategy: CV strategy filter.

    Returns:
        Model base name (e.g., "convnext"), or None.
    """
    suffix = (
        "_stratified" if cv_strategy == "random_stratified" else "_loao"
    )
    best_name: Optional[str] = None
    best_qwk: float = -1.0

    for key, entry in registry.items():
        if not key.endswith(suffix):
            continue
        base_name = key[: -len(suffix)]
        cv_data = entry.get(cv_strategy, {})
        if not cv_data:
            cv_data = list(entry.values())[0] if entry else {}
        mean_qwk = cv_data.get("mean_qwk", 0.0)
        if mean_qwk > best_qwk:
            best_qwk = mean_qwk
            best_name = base_name

    return best_name


def _build_continuous_score_grid(
    slide_df: pd.DataFrame,
) -> np.ndarray:
    """Build a 2D grid of continuous inflammation scores.

    Computes weighted average: score = sum(p_i * i) for i in 0..3,
    using only classes 0-3 (re-normalized after excluding class 4).

    Args:
        slide_df: DataFrame with confidence_0..confidence_3 columns.

    Returns:
        Float array of shape (grid_h, grid_w) with NaN for empty cells.
    """
    x_min, x_max = slide_df["x"].min(), slide_df["x"].max()
    y_min, y_max = slide_df["y"].min(), slide_df["y"].max()
    grid_w = x_max - x_min + 1
    grid_h = y_max - y_min + 1

    grid = np.full((grid_h, grid_w), np.nan)

    conf_cols = [f"confidence_{i}" for i in range(4)]
    has_conf = all(c in slide_df.columns for c in conf_cols)

    for _, row in slide_df.iterrows():
        gx = row["x"] - x_min
        gy = row["y"] - y_min

        if has_conf:
            probs = np.array([row[c] for c in conf_cols])
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            score = sum(probs[i] * i for i in range(4))
        else:
            score = float(row.get("prediction", 0))

        grid[gy, gx] = score

    return grid


def plot_continuous_inflammation_heatmaps(
    project_root: Path,
    cv_strategy: str = "random_stratified",
    output_dir: Optional[Path] = None,
    experiments_dir: Optional[Path] = None,
    thumbnail_size: int = THUMBNAIL_SIZE,
) -> List[Path]:
    """Generate continuous inflammation score heatmaps (Fig 9-style).

    For each animal, creates a two-panel figure:
      A. Original H&E tissue mosaic (downsampled tiles)
      B. Continuous inflammation score heatmap (0.0-3.0, blue-to-red)

    Uses the single best model (highest mean_qwk) for predictions.

    Args:
        project_root: Project root directory.
        cv_strategy: CV strategy for predictions.
        output_dir: Directory to save figures.
        experiments_dir: Override experiments directory.
        thumbnail_size: Pixels per tile in tissue mosaic.

    Returns:
        List of paths to saved PNGs.
    """
    registry = _load_registry(project_root)
    output_dir = output_dir or project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model = _get_best_overall_model(registry, cv_strategy)
    if not best_model:
        logger.warning("No best model found for continuous heatmaps")
        return []

    display = DISPLAY_NAMES.get(best_model, best_model)
    logger.info(
        "Continuous heatmaps: using %s (%s)", display, cv_strategy,
    )

    df = _collect_predictions_for_model(
        project_root, registry, best_model, cv_strategy,
        experiments_dir=experiments_dir,
    )
    if df.empty:
        logger.warning("No predictions for %s", best_model)
        return []

    animal_slides = _get_largest_slide_per_animal(df)
    data_roots = _get_data_roots(project_root)

    saved_paths: List[Path] = []
    for animal_id in sorted(animal_slides.keys()):
        slide_id = animal_slides[animal_id]
        slide_df = df[
            (df["animal_id"] == animal_id) & (df["slide_id"] == slide_id)
        ]
        if slide_df.empty:
            continue

        path = _render_continuous_figure(
            slide_df=slide_df,
            animal_id=animal_id,
            slide_id=slide_id,
            model_display=display,
            data_roots=data_roots,
            output_dir=output_dir,
            cv_strategy=cv_strategy,
            thumbnail_size=thumbnail_size,
        )
        if path is not None:
            saved_paths.append(path)

    logger.info(
        "Generated %d continuous heatmaps in %s",
        len(saved_paths), output_dir,
    )
    return saved_paths


def _render_continuous_figure(
    *,
    slide_df: pd.DataFrame,
    animal_id: str,
    slide_id: str,
    model_display: str,
    data_roots: List[Path],
    output_dir: Path,
    cv_strategy: str,
    thumbnail_size: int,
) -> Optional[Path]:
    """Render Fig 9-style A/B figure for one animal.

    A = H&E tissue mosaic, B = continuous score heatmap.

    Args:
        slide_df: Predictions for one slide.
        animal_id: Animal identifier.
        slide_id: Slide identifier.
        model_display: Display name of model used.
        data_roots: Tile image root directories.
        output_dir: Save directory.
        cv_strategy: CV strategy label.
        thumbnail_size: Tile thumbnail size in pixels.

    Returns:
        Path to saved PNG, or None on failure.
    """
    mosaic = _build_tissue_mosaic(slide_df, data_roots, thumbnail_size)
    score_grid = _build_continuous_score_grid(slide_df)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(14, 6),
    )

    # Panel A: H&E mosaic
    ax_a.imshow(mosaic)
    ax_a.set_title("A", fontweight="bold", fontsize=16, loc="left")
    ax_a.set_xlabel("x")
    ax_a.set_ylabel("y")
    ax_a.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False,
    )

    # Panel B: Continuous score heatmap
    masked = np.ma.masked_invalid(score_grid)
    im = ax_b.imshow(
        masked,
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=3.0,
        interpolation="nearest",
    )
    ax_b.set_title("B", fontweight="bold", fontsize=16, loc="left")
    ax_b.set_xlabel("x")
    ax_b.set_ylabel("y")

    cbar = fig.colorbar(im, ax=ax_b, shrink=0.85, pad=0.02)
    cbar.set_label("Inflammation score", fontsize=11)
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    cv_label = "Stratified" if "stratified" in cv_strategy else "LOAO"
    fig.suptitle(
        f"Spatial Inflammation Score -- Animal {animal_id} "
        f"(Slide {slide_id}, {model_display}, {cv_label})",
        fontweight="bold", fontsize=13,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = (
        output_dir
        / f"continuous_inflammation_{animal_id}_{cv_strategy}.png"
    )
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved: %s", path)
    return path
