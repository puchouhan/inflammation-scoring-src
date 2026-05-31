"""
TensorBoard Event Extraction Module.

Extracts scalar metrics from TensorBoard event files into DataFrames
for use in cross-model comparison plots (learning curves, etc.).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _find_event_file(fold_tb_dir: Path) -> Optional[Path]:
    """Find the TensorBoard event file in a fold directory.

    Args:
        fold_tb_dir: Path to fold-level TensorBoard directory.

    Returns:
        Path to event file, or None if not found.
    """
    event_files = list(fold_tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    return event_files[0]


def _find_flat_event_files(
    tb_dir: Path,
    n_folds: int,
    min_size_bytes: int = 1024,
) -> Dict[int, Path]:
    """Map fold indices to event files in a flat TensorBoard directory.

    Historical runs stored all fold event files flat in one directory.
    Each fold produces a main event file with suffix ``fold_idx * 3``.
    Small finalization files (< min_size_bytes) are ignored.

    Args:
        tb_dir: Path to the TensorBoard directory.
        n_folds: Expected number of folds.
        min_size_bytes: Minimum file size to consider as real data.

    Returns:
        Dict mapping fold_idx to the corresponding event file Path.
    """
    all_events = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not all_events:
        return {}

    # Try to filter out tiny finalization files by size.
    # Google Drive FUSE may report st_size=0 for unsynced files,
    # so fall back to all events if size filtering removes everything.
    try:
        sized = [(f, f.stat().st_size) for f in all_events]
        real_events = [f for f, sz in sized if sz >= min_size_bytes]
        if not real_events:
            # Size filter removed everything (Drive FUSE issue or all small)
            logger.warning(
                f"Size filter ({min_size_bytes}B) removed all "
                f"{len(all_events)} event files in {tb_dir} -- "
                f"using all files (Drive FUSE stat issue?)"
            )
            real_events = all_events
    except OSError as exc:
        logger.warning(f"stat() failed in {tb_dir}: {exc} -- using all files")
        real_events = list(all_events)

    # Sort by last numeric suffix (fold_idx * 3 pattern)
    def _suffix(p: Path) -> int:
        parts = p.name.split(".")
        try:
            return int(parts[-1])
        except (ValueError, IndexError):
            return 0

    real_events.sort(key=_suffix)

    fold_map: Dict[int, Path] = {}
    for fold_idx in range(min(n_folds, len(real_events))):
        fold_map[fold_idx] = real_events[fold_idx]

    return fold_map


def extract_scalars_for_fold(
    event_file: Path,
    metric_names: List[str],
    try_parent_dir: bool = False,
) -> pd.DataFrame:
    """Extract scalar metrics from a single TensorBoard event file.

    Args:
        event_file: Path to TensorBoard event file.
        metric_names: List of scalar tag names to extract.
        try_parent_dir: If True, fall back to loading the parent
            directory when file-based loading yields no tags.
            Safe for structured (per-fold) directories; avoid for
            flat layouts where the parent contains all folds.

    Returns:
        DataFrame with columns: epoch, metric, value.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        logger.warning("tensorboard not installed -- cannot extract scalars")
        return pd.DataFrame(columns=["epoch", "metric", "value"])

    attempts = [str(event_file)]
    if try_parent_dir:
        attempts.append(str(event_file.parent))

    for attempt_path in attempts:
        try:
            ea = event_accumulator.EventAccumulator(attempt_path)
            ea.Reload()
        except Exception as e:
            logger.error(f"Failed to load {attempt_path}: {e}")
            continue

        available_tags = ea.Tags().get("scalars", [])
        if available_tags:
            break
    else:
        # All attempts failed or returned no tags
        logger.warning(
            f"No scalar tags found for {event_file.name} "
            f"(tried: {attempts})"
        )
        return pd.DataFrame(columns=["epoch", "metric", "value"])

    logger.debug(f"Available tags in {event_file.name}: {available_tags}")
    rows: List[Dict] = []

    for metric in metric_names:
        # Direct match first, then try fold-prefixed variants
        if metric in available_tags:
            tag_to_use = metric
        else:
            # ExperimentTracker stores as fold_X/metric -- try suffix match
            matching = [t for t in available_tags if t.endswith(f"/{metric}")]
            if matching:
                tag_to_use = matching[0]
                logger.debug(
                    f"Metric '{metric}' matched via prefix: '{tag_to_use}'"
                )
            else:
                logger.debug(f"Metric '{metric}' not in {event_file}")
                continue

        for scalar_event in ea.Scalars(tag_to_use):
            rows.append({
                "epoch": scalar_event.step,
                "metric": metric,
                "value": scalar_event.value,
            })

    if not rows:
        logger.warning(
            f"Requested metrics {metric_names} not found among "
            f"available tags {available_tags} in {event_file.name}"
        )

    return pd.DataFrame(rows)


def extract_model_training_history(
    experiment_dir: Path,
    model_name: str,
    n_folds: int,
    tb_subdir: str = "tensorboard",
    metric_names: Optional[List[str]] = None,
) -> Dict[int, pd.DataFrame]:
    """Extract training history for all folds of a model.

    Supports two directory layouts:
    - **Structured**: ``tensorboard/fold_N/events.out.tfevents.*``
    - **Flat** (legacy): all event files directly in ``tensorboard/``

    Args:
        experiment_dir: Path to experiment run directory
            (e.g., experiments/2026-03-29_.../densenet).
        model_name: Model name (for logging).
        n_folds: Number of folds to look for.
        tb_subdir: Name of TensorBoard subdirectory.
        metric_names: Scalar tags to extract.
            Defaults to train_loss_epoch, val_loss, val_qwk, val_macro_f1.

    Returns:
        Dict mapping fold_idx to DataFrame with columns: epoch, metric, value.
    """
    if metric_names is None:
        metric_names = ["train_loss_epoch", "val_loss", "val_qwk", "val_macro_f1"]

    tb_dir = experiment_dir / tb_subdir
    if not tb_dir.exists():
        logger.warning(f"TensorBoard dir not found: {tb_dir}")
        return {}

    fold_data: Dict[int, pd.DataFrame] = {}

    # Try structured layout first (fold_N/ subdirectories)
    has_fold_subdirs = any(
        (tb_dir / f"fold_{i}").is_dir() for i in range(n_folds)
    )

    if has_fold_subdirs:
        fold_data = _extract_from_structured(
            tb_dir, model_name, n_folds, metric_names,
        )
    else:
        fold_data = _extract_from_flat(
            tb_dir, model_name, n_folds, metric_names,
        )

    if not fold_data:
        # Fallback: try Lightning CSV logs for per-epoch data
        fold_data = _extract_from_csv_logs(
            experiment_dir, model_name, n_folds, metric_names,
        )

    if not fold_data:
        logger.warning(f"No TensorBoard data found for {model_name}")
        return fold_data

    # Check if data has per-epoch granularity (>1 epoch per fold)
    # ExperimentTracker summary logs have only 1 data point per fold
    max_epochs = max(
        len(df) for df in fold_data.values()
    ) if fold_data else 0
    if max_epochs <= len(metric_names):
        logger.debug(
            f"TensorBoard data for {model_name} appears to be summary-only "
            f"({max_epochs} rows). Trying CSV logs for per-epoch data."
        )
        csv_data = _extract_from_csv_logs(
            experiment_dir, model_name, n_folds, metric_names,
        )
        if csv_data:
            fold_data = csv_data

    return fold_data


def _extract_from_structured(
    tb_dir: Path,
    model_name: str,
    n_folds: int,
    metric_names: List[str],
) -> Dict[int, pd.DataFrame]:
    """Extract from structured layout with fold_N/ subdirectories."""
    fold_data: Dict[int, pd.DataFrame] = {}
    for fold_idx in range(n_folds):
        fold_tb_dir = tb_dir / f"fold_{fold_idx}"
        if not fold_tb_dir.exists():
            continue
        event_file = _find_event_file(fold_tb_dir)
        if event_file is None:
            continue
        df = extract_scalars_for_fold(
            event_file, metric_names, try_parent_dir=True,
        )
        if not df.empty:
            df["fold"] = fold_idx
            fold_data[fold_idx] = df
    return fold_data


def _extract_from_flat(
    tb_dir: Path,
    model_name: str,
    n_folds: int,
    metric_names: List[str],
) -> Dict[int, pd.DataFrame]:
    """Extract from flat layout (all event files in one directory)."""
    fold_map = _find_flat_event_files(tb_dir, n_folds)
    if not fold_map:
        logger.debug(f"No usable event files in flat dir for {model_name}")
        return {}

    fold_data: Dict[int, pd.DataFrame] = {}
    for fold_idx, event_file in fold_map.items():
        df = extract_scalars_for_fold(event_file, metric_names)
        if not df.empty:
            df["fold"] = fold_idx
            fold_data[fold_idx] = df
            logger.debug(
                f"Extracted {len(df)} rows for {model_name} "
                f"fold {fold_idx} (flat)"
            )
    return fold_data


# Mapping from requested metric names to Lightning CSV column names.
# Lightning appends _epoch suffix for metrics logged with on_epoch=True.
_CSV_COLUMN_MAP: Dict[str, List[str]] = {
    "train_loss_epoch": ["train_loss_epoch", "train_loss"],
    "val_loss": ["val_loss"],
    "val_qwk": ["val_kappa", "val_qwk"],
    "val_macro_f1": ["val_macro_f1"],
}


def _extract_from_csv_logs(
    experiment_dir: Path,
    model_name: str,
    n_folds: int,
    metric_names: List[str],
) -> Dict[int, pd.DataFrame]:
    """Extract per-epoch metrics from Lightning CSVLogger output.

    Looks for CSV files at ``experiment_dir/csv_logs/fold_N/metrics.csv``.

    Args:
        experiment_dir: Path to model experiment directory.
        model_name: Model name (for logging).
        n_folds: Number of folds to look for.
        metric_names: Canonical metric names to extract.

    Returns:
        Dict mapping fold_idx to DataFrame with columns: epoch, metric, value.
    """
    csv_base = experiment_dir / "csv_logs"
    if not csv_base.exists():
        logger.debug(f"No csv_logs directory for {model_name}: {csv_base}")
        return {}

    fold_data: Dict[int, pd.DataFrame] = {}

    for fold_idx in range(n_folds):
        csv_dir = csv_base / f"fold_{fold_idx}"
        csv_file = csv_dir / "metrics.csv"
        if not csv_file.exists():
            continue

        try:
            raw_df = pd.read_csv(csv_file)
        except Exception as exc:
            logger.warning(f"Failed to read {csv_file}: {exc}")
            continue

        if raw_df.empty:
            continue

        rows: List[Dict] = []
        for metric in metric_names:
            # Try canonical name and known aliases
            candidates = _CSV_COLUMN_MAP.get(metric, [metric])
            col = next((c for c in candidates if c in raw_df.columns), None)
            if col is None:
                continue

            series = raw_df[["epoch", col]].dropna(subset=[col])
            for _, row in series.iterrows():
                rows.append({
                    "epoch": int(row["epoch"]),
                    "metric": metric,
                    "value": float(row[col]),
                })

        if rows:
            df = pd.DataFrame(rows)
            df["fold"] = fold_idx
            fold_data[fold_idx] = df
            logger.debug(
                f"Extracted {len(df)} rows for {model_name} "
                f"fold {fold_idx} (csv_logs)"
            )

    if fold_data:
        logger.info(
            f"Loaded per-epoch data from CSV logs for {model_name} "
            f"({len(fold_data)} folds)"
        )

    return fold_data


def extract_all_models_training_history(
    project_root: Path,
    registry: Dict,
    cv_strategy: str,
    metric_names: Optional[List[str]] = None,
    experiments_dir: Optional[Path] = None,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """Extract training history for all models in the registry.

    Args:
        project_root: Project root directory.
        registry: Loaded best_models_registry dict.
        cv_strategy: CV strategy filter.
        metric_names: Scalar tags to extract.
        experiments_dir: Override directory containing experiment runs.
            Defaults to project_root / 'experiments'.

    Returns:
        Nested dict: {model_name: {fold_idx: DataFrame}}.
    """
    suffix = "_stratified" if cv_strategy == "random_stratified" else "_loao"
    base_dir = experiments_dir or (project_root / "experiments")
    all_data: Dict[str, Dict[int, pd.DataFrame]] = {}

    for key, entry in registry.items():
        if not key.endswith(suffix):
            continue
        cv_data = list(entry.values())[0]
        if cv_data.get("cv_strategy") != cv_strategy:
            continue

        base_name = key[: -len(suffix)]
        run_id = cv_data.get("run_id", "")
        fold_models = cv_data.get("fold_models", {})
        n_folds = len(fold_models)

        if not run_id or n_folds == 0:
            continue

        exp_dir = base_dir / run_id / base_name
        if not exp_dir.exists():
            logger.warning(
                f"Experiment dir not found for {base_name} "
                f"({cv_strategy}): {exp_dir}"
            )
            continue

        fold_data = extract_model_training_history(
            experiment_dir=exp_dir,
            model_name=base_name,
            n_folds=n_folds,
            metric_names=metric_names,
        )

        if fold_data:
            all_data[base_name] = fold_data

    logger.info(
        f"Extracted training history for {len(all_data)}/{len(registry)} "
        f"models ({cv_strategy})"
    )
    return all_data
