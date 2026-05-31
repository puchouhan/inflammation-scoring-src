"""
Held-out inference on animal 15_304.

This script loads all trained LOAO fold checkpoints, runs inference on the
84 tiles from animal 15_304, and reports per-architecture QWK. Run this on
the same machine / Colab environment where the experiment checkpoints live.

Usage (from project root):
    python scripts/run_15304_inference.py --run_dir experiments/<run_id>

Output: results_15304.csv printed to stdout and saved alongside this script.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

from configs.utils import load_config
from src.data.inflammation_dataset import InflammationDataset, create_dataframe
from src.models.base_model import InflammationModel
from src.utils.seeds_logging import get_logger

logger = get_logger(__name__)

ANIMAL_15304 = "15_304"

# ImageNet normalisation (matches training-time val pipeline)
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 256

VAL_TRANSFORM = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def build_15304_loader(cfg: dict) -> Tuple[torch.utils.data.DataLoader, pd.DataFrame]:
    """Build a DataLoader containing only tiles from animal 15_304.

    Args:
        cfg: Full project config dict.

    Returns:
        DataLoader and the underlying DataFrame subset.

    Raises:
        RuntimeError: If no tiles for animal 15_304 are found.
    """
    data_dir: str = cfg["directories"]["data_dir"]
    df_all = create_dataframe(data_dir)
    df_15304 = df_all[df_all["animal_id"] == ANIMAL_15304].reset_index(drop=True)

    if df_15304.empty:
        raise RuntimeError(
            f"No tiles for animal {ANIMAL_15304} found in {data_dir}. "
            "Check that dataset_norm/training/ contains 15_304 tiles."
        )

    logger.info(f"Found {len(df_15304)} tiles for animal {ANIMAL_15304}.")
    dataset = InflammationDataset(df_15304, data_dir, transform=VAL_TRANSFORM)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get("training", {}).get("batch_size", 32),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, df_15304


def infer_checkpoint(
    checkpoint_path: str,
    loader: torch.utils.data.DataLoader,
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference for one checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file.
        loader: DataLoader with 15_304 tiles.
        cfg: Full project config dict.

    Returns:
        Tuple of (y_true, y_pred) numpy arrays (ignore class excluded).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InflammationModel.load_from_checkpoint(checkpoint_path, cfg=cfg, map_location=device)
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    ignore_idx: int = cfg.get("ignore_class_index", 4)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    valid = y_true != ignore_idx
    y_true = y_true[valid]
    y_pred = np.clip(y_pred[valid], 0, 3)
    return y_true, y_pred


def run_inference(run_dir: Path, cfg: dict, models_to_eval: List[str]) -> pd.DataFrame:
    """Evaluate all models on animal 15_304.

    Args:
        run_dir: Path to the experiment run directory (contains one sub-dir per model).
        cfg: Full project config dict.
        models_to_eval: List of model names to evaluate.

    Returns:
        DataFrame with columns: Model, Fold0 QWK, Fold1 QWK, Mean QWK, Best QWK.
    """
    loader, df_15304 = build_15304_loader(cfg)
    rows = []

    for model_name in models_to_eval:
        model_dir = run_dir / model_name
        ckpt_file = model_dir / "checkpoints.json"

        if not ckpt_file.exists():
            logger.warning(f"checkpoints.json missing for {model_name}, skipping.")
            continue

        with open(ckpt_file, "r") as fh:
            meta = json.load(fh)

        fold_qwks: Dict[int, float] = {}
        for fold_info in meta.get("folds", []):
            fold_idx: int = fold_info["fold_idx"]
            ckpt_path: str = fold_info["checkpoint_path"]

            try:
                y_true, y_pred = infer_checkpoint(ckpt_path, loader, cfg)
                qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
                fold_qwks[fold_idx] = qwk
                logger.info(f"{model_name} fold {fold_idx}: QWK={qwk:.4f} on 15_304")
            except Exception as exc:
                logger.error(f"{model_name} fold {fold_idx} failed: {exc}")

        if not fold_qwks:
            continue

        qwk_values = list(fold_qwks.values())
        rows.append({
            "Model": model_name,
            "Fold0 QWK": fold_qwks.get(0, float("nan")),
            "Fold1 QWK": fold_qwks.get(1, float("nan")),
            "Mean QWK": float(np.mean(qwk_values)),
            "Best QWK": float(np.max(qwk_values)),
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Entry point: parse args, run inference, print and save results."""
    parser = argparse.ArgumentParser(description="Run inference on held-out animal 15_304.")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to experiment run directory (e.g. experiments/20250101_120000_loao).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="Config model name to load (default: base). Change if using a specific model config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.model)
    run_dir = Path(args.run_dir)
    models_to_eval: List[str] = cfg.get("models_to_train", [])

    if not models_to_eval:
        raise ValueError("No models found in config key 'models_to_train'.")

    logger.info(f"Evaluating {len(models_to_eval)} models on animal {ANIMAL_15304}.")
    results = run_inference(run_dir, cfg, models_to_eval)

    print("\n=== 15_304 Held-out Inference Results ===")
    print(results.to_string(index=False))

    out_path = run_dir / "results_15304.csv"
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
