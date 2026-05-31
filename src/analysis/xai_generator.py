# src/analysis/xai_generator.py
import argparse
import sys
from pathlib import Path
import json
import numpy as np
import cv2
import torch
import random
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
from matplotlib.colors import Normalize
# WARNING: Requires 'pip install grad-cam'
try:
    from pytorch_grad_cam import GradCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Error: pytorch-grad-cam not installed. Please install it using 'pip install grad-cam'")
    sys.exit(1)

# Local imports hack
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path: sys.path.append(project_root)

# We need to add the parent directory to sys.path to run this as a script
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import InflammationModel
from src.data.inflammation_dataset import InflammationDataset
# from src.utils.logging_config import get_logger
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reshape_transform_vit(tensor):
    """Helper for ViT attention maps in pytorch-grad-cam."""
    result = tensor[:, 1:, :]  # Skip CLS token
    result = result.transpose(1, 2)
    side = int(np.sqrt(result.shape[2]))
    result = result.reshape(tensor.size(0), result.size(1), side, side)
    return result

def reshape_transform_swin(tensor):
    """Reshape transform for Swin Transformer.

    Swin norm1 outputs (B, H, W, C), GradCAM expects (B, C, H, W).
    """
    return tensor.permute(0, 3, 1, 2)

# Human-readable class labels (Class 4 = Ignore artifact, excluded from scoring)
_CLASS_LABELS: dict = {
    0: "Class 0 (None)",
    1: "Class 1 (Mild)",
    2: "Class 2 (Moderate)",
    3: "Class 3 (Severe)",
    4: "Class 4 (Ignore)",
}


def _save_gradcam_figure(
    rgb_img: np.ndarray,
    grayscale_cam: np.ndarray,
    overlay: np.ndarray,
    label: int,
    predicted: int,
    softmax_probs: np.ndarray,
    arch: str,
    animal_id: str,
    fold_idx: int,
    idx: int,
    save_path: Path,
) -> None:
    """Save a 3-panel Grad-CAM figure with full scientific annotations.

    Panels: Original H&E | Grad-CAM Overlay | Activation Map (grayscale).
    Title includes architecture, fold, true/predicted class labels, and top-class confidence.
    A shared colorbar on the right encodes activation intensity (0.0 - 1.0).

    Args:
        rgb_img: Normalised RGB image array (H x W x 3), values in [0, 1].
        grayscale_cam: Raw Grad-CAM activation map (H x W), values in [0, 1].
        overlay: Grad-CAM overlay on RGB image (H x W x 3), uint8.
        label: Ground-truth class index.
        predicted: Predicted class index.
        softmax_probs: Softmax probability vector (num_classes,).
        arch: TIMM backbone name, e.g. 'swin_tiny_patch4_window7_224'.
        animal_id: Animal identifier parsed from filename.
        fold_idx: Cross-validation fold index.
        idx: Dataset sample index.
        save_path: Destination PNG path.
    """
    true_label_str = _CLASS_LABELS.get(int(label), f"Class {label}")
    pred_label_str = _CLASS_LABELS.get(int(predicted), f"Class {predicted}")
    confidence = float(softmax_probs[int(predicted)]) * 100.0
    correct_str = "correct" if int(label) == int(predicted) else "WRONG"

    # figsize=(11, 6): narrower than the old (15, 5) so fonts scale up when
    # the figure is included at 0.85\linewidth in the thesis (single-column A4,
    # text width ~16 cm). Scale factor becomes ~0.48 instead of ~0.35, raising
    # effective print size of 16pt panel titles from ~5.6pt to ~7.7pt and
    # 14pt axis labels to ~6.7pt -- comfortably legible in a printed thesis.
    fig, axes = plt.subplots(1, 3, figsize=(11, 6))
    fig.patch.set_facecolor("white")

    # --- Panel 1: Original image ---
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original H&E", fontsize=16, fontweight="bold")
    axes[0].set_xlabel(f"Animal: {animal_id}", fontsize=14)
    axes[0].set_ylabel("pixels", fontsize=14)
    axes[0].tick_params(labelsize=12)

    # --- Panel 2: Grad-CAM overlay ---
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay", fontsize=16, fontweight="bold")
    axes[1].set_xlabel("High activation = red", fontsize=14)
    axes[1].tick_params(labelsize=12)
    axes[1].set_yticks([])

    # --- Panel 3: Raw activation map (jet colormap for print clarity) ---
    norm = Normalize(vmin=0.0, vmax=1.0)
    im = axes[2].imshow(grayscale_cam, cmap="jet", norm=norm)
    axes[2].set_title("Activation Map", fontsize=16, fontweight="bold")
    axes[2].set_xlabel("Activation intensity", fontsize=14)
    axes[2].tick_params(labelsize=12)
    axes[2].set_yticks([])

    # --- Shared colorbar ---
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Grad-CAM activation (0 = low, 1 = high)", fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # --- Suptitle with all metadata ---
    arch_short = arch.split("_patch")[0].replace("_", " ")
    title = (
        f"Architecture: {arch_short}  |  Fold {fold_idx}  |  "
        f"True: {true_label_str}  |  Pred: {pred_label_str}  ({confidence:.1f}%)  [{correct_str}]"
    )
    fig.suptitle(title, fontsize=15, y=1.02)

    # --- Probability bar below panels ---
    prob_str = "  ".join(
        [f"C{i}={float(softmax_probs[i])*100:.1f}%" for i in range(len(softmax_probs))]
    )
    fig.text(0.5, -0.01, f"Softmax: {prob_str}", ha="center", fontsize=13, color="dimgray")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_xai_analysis(
    checkpoint_path: str,
    output_dir: str,
    num_images: int = 5,
    image_indices: list = None,
    fold_idx: int = 0,
) -> list:
    """Generate annotated 3-panel Grad-CAM figures for model explainability.

    Each saved PNG contains: Original H&E | Grad-CAM overlay | Raw activation map,
    with colorbar, true/predicted class labels, softmax confidence, and fold metadata.

    Args:
        checkpoint_path: Path to model checkpoint (.ckpt).
        output_dir: Output directory for PNG figures.
        num_images: Number of random images if image_indices is None.
        image_indices: Specific dataset indices to visualize (overrides num_images).
        fold_idx: CV fold index used to select the correct validation split.

    Returns:
        List of generated PNG file paths.
    """
    ckpt_path = Path(checkpoint_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Reload Model
    logger.info(f"Loading checkpoint: {ckpt_path}")
    try:
        model_wrapper = InflammationModel.load_from_checkpoint(ckpt_path)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return []

    model_wrapper.eval()
    model = model_wrapper.backbone  # Access the TIMM backbone, not .model
    cfg = model_wrapper.cfg if hasattr(model_wrapper, 'cfg') else model_wrapper.config
    
    # Heuristic for target layers
    target_layers = []
    transform_func = None
    arch = cfg['model']['backbone']
    logger.info(f"Analyzing architecture: {arch}")
    
    if "maxvit" in arch:
        try:
            target_layers = [model.stages[-1].blocks[-1].conv.norm2]
        except Exception:
            logger.warning("Could not find MaxViT conv.norm2 layer.")
    elif "swin" in arch:
        try:
            target_layers = [model.layers[-1].blocks[-1].norm1]
            transform_func = reshape_transform_swin
        except Exception:
            logger.warning("Could not find Swin layers. Trying auto-detection.")
    elif "tnt" in arch:
        try:
            target_layers = [model.blocks[-1].norm_out]
            transform_func = reshape_transform_vit
        except Exception:
            logger.warning("Could not find TNT blocks. Trying auto-detection.")
    elif "dino" in arch or "simclr" in arch:
        # DINO and SimCLR use ViT backbones; treat identically to plain ViT
        try:
            target_layers = [model.blocks[-1].norm1]
            transform_func = reshape_transform_vit
        except Exception:
            logger.warning("Could not find ViT blocks for DINO/SimCLR backbone. Trying auto-detection.")
    elif "vit" in arch or "convit" in arch:
        try:
            target_layers = [model.blocks[-1].norm1]
            transform_func = reshape_transform_vit
        except Exception:
            logger.warning("Could not find standard ViT blocks. Trying auto-detection.")
    elif "gnn" in arch:
        # Graph Neural Networks operate on graph topology, not a regular 2D feature map.
        # Spatial GradCAM is not applicable; skip gracefully.
        logger.info(
            "Architecture '%s' is a Graph Neural Network. "
            "Spatial GradCAM is not applicable to GNN architectures. Skipping xAI generation.",
            arch,
        )
        return []
    elif "convnext" in arch:
        try:
            target_layers = [model.stages[-1].blocks[-1]]
        except Exception:
            pass
    elif "efficientnet" in arch:
        try:
            target_layers = [model.conv_head]
        except Exception:
            pass
    elif "densenet" in arch:
        try:
            target_layers = [model.features[-1]]
        except Exception:
            logger.warning("Could not find DenseNet features layer.")
    elif "regnety" in arch or "regnet" in arch:
        try:
            target_layers = [model.s4]
        except Exception:
            logger.warning("Could not find RegNetY s4 layer.")
    else:
        # Fallback for generic CNNs (ResNet style)
        try:
            target_layers = [model.layer4[-1]]
        except Exception:
            pass

    if not target_layers:
        logger.warning("No target layers defined. GradCAM might fail or default to last layer.")

    # Get Data - use the held-out test set (dataset_norm/val), always available in Colab
    from src.data.inflammation_dataset import get_test_dataloader
    test_loader = get_test_dataloader(cfg)
    ds = test_loader.dataset
    logger.info(f"Loaded test dataset with {len(ds)} images")

    if len(ds) == 0:
        logger.error("Dataset is empty.")
        return []

    if image_indices is not None:
        indices = image_indices
        logger.info(f"Using provided indices: {indices}")
    else:
        indices = np.random.choice(len(ds), min(num_images, len(ds)), replace=False)

    generated_files = []

    for idx in indices:
        img_tensor, label = ds[idx]

        row = ds.df.iloc[idx]
        animal_id = str(row.get('animal_id', 'unknown'))

        input_tensor = img_tensor.unsqueeze(0)
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            logits = model_wrapper(input_tensor)
            softmax_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        predicted = int(np.argmax(softmax_probs))

        try:
            cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=transform_func)
            targets = [ClassifierOutputTarget(predicted)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

            rgb_img = img_tensor.permute(1, 2, 0).cpu().numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)

            overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            fname = f"idx{idx}_{animal_id}_lbl{label}_pred{predicted}.png"
            save_path = out_path / fname

            _save_gradcam_figure(
                rgb_img=rgb_img,
                grayscale_cam=grayscale_cam,
                overlay=overlay,
                label=label,
                predicted=predicted,
                softmax_probs=softmax_probs,
                arch=arch,
                animal_id=animal_id,
                fold_idx=fold_idx,
                idx=int(idx),
                save_path=save_path,
            )

            generated_files.append(save_path)
            logger.info(f"Saved: {save_path}")
        except Exception as e:
            logger.error(f"XAI Failed for image {idx}: {e}")

    return generated_files

def _process_heatmap_files(heatmap_files: list) -> tuple:
    """Load heatmap PNGs and accumulate grid activations, peak values, and entropies.

    Args:
        heatmap_files: List of Path objects pointing to *_heatmap.png files.

    Returns:
        Tuple of (grid_accumulator, peak_activations, entropies, loaded_count).
    """
    grid_accumulator = np.zeros((3, 3), dtype=np.float64)
    peak_activations: list = []
    entropies: list = []
    loaded_count: int = 0

    for hmap_path in heatmap_files:
        img_bgr = cv2.imread(str(hmap_path))
        if img_bgr is None:
            logger.warning(f"Could not read heatmap: {hmap_path}")
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        h, w = gray.shape
        row_size, col_size = h // 3, w // 3
        for row in range(3):
            for col in range(3):
                r_end = (row + 1) * row_size if row < 2 else h
                c_end = (col + 1) * col_size if col < 2 else w
                grid_accumulator[row, col] += gray[row * row_size:r_end, col * col_size:c_end].mean()
        peak_activations.append(float(gray.max()))
        flat = gray.flatten()
        total = flat.sum()
        p = flat / total if total > 0 else np.ones_like(flat) / len(flat)
        entropies.append(float(-np.sum(p * np.log(p + 1e-8))))
        loaded_count += 1

    return grid_accumulator, peak_activations, entropies, loaded_count


def compute_attribution_statistics(
    heatmap_dir: Path,
    output_path: Path,
) -> dict:
    """Compute aggregate statistics over saved GradCAM heatmap PNG files.

    Loads all *_heatmap.png files from heatmap_dir, converts them to grayscale
    activation maps, and computes:
    - Mean activation per image region (9-region grid: 3x3)
    - Top activated spatial regions across all heatmaps
    - Mean/std of peak activation value
    - Mean/std of activation entropy (spread of attention)
    - Fraction of activation in center vs. border regions

    Args:
        heatmap_dir: Directory containing *_heatmap.png files.
        output_path: Path to save JSON statistics artifact.

    Returns:
        Dictionary with attribution statistics.
    """
    _REGION_NAMES: list = [
        ["top-left", "top-center", "top-right"],
        ["middle-left", "center", "middle-right"],
        ["bottom-left", "bottom-center", "bottom-right"],
    ]

    heatmap_files: list = list(Path(heatmap_dir).rglob("*_heatmap.png"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _save(data: dict) -> dict:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        return data

    if not heatmap_files:
        logger.warning(f"No *_heatmap.png files found in {heatmap_dir}")
        return _save({
            "n_heatmaps": 0,
            "error": "No heatmap files found",
            "mean_activation_grid_3x3": None,
            "top_activated_region": None,
            "mean_peak_activation": None,
            "std_peak_activation": None,
            "mean_entropy": None,
            "std_entropy": None,
            "center_vs_border_ratio": None,
        })

    grid_accumulator, peak_activations, entropies, loaded_count = _process_heatmap_files(
        heatmap_files
    )

    if loaded_count == 0:
        logger.warning(f"All heatmap files failed to load in {heatmap_dir}")
        return _save({"n_heatmaps": 0, "error": "All heatmap files failed to load"})

    mean_grid_np = grid_accumulator / loaded_count
    mean_grid: list = mean_grid_np.tolist()
    top_row, top_col = np.unravel_index(np.argmax(mean_grid_np), mean_grid_np.shape)
    top_region: str = _REGION_NAMES[int(top_row)][int(top_col)]

    border_sum: float = sum(
        mean_grid[r][c] for r in range(3) for c in range(3) if not (r == 1 and c == 1)
    )
    center_vs_border: float = float(mean_grid[1][1] / (border_sum / 8.0 + 1e-8))

    result: dict = {
        "n_heatmaps": loaded_count,
        "mean_activation_grid_3x3": mean_grid,
        "top_activated_region": top_region,
        "mean_peak_activation": float(np.mean(peak_activations)),
        "std_peak_activation": float(np.std(peak_activations)),
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies)),
        "center_vs_border_ratio": center_vs_border,
    }

    logger.info(f"Attribution statistics saved to {output_path} (n={loaded_count})")
    return _save(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="analysis_results/heatmaps")
    parser.add_argument("--num_images", type=int, default=5)
    # Note: Pass list as string "1,2,3" if needed via CLI, but main use is import
    args = parser.parse_args()
    run_xai_analysis(args.checkpoint, args.output, args.num_images)
