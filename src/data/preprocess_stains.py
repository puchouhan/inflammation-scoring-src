import os
import cv2
import numpy as np
import torch
import torchstain
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import sys
from typing import Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.seeds_logging import get_logger

logger = get_logger("Preprocessing")


def _to_chw_float255(img_rgb: np.ndarray) -> torch.Tensor:
    # torchstain 1.3.0 expects CHW tensors scaled to [0, 255]
    return transforms.ToTensor()(img_rgb).float() * 255.0


def _norm_to_uint8_rgb(norm_img) -> np.ndarray:
    """Convert torchstain output to uint8 RGB (H, W, 3).

    torchstain 1.3.0 returns HWC int32 in [0,255] when input is CHW float in [0,255].
    Older/newer variants may return CHW float in [0,1] or [0,255]. This function
    handles these cases conservatively.
    """
    if isinstance(norm_img, tuple):
        norm_img = norm_img[0]

    if isinstance(norm_img, torch.Tensor):
        norm_img = norm_img.detach().cpu()
        if norm_img.ndim != 3:
            raise ValueError(f"Unexpected normalized tensor shape: {tuple(norm_img.shape)}")

        # CHW -> HWC
        if norm_img.shape[0] == 3 and norm_img.shape[-1] != 3:
            norm_np = norm_img.permute(1, 2, 0).numpy()
        # HWC
        elif norm_img.shape[-1] == 3:
            norm_np = norm_img.numpy()
        else:
            raise ValueError(f"Unexpected normalized tensor layout: {tuple(norm_img.shape)}")
    else:
        norm_np = np.asarray(norm_img)

    if norm_np.ndim != 3 or norm_np.shape[-1] != 3:
        raise ValueError(f"Unexpected normalized array shape: {norm_np.shape}")

    # If float in [0,1], scale up; otherwise assume it's already [0,255]
    if np.issubdtype(norm_np.dtype, np.floating):
        max_val = float(np.nanmax(norm_np))
        if max_val <= 1.5:
            norm_np = norm_np * 255.0
        norm_np = np.clip(norm_np, 0, 255).astype(np.uint8)
    else:
        norm_np = np.clip(norm_np, 0, 255).astype(np.uint8)

    return norm_np

def preprocess_stains(input_dir: str, output_dir: str, target_img_path: Optional[str] = None):
    """
    Normalizes H&E stained images using Macenko method.
    Reads from input_dir, saves to output_dir maintaining structure.
    
    For reproducibility, this uses a fixed reference image (reference_patch.png)
    stored in the data directory. If not found, it creates one automatically.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory {input_path} does not exist.")
        return

    # 1. Setup Normalizer with FIXED reference for reproducibility
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    
    # Default reference image path (checked into git for reproducibility)
    default_ref_path = Path(__file__).parent / "reference_patch.png"
    
    if target_img_path and os.path.exists(target_img_path):
        logger.info(f"Fitting normalizer to target: {target_img_path}")
        ref_path = target_img_path
    elif default_ref_path.exists():
        logger.info(f"Using fixed reference image: {default_ref_path}")
        ref_path = str(default_ref_path)
    else:
        # Create reference image if it doesn't exist (first-time setup)
        logger.warning(f"Reference image not found. Creating {default_ref_path}...")
        potential_targets = list(input_path.glob("**/2/*.png")) + list(input_path.glob("**/3/*.png"))
        if not potential_targets:
            potential_targets = list(input_path.glob("**/*.png"))
        
        ref_img_src = None
        for candidate in potential_targets:
            candidate_bgr = cv2.imread(str(candidate))
            if candidate_bgr is not None:
                ref_img_src = candidate_bgr
                logger.info(f"Selected {candidate} as reference (saved to {default_ref_path})")
                # Save this as the permanent reference
                cv2.imwrite(str(default_ref_path), candidate_bgr)
                break
        
        if ref_img_src is None:
            logger.error("Could not find any valid images to create reference.")
            return
        
        ref_path = str(default_ref_path)
    
    # Load and fit the reference
    target_bgr = cv2.imread(ref_path)
    if target_bgr is None:
        logger.error(f"Could not read reference image: {ref_path}")
        return
    
    target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    target_tensor = _to_chw_float255(target)
    normalizer.fit(target_tensor)
    logger.info("SUCCESS: Normalizer fitted to reference image (reproducible)")

    # 2. Process Images
    image_files = list(input_path.glob("**/*.png"))
    logger.info(f"Found {len(image_files)} images to process.")

    normalized_ok = 0
    copied_original = 0
    failed_total = 0

    for img_file in tqdm(image_files, desc="Normalizing Stains"):
        # Create output directory structure
        rel_path = img_file.relative_to(input_path)
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        img_bgr = cv2.imread(str(img_file))
        if img_bgr is None:
            failed_total += 1
            logger.warning(f"Could not read image: {img_file}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = _to_chw_float255(img_rgb)

        try:
            norm_img, _, _ = normalizer.normalize(I=img_tensor, stains=True)
            norm_rgb_u8 = _norm_to_uint8_rgb(norm_img)
            norm_bgr = cv2.cvtColor(norm_rgb_u8, cv2.COLOR_RGB2BGR)

            ok = cv2.imwrite(str(out_file), norm_bgr)
            if not ok:
                raise IOError("cv2.imwrite returned False")
            normalized_ok += 1
        except Exception as e:
            failed_total += 1
            logger.warning(f"Normalization failed for {img_file}. Copying original. Error: {e}")
            ok = cv2.imwrite(str(out_file), img_bgr)
            if ok:
                copied_original += 1

    logger.info(
        "Normalization complete. Saved to %s (normalized=%d, copied=%d, failed=%d)",
        output_dir,
        normalized_ok,
        copied_original,
        failed_total,
    )

if __name__ == "__main__":
    # Default paths
    INPUT_DIR = "dataset"
    OUTPUT_DIR = "dataset_norm"
    
    preprocess_stains(INPUT_DIR, OUTPUT_DIR)
