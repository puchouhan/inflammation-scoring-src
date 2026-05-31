"""Data exploration utilities for inflammation dataset."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from configs.utils import load_config
from src.data.inflammation_dataset import get_dataloaders
from src.utils.seeds_logging import seed_everything

logger = logging.getLogger(__name__)


class DataExplorer:
    """Explore and visualize the inflammation dataset."""

    def __init__(self, config_path: Optional[str] = None, save_dir: Optional[Path] = None):
        """Initialize data explorer.
        
        Args:
            config_path: Path to config file, defaults to base config
            save_dir: Directory to save plots (if None, only displays plots)
        """
        self.config = load_config() if config_path is None else load_config(config_path)
        seed_everything(self.config['seed'])
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def check_and_prepare_normalized_dataset(
        self,
        norm_dir: Path = Path("../../dataset_norm"),
        raw_dir: Path = Path("../../dataset"),
        show_plots: bool = False
    ) -> bool:
        """Check if normalized dataset exists and create it if necessary.
        
        Args:
            norm_dir: Path to normalized dataset directory
            raw_dir: Path to raw dataset directory
            show_plots: If True, automatically show data exploration plots after creation
            
        Returns:
            True if dataset exists or was created successfully, False otherwise
        """
        dataset_was_created = False
        
        if not norm_dir.exists() or not any(norm_dir.rglob("*.png")):
            logger.warning("=" * 80)
            logger.warning("NORMALIZED DATASET NOT FOUND")
            logger.warning("=" * 80)
            logger.warning("Creating normalized dataset from raw images...")
            logger.warning("This may take several minutes on first run.\n")
            
            try:
                # Import and run preprocessing
                from src.data.preprocess_stains import preprocess_stains
                
                logger.info("Starting stain normalization...")
                logger.info("  Input:  %s", raw_dir)
                logger.info("  Output: %s\n", norm_dir)
                
                preprocess_stains(str(raw_dir), str(norm_dir))
                
                logger.info("\n%s", "=" * 80)
                logger.info("NORMALIZED DATASET CREATED SUCCESSFULLY")
                logger.info("=" * 80)
                logger.info("Location: %s", norm_dir)
                logger.info("Total images: %d\n", len(list(norm_dir.rglob("*.png"))))
                
                dataset_was_created = True
                
            except Exception as e:
                logger.error("=" * 80)
                logger.error("FAILED TO CREATE NORMALIZED DATASET")
                logger.error("=" * 80)
                logger.exception("Error details:")
                raise RuntimeError(
                    f"Could not create normalized dataset. Please check that:\n"
                    f"1. Raw dataset exists at: {raw_dir}\n"
                    f"2. Dataset contains .png images\n"
                    f"3. You have write permissions for: {norm_dir}\n"
                    f"Original error: {e}"
                )
        else:
            logger.info("Normalized dataset found: %s", norm_dir)
            num_images = len(list(norm_dir.rglob("*.png")))
            logger.info("Total normalized images: %d\n", num_images)
            
            if num_images == 0:
                logger.warning("Warning: Normalized directory exists but contains no images!")
                logger.warning("  Consider deleting %s and re-running this method.\n", norm_dir)
                return False
        
        # Show plots if requested (regardless of whether dataset was just created)
        if show_plots:
            logger.info("\n%s", "=" * 80)
            logger.info("EXPLORING NORMALIZED DATASET")
            logger.info("%s\n", "=" * 80)
            
            try:
                # Verify dataset has images before attempting visualization
                if not any(norm_dir.rglob("*.png")):
                    logger.warning("Dataset directory exists but no images found. Skipping visualization.")
                    return True
                
                # Update config to point to the normalized directory
                self.config['data']['norm_dir'] = str(norm_dir)
                
                # Ensure img_size exists in config (needed for transforms)
                if 'img_size' not in self.config.get('data', {}):
                    self.config['data']['img_size'] = 256  # Default value
                
                # Load data and show visualizations
                logger.info("Loading data for visualization...")
                try:
                    self.load_data(fold_idx=0)
                except Exception as load_error:
                    logger.warning("Could not load data: %s", str(load_error))
                    logger.exception("Full error traceback:")
                    return True
                
                # Verify loaders were created successfully
                if self.train_loader is None or self.val_loader is None:
                    logger.warning("Could not create dataloaders. Skipping visualization.")
                    return True
                
                # Check if loaders have data
                try:
                    train_size = len(self.train_loader.dataset)
                    val_size = len(self.val_loader.dataset)
                    if train_size == 0 or val_size == 0:
                        logger.warning("Dataloaders are empty. Skipping visualization.")
                        return True
                except Exception as size_error:
                    logger.warning("Could not determine dataset size: %s", str(size_error))
                    logger.exception("Full error traceback:")
                    return True
                
                logger.info("\nGenerating dataset statistics...")
                stats_df = self.get_dataset_statistics()
                print("\n" + stats_df.to_string(index=False))
                
                logger.info("\nGenerating class distribution plot...")
                self.show_class_distribution()
                
                # Show normalization comparison for 3 random images
                logger.info("\nGenerating normalization comparison plot...")
                self.show_normalization_comparison(raw_dir=raw_dir, norm_dir=norm_dir, n_samples=3)
                
                logger.info("\nDataset exploration complete!\n")
                
            except Exception as e:
                logger.warning("Could not generate plots: %s", str(e))
                logger.exception("Full error traceback:")
                logger.warning("You can manually run explorer.run() to see visualizations.\n")
        
        return True
    
    def load_data(self, fold_idx: int = 0) -> None:
        """Load training and validation data for specified fold."""
        # Update config to use the correct normalized directory
        if 'data' in self.config and 'norm_dir' in self.config['data']:
            logger.info("Loading data from: %s", self.config['data']['norm_dir'])
        
        self.train_loader, self.val_loader = get_dataloaders(
            self.config, fold_idx=fold_idx
        )

    def show_class_distribution(self) -> None:
        """Display class distribution in training and validation sets."""
        if self.train_loader is None:
            self.load_data()

        train_labels = []
        for _, labels in self.train_loader:
            train_labels.extend(labels.numpy())

        val_labels = []
        for _, labels in self.val_loader:
            val_labels.extend(labels.numpy())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.hist(train_labels, bins=4, edgecolor='black')
        ax1.set_title('Training Set Distribution')
        ax1.set_xlabel('Inflammation Grade')
        ax1.set_ylabel('Count')
        
        ax2.hist(val_labels, bins=4, edgecolor='black')
        ax2.set_title('Validation Set Distribution')
        ax2.set_xlabel('Inflammation Grade')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / "class_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved class distribution plot: {save_path}")
            plt.close(fig)
        else:
            plt.show()  # Display in notebook

    def show_sample_images(self, num_samples: int = 8) -> None:
        """Display sample images from each class."""
        if self.train_loader is None:
            self.load_data()

        images, labels = next(iter(self.train_loader))
        
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        axes = axes.ravel()
        
        for idx in range(min(num_samples, len(images))):
            img = images[idx].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'Grade {labels[idx].item()}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if self.save_dir:
            save_path = self.save_dir / "sample_images.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved sample images plot: {save_path}")
            plt.close(fig)
        else:
            plt.show()  # Display in notebook

    def show_normalization_comparison(self, raw_dir: Path, norm_dir: Path, n_samples: int = 3) -> None:
        """Display comparison of original, normalized, and difference images.
        
        Args:
            raw_dir: Path to raw dataset directory
            norm_dir: Path to normalized dataset directory
            n_samples: Number of image samples to show (default: 3)
        """
        import cv2
        import random
        
        # Find matching images in both s
        raw_images = list(raw_dir.rglob("*.png"))
        if len(raw_images) == 0:
            logger.warning("No raw images found for comparison")
            return
        
        # Sample random images
        sampled_paths = random.sample(raw_images, min(n_samples, len(raw_images)))
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, raw_path in enumerate(sampled_paths):
            # Find corresponding normalized image
            rel_path = raw_path.relative_to(raw_dir)
            norm_path = norm_dir / rel_path
            
            if not norm_path.exists():
                logger.warning(f"Normalized image not found: {norm_path}")
                continue
            
            # Load images
            raw_img = cv2.imread(str(raw_path))
            norm_img = cv2.imread(str(norm_path))
            
            if raw_img is None or norm_img is None:
                logger.warning(f"Could not load images: {raw_path}")
                continue
            
            # Convert BGR to RGB
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)

            # Compute difference (absolute) with contrast stretching for visibility
            diff_float = np.abs(raw_img.astype(float) - norm_img.astype(float))
            diff_max = diff_float.max()
            if diff_max > 0:
                diff_img = (diff_float / diff_max * 255).astype(np.uint8)
            else:
                diff_img = diff_float.astype(np.uint8)

            # Display original
            axes[idx, 0].imshow(raw_img)
            axes[idx, 0].set_title(f'Original #{idx+1}', fontsize=12, fontweight='bold')
            axes[idx, 0].axis('off')

            # Display normalized
            axes[idx, 1].imshow(norm_img)
            axes[idx, 1].set_title(f'Normalized #{idx+1}', fontsize=12, fontweight='bold')
            axes[idx, 1].axis('off')

            # Display difference (contrast-stretched: bright = large change)
            axes[idx, 2].imshow(diff_img)
            axes[idx, 2].set_title(f'Difference #{idx+1} (amplified)', fontsize=12, fontweight='bold')
            axes[idx, 2].axis('off')
        
        plt.suptitle('Stain Normalization Comparison: Original vs Normalized vs Difference', 
                     fontsize=16, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        
        if self.save_dir:
            save_path = self.save_dir / "normalization_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.25)
            print(f"Saved normalization comparison plot: {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def show_rgb_histograms(self, raw_dir: Path, norm_dir: Path, n_samples: int = 3) -> None:
        """Display RGB histograms as bar charts for original and normalized images.
        Creates separate plots for each color channel (R, G, B).
        
        Args:
            raw_dir: Path to raw dataset directory
            norm_dir: Path to normalized dataset directory
            n_samples: Number of image samples to show (default: 3)
        """
        import cv2
        import random
        
        # Find matching images in both directories
        raw_images = list(raw_dir.rglob("*.png"))
        if len(raw_images) == 0:
            logger.warning("No raw images found for histogram comparison")
            return
        
        # Sample random images (same seed for consistency)
        sampled_paths = random.sample(raw_images, min(n_samples, len(raw_images)))
        
        # Define channel info: (channel_idx, color_name, display_color)
        channels_info = [
            (0, 'R', 'red'),
            (1, 'G', 'green'),
            (2, 'B', 'blue')
        ]
        
        # Create separate figure for each color channel
        for channel_idx, channel_name, channel_color in channels_info:
            fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)
            
            for idx, raw_path in enumerate(sampled_paths):
                # Find corresponding normalized image
                rel_path = raw_path.relative_to(raw_dir)
                norm_path = norm_dir / rel_path
                
                if not norm_path.exists():
                    logger.warning(f"Normalized image not found: {norm_path}")
                    continue
                
                # Load images
                raw_img = cv2.imread(str(raw_path))
                norm_img = cv2.imread(str(norm_path))
                
                if raw_img is None or norm_img is None:
                    logger.warning(f"Could not load images: {raw_path}")
                    continue
                
                # Convert BGR to RGB
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
                
                # Calculate histogram for original image (current channel)
                hist_raw = cv2.calcHist([raw_img], [channel_idx], None, [256], [0, 256]).flatten()
                
                # Plot bar chart for original image
                axes[idx, 0].bar(range(256), hist_raw, color=channel_color, alpha=0.7, width=1.0)
                axes[idx, 0].set_title(f'Original #{idx+1} - {channel_name}-Kanal Histogramm', 
                                      fontsize=11, fontweight='bold')
                axes[idx, 0].set_xlabel('Pixel Intensity')
                axes[idx, 0].set_ylabel('Frequency')
                axes[idx, 0].grid(True, alpha=0.3, axis='y')
                axes[idx, 0].set_xlim([0, 255])
                
                # Calculate histogram for normalized image (current channel)
                hist_norm = cv2.calcHist([norm_img], [channel_idx], None, [256], [0, 256]).flatten()
                
                # Plot bar chart for normalized image
                axes[idx, 1].bar(range(256), hist_norm, color=channel_color, alpha=0.7, width=1.0)
                axes[idx, 1].set_title(f'Normalisiert #{idx+1} - {channel_name}-Kanal Histogramm', 
                                      fontsize=11, fontweight='bold')
                axes[idx, 1].set_xlabel('Pixel Intensity')
                axes[idx, 1].set_ylabel('Frequency')
                axes[idx, 1].grid(True, alpha=0.3, axis='y')
                axes[idx, 1].set_xlim([0, 255])
            
            plt.suptitle(f'{channel_name}-Kanal Histogram Comparison: Original vs Normalisiert', 
                         fontsize=16, fontweight='bold', y=0.998)
            plt.tight_layout()
            
            if self.save_dir:
                save_path = self.save_dir / f"histogram_{channel_name}_channel.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved {channel_name}-channel histogram plot: {save_path}")
                plt.close(fig)
            else:
                plt.show()

    def get_dataset_statistics(self) -> pd.DataFrame:
        """Compute and return dataset statistics."""
        if self.train_loader is None:
            self.load_data()

        stats = {
            'Split': ['Training', 'Validation'],
            'Samples': [
                len(self.train_loader.dataset),
                len(self.val_loader.dataset)
            ],
            'Batch Size': [
                self.train_loader.batch_size,
                self.val_loader.batch_size
            ]
        }
        
        return pd.DataFrame(stats)


if __name__ == '__main__':
    explorer = DataExplorer()