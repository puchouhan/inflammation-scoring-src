"""Performance visualization utilities for trained models.

This module provides comprehensive visualization and analysis of model performance
including confusion matrices, per-class metrics, fold consistency, and training curves.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def visualize_model_performance(
    results: Dict[str, Any],
    run_id: str,
    model_name: str,
    config: Dict[str, Any],
    project_root: Path
) -> None:
    """Generate comprehensive performance visualizations for a trained model.
    
    This function creates 4 types of visualizations:
    1. Confusion matrices for all folds (side-by-side comparison)
    2. Per-class performance metrics (Precision, Recall, F1)
    3. Fold-wise QWK comparison with consistency metrics
    4. Training curves from TensorBoard logs (Loss & QWK over epochs)
    
    Args:
        results: Training results dictionary containing fold metrics and paths
        run_id: Unique run identifier (timestamp-based)
        model_name: Name of the model architecture
        config: Configuration dictionary with directory paths
        project_root: Absolute path to project root directory
        
    Example:
        >>> from src.utils.performance_visualizer import visualize_model_performance
        >>> 
        >>> visualize_model_performance(
        ...     results=results,
        ...     run_id=run_id,
        ...     model_name='densenet',
        ...     config=config,
        ...     project_root=PROJECT_ROOT
        ... )
    """
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # Get experiment directory
    exp_dir = project_root / config['directories']['experiments_dir'] / run_id / model_name
    figures_dir = exp_dir / config['directories']['figures_subdir']
    
    # Auto-detect number of folds from results
    n_folds = len(results['fold_results'])
    
    print("=" * 80)
    print(f"PERFORMANCE ANALYSIS: {model_name.upper()}")
    print("=" * 80)
    print(f"Run ID: {run_id}")
    print(f"Mean Validation QWK: {results['mean_qwk']:.4f}")
    print(f"Std Validation QWK: {results['std_qwk']:.4f}")
    print("=" * 80)
    
    # === 1. CONFUSION MATRICES (ALL FOLDS) ===
    _plot_confusion_matrices(results, figures_dir, n_folds)
    
    # === 2. PER-CLASS PERFORMANCE METRICS ===
    _plot_per_class_metrics(figures_dir, n_folds)
    
    # === 3. FOLD-WISE QWK COMPARISON ===
    _plot_fold_consistency(results, n_folds)
    
    # === 4. TRAINING CURVES FROM TENSORBOARD ===
    _plot_training_curves(exp_dir, config, n_folds)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 80)


def _plot_confusion_matrices(
    results: Dict[str, Any],
    figures_dir: Path,
    n_folds: int
) -> None:
    """Plot confusion matrices for all folds side-by-side."""
    print("\n1. Confusion Matrices (All Folds)")
    fig, axes = plt.subplots(1, n_folds, figsize=(5*n_folds, 4))
    if n_folds == 1:
        axes = [axes]
    
    for fold_idx in range(n_folds):
        cm_path = figures_dir / f"confusion_matrix_fold_{fold_idx}.png"
        if cm_path.exists():
            img = Image.open(cm_path)
            axes[fold_idx].imshow(img)
            axes[fold_idx].axis('off')
            axes[fold_idx].set_title(
                f"Fold {fold_idx} (QWK: {results['fold_results'][fold_idx]['val_qwk']:.3f})", 
                fontsize=12, fontweight='bold'
            )
        else:
            axes[fold_idx].text(0.5, 0.5, f"Fold {fold_idx}\nNot Found", 
                               ha='center', va='center', fontsize=12)
            axes[fold_idx].axis('off')
    
    plt.tight_layout()
    plt.close(fig)


def _plot_per_class_metrics(figures_dir: Path, n_folds: int) -> None:
    """Plot per-class performance metrics averaged across folds."""
    print("\n2. Per-Class Performance (Averaged Across Folds)")
    
    # Load per-class metrics from JSON files
    all_metrics = []
    for fold_idx in range(n_folds):
        metrics_path = figures_dir / f"per_class_metrics_fold_{fold_idx}.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                fold_metrics = json.load(f)
                all_metrics.append(fold_metrics)
    
    if all_metrics:
        # Calculate mean metrics across folds
        classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Ignore']
        metrics_names = ['precision', 'recall', 'f1-score']
        
        mean_metrics = {cls: {m: [] for m in metrics_names} for cls in classes}
        
        for fold_data in all_metrics:
            for cls in classes:
                if cls in fold_data:
                    for metric in metrics_names:
                        if metric in fold_data[cls]:
                            mean_metrics[cls][metric].append(fold_data[cls][metric])
        
        # Compute averages
        avg_metrics = {
            cls: {
                m: np.mean(mean_metrics[cls][m]) if mean_metrics[cls][m] else 0 
                for m in metrics_names
            } 
            for cls in classes
        }
        
        # Plot bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics_names):
            values = [avg_metrics[cls][metric] for cls in classes]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Inflammation Grade', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance (Mean Across Folds)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.close(fig)
        
        # Print numeric values
        print("\nNumeric Values:")
        for cls in classes:
            print(f"  {cls:10s}: Precision={avg_metrics[cls]['precision']:.3f}, "
                  f"Recall={avg_metrics[cls]['recall']:.3f}, "
                  f"F1={avg_metrics[cls]['f1-score']:.3f}")
    else:
        print("  No per-class metrics found.")


def _plot_fold_consistency(results: Dict[str, Any], n_folds: int) -> None:
    """Plot fold-wise QWK comparison to assess model consistency."""
    print("\n3. Fold-wise QWK Comparison")
    
    fold_qwks = [results['fold_results'][i]['val_qwk'] for i in range(n_folds)]
    fold_labels = [f"Fold {i}" for i in range(n_folds)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71' if qwk == max(fold_qwks) else '#3498db' for qwk in fold_qwks]
    bars = ax.bar(fold_labels, fold_qwks, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, qwk in zip(bars, fold_qwks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{qwk:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add mean line
    ax.axhline(
        y=results['mean_qwk'], 
        color='r', 
        linestyle='--', 
        linewidth=2, 
        label=f"Mean: {results['mean_qwk']:.3f} ± {results['std_qwk']:.3f}"
    )
    
    ax.set_ylabel('Quadratic Weighted Kappa', fontsize=12, fontweight='bold')
    ax.set_title('Fold-wise Performance Consistency', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.close(fig)
    
    print(f"\nConsistency: Mean QWK = {results['mean_qwk']:.4f} ± {results['std_qwk']:.4f}")
    print(f"Range: [{min(fold_qwks):.4f}, {max(fold_qwks):.4f}]")
    print(f"Coefficient of Variation: {(results['std_qwk']/results['mean_qwk'])*100:.2f}%")


def _plot_training_curves(exp_dir: Path, config: Dict[str, Any], n_folds: int) -> None:
    """Plot training curves (Loss & QWK) from TensorBoard logs."""
    print("\n4. Training Curves (from TensorBoard Logs)")
    
    try:
        from tensorboard.backend.event_processing import event_accumulator
        
        # Get TensorBoard directory
        tb_dir = exp_dir / config['directories']['tensorboard_subdir']
        
        print(f"  Looking for TensorBoard logs in: {tb_dir}")
        print(f"  Directory exists: {tb_dir.exists()}")
        
        if tb_dir.exists():
            # Collect data from all folds
            all_train_loss = {}
            all_val_qwk = {}
            
            for fold_idx in range(n_folds):
                fold_tb_dir = tb_dir / f"fold_{fold_idx}"
                print(f"  Checking fold {fold_idx}: {fold_tb_dir}")
                
                if not fold_tb_dir.exists():
                    print(f"    Fold directory not found")
                    continue
                    
                # Find event file
                event_files = list(fold_tb_dir.glob("events.out.tfevents.*"))
                print(f"    Found {len(event_files)} event files")
                
                if not event_files:
                    continue
                
                # Load TensorBoard data
                try:
                    ea = event_accumulator.EventAccumulator(str(event_files[0]))
                    ea.Reload()
                    
                    available_tags = ea.Tags()['scalars']
                    print(f"    Available metrics: {available_tags}")
                    
                    # Extract metrics
                    if 'train_loss_epoch' in available_tags:
                        train_loss = ea.Scalars('train_loss_epoch')
                        all_train_loss[fold_idx] = [(s.step, s.value) for s in train_loss]
                        print(f"    Loaded {len(train_loss)} training loss points")
                    
                    if 'val_qwk' in available_tags:
                        val_qwk = ea.Scalars('val_qwk')
                        all_val_qwk[fold_idx] = [(s.step, s.value) for s in val_qwk]
                        print(f"    Loaded {len(val_qwk)} validation QWK points")
                except Exception as e:
                    print(f"    Error loading event file: {e}")
                    continue
            
            # Plot training curves
            if all_train_loss or all_val_qwk:
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                
                # Plot Training Loss
                if all_train_loss:
                    ax = axes[0]
                    for fold_idx, data in all_train_loss.items():
                        steps, values = zip(*data)
                        ax.plot(steps, values, marker='o', label=f'Fold {fold_idx}', 
                               linewidth=2, markersize=4)
                    
                    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
                    ax.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                else:
                    axes[0].text(0.5, 0.5, 'Training Loss\nNot Available', 
                                ha='center', va='center', fontsize=12)
                    axes[0].axis('off')
                
                # Plot Validation QWK
                if all_val_qwk:
                    ax = axes[1]
                    for fold_idx, data in all_val_qwk.items():
                        steps, values = zip(*data)
                        ax.plot(steps, values, marker='o', label=f'Fold {fold_idx}', 
                               linewidth=2, markersize=4)
                    
                    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Validation QWK', fontsize=12, fontweight='bold')
                    ax.set_title('Validation QWK Over Epochs', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    ax.set_ylim(0, 1.0)
                else:
                    axes[1].text(0.5, 0.5, 'Validation QWK\nNot Available', 
                                ha='center', va='center', fontsize=12)
                    axes[1].axis('off')
                
                plt.tight_layout()
                plt.close(fig)
                
                print("  Training curves loaded successfully")
            else:
                print("  No TensorBoard data found in expected format")
        else:
            print(f"  TensorBoard directory not found: {tb_dir}")
            
    except ImportError:
        print("  tensorboard package not installed. Install with: pip install tensorboard")
    except Exception as e:
        logger.warning(f"Error loading TensorBoard data: {e}")
        print(f"  Error loading TensorBoard data: {e}")
