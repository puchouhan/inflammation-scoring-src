"""
Optimized visualization helpers - Single inference pass for all outputs.
Reduces evaluation time from ~15 min to ~5 min per fold by eliminating redundant inference.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from typing import Optional, Tuple
import json
import pandas as pd


def run_single_inference(
    model,
    dataloader,
    extract_filepaths: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[list]]:
    """
    Run model inference ONCE and cache all results.
    
    This eliminates redundant inference passes - instead of:
    - save_confusion_matrix: full inference
    - save_per_class_metrics: full inference  
    - save_predictions_csv: full inference
    
    We do: single inference → generate all outputs from cached results.
    
    Args:
        model: Trained model (Lightning module or PyTorch model)
        dataloader: DataLoader for evaluation
        extract_filepaths: Whether to extract filepaths (needed only for CSV)
        
    Returns:
        labels: Ground truth labels (N,)
        preds: Predicted class indices (N,)
        probs: Prediction probabilities (N, num_classes)
        filepaths: Optional list of filepaths
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_labels = []
    all_probs = []
    filepaths = [] if extract_filepaths else None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                images, labels = batch
                if extract_filepaths and hasattr(dataloader.dataset, 'df'):
                    start_idx = batch_idx * dataloader.batch_size
                    end_idx = start_idx + len(labels)
                    batch_paths = dataloader.dataset.df.iloc[start_idx:end_idx]['filepath'].tolist()
                    filepaths.extend(batch_paths)
            else:
                images = batch['image']
                labels = batch['label']
                if extract_filepaths:
                    filepaths.extend(batch.get('filepath', [f"sample_{batch_idx}_{i}" for i in range(len(labels))]))
            
            images = images.to(device)
            
            # Get predictions
            if hasattr(model, 'model'):
                outputs = model.model(images)
            else:
                outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        filepaths
    )


def save_confusion_matrix_from_cache(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Path,
    class_names: list = None,
    fold_idx: Optional[int] = None
):
    """Generate confusion matrix from cached predictions."""
    if class_names is None:
        class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    
    cm = confusion_matrix(labels, preds)
    n_classes = cm.shape[0]
    
    # Handle class name mismatches
    if len(class_names) > n_classes:
        class_names = class_names[:n_classes]
    elif len(class_names) < n_classes:
        class_names = class_names + [f'Class {i}' for i in range(len(class_names), n_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='.0f')
    
    ax.set_title(f'Confusion Matrix - Fold {fold_idx}' if fold_idx is not None else 'Confusion Matrix')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Confusion matrix saved: {save_path}")
    return cm


def save_per_class_metrics_from_cache(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Path,
    class_names: list = None,
    fold_idx: Optional[int] = None
):
    """Calculate per-class metrics from cached predictions."""
    if class_names is None:
        class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # Keep plotting and JSON generation robust even if metrics and class_names
    # differ in length (e.g., ignore class present in predictions).
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    f1 = np.asarray(f1)
    support = np.asarray(support)
    n_display = int(min(precision.shape[0], len(class_names)))
    precision = precision[:n_display]
    recall = recall[:n_display]
    f1 = f1[:n_display]
    support = support[:n_display]
    class_names = class_names[:n_display]
    
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    # Save JSON
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(per_class_metrics, f, indent=2)
    
    # Generate plot
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Class')
    ax.set_title(f'Per-Class Metrics - Fold {fold_idx}' if fold_idx is not None else 'Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_path.parent / save_path.name.replace('.json', '.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Per-class metrics saved: {save_path}")
    print(f"Per-class plot saved: {plot_path}")
    
    return per_class_metrics


def save_predictions_csv_from_cache(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
    filepaths: Optional[list] = None,
    fold_idx: Optional[int] = None
):
    """Save detailed predictions CSV from cached results."""
    results = []
    
    for i in range(len(labels)):
        result = {
            'filepath': filepaths[i] if filepaths else f"sample_{i}",
            'ground_truth': int(labels[i]),
            'prediction': int(preds[i]),
            'correct': int(labels[i]) == int(preds[i])
        }
        
        # Add confidence scores for all classes
        for class_idx in range(probs.shape[1]):
            result[f'confidence_{class_idx}'] = float(probs[i, class_idx])
        
        results.append(result)
    
    df = pd.DataFrame(results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    
    accuracy = df['correct'].mean()
    total = len(df)
    correct = df['correct'].sum()
    
    print(f"Predictions saved: {save_path}")
    print(f"  Total: {total}, Correct: {correct} ({accuracy*100:.2f}%), Incorrect: {total - correct} ({(1-accuracy)*100:.2f}%)")
    
    return df


def generate_all_visualizations_optimized(
    model,
    val_loader,
    metrics_dir: Path,
    predictions_dir: Path,
    fold_idx: Optional[int] = None,
    class_names: list = None,
    include_predictions_csv: bool = False,
    exclude_ignore_class: bool = True,
    ignore_class_idx: int = 4,
):
    """
    Generate all visualizations with SINGLE inference pass (3x faster).
    
    Performance comparison:
    - Old: 3 separate inference passes (~15 min)
    - New: 1 inference pass (~5 min)
    - Speedup: 3x faster
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        metrics_dir: Directory to save metrics and plots
        predictions_dir: Directory to save predictions CSV
        fold_idx: Optional fold index
        class_names: List of class names
        include_predictions_csv: If True, generate CSV (slower due to filepath extraction)
        exclude_ignore_class: If True, remove ignore-class samples before plotting/metrics
        ignore_class_idx: Class index used for ignore/artifact class
    """
    if class_names is None:
        class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']
    
    print(f"\nGenerating visualizations for fold {fold_idx} (optimized - single inference)...")
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # SINGLE INFERENCE PASS
    labels, preds, probs, filepaths = run_single_inference(
        model, val_loader, extract_filepaths=include_predictions_csv
    )

    # Keep evaluation outputs aligned with ordinal inflammation scoring (classes 0-3)
    # by default excluding ignore/artifact class samples from post-training plots.
    if exclude_ignore_class:
        keep_mask = labels != ignore_class_idx
        removed_count = int((~keep_mask).sum())
        if removed_count > 0:
            print(
                f"Excluding {removed_count} ignore-class samples (class {ignore_class_idx}) "
                f"for fold {fold_idx} visualization outputs."
            )
        labels = labels[keep_mask]
        preds = preds[keep_mask]
        probs = probs[keep_mask]
        if filepaths is not None:
            filepaths = [fp for fp, keep in zip(filepaths, keep_mask) if keep]

    if labels.size == 0:
        print(f"Skipping visualization generation for fold {fold_idx}: no samples after filtering.")
        return
    
    # Generate all outputs from cached results
    # 1. Confusion Matrix (essential)
    cm_path = metrics_dir / (f'fold_{fold_idx}_confusion_matrix.png' if fold_idx is not None else 'confusion_matrix.png')
    save_confusion_matrix_from_cache(labels, preds, cm_path, class_names, fold_idx)
    
    # 2. Per-Class Metrics (essential)
    per_class_path = metrics_dir / (f'fold_{fold_idx}_per_class.json' if fold_idx is not None else 'per_class_metrics.json')
    save_per_class_metrics_from_cache(labels, preds, per_class_path, class_names, fold_idx)
    
    # 3. Predictions CSV (optional - for error analysis)
    if include_predictions_csv:
        pred_path = predictions_dir / (f'fold_{fold_idx}_predictions.csv' if fold_idx is not None else 'predictions.csv')
        save_predictions_csv_from_cache(labels, preds, probs, pred_path, filepaths, fold_idx)
    else:
        print("Skipping predictions CSV (set include_predictions_csv=True to generate)")
    
    print(f"All visualizations saved to {metrics_dir}\n")
