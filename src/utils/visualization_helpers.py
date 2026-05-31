"""
Visualization Helpers for Model Evaluation
Provides functions to generate and save plots for experiment tracking.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional, Dict, Any
import json


def save_confusion_matrix(
    model,
    dataloader,
    save_path: Path,
    class_names: list = None,
    fold_idx: Optional[int] = None
):
    """
    Generate and save confusion matrix plot.
    
    Args:
        model: Trained model (Lightning module or PyTorch model)
        dataloader: DataLoader for evaluation
        save_path: Path to save the plot
        class_names: List of class names for labels
        fold_idx: Optional fold index for filename
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    
    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                images, labels = batch
            else:
                images = batch['image']
                labels = batch['label']
            
            images = images.to(device)
            
            # Get predictions
            if hasattr(model, 'model'):
                # Lightning module
                outputs = model.model(images)
            else:
                # Regular PyTorch model
                outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Robustly handle class label mismatches
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    else:
        # If class_names is longer than cm, trim; if shorter, extend
        if len(class_names) > n_classes:
            class_names = class_names[:n_classes]
        elif len(class_names) < n_classes:
            class_names = class_names + [f'Class {i}' for i in range(len(class_names), n_classes)]

    # Warn if ignore class is present but not in confusion matrix
    if 'Ignore' in class_names and n_classes < len(class_names):
        import warnings
        warnings.warn("'Ignore' class label present in class_names but not in confusion matrix. Skipping.")
        class_names = class_names[:n_classes]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='.0f')

    if fold_idx is not None:
        ax.set_title(f'Confusion Matrix - Fold {fold_idx}')
    else:
        ax.set_title('Confusion Matrix')

    plt.tight_layout()

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Confusion matrix saved: {save_path}")

    return cm


def save_per_class_metrics(
    model,
    dataloader,
    save_path: Path,
    class_names: list = None,
    fold_idx: Optional[int] = None
):
    """
    Calculate and save per-class metrics as JSON and visualization.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        save_path: Path to save the JSON (PNG will be saved alongside)
        class_names: List of class names
        fold_idx: Optional fold index
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    
    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                images, labels = batch
            else:
                images = batch['image']
                labels = batch['label']
            
            images = images.to(device)
            
            if hasattr(model, 'model'):
                outputs = model.model(images)
            else:
                outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate per-class metrics
    cm = confusion_matrix(all_labels, all_preds)
    n_classes = len(class_names)
    
    per_class_metrics = {}
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[str(i)] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(cm[i, :].sum())
        }
    
    # Save JSON
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(per_class_metrics, f, indent=2)
    
    # Create visualization
    metrics_data = {
        'Precision': [per_class_metrics[str(i)]['precision'] for i in range(n_classes)],
        'Recall': [per_class_metrics[str(i)]['recall'] for i in range(n_classes)],
        'F1-Score': [per_class_metrics[str(i)]['f1'] for i in range(n_classes)]
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(n_classes)
    width = 0.25
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax.bar(x + idx * width, values, width, label=metric_name)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    if fold_idx is not None:
        ax.set_title(f'Per-Class Metrics - Fold {fold_idx}')
    else:
        ax.set_title('Per-Class Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure alongside JSON
    plot_path = save_path.parent / save_path.name.replace('.json', '.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Per-class metrics saved: {save_path}")
    print(f"Per-class plot saved: {plot_path}")
    
    return per_class_metrics


def save_training_curves(
    trainer,
    save_dir: Path,
    fold_idx: Optional[int] = None
):
    """
    Extract and plot training curves from Lightning trainer logs.
    
    Args:
        trainer: PyTorch Lightning Trainer instance
        save_dir: Directory to save the plots
        fold_idx: Optional fold index
    """
    if not hasattr(trainer, 'logged_metrics'):
        print("WARNING: No logged metrics found in trainer")
        return
    
    # Try to extract metrics from callbacks
    metrics_history = {}
    
    # Check if we have a logger with metrics
    if trainer.logger is not None and hasattr(trainer.logger, 'experiment'):
        try:
            # For TensorBoard logger
            if hasattr(trainer.logger.experiment, 'scalar_dict'):
                metrics_history = trainer.logger.experiment.scalar_dict
        except:
            pass
    
    # If no metrics found, skip
    if not metrics_history:
        print("WARNING: Could not extract training curves")
        return
    
    # Plot available metrics
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metric_names = ['train_loss', 'val_loss', 'val_acc', 'val_kappa']
    
    for idx, metric_name in enumerate(metric_names):
        if metric_name in metrics_history:
            values = metrics_history[metric_name]
            axes[idx].plot(values)
            axes[idx].set_title(metric_name.replace('_', ' ').title())
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3)
        else:
            axes[idx].text(0.5, 0.5, f'{metric_name}\nnot available',
                          ha='center', va='center', fontsize=12)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
    
    if fold_idx is not None:
        fig.suptitle(f'Training Curves - Fold {fold_idx}', fontsize=16)
    else:
        fig.suptitle('Training Curves', fontsize=16)
    
    plt.tight_layout()
    
    save_path = save_dir / f'training_curves_fold_{fold_idx}.png' if fold_idx is not None else save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Training curves saved: {save_path}")


def save_predictions_csv(
    model,
    dataloader,
    save_path: Path,
    fold_idx: Optional[int] = None
):
    """
    Save predictions as CSV for detailed error analysis.
    
    CSV Format:
    filepath, ground_truth, prediction, correct, confidence_0, confidence_1, ...
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        save_path: Path to save CSV file
        fold_idx: Optional fold index
    """
    import pandas as pd
    
    model.eval()
    device = next(model.parameters()).device
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (tuple, list)):
                images, labels = batch
                # Try to get filepaths from dataset
                if hasattr(dataloader.dataset, 'df'):
                    start_idx = batch_idx * dataloader.batch_size
                    end_idx = start_idx + len(labels)
                    filepaths = dataloader.dataset.df.iloc[start_idx:end_idx]['filepath'].tolist()
                else:
                    filepaths = [f"image_{batch_idx}_{i}" for i in range(len(labels))]
            else:
                images = batch['image']
                labels = batch['label']
                filepaths = batch.get('filepath', [f"image_{batch_idx}_{i}" for i in range(len(labels))])
            
            images = images.to(device)
            
            # Get predictions
            if hasattr(model, 'model'):
                outputs = model.model(images)
            else:
                outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Store results
            for i in range(len(labels)):
                result = {
                    'filepath': filepaths[i] if i < len(filepaths) else f"unknown_{batch_idx}_{i}",
                    'ground_truth': int(labels[i].cpu().item()),
                    'prediction': int(preds[i].cpu().item()),
                    'correct': int(labels[i].cpu().item()) == int(preds[i].cpu().item()),
                }
                
                # Add confidence scores for each class
                for class_idx in range(probs.shape[1]):
                    result[f'confidence_{class_idx}'] = float(probs[i, class_idx].cpu().item())
                
                results.append(result)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    
    # Print summary
    accuracy = df['correct'].mean()
    total = len(df)
    correct = df['correct'].sum()
    
    print(f"Predictions saved: {save_path}")
    print(f"  Total samples: {total}")
    print(f"  Correct: {correct} ({accuracy*100:.2f}%)")
    print(f"  Incorrect: {total - correct} ({(1-accuracy)*100:.2f}%)")
    
    return df


def plot_training_curves_from_csv(
    metrics_csv_path: Path,
    save_dir: Path,
    fold_idx: Optional[int] = None
):
    """
    Plot training curves from Lightning CSV logger output.
    
    Args:
        metrics_csv_path: Path to metrics.csv from Lightning logger
        save_dir: Directory to save plots
        fold_idx: Optional fold index
    """
    import pandas as pd
    
    if not metrics_csv_path.exists():
        print(f"WARNING: Metrics CSV not found: {metrics_csv_path}")
        return
    
    df = pd.read_csv(metrics_csv_path)
    
    # Define metrics to plot
    metric_pairs = [
        ('train_loss_epoch', 'val_loss', 'Loss'),
        ('train_acc_epoch', 'val_acc', 'Accuracy'),
        (None, 'val_kappa', 'Quadratic Weighted Kappa'),
        (None, 'val_macro_f1', 'Macro F1-Score')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (train_metric, val_metric, title) in enumerate(metric_pairs):
        ax = axes[idx]
        
        has_data = False
        
        if train_metric and train_metric in df.columns:
            train_data = df[train_metric].dropna()
            if len(train_data) > 0:
                ax.plot(train_data.index, train_data.values, label='Train', marker='o', markersize=3)
                has_data = True
        
        if val_metric in df.columns:
            val_data = df[val_metric].dropna()
            if len(val_data) > 0:
                ax.plot(val_data.index, val_data.values, label='Validation', marker='s', markersize=3)
                has_data = True
        
        if has_data:
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{title}\nNo data available',
                   ha='center', va='center', fontsize=10, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    
    if fold_idx is not None:
        fig.suptitle(f'Training Curves - Fold {fold_idx}', fontsize=16, y=1.00)
    else:
        fig.suptitle('Training Curves', fontsize=16, y=1.00)
    
    plt.tight_layout()
    
    save_path = save_dir / (f'fold_{fold_idx}_training_curves.png' if fold_idx is not None else 'training_curves.png')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Training curves saved: {save_path}")


def save_all_visualizations(
    model,
    trainer,
    val_loader,
    metrics_dir: Path,
    predictions_dir: Path,
    fold_idx: Optional[int] = None,
    class_names: list = None
):
    """
    Save all standard visualizations for a trained model.
    
    Generates:
    - Confusion matrix (PNG)
    - Per-class metrics (JSON + PNG)
    - Predictions CSV (for error analysis)
    - Training curves (PNG, if available)
    
    Args:
        model: Trained model
        trainer: Lightning Trainer instance
        val_loader: Validation DataLoader
        metrics_dir: Directory to save metrics and plots
        predictions_dir: Directory to save predictions CSV
        fold_idx: Optional fold index
        class_names: List of class names
    """
    if class_names is None:
        class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']
    
    print(f"\nGenerating visualizations for fold {fold_idx}...")
    
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm_path = metrics_dir / (f'fold_{fold_idx}_confusion_matrix.png' if fold_idx is not None else 'confusion_matrix.png')
    save_confusion_matrix(model, val_loader, cm_path, class_names, fold_idx)
    
    # 2. Per-Class Metrics
    per_class_path = metrics_dir / (f'fold_{fold_idx}_per_class.json' if fold_idx is not None else 'per_class_metrics.json')
    save_per_class_metrics(model, val_loader, per_class_path, class_names, fold_idx)
    
    # 3. Predictions CSV
    pred_path = predictions_dir / (f'fold_{fold_idx}_predictions.csv' if fold_idx is not None else 'predictions.csv')
    save_predictions_csv(model, val_loader, pred_path, fold_idx)
    
    # 4. Training Curves (from Lightning logger)
    try:
        # Try to find metrics.csv from Lightning CSV logger
        if trainer.logger is not None:
            log_dir = Path(trainer.logger.log_dir) if hasattr(trainer.logger, 'log_dir') else None
            if log_dir and log_dir.exists():
                metrics_csv = log_dir / 'metrics.csv'
                if metrics_csv.exists():
                    plot_training_curves_from_csv(metrics_csv, metrics_dir, fold_idx)
                else:
print(f"WARNING: Metrics CSV not found at {metrics_csv}")
            else:
                print("WARNING: Could not locate Lightning log directory")
        except Exception as e:
        print(f"WARNING: Could not save training curves: {e}")
    
    print(f"All visualizations saved to {metrics_dir}\n")
