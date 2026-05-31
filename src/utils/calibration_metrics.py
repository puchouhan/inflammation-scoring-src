"""
Calibration and Confidence Metrics
Measures prediction confidence quality and calibration for clinical deployment.
"""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss
from src.utils.seeds_logging import get_logger

logger = get_logger("Calibration")


def compute_confidence_metrics(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute confidence-related metrics.
    
    Args:
        predictions: Predicted class indices (N,)
        probabilities: Predicted probabilities (N, num_classes)
        targets: Ground truth labels (N,)
        
    Returns:
        Dictionary with confidence metrics
    """
    # Maximum probability (confidence) for each prediction
    confidences = probabilities.max(axis=1)
    
    # Accuracy for each confidence level
    correct = (predictions == targets)
    
    metrics = {
        "mean_confidence": float(np.mean(confidences)),
        "std_confidence": float(np.std(confidences)),
        "min_confidence": float(np.min(confidences)),
        "max_confidence": float(np.max(confidences)),
        "median_confidence": float(np.median(confidences)),
        
        # Confidence when correct vs incorrect
        "mean_confidence_correct": float(np.mean(confidences[correct])) if correct.sum() > 0 else 0.0,
        "mean_confidence_incorrect": float(np.mean(confidences[~correct])) if (~correct).sum() > 0 else 0.0,
        
        # Accuracy stats
        "accuracy": float(correct.mean()),
        "num_samples": int(len(targets)),
    }
    
    # Confidence gap (should be large for well-calibrated models)
    metrics["confidence_gap"] = metrics["mean_confidence_correct"] - metrics["mean_confidence_incorrect"]
    
    return metrics


def compute_expected_calibration_error(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy across bins.
    Lower is better (0 = perfect calibration).
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
        predictions: Predicted class indices (N,)
        targets: Ground truth labels (N,)
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with ECE and calibration data
    """
    confidences = probabilities.max(axis=1)
    correct = (predictions == targets)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # ECE contribution from this bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_center": float((bin_lower + bin_upper) / 2),
                "accuracy": float(accuracy_in_bin),
                "confidence": float(avg_confidence_in_bin),
                "count": int(in_bin.sum()),
                "proportion": float(prop_in_bin),
            })
        else:
            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_center": float((bin_lower + bin_upper) / 2),
                "accuracy": 0.0,
                "confidence": 0.0,
                "count": 0,
                "proportion": 0.0,
            })
    
    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "bin_data": bin_data,
        "total_samples": int(len(targets)),
    }


def compute_brier_score(
    probabilities: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> float:
    """
    Compute Brier Score (lower is better).
    
    Measures mean squared error between predicted probabilities and true labels.
    
    Args:
        probabilities: Predicted probabilities (N, num_classes)
        targets: Ground truth labels (N,)
        num_classes: Number of classes
        
    Returns:
        Brier score (0 = perfect, 1 = worst)
    """
    # One-hot encode targets
    targets_onehot = np.zeros((len(targets), num_classes))
    targets_onehot[np.arange(len(targets)), targets] = 1
    
    # Compute mean squared error
    brier = np.mean(np.sum((probabilities - targets_onehot) ** 2, axis=1))
    
    return float(brier)


def evaluate_calibration(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 4,
    ignore_index: Optional[int] = None,
    n_bins: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Full calibration evaluation on a dataset.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on
        num_classes: Number of classes (excluding ignore)
        ignore_index: Index of ignore class to exclude
        n_bins: Number of bins for ECE
        verbose: Whether to print results
        
    Returns:
        Dictionary with all calibration metrics
    """
    model.eval()
    
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Get predictions
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            
            # Filter out ignore class if specified
            if ignore_index is not None:
                valid_mask = batch_y != ignore_index
                batch_y = batch_y[valid_mask]
                probs = probs[valid_mask]
                
                # Remove ignore class from probabilities
                if ignore_index < probs.size(1):
                    probs_list = [probs[:, :ignore_index], probs[:, ignore_index+1:]]
                    probs = torch.cat(probs_list, dim=1)
                
                # Renormalize
                probs = probs / probs.sum(dim=1, keepdim=True)
            
            preds = probs.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    confidence_metrics = compute_confidence_metrics(all_preds, all_probs, all_targets)
    ece_metrics = compute_expected_calibration_error(all_probs, all_preds, all_targets, n_bins)
    brier = compute_brier_score(all_probs, all_targets, num_classes)
    
    results = {
        "confidence": confidence_metrics,
        "calibration": ece_metrics,
        "brier_score": brier,
    }
    
    if verbose:
        logger.info("=" * 60)
        logger.info("CALIBRATION METRICS")
        logger.info("=" * 60)
        logger.info(f"Samples:                 {confidence_metrics['num_samples']}")
        logger.info(f"Accuracy:                {confidence_metrics['accuracy']:.4f}")
        logger.info("")
        logger.info("CONFIDENCE:")
        logger.info(f"  Mean:                  {confidence_metrics['mean_confidence']:.4f}")
        logger.info(f"  Mean (correct):        {confidence_metrics['mean_confidence_correct']:.4f}")
        logger.info(f"  Mean (incorrect):      {confidence_metrics['mean_confidence_incorrect']:.4f}")
        logger.info(f"  Confidence Gap:        {confidence_metrics['confidence_gap']:.4f}")
        logger.info("")
        logger.info("CALIBRATION:")
        logger.info(f"  ECE:                   {ece_metrics['ece']:.4f}")
        logger.info(f"  Brier Score:           {brier:.4f}")
        logger.info("=" * 60)
    
    return results


def plot_calibration_curve(
    ece_metrics: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Calibration Curve"
) -> Any:
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        ece_metrics: ECE metrics from compute_expected_calibration_error
        save_path: Optional path to save figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    
    bin_data = ece_metrics["bin_data"]
    
    # Extract data for plotting
    bin_centers = [bd["bin_center"] for bd in bin_data]
    accuracies = [bd["accuracy"] for bd in bin_data]
    confidences = [bd["confidence"] for bd in bin_data]
    counts = [bd["count"] for bd in bin_data]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curve
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
    ax1.bar(bin_centers, accuracies, width=0.08, alpha=0.3, label='Accuracy', edgecolor='black')
    ax1.plot(bin_centers, confidences, 'ro-', label='Confidence', linewidth=2)
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece_metrics["ece"]:.4f}', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Sample distribution
    ax2.bar(bin_centers, counts, width=0.08, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration curve saved to {save_path}")
    
    return fig


def compare_model_calibration(
    models: Dict[str, nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 4,
    ignore_index: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare calibration across multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        dataloader: DataLoader for evaluation
        device: Device to run on
        num_classes: Number of classes
        ignore_index: Index of ignore class
        
    Returns:
        Dictionary of model_name -> calibration_metrics
    """
    results = {}
    
    logger.info(f"Comparing calibration of {len(models)} models...")
    
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        metrics = evaluate_calibration(
            model, dataloader, device, num_classes, ignore_index, verbose=False
        )
        results[model_name] = metrics
    
    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("CALIBRATION COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Model':<15} {'Accuracy':<12} {'Mean Conf':<12} {'ECE':<12} {'Brier':<12}")
    logger.info("-" * 80)
    
    for model_name, metrics in results.items():
        acc = metrics["confidence"]["accuracy"]
        conf = metrics["confidence"]["mean_confidence"]
        ece = metrics["calibration"]["ece"]
        brier = metrics["brier_score"]
        
        logger.info(f"{model_name:<15} {acc:<12.4f} {conf:<12.4f} {ece:<12.4f} {brier:<12.4f}")
    
    logger.info("=" * 80 + "\n")
    
    return results
