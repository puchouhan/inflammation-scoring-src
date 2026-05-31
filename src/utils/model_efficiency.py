"""
Model Efficiency Metrics
Calculates computational complexity, inference time, and memory usage.
"""
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from src.utils.seeds_logging import get_logger

logger = get_logger("ModelEfficiency")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def measure_inference_time(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256),
    num_warmup: int = 10,
    num_iterations: int = 100,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Measure inference time with warmup.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        num_warmup: Number of warmup iterations
        num_iterations: Number of measurement iterations
        device: Device to run on (auto-detects if None)
        
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Ensure model is on the target device
    model = model.to(device)
    
    dummy_input = torch.randn(input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Synchronize for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            
            # Synchronize
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
    }


def measure_memory_usage(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256),
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Measure memory usage during inference.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Device to run on (auto-detects if None)
        
    Returns:
        Dictionary with memory statistics in MB
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    # Ensure model is on the target device
    model = model.to(device)
    
    memory_stats = {}
    
    if device.type == 'cuda':
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        # Measure baseline
        baseline = torch.cuda.memory_allocated(device) / (1024 ** 2)
        
        # Run inference
        dummy_input = torch.randn(input_size, device=device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Get stats
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        
        memory_stats = {
            "allocated_mb": float(allocated),
            "peak_mb": float(peak),
            "baseline_mb": float(baseline),
            "inference_mb": float(allocated - baseline),
        }
    elif device.type == 'mps':
        # MPS doesn't have detailed memory APIs
        memory_stats = {
            "allocated_mb": 0.0,
            "peak_mb": 0.0,
            "baseline_mb": 0.0,
            "inference_mb": 0.0,
            "note": "MPS memory tracking not available",
        }
    else:
        memory_stats = {
            "allocated_mb": 0.0,
            "peak_mb": 0.0,
            "baseline_mb": 0.0,
            "inference_mb": 0.0,
            "note": "CPU memory tracking not implemented",
        }
    
    return memory_stats


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size on disk in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    import tempfile
    
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_mb = tmp.tell() / (1024 ** 2)
    
    return float(size_mb)


def compute_efficiency_metrics(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256),
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute all efficiency metrics for a model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size for single image (1, C, H, W)
        batch_size: Batch size for throughput calculation
        device: Device to run on (auto-detects if None)
        verbose: Whether to print results
        
    Returns:
        Dictionary with all efficiency metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    logger.info("Computing efficiency metrics...")
    
    # 1. Parameter count
    total_params, trainable_params = count_parameters(model)
    
    # 2. Model size
    model_size_mb = get_model_size_mb(model)
    
    # 3. Inference time (single image)
    timing_single = measure_inference_time(model, input_size, device=device)
    
    # 4. Batch inference time
    batch_input_size = (batch_size, *input_size[1:])
    timing_batch = measure_inference_time(
        model, batch_input_size, num_iterations=50, device=device
    )
    
    # 5. Memory usage
    memory_stats = measure_memory_usage(model, batch_input_size, device=device)
    
    # 6. Throughput (images per second)
    throughput = (batch_size / timing_batch["mean_ms"]) * 1000  # images/sec
    
    # Compile results
    metrics = {
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "total_millions": round(total_params / 1e6, 2),
        },
        "model_size": {
            "mb": round(model_size_mb, 2),
        },
        "inference_time": {
            "single_image_ms": {
                "mean": round(timing_single["mean_ms"], 3),
                "std": round(timing_single["std_ms"], 3),
                "median": round(timing_single["median_ms"], 3),
            },
            "batch_ms": {
                "mean": round(timing_batch["mean_ms"], 3),
                "std": round(timing_batch["std_ms"], 3),
                "median": round(timing_batch["median_ms"], 3),
                "batch_size": batch_size,
            },
        },
        "throughput": {
            "images_per_second": round(throughput, 2),
            "batch_size": batch_size,
        },
        "memory": memory_stats,
        "device": str(device),
        "input_size": list(input_size),
    }
    
    if verbose:
        logger.info("=" * 60)
        logger.info("MODEL EFFICIENCY METRICS")
        logger.info("=" * 60)
        logger.info(f"Parameters:        {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"Trainable:         {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logger.info(f"Model Size:        {model_size_mb:.2f} MB")
        logger.info(f"Inference (1 img): {timing_single['mean_ms']:.3f} ± {timing_single['std_ms']:.3f} ms")
        logger.info(f"Inference (batch): {timing_batch['mean_ms']:.3f} ms for {batch_size} images")
        logger.info(f"Throughput:        {throughput:.2f} images/sec")
        if memory_stats.get("peak_mb", 0) > 0:
            logger.info(f"Memory (peak):     {memory_stats['peak_mb']:.2f} MB")
        logger.info("=" * 60)
    
    return metrics


def compare_model_efficiency(
    models: Dict[str, nn.Module],
    input_size: Tuple[int, int, int, int] = (1, 3, 256, 256),
    batch_size: int = 32,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare efficiency metrics across multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        input_size: Input tensor size
        batch_size: Batch size for throughput
        device: Device to run on
        
    Returns:
        Dictionary of model_name -> efficiency_metrics
    """
    results = {}
    
    logger.info(f"Comparing efficiency of {len(models)} models...")
    
    for model_name, model in models.items():
        logger.info(f"\nAnalyzing {model_name}...")
        metrics = compute_efficiency_metrics(
            model, input_size, batch_size, device, verbose=False
        )
        results[model_name] = metrics
    
    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("EFFICIENCY COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Model':<15} {'Params (M)':<12} {'Size (MB)':<12} {'Infer (ms)':<12} {'Throughput':<12}")
    logger.info("-" * 80)
    
    for model_name, metrics in results.items():
        params_m = metrics["parameters"]["total_millions"]
        size_mb = metrics["model_size"]["mb"]
        infer_ms = metrics["inference_time"]["single_image_ms"]["mean"]
        throughput = metrics["throughput"]["images_per_second"]
        
        logger.info(f"{model_name:<15} {params_m:<12.2f} {size_mb:<12.2f} {infer_ms:<12.3f} {throughput:<12.2f}")
    
    logger.info("=" * 80 + "\n")
    
    return results
