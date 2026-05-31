import os
import random
import numpy as np
import torch
import logging
from rich.logging import RichHandler
from rich.console import Console

def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers for PyTorch, numpy and random.
    Ensures 100% reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    get_logger().info(f"Global seed set to: {seed}")

def get_logger(name: str = "InflammationProject"):
    """
    Returns a Rich logger with formatted output.
    """
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
    return logging.getLogger(name)

def log_config(config: dict):
    """
    Logs the configuration dictionary in a readable format.
    """
    console = Console()
    console.print(config, style="bold cyan")
