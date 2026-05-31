"""
Logging configuration for the project.
Sets up structured logging across all modules.
"""
import logging
import sys
from pathlib import Path


def setup_logging(level=logging.INFO, log_file=None, output_mode="both"):
    """
    Configure logging for the entire project.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
        output_mode: Where to output logs - "console", "file", or "both" (default: "both")
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if output_mode in ["console", "both"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if output_mode in ["file", "both"] and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


# Initialize logging when module is imported
setup_logging()
