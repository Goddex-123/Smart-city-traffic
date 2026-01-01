"""
Logging utilities for Smart City Traffic System.
Provides a configured logger using loguru.
"""

import sys
from pathlib import Path
from loguru import logger
from .config import get_config


def setup_logger(name: str = "smart_traffic", log_file: str = None) -> logger:
    """
    Setup and configure logger.
    
    Args:
        name: Name of the logger
        log_file: Optional log file name (without path)
        
    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()
    
    # Get configuration
    config = get_config()
    log_config = config.get('logging', {})
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        format=log_config.get(
            'format',
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_config.get('level', 'INFO'),
        colorize=True
    )
    
    # File handler if log file specified
    if log_file:
        log_dir = config.get_path('logs')
        log_path = log_dir / log_file
        
        logger.add(
            str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
            level=log_config.get('level', 'INFO'),
            rotation=log_config.get('rotation', '10 MB'),
            retention=log_config.get('retention', '30 days'),
            compression="zip"
        )
    
    return logger


def get_logger(name: str = None, log_file: str = None) -> logger:
    """
    Get a logger instance.
    
    Args:
        name: Name for the logger (usually __name__)
        log_file: Optional log file name
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Setup default logger
default_logger = setup_logger("smart_traffic", "smart_traffic.log")


if __name__ == "__main__":
    # Test logging
    test_logger = get_logger("test_module")
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.success("This is a success message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    print("\nLogger setup successful!")
