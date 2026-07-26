"""Logging configuration."""
import sys
from pathlib import Path
from loguru import logger

def get_logger(name: str):
    """
    Configure loguru logger to output to console and a file in results/logs/{name}.log.
    
    Args:
        name (str): Name of the log file (without .log extension).
        
    Returns:
        logger: Configured loguru logger instance.
    """
    logger.remove()
    
    logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"
    
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", rotation="10 MB")
    
    return logger
