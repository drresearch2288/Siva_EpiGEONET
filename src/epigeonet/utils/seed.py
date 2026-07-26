"""Seed utilities for reproducibility."""
import random
import os
import numpy as np
import torch
from loguru import logger

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.backends.mps.is_available():
        if hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)
            
    # Some algorithms might not have deterministic implementations in MPS yet,
    # but we set it where possible to enforce reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info(f"Global seed set to {seed}")
