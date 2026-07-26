"""Device configuration and utilities."""
import os
from typing import Any, Dict, List
import torch
from loguru import logger

from epigeonet.utils.config import load_config

_DEVICE_INIT_DONE = False

def get_device(prefer: str = "auto") -> torch.device:
    """
    Get the appropriate PyTorch device (MPS if available, else CPU).
    Reads configuration from config/device.yaml on first call and sets up environment.

    Args:
        prefer (str): Preferred device ('auto', 'mps', or 'cpu').

    Returns:
        torch.device: The selected PyTorch device.
    """
    global _DEVICE_INIT_DONE
    
    if not _DEVICE_INIT_DONE:
        device_config = load_config("config/device.yaml").get("device", {})
        
        fallback = device_config.get("enable_mps_fallback", True)
        if fallback:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
        dtype_str = device_config.get("dtype", "float32")
        if dtype_str == "float32":
            torch.set_default_dtype(torch.float32)
        elif dtype_str == "float64":
            torch.set_default_dtype(torch.float64)
            
        _DEVICE_INIT_DONE = True

    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_device(obj: Any, device: torch.device) -> Any:
    """
    Move tensors, PyG Data, or dictionaries to the specified device.
    Recursively processes dictionaries and lists.

    Args:
        obj (Any): The object to move (Tensor, PyG Data, Dict, List, etc.).
        device (torch.device): Target device.

    Returns:
        Any: The object moved to the device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    elif hasattr(obj, "to") and callable(getattr(obj, "to")):
        return obj.to(device)
    return obj

def mps_peak_memory_mb() -> float:
    """
    Get current peak allocated memory on MPS in MB.

    Returns:
        float: Allocated memory in MB, or 0.0 if not on MPS.
    """
    if torch.backends.mps.is_available():
        # current_allocated_memory returns bytes
        return torch.mps.current_allocated_memory() / 1e6
    return 0.0

def log_device_banner() -> None:
    """Log banner with device, dtype, fallback flag, and torch version information."""
    dev = get_device()
    dtype = torch.get_default_dtype()
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "Not set")
    version = torch.__version__
    
    logger.info("--- PyTorch Environment ---")
    logger.info(f"PyTorch Version: {version}")
    logger.info(f"Active Device:   {dev}")
    logger.info(f"Default Dtype:   {dtype}")
    logger.info(f"MPS Fallback:    {fallback}")
    logger.info("---------------------------")
