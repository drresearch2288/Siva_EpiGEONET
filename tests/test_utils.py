"""Tests for utility modules."""
import torch
import pytest
from epigeonet.utils.device import get_device, to_device
from epigeonet.utils.seed import set_seed
from epigeonet.utils.config import load_all

def test_get_device():
    """Assert get_device() returns a torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["mps", "cpu"]

def test_to_device():
    """Assert a float32 tensor round-trips to device and back."""
    device = get_device()
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    
    tensor_dev = to_device(tensor, device)
    assert tensor_dev.device.type == device.type
    
    tensor_cpu = to_device(tensor_dev, torch.device("cpu"))
    assert tensor_cpu.device.type == "cpu"
    assert torch.allclose(tensor, tensor_cpu)

def test_set_seed():
    """Assert set_seed makes two torch.randn calls reproducible."""
    set_seed(42)
    tensor1 = torch.randn(5, 5)
    
    set_seed(42)
    tensor2 = torch.randn(5, 5)
    
    assert torch.allclose(tensor1, tensor2)
