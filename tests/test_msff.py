"""Tests for MSFF layer."""
import pytest
import torch
from epigeonet.models.layers.msff import MSFF

def test_msff_forward_shape():
    """Test the output shape of MSFF forward pass."""
    B, T, N, in_dim = 2, 12, 10, 24
    hidden = 64
    x = torch.randn(B, T, N, in_dim)
    
    msff = MSFF(in_dim=in_dim, hidden=hidden)
    
    out = msff(x)
    assert out.shape == (B, T, N, hidden)

def test_msff_broadcast_static():
    """Test that static broadcasting is timestep-invariant and shapes match."""
    B, T, N, in_dim = 2, 12, 10, 24
    hidden = 64
    static_dim = 1
    
    x = torch.randn(B, T, N, in_dim)
    static = torch.randn(N, static_dim)
    
    msff = MSFF(in_dim=in_dim, hidden=hidden, static_dim=static_dim)
    msff.eval() 
    
    h = msff(x)
    h_fused = msff.broadcast_static(h, static)
    
    assert h_fused.shape == (B, T, N, hidden)
    
    diff = h_fused - h
    
    # Check variance along T dimension (dim=1)
    diff_var_T = diff.var(dim=1)
    assert torch.allclose(diff_var_T, torch.zeros_like(diff_var_T), atol=1e-6)
    
    # Check variance along B dimension (dim=0)
    diff_var_B = diff.var(dim=0)
    assert torch.allclose(diff_var_B, torch.zeros_like(diff_var_B), atol=1e-6)
