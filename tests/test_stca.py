"""Tests for Spatio-Temporal Cross-Attention (STCA)."""
import pytest
import torch
from epigeonet.models.layers.stca import STCA

def test_stca_shape_and_attention():
    """Test output shapes and attention weight properties."""
    N = 10
    T = 12
    dim = 64
    out_dim = 128
    
    spatial = torch.randn(N, T, dim)
    temporal = torch.randn(N, T, dim)
    
    model = STCA(dim=dim, heads=4, out=out_dim)
    
    z, attn = model(spatial, temporal)
    
    # Check shapes
    assert z.shape == (N, out_dim)
    assert attn.shape == (N, T)
    
    # Check that attention rows sum to ~1 over T
    attn_sums = attn.sum(dim=1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5)
