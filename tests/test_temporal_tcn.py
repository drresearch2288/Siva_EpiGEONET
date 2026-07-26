"""Tests for the Temporal TCN Encoder."""
import pytest
import torch
from epigeonet.models.layers.temporal_tcn import TemporalEncoder

def test_tcn_shape_and_rf():
    """Test output shape and receptive field coverage."""
    B, N, T, C = 2, 5, 12, 64
    x = torch.randn(B, N, T, C)
    
    model = TemporalEncoder(channels=C, layers=4, kernel=3, dilations=(1, 2, 4, 8))
    
    out = model(x)
    
    assert out.shape == (B, N, T, C)
    assert model.receptive_field >= 12

def test_tcn_causality():
    """Test causality: perturbing t+1 doesn't change output at t."""
    B, N, T, C = 1, 1, 12, 64
    x1 = torch.randn(B, N, T, C)
    x2 = x1.clone()
    
    # Perturb time step t=6 in x2
    x2[0, 0, 6, :] += 10.0
    
    model = TemporalEncoder(channels=C, layers=4, kernel=3, dilations=(1, 2, 4, 8))
    model.eval()
    
    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)
        
    # Outputs at t < 6 should be exactly the same
    assert torch.allclose(out1[:, :, :6, :], out2[:, :, :6, :], atol=1e-6)
    
    # Outputs at t >= 6 should be different
    assert not torch.allclose(out1[:, :, 6:, :], out2[:, :, 6:, :], atol=1e-6)
