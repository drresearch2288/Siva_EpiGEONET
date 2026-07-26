"""Tests for Spatial GATv2 Encoder."""
import pytest
import torch
from epigeonet.models.layers.spatial_gatv2 import SpatialEncoder, BatchedSpatialEncoder

def test_spatial_encoder():
    """Test output shape and attention extraction of SpatialEncoder."""
    N = 10
    in_dim = 64
    hidden = 64
    
    x = torch.randn(N, in_dim)
    edge_index = torch.randint(0, N, (2, 20))
    edge_weight = torch.rand(20)
    
    encoder = SpatialEncoder(in_dim=in_dim, hidden=hidden, heads=4, layers=2, edge_dim=1)
    
    h, attns = encoder(x, edge_index, edge_weight)
    
    assert h.shape == (N, hidden)
    assert len(attns) == 2
    
    for edge_idx, alpha in attns:
        assert torch.isfinite(alpha).all()
        assert alpha.shape[1] == 4  # 4 heads

def test_batched_spatial_encoder():
    """Test the Batched wrapper loops correctly."""
    T = 3
    N = 10
    in_dim = 64
    hidden = 64
    
    x = torch.randn(T, N, in_dim)
    edge_indices = [torch.randint(0, N, (2, 15)) for _ in range(T)]
    edge_weights = [torch.rand(15) for _ in range(T)]
    
    encoder = SpatialEncoder(in_dim=in_dim, hidden=hidden, heads=4, layers=2)
    batched = BatchedSpatialEncoder(encoder)
    
    h_out, all_attns = batched(x, edge_indices, edge_weights)
    
    assert h_out.shape == (T, N, hidden)
    assert len(all_attns) == T
    assert len(all_attns[0]) == 2
