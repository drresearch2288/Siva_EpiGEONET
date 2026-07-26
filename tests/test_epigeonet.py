"""Tests for the full EpiGeoNet model."""
import pytest
import torch
from epigeonet.models.epigeonet import EpiGeoNet
from epigeonet.utils.device import get_device

def test_epigeonet_forward_backward():
    """Test tiny forward/backward pass on a 6-district synthetic batch."""
    device = get_device()
    N = 10
    B = 2
    T = 12
    in_dim = 24
    
    model = EpiGeoNet(n_nodes=N, in_dim=in_dim, explain=True).to(device)
    
    batch = {
        'x': torch.randn(B, T, N, in_dim, device=device),
        'static': torch.randn(N, 1, device=device),
        'w_clim': torch.rand(B, T, N, N, device=device),
        'w_case': torch.rand(B, T, N, N, device=device),
        'graph_static': {
            'A_geo': torch.eye(N, device=device),
            'w_dist': torch.rand(N, N, device=device)
        }
    }
    
    preds, explanations = model(batch)
    
    assert 'reg' in preds
    assert 'risk_logits' in preds
    assert 'alert_logit' in preds
    
    assert preds['reg'].shape == (B, N, 3)
    assert preds['risk_logits'].shape == (B, N, 4)
    assert preds['alert_logit'].shape == (B, N, 1)
    
    assert explanations is not None
    assert 'spatial_attn' in explanations
    assert 'temporal_attn' in explanations
    assert 'fusion_weights' in explanations
    
    # Backward pass
    loss = preds['reg'].sum() + preds['risk_logits'].sum() + preds['alert_logit'].sum()
    loss.backward()
    
    # Check ASCGC params have grads
    assert model.ascgc.fusion_logits.grad is not None
    assert torch.isfinite(model.ascgc.fusion_logits.grad).all()
    
    # Check parameter count is reasonable and positive
    assert model.count_parameters() > 1000
