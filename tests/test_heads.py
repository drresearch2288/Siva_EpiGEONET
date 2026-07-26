"""Tests for multi-task heads."""
import pytest
import torch
from epigeonet.models.layers.heads import MultiTaskHeads

def test_multitask_heads():
    """Test output shapes and non-negativity of regression outputs."""
    N = 10
    in_dim = 128
    
    z = torch.randn(N, in_dim)
    
    model = MultiTaskHeads(in_dim=in_dim, horizons=3, n_classes=4)
    
    outputs = model(z)
    
    assert 'reg' in outputs
    assert 'risk_logits' in outputs
    assert 'alert_logit' in outputs
    
    reg = outputs['reg']
    risk = outputs['risk_logits']
    alert = outputs['alert_logit']
    
    assert reg.shape == (N, 3)
    assert risk.shape == (N, 4)
    assert alert.shape == (N, 1)
    
    # Regression outputs must be non-negative (thanks to Softplus)
    assert torch.all(reg >= 0)
    
    # Alert head threshold should exist as a buffer
    assert hasattr(model.alert_head, 'threshold')
