"""Tests for loss functions."""
import pytest
import torch
import torch.nn.functional as F
from epigeonet.models.losses import (
    huber_reg, weighted_ce, focal_alert, laplacian_smoothness, attention_entropy, total_loss
)

def test_losses_non_negative():
    """Test that all loss terms are non-negative."""
    pred = torch.randn(2, 3)
    target = torch.randn(2, 3)
    assert huber_reg(pred, target) >= 0
    
    logits = torch.randn(2, 4)
    target_cls = torch.randint(0, 4, (2,))
    assert weighted_ce(logits, target_cls) >= 0
    
    logit = torch.randn(2, 1)
    target_alert = torch.randint(0, 2, (2, 1)).float()
    assert focal_alert(logit, target_alert) >= 0
    
    risk_probs = F.softmax(torch.randn(2, 5, 4), dim=-1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_weight = torch.tensor([0.5, 0.5, 0.5])
    assert laplacian_smoothness(risk_probs, edge_index, edge_weight) >= 0
    
    attn = F.softmax(torch.randn(2, 12), dim=-1)
    assert attention_entropy(attn) >= 0

def test_focal_downweights_easy_negatives():
    """Test that focal loss down-weights easy negatives compared to BCE."""
    # easy negative: target=0, prediction strongly negative (logit=-5)
    logit = torch.tensor([-5.0])
    target = torch.tensor([0.0])
    
    fl = focal_alert(logit, target, gamma=2, alpha=0.25)
    bce = F.binary_cross_entropy_with_logits(logit, target)
    
    # Since alpha=0.25 for negatives it would be (1-0.25)=0.75 weight, but pt is close to 1
    # so (1-pt)^2 is close to 0. FL should be much smaller than BCE.
    assert fl < bce

def test_laplacian_constant_risk():
    """Laplacian smoothness should be 0 for a constant risk field."""
    B, N, C = 2, 5, 4
    # Constant risk field: all nodes have the same probabilities
    risk_probs = torch.ones(B, N, C) * 0.25
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    edge_weight = torch.ones(5)
    
    loss = laplacian_smoothness(risk_probs, edge_index, edge_weight)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
    
def test_total_loss():
    outputs = {
        'reg': torch.randn(2, 5, 3),
        'risk_logits': torch.randn(2, 5, 4),
        'alert_logit': torch.randn(2, 5, 1),
        'temporal_attn': F.softmax(torch.randn(2, 5, 12), dim=-1)
    }
    targets = {
        'reg': torch.randn(2, 5, 3),
        'risk': torch.randint(0, 4, (2, 5)),
        'alert': torch.randint(0, 2, (2, 5, 1)).float()
    }
    graph = {
        'edge_index': torch.tensor([[0, 1], [1, 0]]),
        'edge_weight': torch.tensor([1.0, 1.0])
    }
    
    loss, components = total_loss(outputs, targets, graph)
    assert loss >= 0
    assert 'loss_total' in components
