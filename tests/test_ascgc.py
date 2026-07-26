"""Tests for the ASCGC graph module."""
import pytest
import torch
from epigeonet.graph.ascgc import ASCGC

def test_ascgc_initialization():
    """Test fusion weights sum to 1 and initialize correctly."""
    n_nodes = 10
    model = ASCGC(n_nodes=n_nodes, k=3, init=(0.5, 0.2, 0.2, 0.1))
    
    weights = model.fusion_weights()
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    # Softmax of log(init) should recover init closely
    assert torch.allclose(weights, torch.tensor([0.5, 0.2, 0.2, 0.1]), atol=1e-6)

def test_ascgc_forward():
    """Test top-k sparsification and normalization."""
    n_nodes = 5
    k = 2
    model = ASCGC(n_nodes=n_nodes, k=k)
    
    w_clim = torch.rand(n_nodes, n_nodes)
    w_case = torch.rand(n_nodes, n_nodes)
    static = {
        'A_geo': torch.eye(n_nodes),
        'w_dist': torch.rand(n_nodes, n_nodes)
    }
    
    edge_index, edge_weight = model(w_clim, w_case, static)
    
    E = n_nodes * k
    assert edge_index.shape == (2, E)
    assert edge_weight.shape == (E,)
    
    # Check row normalization
    ew_matrix = edge_weight.view(n_nodes, k)
    assert torch.allclose(ew_matrix.sum(dim=1), torch.ones(n_nodes))
    
def test_ascgc_gradients():
    """Test that gradients flow to alpha..delta."""
    n_nodes = 5
    model = ASCGC(n_nodes=n_nodes, k=2)
    
    w_clim = torch.rand(n_nodes, n_nodes, requires_grad=False)
    w_case = torch.rand(n_nodes, n_nodes, requires_grad=False)
    static = {
        'A_geo': torch.eye(n_nodes),
        'w_dist': torch.rand(n_nodes, n_nodes)
    }
    
    edge_index, edge_weight = model(w_clim, w_case, static)
    
    loss = edge_weight.sum()
    # The sum of normalized weights per row is always exactly 1.0 (k*1.0),
    # so the derivative of the sum w.r.t logits might be zero theoretically.
    # Let's use a non-trivial loss function to ensure gradient flow
    loss = (edge_weight * torch.arange(len(edge_weight)).float()).sum()
    loss.backward()
    
    assert model.fusion_logits.grad is not None
    assert model.fusion_logits.grad.shape == (4,)
