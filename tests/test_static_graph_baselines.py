"""Tests for static graph baselines B5 & B6."""
import pytest
import torch
from torch.utils.data import DataLoader
from epigeonet.models.baselines.st_gcn_gru import ST_GCN_GRU
from epigeonet.models.baselines.dcrnn import DCRNN
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
from epigeonet.models.losses import total_loss
from epigeonet.utils.device import get_device

@pytest.fixture
def dummy_batch():
    device = get_device()
    N = 6
    T = 12
    in_dim = 24
    
    sample = {
        'x': torch.randn(T, N, in_dim),
        'static': torch.randn(N, 1),
        'w_clim': torch.rand(T, N, N),
        'w_case': torch.rand(T, N, N),
        'graph_static': {
            'edge_index': torch.tensor([[0, 1, 2], [1, 2, 0]]),
            'edge_weight': torch.tensor([1.0, 1.0, 1.0]),
            'A_geo': torch.eye(N),
            'w_dist': torch.rand(N, N)
        },
        'targets': {
            'reg': torch.ones(N, 3),
            'risk': torch.zeros(N, dtype=torch.long),
            'alert': torch.zeros(N, 1).float()
        }
    }
    
    ds = EpiGeoDataset()
    ds.samples = [sample, sample]
    
    loader = DataLoader(ds, batch_size=2, collate_fn=epigeonet_collate, shuffle=True, pin_memory=False)
    
    batch = next(iter(loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    for k, v in batch['targets'].items():
        batch['targets'][k] = v.to(device)
    for k, v in batch['graph_static'].items():
        batch['graph_static'][k] = v.to(device)
        
    return batch

def test_st_gcn_gru_forward_and_grads(dummy_batch):
    device = get_device()
    model = ST_GCN_GRU(in_dim=24, hidden=64, num_layers=1).to(device)
    
    model.train()
    preds, _ = model(dummy_batch)
    loss, _ = total_loss(preds, dummy_batch['targets'], dummy_batch['graph_static'])
    
    loss.backward()
    
    # Check finite grads
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all()
    assert has_grad

def test_dcrnn_forward_and_grads(dummy_batch):
    device = get_device()
    model = DCRNN(in_dim=24, hidden=64, K=2).to(device)
    
    model.train()
    preds, _ = model(dummy_batch)
    loss, _ = total_loss(preds, dummy_batch['targets'], dummy_batch['graph_static'])
    
    loss.backward()
    
    # Check finite grads
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all()
    assert has_grad
