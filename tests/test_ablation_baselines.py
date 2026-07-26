"""Tests for ablation baselines B7 & B8."""
import pytest
import torch
from torch.utils.data import DataLoader
from epigeonet.models.baselines.transformer_only import TransformerOnly
from epigeonet.models.baselines.gat_only import GATOnly
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
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

def test_transformer_only(dummy_batch):
    device = get_device()
    model = TransformerOnly().to(device)
    model.eval()
    with torch.no_grad():
        preds, _ = model(dummy_batch)
    assert 'reg' in preds
    assert preds['reg'].shape == (2, 6, 3)

def test_gat_only(dummy_batch):
    device = get_device()
    model = GATOnly().to(device)
    model.eval()
    with torch.no_grad():
        preds, _ = model(dummy_batch)
    assert 'reg' in preds
    assert preds['reg'].shape == (2, 6, 3)
