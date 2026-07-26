"""Tests for the Trainer."""
import pytest
import torch
from torch.utils.data import DataLoader
from epigeonet.models.epigeonet import EpiGeoNet
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
from epigeonet.training.trainer import Trainer
from epigeonet.utils.device import get_device

def test_trainer_loss_decreases():
    device = get_device()
    N = 10
    T = 12
    in_dim = 24
    
    ds = EpiGeoDataset()
    
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
    ds.samples = [sample, sample] # batch size 2
    
    loader = DataLoader(ds, batch_size=2, collate_fn=epigeonet_collate, shuffle=True)
    
    model = EpiGeoNet(n_nodes=N, in_dim=in_dim, explain=False).to(device)
    
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        for k, v in batch['targets'].items():
            batch['targets'][k] = v.to(device)
        for k, v in batch['graph_static'].items():
            batch['graph_static'][k] = v.to(device)
            
        preds, _ = model(batch)
        from epigeonet.models.losses import total_loss
        initial_loss, _ = total_loss(preds, batch['targets'], batch['graph_static'])
        
    trainer = Trainer(model, device, lr=0.01, epochs=2, save_dir='tests/models')
    
    best_val_loss = trainer.fit(loader, loader)
    
    assert best_val_loss < initial_loss
