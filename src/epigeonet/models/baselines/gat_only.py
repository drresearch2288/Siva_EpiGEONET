"""GAT-only Baseline (B8)."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from epigeonet.models.layers.spatial_gatv2 import SpatialEncoder
from epigeonet.models.layers.heads import MultiTaskHeads
from epigeonet.training.trainer import Trainer
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
from torch.utils.data import DataLoader
from epigeonet.utils.device import get_device

class GATOnly(nn.Module):
    def __init__(self, in_dim=24, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.gat = SpatialEncoder(in_dim, hidden, heads=4)
        self.head = MultiTaskHeads(hidden)
        
    def forward(self, batch):
        x = batch['x'] # [B, T, N, D]
        edge_index = batch['graph_static']['edge_index']
        edge_weight = batch['graph_static']['edge_weight']
        B, T, N, D = x.shape
        
        # Use only the last timestep (forecast week)
        x_last = x[:, -1, :, :] # [B, N, D]
        
        out = []
        for b in range(B):
            h_b, _ = self.gat(x_last[b], edge_index, edge_weight) # [N, hidden]
            out.append(h_b)
            
        h_out = torch.stack(out) # [B, N, hidden]
        
        preds = self.head(h_out)
        return preds, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    device = get_device()
    
    ds_train = EpiGeoDataset(split='train')
    ds_val = EpiGeoDataset(split='val')
    
    if len(ds_train) > 0:
        train_loader = DataLoader(ds_train, batch_size=32, collate_fn=epigeonet_collate, pin_memory=False)
        val_loader = DataLoader(ds_val, batch_size=32, collate_fn=epigeonet_collate, pin_memory=False)
        
        model = GATOnly().to(device)
        trainer = Trainer(model, device, epochs=args.epochs, save_dir='results/models')
        
        trainer.fit(train_loader, val_loader)
        
    print("GAT-only baseline script ready.")

if __name__ == '__main__':
    main()
