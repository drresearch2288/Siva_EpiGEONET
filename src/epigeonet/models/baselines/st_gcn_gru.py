"""ST-GCN-GRU Baseline."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.nn import GCNConv
from epigeonet.models.layers.heads import MultiTaskHeads
from epigeonet.training.trainer import Trainer
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
from torch.utils.data import DataLoader
from epigeonet.utils.device import get_device

class ST_GCN_GRU(nn.Module):
    def __init__(self, in_dim=24, hidden=64, num_layers=2):
        super().__init__()
        self.hidden = hidden
        self.gcn = GCNConv(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers, batch_first=True)
        self.head = MultiTaskHeads(hidden)
        
    def forward(self, batch):
        x = batch['x'] # [B, T, N, D]
        edge_index = batch['graph_static']['edge_index']
        edge_weight = batch['graph_static']['edge_weight']
        
        B, T, N, D = x.shape
        
        # Apply GCN per timestep
        # We can reshape x to [B*T*N, D] or apply it in a loop
        # For simplicity and batching in PyG, we can do it step-by-step or reshape carefully.
        # But edge_index is for N nodes. If we want to process B*T graphs, we need batching.
        # Let's do a loop over T for simplicity since T=12 is small, or loop over B*T.
        
        out = []
        for b in range(B):
            b_out = []
            for t in range(T):
                xt = x[b, t] # [N, D]
                ht = self.gcn(xt, edge_index, edge_weight) # [N, hidden]
                b_out.append(ht)
            out.append(torch.stack(b_out)) # [T, N, hidden]
            
        out = torch.stack(out) # [B, T, N, hidden]
        
        # Flatten B, N for GRU
        out_flat = out.transpose(1, 2).reshape(B * N, T, self.hidden) # [B*N, T, hidden]
        
        gru_out, _ = self.gru(out_flat)
        
        h_last = gru_out[:, -1, :].view(B, N, self.hidden) # [B, N, hidden]
        
        preds = self.head(h_last)
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
        
        model = ST_GCN_GRU(in_dim=24, hidden=64, num_layers=2).to(device)
        trainer = Trainer(model, device, epochs=args.epochs, save_dir='results/models')
        
        trainer.fit(train_loader, val_loader)
        
    print("ST-GCN-GRU baseline script ready.")

if __name__ == '__main__':
    main()
