"""DCRNN Baseline."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from epigeonet.models.layers.heads import MultiTaskHeads
from epigeonet.training.trainer import Trainer
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
from torch.utils.data import DataLoader
from epigeonet.utils.device import get_device

class DiffusionConv(nn.Module):
    def __init__(self, in_dim, out_dim, K=2):
        super().__init__()
        self.K = K
        self.proj = nn.Linear(in_dim * (2 * K + 1), out_dim)
        
    def forward(self, x, A):
        # x: [B, N, D], A: [N, N]
        B, N, D = x.shape
        device = x.device
        
        # Calculate transition matrices
        # Forward: D_out^-1 A
        # Backward: D_in^-1 A^T
        
        # Add self loops
        A_tilde = A + torch.eye(N, device=device)
        
        D_out = A_tilde.sum(dim=1)
        D_in = A_tilde.sum(dim=0)
        
        P_f = A_tilde / D_out.unsqueeze(1).clamp(min=1e-8)
        P_b = A_tilde.T / D_in.unsqueeze(1).clamp(min=1e-8)
        
        supports = [x]
        
        # Forward and backward diffusions
        # We process as dense matrices for MPS safety and simplicity
        x_f = x
        x_b = x
        for k in range(1, self.K + 1):
            x_f = torch.bmm(P_f.unsqueeze(0).expand(B, N, N), x_f)
            x_b = torch.bmm(P_b.unsqueeze(0).expand(B, N, N), x_b)
            supports.append(x_f)
            supports.append(x_b)
            
        out = torch.cat(supports, dim=-1) # [B, N, in_dim * (2K + 1)]
        return self.proj(out)

class DCRNN(nn.Module):
    def __init__(self, in_dim=24, hidden=64, K=2):
        super().__init__()
        self.hidden = hidden
        
        # A simple DCRNN-like structure: Diffusion Conv then GRU
        # A full DCRNN uses DCGRUCell, but for a baseline this is conceptually identical
        self.dc = DiffusionConv(in_dim, hidden, K)
        self.gru = nn.GRU(hidden, hidden, 1, batch_first=True)
        self.head = MultiTaskHeads(hidden)
        
    def forward(self, batch):
        x = batch['x'] # [B, T, N, D]
        # We need dense A
        edge_index = batch['graph_static']['edge_index']
        edge_weight = batch['graph_static']['edge_weight']
        B, T, N, D = x.shape
        
        device = x.device
        A = torch.zeros((N, N), device=device)
        A[edge_index[0], edge_index[1]] = edge_weight
        
        # Apply DiffusionConv over time
        x_flat_T = x.view(B * T, N, D)
        dc_out = self.dc(x_flat_T, A) # [B*T, N, hidden]
        
        dc_out = dc_out.view(B, T, N, self.hidden)
        
        # GRU over time
        dc_out_flat = dc_out.transpose(1, 2).reshape(B * N, T, self.hidden)
        gru_out, _ = self.gru(dc_out_flat)
        
        h_last = gru_out[:, -1, :].view(B, N, self.hidden)
        
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
        
        model = DCRNN(in_dim=24, hidden=64, K=2).to(device)
        trainer = Trainer(model, device, epochs=args.epochs, save_dir='results/models')
        
        trainer.fit(train_loader, val_loader)
        
    print("DCRNN baseline script ready.")

if __name__ == '__main__':
    main()
