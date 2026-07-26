"""LSTM Baseline."""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from epigeonet.models.layers.heads import MultiTaskHeads
from epigeonet.training.trainer import Trainer
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
from torch.utils.data import DataLoader
from epigeonet.utils.device import get_device

class LSTMBaseline(nn.Module):
    def __init__(self, in_dim=24, hidden=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers, batch_first=True)
        self.head = MultiTaskHeads(hidden)
        
    def forward(self, batch):
        x = batch['x'] # [B, T, N, in_dim]
        B, T, N, D = x.shape
        
        # Flatten B and N to treat each district independently
        x_flat = x.transpose(1, 2).reshape(B * N, T, D) # [B*N, T, D]
        
        # LSTM
        out, (hn, cn) = self.lstm(x_flat)
        
        # Take last timestep
        h_last = out[:, -1, :] # [B*N, hidden]
        
        # Reshape back to [B, N, hidden]
        h_last = h_last.view(B, N, -1)
        
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
        
        model = LSTMBaseline(in_dim=24, hidden=64, num_layers=2).to(device)
        trainer = Trainer(model, device, epochs=args.epochs, save_dir='results/models')
        
        trainer.fit(train_loader, val_loader)
        # In a real pipeline, we'd then predict and save to parquet
        
    print("LSTM baseline script ready.")

if __name__ == '__main__':
    main()
