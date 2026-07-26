"""Transformer-only Baseline (B7)."""
import argparse
import math
import torch
import torch.nn as nn
from pathlib import Path
from epigeonet.models.layers.heads import MultiTaskHeads
from epigeonet.training.trainer import Trainer
from epigeonet.training.dataset import EpiGeoDataset, epigeonet_collate
from torch.utils.data import DataLoader
from epigeonet.utils.device import get_device

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerOnly(nn.Module):
    def __init__(self, in_dim=24, hidden=64, num_layers=4, nhead=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pos_encoder = PositionalEncoding(hidden)
        encoder_layers = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=hidden*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.head = MultiTaskHeads(hidden)
        
    def forward(self, batch):
        x = batch['x'] # [B, T, N, D]
        B, T, N, D = x.shape
        
        # Flatten across districts
        x_flat = x.transpose(1, 2).reshape(B * N, T, D) # [B*N, T, D]
        
        h = self.proj(x_flat)
        h = self.pos_encoder(h)
        out = self.transformer(h) # [B*N, T, hidden]
        
        h_last = out[:, -1, :] # [B*N, hidden]
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
        
        model = TransformerOnly().to(device)
        trainer = Trainer(model, device, epochs=args.epochs, save_dir='results/models')
        
        trainer.fit(train_loader, val_loader)
        
    print("Transformer baseline script ready.")

if __name__ == '__main__':
    main()
