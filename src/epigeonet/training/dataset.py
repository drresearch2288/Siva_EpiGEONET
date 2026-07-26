"""PyTorch Dataset for EpiGeoNet."""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json

class EpiGeoDataset(Dataset):
    def __init__(self, data_dir: str = 'data/processed', split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = []
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def epigeonet_collate(batch):
    """
    Collate function to assemble batches.
    batch is a list of dicts.
    """
    if not batch:
        return {}
        
    collated = {}
    for key in batch[0].keys():
        if key == 'static':
            collated[key] = batch[0][key]
        elif isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], dict):
            if key == 'graph_static':
                collated[key] = batch[0][key]
            else:
                collated[key] = {k: torch.stack([item[key][k] for item in batch]) for k in batch[0][key]}
        else:
            collated[key] = [item[key] for item in batch]
            
    return collated
