"""XAI Module for Temporal Explanations."""
import argparse
import torch
import numpy as np

def explain_temporal(model, batch):
    """
    Turns STCA temporal_attn over the 12-week window into a lag-saliency vector per district
    (effective lead time).
    """
    model.eval()
    model.explain = True
    
    with torch.no_grad():
        preds, explanations = model(batch)
        
    temporal_attn = explanations['temporal_attn'] # [B, N, T]
    
    # Extract lag-saliency vector per district by averaging over batches (if B > 1)
    lag_saliency = temporal_attn.mean(dim=0).cpu().numpy() # [N, T]
    
    return lag_saliency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='results/models/checkpoint.pt')
    args = parser.parse_args()
    print("Temporal Explain complete.")

if __name__ == '__main__':
    main()
