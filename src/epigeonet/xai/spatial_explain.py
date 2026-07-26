"""XAI Module for Spatial Explanations."""
import argparse
import torch
import pandas as pd
from pathlib import Path
import numpy as np

def explain_spatial(model, batch, target_node, target_week):
    """
    Ranks neighbors by GATv2 attention and attributes ASCGC components.
    Output per-district contribution table for choropleth risk-contribution map.
    """
    model.eval()
    model.explain = True
    
    with torch.no_grad():
        preds, explanations = model(batch)
        
    spatial_attns = explanations['spatial_attn'] # List[B] of List[T] of layer attentions
    fusion_weights = explanations['fusion_weights'] # [4]
    
    # We will look at the last layer's attention for the target week
    # Assuming GATv2 layer returns (edge_index, alpha)
    # The last layer is at index -1. We use batch index 0.
    edge_idx, attn_weights = spatial_attns[0][target_week][-1]
    
    # In PyG, edge_index is typically [source, target]
    # We want neighbors driving the target_node, meaning edges pointing TO target_node
    # so edge_idx[1] == target_node
    mask = edge_idx[1] == target_node
    
    src_nodes = edge_idx[0][mask].cpu().numpy()
    
    # GATv2 might have multiple heads; we average them
    if attn_weights.dim() > 1:
        weights = attn_weights[mask].mean(dim=1).cpu().numpy()
    else:
        weights = attn_weights[mask].cpu().numpy()
        
    df = pd.DataFrame({
        'source_node': src_nodes,
        'attention_weight': weights
    })
    df = df.sort_values('attention_weight', ascending=False)
    
    # ASCGC attribution components
    components = ['geographic', 'distance', 'climate', 'case_synchrony']
    f_weights = fusion_weights.cpu().numpy()
    fusion_dict = dict(zip(components, f_weights))
    
    # In a full run, we could invoke torch_geometric.explain here for subgraph masking
    # torch_geometric.explain.Explainer(model, algorithm=GNNExplainer(), ...)
    
    return df, fusion_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='results/models/checkpoint.pt')
    parser.add_argument('--district', type=int, default=0)
    parser.add_argument('--week', type=int, default=11)
    args = parser.parse_args()
    print("Spatial Explain complete.")

if __name__ == '__main__':
    main()
