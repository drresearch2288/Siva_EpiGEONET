"""Spatial Encoder using GATv2 for EpiGeoNet."""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from typing import Tuple, List

class SpatialEncoder(nn.Module):
    """
    2-layer, 4-head GATv2 operating on the ASCGC weekly graph.
    Produces spatially-contextualised embeddings and returns attention weights for XAI.
    """
    def __init__(self, in_dim: int = 64, hidden: int = 64, heads: int = 4, layers: int = 2, edge_dim: int = 1):
        super().__init__()
        self.layers = layers
        self.hidden = hidden
        
        assert hidden % heads == 0, "Hidden dimension must be divisible by heads."
        head_dim = hidden // heads
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(layers):
            conv = GATv2Conv(
                in_channels=in_dim if i == 0 else hidden,
                out_channels=head_dim,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                # Retaining default add_self_loops=True to prevent NaNs on isolated nodes
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden))
            
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for a single week's graph.
        
        Args:
            x (torch.Tensor): Node features [N, in_dim]
            edge_index (torch.Tensor): [2, E]
            edge_weight (torch.Tensor): [E]
            
        Returns:
            h (torch.Tensor): [N, hidden]
            attentions (List[Tuple[edge_index, alpha]]): List of attention tuples from each layer.
        """
        attentions = []
        h = x
        
        # Ensure edge_weight has correct shape for edge_attr [E, 1]
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        
        for i in range(self.layers):
            h_new, attn = self.convs[i](h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            
            h_new = self.norms[i](h_new)
            h_new = torch.relu(h_new)
            
            # Residual connection
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new
                
            attentions.append(attn)
            
        return h, attentions

class BatchedSpatialEncoder(nn.Module):
    """
    Wrapper that applies SpatialEncoder over T weeks of a window (vmap-style loop).
    """
    def __init__(self, encoder: SpatialEncoder):
        super().__init__()
        self.encoder = encoder
        
    def forward(self, x: torch.Tensor, edge_indices: List[torch.Tensor], edge_weights: List[torch.Tensor]) -> Tuple[torch.Tensor, List[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Args:
            x (torch.Tensor): [T, N, in_dim]
            edge_indices: List of length T, each [2, E]
            edge_weights: List of length T, each [E]
            
        Returns:
            h_out (torch.Tensor): [T, N, hidden]
            all_attentions: List of length T, containing layer-wise attentions for each week.
        """
        T = x.size(0)
        h_out = []
        all_attentions = []
        
        for t in range(T):
            h_t, attn_t = self.encoder(x[t], edge_indices[t], edge_weights[t])
            h_out.append(h_t)
            all_attentions.append(attn_t)
            
        return torch.stack(h_out, dim=0), all_attentions
