"""Spatio-Temporal Cross-Attention (STCA) for EpiGeoNet."""
import torch
import torch.nn as nn
from typing import Tuple

class STCA(nn.Module):
    """
    Lightweight multi-head cross-attention block fusing spatial and temporal embeddings.
    
    Query: temporal summary (e.g., the final timestep of the TCN embedding)
    Key/Value: spatial embeddings across the T weeks
    
    This yields attention weights over the input window, serving as temporal
    explainability (lag-saliency) indicating which past weeks matter most.
    """
    def __init__(self, dim: int = 64, heads: int = 4, out: int = 128):
        super().__init__()
        self.dim = dim
        
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, out),
            nn.GELU(),
            nn.Linear(out, out)
        )
        
    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spatial (torch.Tensor): [N, T, dim] 
            temporal (torch.Tensor): [N, T, dim]
            
        Returns:
            z (torch.Tensor): [N, out] Spatio-temporal representation.
            temporal_attn (torch.Tensor): [N, T] Attention weights over the input window.
        """
        # Query: The final timestep of the temporal sequence summarizes the history up to now
        query = temporal[:, -1:, :]  # [N, 1, dim]
        
        # Key/Value: The spatial embeddings across all T weeks
        key = spatial   # [N, T, dim]
        value = spatial # [N, T, dim]
        
        # Multihead Cross-Attention
        attn_output, attn_weights = self.mha(query, key, value)
        
        # Residual + Norm
        x = self.norm1(query + attn_output)
        
        # Squeeze the time dimension which is now 1
        x = x.squeeze(1) # [N, dim]
        x = self.norm2(x)
        
        # Map to 128-D
        z = self.mlp(x) # [N, out]
        
        # Return attention weights [N, T] for explainability
        temporal_attn = attn_weights.squeeze(1)
        
        return z, temporal_attn
