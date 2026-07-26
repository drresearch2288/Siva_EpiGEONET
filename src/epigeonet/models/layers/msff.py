"""Multi-Scale Feature Fusion (MSFF) layer."""
import torch
import torch.nn as nn
from typing import Optional

class MSFF(nn.Module):
    """
    Maps raw feature vectors to latent embeddings, with support for static feature injection.
    """
    def __init__(self, in_dim: int = 24, hidden: int = 64, static_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        
        self.static_proj = nn.Linear(static_dim, hidden) if static_dim > 0 else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for dynamic features.
        
        Args:
            x (torch.Tensor): Tensor of shape [B, T, N, in_dim] or [T, N, in_dim].
                              
        Returns:
            torch.Tensor: Embedded tensor of shape [..., hidden].
        """
        h = self.proj(x)
        h = self.act(h)
        h = self.norm(h)
        h = self.dropout(h)
        return h

    def broadcast_static(self, h: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        Adds a projected static embedding to every timestep.
        
        Args:
            h (torch.Tensor): Dynamic embeddings, shape [B, T, N, hidden].
            static (torch.Tensor): Static features, shape [N, static_dim].
                                   
        Returns:
            torch.Tensor: Fused embeddings with the same shape as h.
        """
        if self.static_proj is None:
            return h
            
        static_emb = self.static_proj(static) # [N, hidden]
        
        # Reshape to [1, 1, ..., N, hidden] to broadcast over batch/time dims
        view_shape = [1] * (h.dim() - 2) + [static_emb.size(0), static_emb.size(1)]
        static_emb_bcast = static_emb.view(*view_shape)
        
        return h + static_emb_bcast
