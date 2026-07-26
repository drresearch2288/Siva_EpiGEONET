"""Adaptive Spatio-Temporal Climate Graph Convolution (ASCGC)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASCGC(nn.Module):
    """
    Adaptive graph module fusing geographic, distance, climate, and case synchrony edges.
    """
    def __init__(self, n_nodes: int, k: int = 8, init: tuple = (0.4, 0.2, 0.2, 0.2)):
        super().__init__()
        self.n_nodes = n_nodes
        self.k = k
        
        init_tensor = torch.tensor(init, dtype=torch.float32)
        logits = torch.log(init_tensor + 1e-8)
        self.fusion_logits = nn.Parameter(logits)
        
    def fusion_weights(self) -> torch.Tensor:
        """Return the current softmax-normalized fusion weights [alpha, beta, gamma, delta]."""
        return F.softmax(self.fusion_logits, dim=0)
        
    def forward(self, w_clim: torch.Tensor, w_case: torch.Tensor, static: dict) -> tuple:
        """
        Forward pass to compute the adaptive dynamic graph for week t.
        
        Args:
            w_clim (torch.Tensor): [N, N] cosine similarity of 4-week climate covariates.
            w_case (torch.Tensor): [N, N] Pearson correlation of 8-week case history.
            static (dict): Dict containing 'A_geo' [N, N] (indicator) and 'w_dist' [N, N].
                           
        Returns:
            edge_index (torch.Tensor): [2, E] shape, PyG style.
            edge_weight (torch.Tensor): [E] shape, row-normalized weights.
        """
        A_geo = static['A_geo']
        w_dist = static['w_dist']
        
        device = self.fusion_logits.device
        A_geo = A_geo.to(device)
        w_dist = w_dist.to(device)
        w_clim = w_clim.to(device)
        w_case = w_case.to(device)
        
        weights = self.fusion_weights()
        a, b, c, d = weights[0], weights[1], weights[2], weights[3]
        
        A_t = a * A_geo + b * w_dist + c * w_clim + d * w_case
        
        # Sparsify (keep top-k edges per node)
        topk_vals, topk_indices = torch.topk(A_t, self.k, dim=1)
        
        N = self.n_nodes
        row = torch.arange(N, device=device).view(-1, 1).expand(-1, self.k).reshape(-1)
        col = topk_indices.reshape(-1)
        
        edge_index = torch.stack([row, col], dim=0)
        
        # Row-normalize
        row_sum = topk_vals.sum(dim=1, keepdim=True) + 1e-8
        normalized_vals = topk_vals / row_sum
        edge_weight = normalized_vals.reshape(-1)
        
        return edge_index, edge_weight
