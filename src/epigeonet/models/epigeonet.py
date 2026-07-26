"""EpiGeoNet complete architecture."""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from epigeonet.models.layers.msff import MSFF
from epigeonet.graph.ascgc import ASCGC
from epigeonet.models.layers.spatial_gatv2 import SpatialEncoder, BatchedSpatialEncoder
from epigeonet.models.layers.temporal_tcn import TemporalEncoder
from epigeonet.models.layers.stca import STCA
from epigeonet.models.layers.heads import MultiTaskHeads
from epigeonet.utils.device import get_device

class EpiGeoNet(nn.Module):
    """
    End-to-End Explainable Spatio-Temporal Graph Attention Network.
    """
    def __init__(self, 
                 n_nodes: int, 
                 in_dim: int = 24, 
                 hidden: int = 64,
                 static_dim: int = 1,
                 ascgc_k: int = 8,
                 gat_heads: int = 4,
                 gat_layers: int = 2,
                 tcn_layers: int = 4,
                 tcn_kernel: int = 3,
                 tcn_dilations: tuple = (1, 2, 4, 8),
                 stca_heads: int = 4,
                 out_dim: int = 128,
                 reg_horizons: int = 3,
                 risk_classes: int = 4,
                 explain: bool = True):
        super().__init__()
        
        self.explain = explain
        self.n_nodes = n_nodes
        
        # 1. MSFF
        self.msff = MSFF(in_dim=in_dim, hidden=hidden, static_dim=static_dim)
        
        # 2. ASCGC Graph
        self.ascgc = ASCGC(n_nodes=n_nodes, k=ascgc_k)
        
        # 3. Spatial Encoder (GATv2)
        base_spatial = SpatialEncoder(in_dim=hidden, hidden=hidden, heads=gat_heads, layers=gat_layers)
        self.spatial = BatchedSpatialEncoder(base_spatial)
        
        # 4. Temporal Encoder (TCN)
        self.temporal = TemporalEncoder(channels=hidden, layers=tcn_layers, kernel=tcn_kernel, dilations=tcn_dilations)
        
        # 5. STCA
        self.stca = STCA(dim=hidden, heads=stca_heads, out=out_dim)
        
        # 6. Heads
        self.heads = MultiTaskHeads(in_dim=out_dim, horizons=reg_horizons, n_classes=risk_classes)
        
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """Respect MPS backend when moving to device."""
        device = kwargs.get('device')
        if not device and len(args) > 0:
            device = args[0]
            
        super().to(*args, **kwargs)
        if device is not None:
            self.ascgc.fusion_logits.data = self.ascgc.fusion_logits.data.to(device)
            if self.ascgc.fusion_logits.grad is not None:
                self.ascgc.fusion_logits.grad.data = self.ascgc.fusion_logits.grad.data.to(device)
        return self

    def forward(self, batch: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Forward pass.
        
        batch keys:
            - 'x': [B, T, N, 24] dynamic features
            - 'static': [N, static_dim]
            - 'w_clim': [B, T, N, N]
            - 'w_case': [B, T, N, N]
            - 'graph_static': dict with 'A_geo' and 'w_dist' [N, N]
        """
        x = batch['x']
        static = batch.get('static')
        w_clim = batch['w_clim']
        w_case = batch['w_case']
        graph_static = batch['graph_static']
        
        B, T, N, _ = x.shape
        
        # 1. MSFF
        h = self.msff(x) # [B, T, N, hidden]
        if static is not None:
            h = self.msff.broadcast_static(h, static)
            
        # 2 & 3. ASCGC -> Spatial Encoder
        spatial_out = []
        spatial_attns = []
        
        for b in range(B):
            b_edge_indices = []
            b_edge_weights = []
            for t in range(T):
                edge_index, edge_weight = self.ascgc(w_clim[b, t], w_case[b, t], graph_static)
                b_edge_indices.append(edge_index)
                b_edge_weights.append(edge_weight)
                
            h_b_spatial, attn_b = self.spatial(h[b], b_edge_indices, b_edge_weights)
            spatial_out.append(h_b_spatial)
            if self.explain:
                spatial_attns.append(attn_b)
                
        h_spatial = torch.stack(spatial_out, dim=0) # [B, T, N, hidden]
        
        # 4. Temporal Encoder (TCN expects [B, N, T, C])
        h_temp_in = h_spatial.permute(0, 2, 1, 3) # [B, N, T, hidden]
        h_temporal = self.temporal(h_temp_in)     # [B, N, T, hidden]
        
        # 5. STCA
        spatial_in = h_spatial.permute(0, 2, 1, 3).reshape(B*N, T, -1)
        temporal_in = h_temporal.reshape(B*N, T, -1)
        
        z, temporal_attn = self.stca(spatial_in, temporal_in) # z: [B*N, 128]
        
        # 6. Heads
        preds = self.heads(z)
        
        # Reshape preds to [B, N, ...]
        reshaped_preds = {}
        for k, v in preds.items():
            reshaped_preds[k] = v.view(B, N, -1)
            
        explanations = None
        if self.explain:
            explanations = {
                'spatial_attn': spatial_attns,
                'temporal_attn': temporal_attn.view(B, N, T),
                'fusion_weights': self.ascgc.fusion_weights()
            }
            
        return reshaped_preds, explanations
