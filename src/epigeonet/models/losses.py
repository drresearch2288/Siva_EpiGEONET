"""Multi-task training objectives for EpiGeoNet."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

def huber_reg(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss for regression over 3 horizons."""
    return F.huber_loss(pred, target, delta=delta, reduction='mean')

def weighted_ce(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor = None) -> torch.Tensor:
    """Weighted Cross Entropy for risk classes."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), weight=class_weights, reduction='mean')

def focal_alert(logit: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    """Focal loss for binary outbreak-onset positive class."""
    bce_loss = F.binary_cross_entropy_with_logits(logit.view(-1), target.float().view(-1), reduction='none')
    pt = torch.exp(-bce_loss)
    
    target_flat = target.float().view(-1)
    alpha_t = alpha * target_flat + (1 - alpha) * (1 - target_flat)
    
    focal_loss = alpha_t * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

def laplacian_smoothness(risk_probs: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """
    Graph-Laplacian smoothness: sum w_ij * ||p_i - p_j||^2.
    risk_probs: [B, N, C]
    edge_index: [2, E]
    edge_weight: [E]
    """
    src, dst = edge_index[0], edge_index[1]
    
    diff = risk_probs[:, src, :] - risk_probs[:, dst, :]
    sq_diff = (diff ** 2).sum(dim=-1) # [B, E]
    
    loss = (edge_weight * sq_diff).sum(dim=-1) # [B]
    return loss.mean()

def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """
    Mean entropy of attention distributions (regulariser for sparsity).
    attn: [..., T] probabilities
    """
    eps = 1e-8
    entropy = -torch.sum(attn * torch.log(attn + eps), dim=-1)
    return entropy.mean()

def total_loss(outputs: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor], 
               graph: Dict[str, torch.Tensor], 
               weights: Dict[str, float] = None,
               class_weights: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes total composite loss.
    L = L_reg + l1*L_cls + l2*L_alert + l3*L_spatial + l4*L_sparse
    """
    if weights is None:
        weights = {'l1': 0.5, 'l2': 0.4, 'l3': 0.05, 'l4': 0.05}
        
    l_reg = huber_reg(outputs['reg'], targets['reg'])
    l_cls = weighted_ce(outputs['risk_logits'], targets['risk'], class_weights)
    l_alert = focal_alert(outputs['alert_logit'], targets['alert'])
    
    risk_probs = F.softmax(outputs['risk_logits'], dim=-1)
    l_spatial = laplacian_smoothness(risk_probs, graph['edge_index'], graph['edge_weight'])
    
    if 'temporal_attn' in outputs:
        l_sparse = attention_entropy(outputs['temporal_attn'])
    else:
        l_sparse = torch.tensor(0.0, device=l_reg.device)
        
    loss = l_reg + weights['l1'] * l_cls + weights['l2'] * l_alert + weights['l3'] * l_spatial + weights['l4'] * l_sparse
    
    components = {
        'loss_total': loss.item(),
        'loss_reg': l_reg.item(),
        'loss_cls': l_cls.item(),
        'loss_alert': l_alert.item(),
        'loss_spatial': l_spatial.item(),
        'loss_sparse': l_sparse.item()
    }
    
    return loss, components
