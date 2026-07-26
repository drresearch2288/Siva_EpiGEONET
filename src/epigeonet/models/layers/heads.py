"""Multi-task prediction heads for EpiGeoNet."""
import torch
import torch.nn as nn
from typing import Dict

class RegressionHead(nn.Module):
    """Predicts future case counts for t+1, t+2, t+4."""
    def __init__(self, in_dim: int = 128, horizons: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, horizons)
        )
        self.softplus = nn.Softplus()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return self.softplus(out)

class RiskHead(nn.Module):
    """Predicts risk class logits (Low/Moderate/High/Severe)."""
    def __init__(self, in_dim: int = 128, n_classes: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class AlertHead(nn.Module):
    """Predicts binary onset alert logit."""
    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        # Calibratable threshold attribute
        self.register_buffer('threshold', torch.tensor(0.5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultiTaskHeads(nn.Module):
    """Wraps all three heads."""
    def __init__(self, in_dim: int = 128, horizons: int = 3, n_classes: int = 4):
        super().__init__()
        self.reg_head = RegressionHead(in_dim, horizons)
        self.risk_head = RiskHead(in_dim, n_classes)
        self.alert_head = AlertHead(in_dim)
        
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            'reg': self.reg_head(z),
            'risk_logits': self.risk_head(z),
            'alert_logit': self.alert_head(z)
        }
