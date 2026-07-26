"""Temporal Convolutional Network (TCN) encoder for EpiGeoNet."""
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    """
    Removes the padding on the right to ensure causality.
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x

class TemporalBlock(nn.Module):
    """
    A single causal TCN block: WeightNorm(Conv1d) -> Chomp1d -> GELU -> Dropout -> Residual.
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv = weight_norm(nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        ))
        self.chomp = Chomp1d(padding)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.chomp(out)
        out = self.act(out)
        out = self.dropout(out)
        return x + out

class TemporalEncoder(nn.Module):
    """
    4-layer dilated TCN whose receptive field covers the full 12-week input window.
    """
    def __init__(self, channels: int = 64, layers: int = 4, kernel: int = 3, dilations: tuple = (1, 2, 4, 8), dropout: float = 0.1):
        super().__init__()
        assert len(dilations) == layers
        
        blocks = []
        for d in dilations:
            blocks.append(TemporalBlock(channels, kernel, d, dropout))
            
        self.network = nn.Sequential(*blocks)
        
        rf = 1 + sum((kernel - 1) * d for d in dilations)
        assert rf >= 12, f"Receptive field ({rf}) is less than the required 12 weeks!"
        self.receptive_field = rf
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, N, T, channels]
            
        Returns:
            torch.Tensor: [B, N, T, channels] keeping the time axis.
        """
        B, N, T, C = x.shape
        
        x_reshaped = x.reshape(B * N, T, C).permute(0, 2, 1)
        
        out = self.network(x_reshaped)
        
        out = out.permute(0, 2, 1).reshape(B, N, T, C)
        
        return out
