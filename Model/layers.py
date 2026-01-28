"""
Core Neural Network Layers for Ego
===================================

This module contains the fundamental building blocks used in the Ego architecture:

1. LayerNorm - Layer Normalization for stabilizing training
2. GELU - Gaussian Error Linear Unit activation function
3. FeedForward - Position-wise feedforward network with expansion factor

These layers are combined in TransformerBlock to form the core of the Ego model.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    A simpler and more efficient alternative to LayerNorm that only scales
    activations based on their root mean square. Used in Llama and Gemma.
    
    Formula:
        y = x / sqrt(mean(x^2) + eps) * scale
    """
    
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm is highly optimized in newer PyTorch versions
        return torch.nn.functional.rms_norm(x, (x.size(-1),), weight=self.scale, eps=self.eps)


class ReLU2(nn.Module):
    """
    ReLU^2 Activation Function
    
    A simple activation function (ReLU followed by squaring) that has shown 
    good results in recent high-performance language models (e.g., nanochat).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x).square()


def apply_rotary_emb(x, cos, sin):
    """
    Apply Rotary Positional Embeddings (RoPE) to queries or keys.
    x shape: (B, T, H, D) or (B, H, T, D)
    cos/sin shape: (1, T, 1, D/2) or matching x
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class FeedForward(nn.Module):
    """
    Position-wise Feedforward Network (FFN)
    
    Updated to support configurable activation functions (GELU or ReLU2).
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        
        self.expansion_layer = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"], bias=False)
        
        # Select activation function based on config
        act_type = cfg.get("activation", "gelu").lower()
        if act_type == "relu2":
            self.activation = ReLU2()
        else:
            self.activation = nn.GELU(approximate='tanh')
            
        self.output_projection = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"], bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expansion_layer(x)
        x = self.activation(x)
        x = self.output_projection(x)
        return x
