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


class LayerNorm(nn.Module):
    """
    Layer Normalization
    
    Normalizes activations across the embedding dimension (last dimension).
    Unlike BatchNorm, LayerNorm normalizes each sample independently,
    making it suitable for variable-length sequences in NLP.
    
    Formula:
        y = (x - mean) / sqrt(var + eps) * scale + shift
    
    Args:
        emb_dim (int): The embedding dimension to normalize over.
    
    Attributes:
        eps (float): Small constant for numerical stability.
        scale (nn.Parameter): Learnable scaling factor (gamma).
        shift (nn.Parameter): Learnable shifting factor (beta).
    """
    
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., emb_dim)
        
        Returns:
            torch.Tensor: Normalized tensor of same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) Activation Function
    
    GELU is a smooth, non-linear activation function used in GPT-2 and BERT.
    Unlike ReLU, GELU allows small negative values, which helps with gradient flow.
    
    This implementation uses the approximate formula from the GPT-2 paper:
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    Properties:
        - Smooth (differentiable everywhere)
        - Non-monotonic for negative values
        - Approximates ReLU for large positive values
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.
        
        Args:
            x (torch.Tensor): Input tensor of any shape.
        
        Returns:
            torch.Tensor: Activated tensor of same shape as input.
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Position-wise Feedforward Network
    
    A two-layer MLP with GELU activation, applied independently to each position.
    The hidden dimension is typically 4x the embedding dimension (expansion factor).
    
    Architecture:
        Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)
    
    This layer enables the model to learn complex feature transformations
    after the attention mechanism has aggregated contextual information.
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - emb_dim (int): The embedding dimension.
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feedforward transformation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, emb_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, emb_dim)
        """
        return self.layers(x)
