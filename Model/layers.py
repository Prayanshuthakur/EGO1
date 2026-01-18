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
        self.eps = 1e-5 # small constant for numerical stability to avoid division by zero
        # gamma and beta are learnable parameters
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
        """
        here the reason for multiplication with gamma 
        and addition with beta is to allow the model to learn the optimal scaling and shifting for each feature
        """
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) Activation Function
    
    GELU is a smooth, non-linear activation function used in GPT-2 and BERT.
    Unlike ReLU, GELU allows small negative values, which helps with gradient flow.
    
    Formula:
        GELU(x) = x * P(X <= x) where X ~ N(0, 1)
    
    This implementation uses the approximate tanh formula from the GPT-2 paper for efficiency.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the GELU activation function.
        """
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # This approximation is faster to compute than the exact error function (erf)
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """
    Position-wise Feedforward Network (FFN)
    
    This component processes each token independently in parallel. It increases the 
    dimensionality (expansion) to learn complex features and then projects it back.
    
    The standard expansion factor is 4x the embedding dimension.
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        
        # ----------------------------------------------------------------
        # 1. Expansion Layer
        # ----------------------------------------------------------------
        # Projects from emb_dim -> 4 * emb_dim
        # Allows the model to map the input to a higher-dimensional space for feature extraction.
        # Variable: self.expansion_layer (formerly fc1/c_fc)
        self.expansion_layer = nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"])
        
        # ----------------------------------------------------------------
        # 2. Activation Function
        # ----------------------------------------------------------------
        # Introduces non-linearity to the network.
        self.activation = GELU()
        
        # ----------------------------------------------------------------
        # 3. Projection Layer (Contraction)
        # ----------------------------------------------------------------
        # Projects from 4 * emb_dim -> emb_dim
        # Compresses the features back to the model's internal width.
        # Variable: self.loss_projection_layer (formerly fc2/c_proj) -- wait, simply 'output_projection'
        self.output_projection = nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward Network.
        
        Step 1: Expand dimensions (Linear)
        Step 2: Apply non-linearity (GELU)
        Step 3: Project back to original dimensions (Linear)
        """
        # Step 1: Expansion
        x = self.expansion_layer(x)
        
        # Step 2: Activation
        x = self.activation(x)
        
        # Step 3: Projection
        x = self.output_projection(x)
        
        return x
