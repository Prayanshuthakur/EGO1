"""
Transformer Block Module
========================

The TransformerBlock is the core repeating unit of the Ego architecture.
Each block applies:
1. Multi-Head Self-Attention (with residual connection)
2. Position-wise FeedForward network (with residual connection)

This implementation uses Pre-LayerNorm (applying LayerNorm before each sub-layer),
which provides better training dynamics compared to the original Post-LayerNorm.

Architecture:
    Input (B, T, emb_dim)
        ↓
    [LayerNorm] → [MultiHeadAttention] → [Dropout] → (+Input)
        ↓
    [LayerNorm] → [FeedForward] → [Dropout] → (+Previous Output)
        ↓
    Output (B, T, emb_dim)
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.layers import LayerNorm, FeedForward
from AttentionMechanism.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    A single Transformer Block (Ego-style).
    
    This block implements the standard transformer decoder architecture with:
    - Pre-LayerNorm: Normalizes input before each sub-layer (not after)
    - Residual Connections: Adds input to output of each sub-layer
    - Dropout: Applied after attention and feedforward for regularization
    
    The Pre-LayerNorm design is used in GPT-2, GPT-3, and most modern LLMs
    because it leads to more stable training compared to Post-LayerNorm.
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - emb_dim (int): Embedding dimension
            - context_length (int): Maximum sequence length
            - n_heads (int): Number of attention heads
            - drop_rate (float): Dropout probability
            - qkv_bias (bool): Whether to use bias in attention projections
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        # Position-wise FeedForward Network
        self.ff = FeedForward(cfg)
        
        # Layer Normalization (Pre-LayerNorm style)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        
        # Dropout for residual connections
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, emb_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, emb_dim)
        
        Processing Flow:
            1. Store input for residual connection
            2. Apply LayerNorm → Attention → Dropout
            3. Add residual (input + output)
            4. Store for second residual
            5. Apply LayerNorm → FeedForward → Dropout
            6. Add residual (previous + output)
        """
        # ============================================
        # Attention Sub-Layer with Residual Connection
        # ============================================
        shortcut = x
        x = self.norm1(x)                    # Pre-LayerNorm
        x = self.att(x)                      # Multi-Head Attention
        x = self.drop_shortcut(x)            # Dropout
        x = x + shortcut                     # Residual Connection
        
        # ============================================
        # FeedForward Sub-Layer with Residual Connection
        # ============================================
        shortcut = x
        x = self.norm2(x)                    # Pre-LayerNorm
        x = self.ff(x)                       # FeedForward
        x = self.drop_shortcut(x)            # Dropout
        x = x + shortcut                     # Residual Connection
        
        return x
