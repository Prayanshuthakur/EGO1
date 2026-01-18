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
        
        # ----------------------------------------------------------------
        # 1. Attention Sub-Block
        # ----------------------------------------------------------------
        # Layer Norm 1: Applied BEFORE attention (Pre-Norm architecture)
        # We normalize the input features to stabilize training.
        # Variable: self.pre_attention_norm
        self.pre_attention_norm = LayerNorm(cfg["emb_dim"])
        
        # Self-Attention: Mixing information between tokens.
        # This allows the model to capture dependencies between distant words.
        # Variable: self.self_attention
        self.self_attention = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        
        # ----------------------------------------------------------------
        # 2. Feed-Forward Sub-Block
        # ----------------------------------------------------------------
        # Layer Norm 2: Applied BEFORE the Feed Forward Network
        # Normalizes the output of the attention block + residual.
        # Variable: self.pre_ffn_norm
        self.pre_ffn_norm = LayerNorm(cfg["emb_dim"])
        
        # Feed-Forward Network: Processing information independently per token.
        # This layer extracts higher-level features from the attended information.
        # Variable: self.feed_forward
        self.feed_forward = FeedForward(cfg)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processing Flow of a Transformer Block:
        
        1. Attention Path:
           Input -> Norm -> Attention -> Add to Input (Residual)
           
        2. Feed-Forward Path:
           Resid Output -> Norm -> FeedForward -> Add to Previous (Residual)
        """
        # ----------------------------------------------------------------
        # 1. Self-Attention with Residual Connection
        # ----------------------------------------------------------------
        # x = x + Attention(Norm(x))
        residual = x
        normalized_x = self.pre_attention_norm(x)
        attention_output = self.self_attention(normalized_x)
        x = residual + attention_output
        
        # ----------------------------------------------------------------
        # 2. Feed-Forward Network with Residual Connection
        # ----------------------------------------------------------------
        # x = x + FeedForward(Norm(x))
        residual = x # we also call it as shortcut connection
        normalized_x = self.pre_ffn_norm(x)
        ffn_output = self.feed_forward(normalized_x)
        x = residual + ffn_output # we also call it as residual connection
        
        return x
