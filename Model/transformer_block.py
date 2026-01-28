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

from Model.layers import RMSNorm, FeedForward
from AttentionMechanism.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    A single Modern Transformer Block.
    
    Upgrades:
    - Pre-RMSNorm: Faster and more stable than LayerNorm
    - Supported GQA, RoPE and Flash Attention (via MultiHeadAttention)
    """
    
    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-attention RMSNorm
        self.pre_attention_norm = RMSNorm(cfg["emb_dim"])
        
        # Self-Attention
        self.self_attention = MultiHeadAttention(cfg, layer_idx)
        
        # Pre-FFN RMSNorm
        self.pre_ffn_norm = RMSNorm(cfg["emb_dim"])
        
        # Feed-Forward Network
        self.feed_forward = FeedForward(cfg)
    
    def forward(self, x: torch.Tensor, cos_sin: tuple, window_size: tuple = None, kv_cache=None) -> torch.Tensor:
        # 1. Self-Attention with Residual Connection
        x = x + self.self_attention(self.pre_attention_norm(x), cos_sin, window_size, kv_cache)
        
        # 2. Feed-Forward Network with Residual Connection
        x = x + self.feed_forward(self.pre_ffn_norm(x))
        
        return x
