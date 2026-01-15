"""
Multi-Head Attention Module
===========================

This module provides the production-ready MultiHeadAttention class for the GPT model.
It implements the optimized multi-head attention with:
- Fused QKV projection (single matrix multiply for Q, K, V)
- Causal masking for autoregressive generation
- Output projection to mix head outputs
- Dropout for regularization

For the learning-oriented implementations, see:
- simplified_attention.py
- self_attention.py
- Causal_Attention.py
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention
    
    This class implements the exact multi-head attention mechanism used in GPT-2.
    Key features:
    1. Parallel attention heads for diverse representation learning
    2. Causal masking prevents attending to future tokens
    3. Scaled dot-product attention for stable training
    4. Output projection mixes information from all heads
    
    Architecture:
        Input (B, T, d_in)
            ↓
        [Linear: d_in -> d_out] × 3 (Q, K, V projections)
            ↓
        Split into num_heads parallel heads
            ↓
        Scaled Dot-Product Attention with Causal Mask
            ↓
        Concatenate heads
            ↓
        [Linear: d_out -> d_out] (output projection)
            ↓
        Output (B, T, d_out)
    
    Args:
        d_in (int): Input embedding dimension.
        d_out (int): Output embedding dimension (must be divisible by num_heads).
        context_length (int): Maximum sequence length for causal mask.
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability.
        qkv_bias (bool): Whether to use bias in Q, K, V projections.
    """
    
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                 num_heads: int, dropout: float, qkv_bias: bool = False):
        super().__init__()
        
        assert d_out % num_heads == 0, \
            f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # Query, Key, Value projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Output projection to mix head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask: upper triangular matrix of ones
        # Will be filled with -inf to mask future tokens
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_in)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_out)
        
        Computation Flow:
            1. Project inputs to Q, K, V vectors
            2. Reshape to separate attention heads
            3. Compute scaled dot-product attention scores
            4. Apply causal mask
            5. Normalize with softmax
            6. Apply dropout
            7. Compute weighted sum of values
            8. Merge heads and project output
        """
        B, T, _ = x.shape
        
        # Step 1: Linear projections
        # Shape: (B, T, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Step 2: Reshape for multi-head attention
        # (B, T, d_out) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        keys = keys.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Step 3: Scaled dot-product attention scores
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        attn_scores = queries @ keys.transpose(-2, -1)
        
        # Step 4: Apply causal mask (set future positions to -inf)
        mask_bool = self.mask.bool()[:T, :T]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # Step 5: Scale and normalize
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        
        # Step 6: Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Step 7: Weighted sum of values
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        context_vec = attn_weights @ values
        
        # Step 8: Merge heads
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, d_out)
        context_vec = context_vec.transpose(1, 2).contiguous().view(B, T, self.d_out)
        
        # Step 9: Output projection
        context_vec = self.out_proj(context_vec)
        
        return context_vec
