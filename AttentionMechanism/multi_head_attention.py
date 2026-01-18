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
        
        # ----------------------------------------------------------------
        # 1. QKV Projection Layer (Fused)
        # ----------------------------------------------------------------
        # Projects input to Query, Key, and Value vectors simultaneously.
        # Shape: d_in -> 3 * d_out
        # Variable: self.qkv_projection (formerly c_attn)
        self.qkv_projection = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        
        # ----------------------------------------------------------------
        # 2. Output Projection Layer
        # ----------------------------------------------------------------
        # Projects the concatenated head outputs back to the original dimension.
        # Shape: d_out -> d_out
        # Variable: self.output_projection (formerly c_proj)
        self.output_projection = nn.Linear(d_out, d_out)
        
        # ----------------------------------------------------------------
        # 3. Regularization
        # ----------------------------------------------------------------
        # Dropout applied to attention scores (probabilities)
        self.attention_dropout = nn.Dropout(dropout)
        # Dropout applied to the final output (residual path)
        self.residual_dropout = nn.Dropout(dropout)
        
        # ----------------------------------------------------------------
        # 4. Causal Mask
        # ----------------------------------------------------------------
        # Upper triangular matrix used to mask future tokens during self-attention.
        # This ensures the model is autoregressive (can't see the future).
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention (Fused Implementation).
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_in)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_out)
        
        Computation Flow:
            1. Project inputs to Q, K, V vectors (Fused for speed)
            2. Split and Reshape to separate attention heads
            3. Compute scaled dot-product attention scores
            4. Apply causal mask (block future tokens)
            5. Normalize with softmax
            6. Apply dropout
            7. Compute weighted sum of values (Attention)
            8. Merge heads and project output
        """
        B, T, C = x.shape
        
        # ----------------------------------------------------------------
        # Step 1: Fused Linear Projection
        # ----------------------------------------------------------------
        # Project input to combined Q, K, V
        # Shape: (B, T, 3 * d_out)
        qkv = self.qkv_projection(x)
        
        # Split into Query, Key, Value
        q, k, v = qkv.split(self.d_out, dim=2)
        
        # ----------------------------------------------------------------
        # Step 2: Reshape for multi-head attention
        # ----------------------------------------------------------------
        # Transform from (Batch, Seq, Dim) to (Batch, Heads, Seq, Head_Dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # ----------------------------------------------------------------
        # Step 3: Scaled dot-product attention scores
        # ----------------------------------------------------------------
        # Dot product of Queries and Keys
        # Scale by 1/sqrt(head_dim) to keep gradients stable
        # Shape: (B, H, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # ----------------------------------------------------------------
        # Step 4: Apply causal mask
        # ----------------------------------------------------------------
        # Fill future positions with negative infinity so softmax makes them zero
        mask_bool = self.mask.bool()[:T, :T]
        attn_scores.masked_fill_(mask_bool, -float('inf'))
        
        # ----------------------------------------------------------------
        # Step 5: Softmax and dropout
        # ----------------------------------------------------------------
        # Convert scores to probabilities
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # Apply dropout to the attention weights
        attn_weights = self.attention_dropout(attn_weights)
        
        # ----------------------------------------------------------------
        # Step 6: Aggregate values
        # ----------------------------------------------------------------
        # Weighted sum of Value vectors
        # Shape: (B, H, T, Head_Dim)
        y = attn_weights @ v 
        
        # ----------------------------------------------------------------
        # Step 7: Merge heads
        # ----------------------------------------------------------------
        # Concatenate all heads back together
        # Shape: (B, T, H, Head_Dim) -> (B, T, d_out)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # ----------------------------------------------------------------
        # Step 8: Output projection and residual dropout
        # ----------------------------------------------------------------
        # Project back to embedding dimension and apply dropout
        y = self.residual_dropout(self.output_projection(y))
        
        return y


