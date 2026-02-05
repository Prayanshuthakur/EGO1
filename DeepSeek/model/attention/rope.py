"""
Rotary Position Embeddings (RoPE)
==================================

RoPE encodes positional information by rotating token embeddings in a high-dimensional space.
More efficient than absolute positional embeddings and enables length extrapolation.

Key advantages:
- Relative positional encoding (distance-aware)
- No learned parameters
- Enables length extrapolation beyond training context
- Used in modern LLMs (LLaMA, DeepSeek, GPT-NeoX)

Reference: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
from typing import Tuple


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the frequency tensor for complex exponentials (cis) for RoPE.
    
    RoPE works by rotating pairs of dimensions by an angle that depends on position.
    For position m and dimension pair (2i, 2i+1), the rotation angle is:
        θ_i = m * θ^(-2i/d)
    
    where θ (theta) is typically 10000.
    
    Args:
        dim: Dimension of the embeddings (must be even)
        seq_len: Maximum sequence length
        theta: Base for frequency computation
        device: Device to create tensors on
        
    Returns:
        freqs_cos: Cosine components [seq_len, dim/2]
        freqs_sin: Sine components [seq_len, dim/2]
    """
    assert dim % 2 == 0, f"Dimension must be even, got {dim}"
    
    # Compute frequencies for each dimension pair
    # freqs[i] = 1 / (theta^(2i/dim)) for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    # Create position indices [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device)
    
    # Outer product: [seq_len, 1] x [1, dim/2] = [seq_len, dim/2]
    # freqs[m, i] = m * (1 / theta^(2i/dim))
    freqs = torch.outer(positions, freqs)
    
    # Compute cos and sin for rotation
    freqs_cos = torch.cos(freqs)  # [seq_len, dim/2]
    freqs_sin = torch.sin(freqs)  # [seq_len, dim/2]
    
    return freqs_cos, freqs_sin


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.
    
    For each pair of dimensions (x[..., 2i], x[..., 2i+1]), we apply a 2D rotation:
        x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
        x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
    
    This is equivalent to multiplying by a rotation matrix:
        [cos(θ)  -sin(θ)]
        [sin(θ)   cos(θ)]
    
    Args:
        x: Input tensor [batch, seq_len, n_heads, head_dim]
        freqs_cos: Cosine frequencies [seq_len, head_dim/2]
        freqs_sin: Sine frequencies [seq_len, head_dim/2]
        
    Returns:
        Rotated tensor of same shape as input
    """
    # Reshape x to separate even and odd dimensions
    # x: [batch, seq_len, n_heads, head_dim]
    # -> [batch, seq_len, n_heads, head_dim/2, 2]
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    
    # Extract even and odd dimensions
    x_even = x_reshaped[..., 0]  # [batch, seq_len, n_heads, head_dim/2]
    x_odd = x_reshaped[..., 1]   # [batch, seq_len, n_heads, head_dim/2]
    
    # Broadcast freqs to match x dimensions
    # freqs_cos/sin: [seq_len, head_dim/2] -> [1, seq_len, 1, head_dim/2]
    freqs_cos = freqs_cos[None, :, None, :]
    freqs_sin = freqs_sin[None, :, None, :]
    
    # Apply rotation
    # x'_even = x_even * cos - x_odd * sin
    # x'_odd  = x_even * sin + x_odd * cos
    x_rotated_even = x_even * freqs_cos - x_odd * freqs_sin
    x_rotated_odd = x_even * freqs_sin + x_odd * freqs_cos
    
    # Stack back together and reshape
    x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
    x_rotated = x_rotated.reshape(*x.shape)
    
    return x_rotated.type_as(x)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    
    Precomputes and caches frequency tensors for efficient application.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
        device: str = "cuda"
    ):
        """
        Args:
            dim: Dimension of embeddings (must be even)
            max_seq_len: Maximum sequence length to precompute
            theta: Base for frequency computation
            device: Device to create tensors on
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute and register as buffers (not trained)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim, max_seq_len, theta, device)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Apply rotary embeddings to input.
        
        Args:
            x: Input tensor [batch, seq_len, n_heads, head_dim]
            seq_len: Actual sequence length (may be < max_seq_len)
            
        Returns:
            Rotated tensor of same shape
        """
        # Slice precomputed frequencies to actual sequence length
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]
        
        return apply_rotary_emb(x, freqs_cos, freqs_sin)


if __name__ == "__main__":
    # Test RoPE
    batch_size = 2
    seq_len = 128
    n_heads = 8
    head_dim = 64
    
    # Create test input
    x = torch.randn(batch_size, seq_len, n_heads, head_dim)
    
    # Create RoPE module
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=512)
    
    # Apply RoPE
    x_rotated = rope(x, seq_len)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_rotated.shape}")
    print(f"Norm preserved: {torch.allclose(x.norm(dim=-1), x_rotated.norm(dim=-1), atol=1e-5)}")
    print(f"✓ RoPE test passed")
