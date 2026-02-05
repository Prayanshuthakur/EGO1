"""
RMSNorm (Root Mean Square Layer Normalization)
===============================================

More efficient than LayerNorm - no mean subtraction, only RMS normalization.
Used in modern LLMs (LLaMA, DeepSeek, etc.)

Formula: y = (x / RMS(x)) * weight
where RMS(x) = sqrt(mean(x^2) + eps)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More computationally efficient than LayerNorm as it doesn't compute mean.
    Only normalizes by the root mean square.
    
    Args:
        dim (int): Dimension of the input
        eps (float): Small constant for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (gamma)
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Normalized tensor of same shape as input
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        # keepdim=True preserves the dimension for broadcasting
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        # x / rms normalizes to unit RMS
        # * self.weight applies learnable scaling
        x_norm = x / rms
        return self.weight * x_norm
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


if __name__ == "__main__":
    # Test RMSNorm
    batch_size, seq_len, dim = 2, 10, 512
    x = torch.randn(batch_size, seq_len, dim)
    
    norm = RMSNorm(dim)
    y = norm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    print(f"Output mean: {y.mean():.4f}, std: {y.std():.4f}")
    print(f"✓ RMSNorm test passed")
