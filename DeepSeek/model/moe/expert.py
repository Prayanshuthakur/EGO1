"""
Expert Network for Mixture of Experts
======================================

Individual expert network using SwiGLU activation.
Each expert is a standard FFN with gated activation.

SwiGLU: Swish-Gated Linear Unit
Formula: SwiGLU(x) = Swish(W_gate(x)) ⊙ W_up(x)
where Swish(x) = x * sigmoid(x)

SwiGLU has been shown to outperform GELU in LLMs.
Reference: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit activation.
    
    More effective than GELU for large language models.
    Used in modern LLMs (LLaMA, DeepSeek, PaLM).
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation.
        
        Args:
            x: Input tensor, expected to be split into two halves along last dim
            
        Returns:
            Activated tensor with half the last dimension
        """
        # Split into gate and value
        x, gate = x.chunk(2, dim=-1)
        
        # Apply Swish to gate: x * sigmoid(x)
        gate = gate * torch.sigmoid(gate)
        
        # Element-wise multiplication
        return x * gate


class Expert(nn.Module):
    """
    Single expert network with SwiGLU activation.
    
    Architecture:
        Input (d_model) 
        -> Gate projection (d_ff)
        -> Up projection (d_ff)
        -> SwiGLU activation
        -> Down projection (d_model)
        -> Output (d_model)
    
    This is the standard FFN used in each MoE expert.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: Model dimension (input/output size)
            d_ff: FFN intermediate dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Gate and up projections (combined for efficiency)
        # Output is 2 * d_ff to split for SwiGLU
        self.gate_up_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        
        # Down projection back to model dimension
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
        # Activation
        self.activation = SwiGLU()
        
        # Dropout (typically 0 for LLMs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through expert network.
        
        Args:
            x: Input tensor [..., d_model]
            
        Returns:
            Output tensor [..., d_model]
        """
        # Up-project and split for gating
        # [..., d_model] -> [..., 2 * d_ff]
        gate_up = self.gate_up_proj(x)
        
        # Apply SwiGLU activation
        # [..., 2 * d_ff] -> [..., d_ff]
        hidden = self.activation(gate_up)
        
        # Down-project back to model dimension
        # [..., d_ff] -> [..., d_model]
        output = self.down_proj(hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class SharedExpert(nn.Module):
    """
    Shared expert that is always activated.
    
    In DeepSeek MoE, shared experts capture common knowledge that all tokens need,
    while routed experts specialize in specific domains.
    
    This is identical to a regular Expert but conceptually separate.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: FFN intermediate dimension
            dropout: Dropout probability
        """
        super().__init__()
        # Shared expert is just a standard expert
        self.expert = Expert(d_model, d_ff, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared expert.
        
        Args:
            x: Input tensor [..., d_model]
            
        Returns:
            Output tensor [..., d_model]
        """
        return self.expert(x)


if __name__ == "__main__":
    # Test Expert
    d_model = 2048
    d_ff = 5632
    batch_size = 2
    seq_len = 128
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create expert
    expert = Expert(d_model, d_ff)
    
    # Forward pass
    output = expert(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in expert.parameters()):,}")
    print(f"✓ Expert test passed")
    
    # Test shared expert
    shared = SharedExpert(d_model, d_ff)
    output_shared = shared(x)
    print(f"Shared expert output shape: {output_shared.shape}")
    print(f"✓ Shared expert test passed")
