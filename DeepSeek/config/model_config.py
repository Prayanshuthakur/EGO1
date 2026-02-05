"""
Model Configuration for DeepSeek
=================================

Defines model architecture hyperparameters following DeepSeek-V3 design.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepSeekConfig:
    """
    Configuration for DeepSeek model architecture.
    
    Key Features:
    - Multi-head Latent Attention (MLA) with low-rank compression
    - Mixture of Experts (MoE) with auxiliary-loss-free load balancing
    - FP8 quantization support
    """
    
    # Model dimensions
    vocab_size: int = 102400  # Vocabulary size
    d_model: int = 2048       # Model dimension (hidden size)
    n_layers: int = 27        # Number of transformer layers
    
    # Multi-head Latent Attention (MLA) parameters
    n_heads: int = 16         # Number of attention heads
    d_head: int = 128         # Dimension per head (d_model / n_heads)
    d_c: int = 512            # Compressed latent dimension (for 32x compression)
    d_rope: int = 64          # Dimension for RoPE component
    
    # MoE parameters
    use_moe: bool = True      # Enable Mixture of Experts
    n_routed_experts: int = 64  # Number of routed experts
    n_shared_experts: int = 2   # Number of shared experts (always active)
    n_activated_experts: int = 6  # Top-K: number of experts to activate per token
    expert_capacity_factor: float = 1.25  # Capacity factor for expert routing
    
    # Router parameters (auxiliary-loss-free)
    router_bias_update_rate: float = 0.01  # Update rate for dynamic bias
    
    # FFN parameters
    d_ff: int = 5632          # FFN intermediate dimension (2.75 * d_model)
    ffn_activation: str = "swiglu"  # Activation function (swiglu or gelu)
    
    # Quantization
    use_fp8: bool = False     # Enable FP8 quantization
    fp8_format: str = "e4m3"  # FP8 format (e4m3 or e5m2)
    fp8_tile_size: int = 128  # Tile size for activation scaling
    fp8_block_size: int = 128 # Block size for weight scaling
    
    # Training parameters
    max_seq_len: int = 4096   # Maximum sequence length
    dropout: float = 0.0      # Dropout probability
    attention_dropout: float = 0.0  # Attention dropout
    
    # Normalization
    norm_eps: float = 1e-6    # Epsilon for RMSNorm
    
    # Initialization
    initializer_range: float = 0.02  # Standard deviation for weight initialization
    
    # Rope
    rope_theta: float = 10000.0  # Base for RoPE frequencies
    rope_scaling: Optional[dict] = None  # RoPE scaling config
    
    # Misc
    tie_word_embeddings: bool = True  # Tie input/output embeddings
    use_cache: bool = True    # Enable KV caching during inference
    
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.d_head == self.d_model // self.n_heads, \
            f"d_head ({self.d_head}) must equal d_model / n_heads ({self.d_model // self.n_heads})"
        assert self.n_activated_experts <= self.n_routed_experts, \
            f"n_activated_experts ({self.n_activated_experts}) must be <= n_routed_experts ({self.n_routed_experts})"
        
        # Calculate compression ratio
        full_kv_dim = self.n_heads * self.d_head * 2  # K + V
        compressed_dim = self.d_c
        self.compression_ratio = full_kv_dim / compressed_dim
        
        print(f"✓ Model config validated")
        print(f"  - KV compression ratio: {self.compression_ratio:.1f}x")
        print(f"  - Total experts: {self.n_routed_experts + self.n_shared_experts}")
        print(f"  - Active experts per token: {self.n_activated_experts + self.n_shared_experts}")


# Predefined configurations
DEEPSEEK_7B_CONFIG = DeepSeekConfig(
    vocab_size=102400,
    d_model=2048,
    n_layers=27,
    n_heads=16,
    d_c=512,
    n_routed_experts=64,
    n_shared_experts=2,
    n_activated_experts=6,
)

DEEPSEEK_16B_CONFIG = DeepSeekConfig(
    vocab_size=102400,
    d_model=2816,
    n_layers=28,
    n_heads=22,
    d_c=512,
    n_routed_experts=128,
    n_shared_experts=2,
    n_activated_experts=6,
)

DEEPSEEK_67B_CONFIG = DeepSeekConfig(
    vocab_size=102400,
    d_model=4096,
    n_layers=60,
    n_heads=32,
    d_c=512,
    n_routed_experts=160,
    n_shared_experts=2,
    n_activated_experts=8,
)
