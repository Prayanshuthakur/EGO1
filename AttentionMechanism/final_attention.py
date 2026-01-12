import torch
import torch.nn as nn
import math

class EngineAttention(nn.Module):
    """
    Engine-grade causal multi-head self-attention.

    This module implements the exact computation used inside GPT-class LLMs,
    optimized for GPU memory locality, kernel fusion, and flash-attention compatibility.

    Input shape  : (B, T, d_in)
    Output shape : (B, T, d_out)
    """

    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # Number of parallel attention heads
        self.num_heads = num_heads

        # Dimensionality per head (D)
        self.head_dim = d_out // num_heads

        # Scaling factor used in dot-product attention
        # Prevents softmax saturation and stabilizes gradients
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # ------------------------------------------------------------------
        # Fused QKV projection:
        # Instead of three separate projections (Wq, Wk, Wv),
        # we use a single linear layer producing [Q | K | V] in one kernel.
        # This reduces memory reads, kernel launches, and improves throughput.
        # ------------------------------------------------------------------
        self.qkv_proj = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)

        # Output projection that mixes information across heads
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # Precomputed additive causal bias:
        # Lower-triangular mask converted into large negative values
        # so future positions receive ~0 probability after softmax.
        # This avoids boolean masking and is flash-attention compatible.
        # ------------------------------------------------------------------
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("causal_bias", (1.0 - mask) * -1e4)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input embeddings of shape (B, T, d_in)

        Returns:
            Tensor: Contextualized embeddings of shape (B, T, d_out)
        """
        B, T, _ = x.shape

        # -------------------------------------------------------------
        # Step 1: Fused QKV Projection
        # Projects inputs into concatenated [Q | K | V] vectors.
        # Result shape: (B, T, 3 * d_out)
        # -------------------------------------------------------------
        qkv = self.qkv_proj(x)

        # -------------------------------------------------------------
        # Step 2: Reshape to separate Q, K, V and attention heads
        # (B, T, 3 * d_out) -> (B, T, 3, H, D)
        # -------------------------------------------------------------
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)

        # -------------------------------------------------------------
        # Step 3: Permute dimensions for attention computation
        # (B, T, 3, H, D) -> (3, B, H, T, D)
        # This layout groups heads and tokens optimally for GPU kernels.
        # -------------------------------------------------------------
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Split into Query, Key, and Value tensors
        q, k, v = qkv.unbind(dim=0)

        # -------------------------------------------------------------
        # Step 4: Scaled Dot-Product Attention Scores
        # Computes similarity between all token pairs per head.
        # Result shape: (B, H, T, T)
        # -------------------------------------------------------------
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # scaled dot product

        # -------------------------------------------------------------
        # Step 5: Apply Causal Bias
        # Adds large negative values to future-token positions so that
        # softmax assigns them zero probability.
        # -------------------------------------------------------------
        attn_scores = attn_scores + self.causal_bias[:T, :T]

        # -------------------------------------------------------------
        # Step 6: Softmax Normalization
        # Converts raw attention scores into probability distributions.
        # -------------------------------------------------------------
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        # -------------------------------------------------------------
        # Step 7: Weighted Aggregation of Value Vectors
        # Produces context vectors per head.
        # Result shape: (B, H, T, D)
        # -------------------------------------------------------------
        out = attn @ v

        # -------------------------------------------------------------
        # Step 8: Merge Attention Heads
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, d_out)
        # -------------------------------------------------------------
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        # -------------------------------------------------------------
        # Step 9: Output Projection
        # Mixes information across heads and applies residual dropout.
        # -------------------------------------------------------------
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out

inputs=torch.tensor(
    [
        [0.43, 0.15, 0.89],        # you
        [0.55, 0.87, 0.66],        # journey
        [0.57, 0.85, 0.64]        # starts
    ]
)
d_in=inputs.shape[1]
d_out=6
context_length=3
num_heads=2
batch=torch.stack([inputs,inputs],dim=0)
dropout=0.0
engine_attention=EngineAttention(d_in,d_out,context_length,num_heads,dropout)
context_vector=engine_attention(batch)
print("your context vector is ",context_vector)
