import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.layers import apply_rotary_emb, RMSNorm
from Model.flash_attention import flash_attn

class MultiHeadAttention(nn.Module):
    """
    Modern Multi-Head Attention (inspired by nanochat/Llama).
    
    Features:
    - Rotary Positional Embeddings (RoPE)
    - Group-Query Attention (GQA) support
    - QK Normalization to prevent attention sinks
    - Flash Attention (with SDPA fallback)
    - Bias-free linear layers
    """
    
    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = cfg["n_heads"]
        self.n_kv_head = cfg.get("n_kv_heads", self.n_head) # Default to MHA
        self.emb_dim = cfg["emb_dim"]
        self.head_dim = self.emb_dim // self.n_head
        
        assert self.emb_dim % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        
        # Projections
        self.c_q = nn.Linear(self.emb_dim, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.emb_dim, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.emb_dim, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        
        # QK Norm (prevents attention instabilities)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        self.dropout = nn.Dropout(cfg.get("drop_rate", 0.0))

    def forward(self, x, cos_sin, window_size=None, kv_cache=None):
        B, T, C = x.size()
        
        # 1. Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # 2. Apply RoPE
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # 3. Apply QK Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # 4. Flash Attention (or SDPA fallback)
        if kv_cache is None:
            # Training or non-cached inference
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference with KV cache
            # Note: EGO1 might need a KVCache manager similar to nanochat's
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)
                
        # 5. Final Output Projection
        y = y.contiguous().view(B, T, C)
        y = self.dropout(self.c_proj(y))
        
        return y
