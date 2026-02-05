"""
DeepSeek Model - Main Model Class
==================================

Complete DeepSeek transformer with:
- Multi-head Latent Attention (MLA)
- Mixture of Experts (MoE)
- RMSNorm
- RoPE

This is the main model class that combines all components.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .attention.mla import MultiHeadLatentAttention
from .attention.rope import precompute_freqs_cis
from .moe.moe_layer import MoELayer
from .layers.rms_norm import RMSNorm


class DeepSeekTransformerBlock(nn.Module):
    """
    Single transformer block with MLA and MoE.
    
    Architecture:
        Input
        -> RMSNorm -> MLA -> Residual
        -> RMSNorm -> MoE -> Residual
        -> Output
    """
    
    def __init__(self, config, layer_idx: int):
        """
        Args:
            config: Model configuration
            layer_idx: Layer index (for debugging/logging)
        """
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-normalization
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # Multi-head Latent Attention
        self.attn = MultiHeadLatentAttention(config)
        
        # Mixture of Experts (or standard FFN if MoE disabled)
        if config.use_moe:
            self.ffn = MoELayer(config)
        else:
            # Standard FFN fallback (not implemented here)
            raise NotImplementedError("Standard FFN not implemented - use MoE")
    
    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        use_cache: bool = False,
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[dict]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input [batch, seq_len, d_model]
            freqs_cos: RoPE cosine frequencies
            freqs_sin: RoPE sine frequencies
            mask: Attention mask
            kv_cache: Cached KV from previous forward
            use_cache: Whether to return cache
            training: Training mode
            
        Returns:
            output: Block output [batch, seq_len, d_model]
            new_cache: Updated KV cache
            moe_stats: MoE statistics
        """
        # === ATTENTION BLOCK ===
        # Pre-norm
        h = self.attn_norm(x)
        
        # MLA with residual
        attn_out, new_cache = self.attn(
            h, freqs_cos, freqs_sin,
            mask=mask,
            kv_cache=kv_cache,
            use_cache=use_cache
        )
        x = x + attn_out
        
        # === FFN BLOCK (MoE) ===
        # Pre-norm
        h = self.ffn_norm(x)
        
        # MoE with residual
        ffn_out, moe_stats = self.ffn(h, training=training)
        x = x + ffn_out
        
        return x, new_cache, moe_stats


class DeepSeekModel(nn.Module):
    """
    Complete DeepSeek Language Model.
    
    Architecture:
        Token Embeddings
        -> N x Transformer Blocks (MLA + MoE)
        -> RMSNorm
        -> LM Head
    """
    
    def __init__(self, config):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        print(f"\n{'='*60}")
        print(f"Initializing DeepSeek Model")
        print(f"{'='*60}")
        print(f"Vocab size: {config.vocab_size}")
        print(f"Model dim: {config.d_model}")
        print(f"Layers: {config.n_layers}")
        print(f"Heads: {config.n_heads}")
        print(f"Max seq len: {config.max_seq_len}")
        
        # === EMBEDDINGS ===
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # === TRANSFORMER BLOCKS ===
        self.layers = nn.ModuleList([
            DeepSeekTransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # === FINAL NORM ===
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # === LM HEAD ===
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights (share embeddings with output)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        
        # === ROPE FREQUENCIES ===
        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.d_rope,
            seq_len=config.max_seq_len,
            theta=config.rope_theta,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"\nTotal parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"{'='*60}\n")
    
    def _init_weights(self, module):
        """Initialize weights following standard practice."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple]] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            kv_caches: List of KV caches for each layer
            use_cache: Whether to return caches
            return_dict: Whether to return dict or tuple
            
        Returns:
            Dictionary with:
                - logits: Output logits [batch, seq_len, vocab_size]
                - kv_caches: Updated caches (if use_cache)
                - moe_stats: MoE statistics
        """
        batch_size, seq_len = input_ids.shape
        
        # === EMBEDDINGS ===
        x = self.embed(input_ids)  # [batch, seq_len, d_model]
        
        # === PREPARE CACHES ===
        if kv_caches is None and use_cache:
            kv_caches = [None] * self.config.n_layers
        
        new_kv_caches = [] if use_cache else None
        all_moe_stats = []
        
        # === TRANSFORMER BLOCKS ===
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            
            x, new_cache, moe_stats = layer(
                x,
                freqs_cos=self.freqs_cos,
                freqs_sin=self.freqs_sin,
                mask=attention_mask,
                kv_cache=kv_cache,
                use_cache=use_cache,
                training=self.training
            )
            
            if use_cache:
                new_kv_caches.append(new_cache)
            if moe_stats is not None:
                all_moe_stats.append(moe_stats)
        
        # === FINAL NORM ===
        x = self.norm(x)
        
        # === LM HEAD ===
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        # === RETURN ===
        if return_dict:
            return {
                "logits": logits,
                "kv_caches": new_kv_caches,
                "moe_stats": all_moe_stats
            }
        else:
            return logits, new_kv_caches, all_moe_stats
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test DeepSeek Model
    from config.model_config import DEEPSEEK_7B_CONFIG
    
    # Create model
    model = DeepSeekModel(DEEPSEEK_7B_CONFIG)
    
    # Create test input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, DEEPSEEK_7B_CONFIG.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids, use_cache=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Number of caches: {len(outputs['kv_caches'])}")
    print(f"Number of MoE stats: {len(outputs['moe_stats'])}")
    print(f"✓ DeepSeek Model test passed")
