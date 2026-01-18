"""
Ego Model Configuration
=======================

This module contains configuration dictionaries for various Ego model sizes.
The configurations define the hyperparameters used to instantiate an EgoModel.
"""

# Ego-124M parameter configuration
EGO_CONFIG_124M = {
    "vocab_size": 50257,      # BPE tokenizer vocabulary size
    "context_length": 1024,   # Maximum context window
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of transformer layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False         # No bias in QKV projections (GPT-2 style)
}

# Ego-45M parameter configuration (Production / H200 Optimized)
EGO_CONFIG_45M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.0,
    "qkv_bias": False
}
