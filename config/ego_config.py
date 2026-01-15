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
