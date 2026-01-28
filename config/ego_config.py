"""
Ego Model Configuration
=======================

This module contains configuration dictionaries for various Ego model sizes.
The configurations define the hyperparameters used to instantiate an EgoModel.
"""

# Ego-124M parameter configuration (Modern SOTA)
EGO_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 2048,   # Increased context
    "emb_dim": 768,
    "n_heads": 12,
    "n_kv_heads": 4,          # GQA
    "n_layers": 12,
    "drop_rate": 0.0,
    "activation": "relu2",    # Modern activation
    "qkv_bias": False
}

# Ego-45M parameter configuration (Fast Production)
EGO_CONFIG_45M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 512,
    "n_heads": 8,
    "n_kv_heads": 2,          # GQA
    "n_layers": 8,
    "drop_rate": 0.0,
    "activation": "relu2",
    "qkv_bias": False
}
