"""
Ego Model Implementation
========================

This module contains the complete Ego (Generative Pre-trained Transformer style) model.
The EgoModel class combines all components into a unified architecture:

1. Token Embeddings - Converts token IDs to dense vectors
2. Positional Embeddings - Adds position information
3. Transformer Blocks - Stack of attention + feedforward layers
4. Final LayerNorm - Stabilizes final representations
5. Output Head - Projects to vocabulary logits

This implementation follows the GPT-2 architecture:
- Pre-LayerNorm in transformer blocks
- GELU activation in feedforward networks
- Causal (autoregressive) attention masking
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.layers import LayerNorm
from Model.transformer_block import TransformerBlock


class EgoModel(nn.Module):
    """
    Ego Model
    
    A decoder-only transformer for autoregressive language modeling.
    Given a sequence of token IDs, the model predicts the probability
    distribution over the vocabulary for the next token at each position.
    
    Architecture Overview:
        Token IDs (B, T)
            ↓
        Token Embedding (vocab_size -> emb_dim)
            +
        Positional Embedding (context_length -> emb_dim)
            ↓
        Dropout
            ↓
        N × TransformerBlock
            ↓
        Final LayerNorm
            ↓
        Linear Output Head (emb_dim -> vocab_size)
            ↓
        Logits (B, T, vocab_size)
    
    Args:
        cfg (dict): Configuration dictionary containing:
            - vocab_size (int): Size of the vocabulary
            - context_length (int): Maximum sequence length
            - emb_dim (int): Embedding dimension
            - n_layers (int): Number of transformer blocks
            - n_heads (int): Number of attention heads per block
            - drop_rate (float): Dropout probability
            - qkv_bias (bool): Whether to use bias in attention
    
    Example:
        >>> cfg = {
        ...     "vocab_size": 50257, "context_length": 1024,
        ...     "emb_dim": 768, "n_layers": 12, "n_heads": 12,
        ...     "drop_rate": 0.1, "qkv_bias": False
        ... }
        >>> model = EgoModel(cfg)
        >>> input_ids = torch.randint(0, 50257, (2, 512))
        >>> logits = model(input_ids)
        >>> print(logits.shape)  # torch.Size([2, 512, 50257])
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        
        # --------------------------------------------------
        # Embedding Layers
        # --------------------------------------------------
        # Token embedding: maps vocabulary indices to vectors
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # Positional embedding: encodes position in sequence
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # Dropout applied after combining embeddings
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # --------------------------------------------------
        # Transformer Blocks
        # --------------------------------------------------
        # Stack of N transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # --------------------------------------------------
        # Output Layers
        # --------------------------------------------------
        # Final layer normalization
        self.final_norm = LayerNorm(cfg["emb_dim"])
        
        # Output projection to vocabulary (no bias as per GPT-2)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Ego model.
        
        Args:
            in_idx (torch.Tensor): Input token IDs of shape (batch_size, seq_len)
                                   Values should be in range [0, vocab_size)
        
        Returns:
            torch.Tensor: Logits over vocabulary of shape (batch_size, seq_len, vocab_size)
                         Apply softmax for probabilities, argmax for predictions.
        
        Processing Steps:
            1. Look up token embeddings from embedding table
            2. Add positional embeddings (absolute position encoding)
            3. Apply dropout for regularization
            4. Pass through stack of transformer blocks
            5. Apply final layer normalization
            6. Project to vocabulary size for next-token prediction
        """
        batch_size, seq_len = in_idx.shape
        
        # Step 1: Token embeddings
        # (B, T) -> (B, T, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        
        # Step 2: Positional embeddings
        # Create position indices [0, 1, 2, ..., seq_len-1]
        # and look up their embeddings
        pos_indices = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_indices)  # (T, emb_dim)
        
        # Step 3: Combine embeddings
        # pos_embeds broadcasts across batch dimension
        x = tok_embeds + pos_embeds  # (B, T, emb_dim)
        x = self.drop_emb(x)
        
        # Step 4: Transformer blocks
        x = self.trf_blocks(x)  # (B, T, emb_dim)
        
        # Step 5: Final normalization
        x = self.final_norm(x)  # (B, T, emb_dim)
        
        # Step 6: Output projection
        logits = self.out_head(x)  # (B, T, vocab_size)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_footprint_mb(self) -> float:
        """Estimate memory footprint in MB (assuming float32)."""
        return self.count_parameters() * 4 / (1024 * 1024)
