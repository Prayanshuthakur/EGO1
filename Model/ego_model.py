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
    Ego - A GPT-2 Style Language Model (Production Ready).
    
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
        >>> model = EgoModel(EGO_CONFIG_45M)
        >>> logits = model(input_ids)
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        
        self.cfg = cfg
        # ----------------------------------------------------------------
        # 1. Embeddings (The Input Layer)
        # ----------------------------------------------------------------
        # Token Embeddings: Converts integer token IDs (0-50k) into dense vectors (512-dim).
        # Variable: self.token_embedding
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # Positional Embeddings: Learns a unique vector for each position (0-511)
        # to give the model a sense of order/sequence not provided by self-attention.
        # Variable: self.position_embedding
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # Dropout: Randomly zeros out elements to prevent overfitting during training.
        self.dropout = nn.Dropout(cfg["drop_rate"])
        
        # ----------------------------------------------------------------
        # 2. Key Architecture Components (The "Hidden" Layers)
        # ----------------------------------------------------------------
        # Transformer Blocks: The core sequential processing units.
        # We store them in a ModuleList so PyTorch can track them properly.
        # Variable: self.transformer_blocks (formerly 'h')
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        # ----------------------------------------------------------------
        # 3. Output Stage (The "Head")
        # ----------------------------------------------------------------
        # Final Layer Norm: Stabilizes the activations before the final prediction.
        # Variable: self.final_layer_norm (formerly 'ln_f')
        self.final_layer_norm = LayerNorm(cfg["emb_dim"])
        
        # Language Model Head: Projects the 512-dim vectors back to 50k vocabulary size
        # to predict the probability of the next token.
        # Variable: self.lm_head
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
        # Weight Tying: We reuse the token embedding weights for the output head.
        # This is a standard practice in GPT models to reduce parameters and improve performance.
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights according to GPT-2 best practices
        self.apply(self._init_weights)
        
        # Apply special scaled initialization for residual projections
        # This keeps the variance constant as network depth increases
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('fc2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg["n_layers"]) ** 0.5)

    def _init_weights(self, module):
        """
        Custom weight initialization based on the GPT-2 paper.
        - Linear layers: Normal distribution (mean=0, std=0.02)
        - Embeddings: Normal distribution (mean=0, std=0.02)
        - Biases: All zeros
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """
        The main Forward Pass (Input -> Output) function.
        """
        B, T = idx.shape # Batch size, Sequence Length
        
        # 1. Embeddings: Get the vectors for tokens and positions
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        
        # Combine and apply input dropout
        x = self.dropout(tok_emb + pos_emb)
        
        # 2. Process through all Transformer Blocks sequentially
        for block in self.transformer_blocks:
            x = block(x)
        
        # 3. Final normalization
        x = self.final_layer_norm(x)
        
        # 4. Project to vocabulary (calculate Logits)
        logits = self.lm_head(x)
        
        # 5. Calculate Loss (if training)
        loss = None
        if targets is not None:
            # Flatten the logits and targets to match CrossEntropyLoss expectations
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return (logits, loss) if loss is not None else logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg["context_length"] else idx[:, -self.cfg["context_length"]:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
