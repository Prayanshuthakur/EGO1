"""
Text Generation Utilities
=========================

This module provides functions for generating text from a trained Ego model.
The main function is `generate_text_simple` which implements greedy decoding.

Greedy Decoding:
    At each step, select the token with the highest probability.
    Simple but deterministic - always produces the same output for the same input.

Future additions could include:
    - Temperature sampling
    - Top-k sampling
    - Top-p (nucleus) sampling
    - Beam search
"""

import torch
import torch.nn as nn


def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int
) -> torch.Tensor:
    """
    Generate text using greedy decoding (select most likely token at each step).
    
    This function implements autoregressive text generation:
    1. Take the current sequence
    2. Get model predictions for the next token
    3. Select the token with highest probability
    4. Append it to the sequence
    5. Repeat until max_new_tokens reached
    
    Args:
        model (nn.Module): A trained Ego model in eval mode.
        idx (torch.Tensor): Starting token IDs of shape (batch, n_tokens).
        max_new_tokens (int): Number of new tokens to generate.
        context_size (int): Maximum context window of the model.
    
    Returns:
        torch.Tensor: Extended token sequence of shape (batch, n_tokens + max_new_tokens)
    
    Example:
        >>> # Assuming 'model' is a trained EgoModel and 'tokenizer' is initialized
        >>> start_tokens = tokenizer.encode("Hello, I am")
        >>> idx = torch.tensor(start_tokens).unsqueeze(0)  # Add batch dimension
        >>> generated = generate_text_simple(model, idx, max_new_tokens=20, context_size=1024)
        >>> output_text = tokenizer.decode(generated[0].tolist())
    
    Note:
        - The model should be in eval mode (model.eval()) before calling this function.
        - For more diverse outputs, consider using temperature sampling or top-k/top-p.
    """
    # Ensure we're not computing gradients during generation
    model.eval()
    
    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's maximum context size
        # E.g., if context_size=5 and current sequence has 10 tokens,
        # we only use the last 5 tokens as input
        idx_cond = idx[:, -context_size:]
        
        # Get model predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on the last token's predictions
        # logits shape: (batch, n_tokens, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]
        
        # Apply softmax to get probability distribution
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        
        # Greedy selection: pick token with highest probability
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        
        # Append the new token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens + 1)
    
    return idx


def generate_text_with_temperature(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: int = None
) -> torch.Tensor:
    """
    Generate text with temperature scaling and optional top-k sampling.
    
    Temperature controls randomness:
    - temperature < 1.0: More deterministic (sharper distribution)
    - temperature = 1.0: Standard probability distribution
    - temperature > 1.0: More random (flatter distribution)
    
    Top-k limits sampling to the k most likely tokens.
    
    Args:
        model (nn.Module): A trained Ego model in eval mode.
        idx (torch.Tensor): Starting token IDs of shape (batch, n_tokens).
        max_new_tokens (int): Number of new tokens to generate.
        context_size (int): Maximum context window of the model.
        temperature (float): Temperature for scaling logits. Default 1.0.
        top_k (int, optional): If set, sample from top-k most likely tokens.
    
    Returns:
        torch.Tensor: Extended token sequence of shape (batch, n_tokens + max_new_tokens)
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        
        # Apply temperature scaling
        logits = logits / temperature
        
        # Apply top-k filtering if specified
        if top_k is not None:
            # Get the top-k values and indices
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            # Get the minimum value among top-k
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            # Set all values below min_top_k to -inf
            logits = torch.where(
                logits < min_top_k,
                torch.full_like(logits, float('-inf')),
                logits
            )
        
        # Convert to probabilities
        probas = torch.softmax(logits, dim=-1)
        
        # Sample from the distribution
        idx_next = torch.multinomial(probas, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
