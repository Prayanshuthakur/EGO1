"""
=============================================================================
EGO MODEL - PRODUCTION TRAINING SCRIPT FOR LIGHTNING.AI / KAGGLE
=============================================================================

This is a SINGLE FILE, SELF-CONTAINED training script for the Ego LLM.
Clone this file alone to Lightning.ai or Kaggle and run it directly.

Features (Industry Best Practices):
1. All model components inline (no external imports needed)
2. Automatic GPU detection (CUDA, MPS, CPU fallback)
3. Gradient clipping to prevent exploding gradients
4. Cosine LR scheduler with linear warmup
5. Full checkpoint saving (model + optimizer + metadata)
6. Periodic text generation to monitor quality
7. Perplexity logging alongside loss
8. Temperature scaling for text generation
9. Top-k sampling for diverse outputs
10. BPE tokenization via tiktoken

Usage:
    python train_lightning.py

Author: EgoAI Project
"""

import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# For BPE tokenization (install via: pip install tiktoken)
import tiktoken


# =============================================================================
# CONFIGURATION
# =============================================================================

EGO_CONFIG = {
    # Model Architecture
    "vocab_size": 50257,      # GPT-2 BPE tokenizer vocabulary size
    "context_length": 256,    # Max sequence length (reduced for demo)
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads  
    "n_layers": 12,           # Number of transformer layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False,        # No bias in QKV projections (GPT-2 style)
    
    # Training Hyperparameters
    "batch_size": 2,          # Batch size (increase if you have more GPU memory)
    "num_epochs":1,         # Number of training epochs (for real use 10)
    "learning_rate": 5e-4,    # Peak learning rate
    "weight_decay": 0.1,      # AdamW weight decay
    "max_grad_norm": 1.0,     # Gradient clipping threshold
    "warmup_steps": 100,      # LR warmup steps
    
    # Data
    "stride": 128,            # Sliding window stride for data creation
    "train_ratio": 0.9,       # Train/val split ratio
    
    # Logging & Saving
    "eval_freq": 5,           # Evaluate every N steps
    "save_freq": 50,          # Save checkpoint every N steps
    "generate_freq": 25,      # Generate sample text every N steps
}


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization with learnable scale and shift."""
    
    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation (GPT-2 style).
    
    FIXED: Uses math.sqrt instead of torch.tensor to avoid device mismatch.
    """
    
    def __init__(self):
        super().__init__()
        # Pre-compute constant as Python float (device-agnostic)
        self.sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            self.sqrt_2_over_pi * (x + 0.044715 * x.pow(3))
        ))


class FeedForward(nn.Module):
    """Position-wise FeedForward Network with 4x expansion."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Causal Self-Attention (GPT-2 style)."""
    
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 num_heads: int, dropout: float, qkv_bias: bool = False):
        super().__init__()
        
        assert d_out % num_heads == 0, f"d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        keys = keys.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(-2, -1)
        mask_bool = self.mask.bool()[:T, :T]
        attn_scores.masked_fill_(mask_bool, float('-inf'))
        
        attn_weights = F.softmax(attn_scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = attn_weights @ values
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_out)
        
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-LayerNorm."""
    
    def __init__(self, cfg: dict):
        super().__init__()
        
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.drop(self.att(self.norm1(x)))
        x = x + shortcut
        
        shortcut = x
        x = self.drop(self.ff(self.norm2(x)))
        x = x + shortcut
        
        return x


class EgoModel(nn.Module):
    """
    Ego - A GPT-2 Style Language Model.
    
    Implements the exact GPT-2 architecture with proper weight initialization
    as described in the original OpenAI paper.
    
    Key GPT-2 Features:
    - Pre-LayerNorm (applied before attention and FFN, not after)
    - GELU activation in feedforward layers
    - Learned absolute positional embeddings
    - Scaled weight initialization for residual connections
    - No bias in QKV projections
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
        # =====================================================================
        # GPT-2 WEIGHT INITIALIZATION (Critical for stable training!)
        # =====================================================================
        # This is THE key difference between amateur and production LLMs.
        # Without proper init, training can diverge or converge very slowly.
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections (GPT-2 paper)
        # Scale by 1/sqrt(2*n_layers) to prevent signal explosion
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('layers.2.weight'):
                # out_proj = attention output, layers.2 = second FFN linear
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layers"]))
        
        print(f"Ego Model initialized with {self.count_parameters():,} parameters")
    
    def _init_weights(self, module):
        """
        GPT-2 style weight initialization.
        
        - Linear layers: Normal(0, 0.02)
        - Embeddings: Normal(0, 0.02)  
        - LayerNorm: scale=1, shift=0 (already default)
        - Biases: Zero
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop_emb(tok_emb + pos_emb)
        
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())



# =============================================================================
# DATASET
# =============================================================================

class TextDataset(Dataset):
    """Text dataset with sliding window for next-token prediction."""
    
    def __init__(self, text: str, tokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []
        
        # Encode text to token IDs
        token_ids = tokenizer.encode(text)
        
        # Create input-target pairs with sliding window
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloaders(text: str, tokenizer, cfg: dict):
    """Create train and validation dataloaders."""
    
    # Split text
    split_idx = int(cfg["train_ratio"] * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, cfg["context_length"], cfg["stride"])
    val_dataset = TextDataset(val_text, tokenizer, cfg["context_length"], cfg["stride"])
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        drop_last=False
    )
    
    return train_loader, val_loader


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    """Compute cross-entropy loss for a batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Compute average loss over a data loader."""
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    
    model.eval()
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    
    return total_loss / num_batches


def get_lr(step, cfg, total_steps):
    """
    Cosine learning rate schedule with linear warmup.
    
    This is the standard schedule used by GPT-3, LLaMA, and most modern LLMs.
    """
    warmup_steps = cfg["warmup_steps"]
    max_lr = cfg["learning_rate"]
    min_lr = max_lr * 0.1  # Decay to 10% of max
    
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # Cosine decay
    if step >= total_steps:
        return min_lr
    
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# TEXT GENERATION
# =============================================================================

def generate_text(model, tokenizer, prompt, device, max_new_tokens=50, 
                  temperature=1.0, top_k=50):
    """
    Generate text with temperature scaling and top-k sampling.
    
    Args:
        model: The EgoModel
        tokenizer: tiktoken tokenizer
        prompt: Starting text
        device: torch device
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Sample from top-k most likely tokens
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    context_length = model.cfg["context_length"]
    
    for _ in range(max_new_tokens):
        # Crop to context length
        idx_cond = idx[:, -context_length:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Get last token logits and apply temperature
        logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k, dim=-1)
            min_val = top_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, 
                                 torch.full_like(logits, float('-inf')), 
                                 logits)
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        idx = torch.cat((idx, idx_next), dim=1)
    
    return tokenizer.decode(idx[0].tolist())


# =============================================================================
# CHECKPOINTING
# =============================================================================

def save_checkpoint(model, optimizer, step, epoch, train_loss, val_loss, cfg, path):
    """Save full checkpoint for resuming training."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": cfg,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train():
    """Main training loop with all best practices."""
    
    print("=" * 70)
    print("EGO MODEL - PRODUCTION TRAINING")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. SETUP DEVICE
    # -------------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
    
    # -------------------------------------------------------------------------
    # 2. LOAD DATA
    # -------------------------------------------------------------------------
    # Try to load dataset
    from datasets import load_dataset

    print("Loading small FineWeb subset for testing...")

    raw_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="CC-MAIN-2023-50",
        split="train",
        streaming=True,
    )

    MAX_SAMPLES = 2000 # 5000(for real training)   # small local test

    def text_stream():
        count = 0
        for example in raw_dataset:
            if count >= MAX_SAMPLES:
                break
            yield example["text"]
            count += 1

    text_data = "\n".join(list(text_stream()))
    print(f"Loaded {len(text_data):,} characters from FineWeb subset.")
    
    # -------------------------------------------------------------------------
    # 3. SETUP TOKENIZER AND DATALOADERS
    # -------------------------------------------------------------------------
    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader = create_dataloaders(text_data, tokenizer, EGO_CONFIG)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # -------------------------------------------------------------------------
    # 4. INITIALIZE MODEL
    # -------------------------------------------------------------------------
    model = EgoModel(EGO_CONFIG).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # -------------------------------------------------------------------------
    # 5. SETUP OPTIMIZER (no scheduler - we manually set LR)
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=EGO_CONFIG["learning_rate"],
        weight_decay=EGO_CONFIG["weight_decay"],
        betas=(0.9, 0.95),  # GPT-3 style betas
        eps=1e-8
    )
    
    # -------------------------------------------------------------------------
    # 6. TRAINING LOOP
    # -------------------------------------------------------------------------
    total_steps = len(train_loader) * EGO_CONFIG["num_epochs"]
    global_step = 0
    best_val_loss = float("inf")
    
    print(f"\nStarting training for {EGO_CONFIG['num_epochs']} epochs ({total_steps} steps)")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(EGO_CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            
            # Update learning rate (cosine schedule with warmup)
            lr = get_lr(global_step, EGO_CONFIG, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            # Forward pass
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (CRITICAL for stable training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), EGO_CONFIG["max_grad_norm"])
            
            # Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # ---- LOGGING ----
            if global_step % EGO_CONFIG["eval_freq"] == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
                
                # Perplexity = exp(loss)
                train_ppl = math.exp(train_loss) if train_loss < 10 else float("inf")
                val_ppl = math.exp(val_loss) if val_loss < 10 else float("inf")
                
                print(f"Ep {epoch+1}/{EGO_CONFIG['num_epochs']} | "
                      f"Step {global_step}/{total_steps} | "
                      f"LR {lr:.2e} | "
                      f"Train Loss {train_loss:.4f} (PPL {train_ppl:.1f}) | "
                      f"Val Loss {val_loss:.4f} (PPL {val_ppl:.1f})")
                
                model.train()
            
            # ---- TEXT GENERATION SAMPLE ----
            if global_step % EGO_CONFIG["generate_freq"] == 0:
                sample = generate_text(model, tokenizer, "Every effort moves you", 
                                       device, max_new_tokens=30, temperature=0.8, top_k=40)
                print(f"  Sample: {sample[:100]}...")
                model.train()
            
            # ---- CHECKPOINTING ----
            if global_step % EGO_CONFIG["save_freq"] == 0:
                val_loss = calc_loss_loader(val_loader, model, device)
                save_checkpoint(model, optimizer, global_step, epoch, 
                               epoch_loss / (batch_idx + 1), val_loss, EGO_CONFIG,
                               f"ego_checkpoint_step{global_step}.pt")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, global_step, epoch,
                                   epoch_loss / (batch_idx + 1), val_loss, EGO_CONFIG,
                                   "ego_best_model.pt")
                
                model.train()
    
    # -------------------------------------------------------------------------
    # 7. FINAL SAVE
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training complete in {elapsed/60:.1f} minutes")
    
    # Save final model (weights only for inference)
    torch.save(model.state_dict(), "ego_model_final.pt")
    print("Final model saved: ego_model_final.pt")
    
    # Save full checkpoint for resuming
    final_val_loss = calc_loss_loader(val_loader, model, device)
    save_checkpoint(model, optimizer, global_step, epoch, 
                   epoch_loss / len(train_loader), final_val_loss, EGO_CONFIG,
                   "ego_checkpoint_final.pt")
    
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    train()
