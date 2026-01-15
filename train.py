#!/usr/bin/env python3
"""
Ego Model Training Demo
=======================

This is the main entry point for the EGO1 LLM training pipeline.
It demonstrates:
1. Loading configuration
2. Initializing the tokenizer
3. Creating a data loader
4. Instantiating the Ego model
5. Running a forward pass
6. Generating sample text (with random weights)

Usage:
    python train.py

Note:
    This script initializes a model with random weights.
    The generated text will be gibberish until the model is trained.
"""

import torch
import tiktoken
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Project imports
from config.ego_config import EGO_CONFIG_124M
from Model.ego_model import EgoModel
from utils.text_generation import generate_text_simple
from Tokenizer.EgoDatasetLoader import create_dataloader


def main():
    EGO_CONFIG_124M={
        "vocab_size": 50257,
        "context_length": 128,
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 2,
        "max_length": 128,
        "batch_size": 1,
        "drop_rate": 0.1,
        "qkv_bias":False
    }
    """Main training demonstration."""
    
    print("=" * 60)
    print("EGO1 - Building Ego Model from Scratch")
    print("=" * 60)
    print()
    
    # =========================================
    # Step 1: Configuration
    # =========================================
    print("[1/6] Loading configuration...")
    print(f"  Model: Ego-124M")
    print(f"  Vocab size: {EGO_CONFIG_124M['vocab_size']:,}")
    print(f"  Context length: {EGO_CONFIG_124M['context_length']}")
    print(f"  Embedding dim: {EGO_CONFIG_124M['emb_dim']}")
    print(f"  Layers: {EGO_CONFIG_124M['n_layers']}")
    print(f"  Attention heads: {EGO_CONFIG_124M['n_heads']}")
    print()
    
    # =========================================
    # Step 2: Tokenizer
    # =========================================
    print("[2/6] Initializing tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"  Tokenizer: GPT-2 BPE")
    print(f"  Vocabulary size: {tokenizer.n_vocab:,}")
    print()
    
    # =========================================
    # Step 3: Sample Data
    # =========================================
    print("[3/6] Creating sample data...")
    
    # Load sample text (using the test dataset if available)
    sample_text_path = os.path.join(os.path.dirname(__file__), "test-datasets.txt")
    if os.path.exists(sample_text_path):
        with open(sample_text_path, "r", encoding="utf-8") as f:
            sample_text = f.read()
        print(f"  Loaded: test-dataset.txt")
        print(f"  Characters: {len(sample_text):,}")
    else:
        sample_text = "Your journey starts with one step. " * 10
        print("  Using default sample text")
    # Create dataloader
    dataloader = create_dataloader(
        sample_text,
        batch_size=2,
        max_length=4,
        stride=4,
        shuffle=False,
        tokenizer=tokenizer
    )
    print(f"  Batches: {len(dataloader)}")
    print()
    
    # =========================================
    # Step 4: Model Initialization
    # =========================================
    print("[4/6] Initializing Ego model...")
    torch.manual_seed(123)
    model = EgoModel(EGO_CONFIG_124M)
    print(f"  Total parameters: {model.count_parameters():,}")
    print(f"  Memory footprint: {model.get_memory_footprint_mb():.2f} MB")
    print()
    
    # =========================================
    # Step 5: Forward Pass Demo
    # =========================================
    print("[5/6] Running forward pass...")
    
    # Get a batch from the dataloader
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    
    print(f"  Output logits shape: {logits.shape}")
    print()
    
    # =========================================
    # Step 6: Text Generation Demo
    # =========================================
    print("[6/6] Generating sample text...")
    
    # Encode a prompt
    prompt = "Hello, I am"
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Token IDs: {encoded}")
    
    # Generate
    generated = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=100,
        context_size=EGO_CONFIG_124M["context_length"]
    )
    
    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"  Generated: '{generated_text}'")
    print()
    
    # =========================================
    # Summary
    # =========================================
    print("=" * 60)
    print("✅ All components working correctly!")
    print()
    print("Note: Generated text is random because the model is untrained.")
    print("Training will be implemented in future updates.")
    print("=" * 60)


if __name__ == "__main__":
    main()
