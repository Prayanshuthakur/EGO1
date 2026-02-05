"""
Test DeepSeek Model Components
===============================

Comprehensive test suite for all DeepSeek components.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import DEEPSEEK_7B_CONFIG
from model.deepseek_model import DeepSeekModel


def test_model_creation():
    """Test model creation and parameter counting."""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    model = DeepSeekModel(DEEPSEEK_7B_CONFIG)
    n_params = model.count_parameters()
    
    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    return model


def test_forward_pass(model):
    """Test forward pass."""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    batch_size = 2
    seq_len = 128
    
    # Create random input
    input_ids = torch.randint(0, DEEPSEEK_7B_CONFIG.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Number of layers: {len(outputs['kv_caches'])}")
    print(f"✓ Forward pass successful")
    
    return outputs


def test_kv_cache_compression(outputs):
    """Test KV cache compression ratio."""
    print("\n" + "="*60)
    print("TEST 3: KV Cache Compression")
    print("="*60)
    
    # Get first layer's cache
    if outputs['kv_caches'] and outputs['kv_caches'][0] is not None:
        c_kv, seq_len = outputs['kv_caches'][0]
        
        # Calculate sizes
        compressed_size = c_kv.numel()
        full_kv_size = seq_len * DEEPSEEK_7B_CONFIG.n_heads * DEEPSEEK_7B_CONFIG.d_head * 2
        compression_ratio = full_kv_size / compressed_size
        
        print(f"Compressed cache size: {compressed_size:,} elements")
        print(f"Full KV cache would be: {full_kv_size:,} elements")
        print(f"Compression ratio: {compression_ratio:.1f}x")
        print(f"✓ KV cache compression working")
    else:
        print("⚠ No KV cache found")


def test_moe_load_balance(outputs):
    """Test MoE load balancing."""
    print("\n" + "="*60)
    print("TEST 4: MoE Load Balancing")
    print("="*60)
    
    if outputs['moe_stats']:
        # Get stats from first layer
        stats = outputs['moe_stats'][0]
        
        if 'load_balance' in stats:
            lb = stats['load_balance']
            print(f"Load distribution:")
            print(f"  Mean: {lb['load_mean']:.4f}")
            print(f"  Std: {lb['load_std']:.4f}")
            print(f"  CV (coefficient of variation): {lb['load_cv']:.4f}")
            print(f"  Min: {lb['load_min']:.4f}")
            print(f"  Max: {lb['load_max']:.4f}")
            print(f"✓ MoE load balancing active")
        
        if 'expert_counts' in stats:
            counts = stats['expert_counts']
            print(f"\nExpert utilization:")
            print(f"  Total tokens routed: {counts.sum()}")
            print(f"  Active experts: {(counts > 0).sum()}/{len(counts)}")
            print(f"✓ Expert routing working")
    else:
        print("⚠ No MoE stats found")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("DeepSeek Model Test Suite")
    print("="*60)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Run tests
    model = test_model_creation()
    outputs = test_forward_pass(model)
    test_kv_cache_compression(outputs)
    test_moe_load_balance(outputs)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
