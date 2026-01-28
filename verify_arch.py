import torch
import sys
import os

# Ensure we can import from EGO1
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.ego_model import EgoModel
from config.ego_config import EGO_CONFIG_45M

def verify_architecture():
    print("🔍 Verifying EGO SOTA Architecture...")
    
    cfg = EGO_CONFIG_45M
    model = EgoModel(cfg)
    
    # Check parameters
    num_params = model.count_parameters()
    print(f"✅ Parameter Count: {num_params/1e6:.2f}M")
    
    # Test forward pass
    x = torch.randint(0, cfg["vocab_size"], (2, 128))
    logits, loss = model(x, targets=x)
    
    print(f"✅ Forward Pass Successful! Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Check for specific components
    assert hasattr(model, "resid_lambdas"), "Missing resid_lambdas"
    assert hasattr(model, "cos"), "Missing RoPE cos buffer"
    
    # Check layer 0 attention
    attn = model.transformer_blocks[0].self_attention
    assert hasattr(attn, "q_norm"), "Missing QK Norm (q_norm)"
    assert attn.n_kv_head < attn.n_head, "GQA not active (n_kv_head should be < n_head for this config)"
    
    print("🚀 Architecture Verification PASSED!")

if __name__ == "__main__":
    verify_architecture()
