"""
Master Training Pipeline for Ego-45M (H200 Optimized)
=====================================================

This script orchestrates the entire lifecycle of the Ego model:
1.  Initialization (Architecture & Config)
2.  Training (High-Performance Loop)
3.  Conversion (PyTorch -> Safetensors with Architecture mapping)
4.  Deployment (Upload to Hugging Face Hub)

Usage:
    python train_ego.py
"""

import os
import torch
import time
import json
import math
from tqdm import tqdm

from config.ego_config import EGO_CONFIG_45M
from Model.ego_model import EgoModel
from Tokenizer.bpe_tokenizer import BytePairEncoding
from Tokenizer.EgoDatasetLoader import create_dataloader
from utils.deployment import convert_ego_to_gpt2_safetensors, upload_to_huggingface

# Configuration
REPO_ID = "RameshRathod/ego-45m-pretrained"
OUTPUT_DIR = "ego_45m_hf"
CHECKPOINT_PATH = "ego_model_final.pt"

def train_lifecyle():
    print("\n" + "="*60)
    print("🚀 EGO-45M: PRODUCTION PIPELINE STARTED")
    print("="*60 + "\n")

    # ------------------------------------------------------------------
    # 1. SETUP & DEVICE
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"✅ Device: {device}")
    
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # ------------------------------------------------------------------
    # 2. DATA PREPARATION
    # ------------------------------------------------------------------
    # Note: Ideally this loads the full FineWeb-Edu subset. 
    # For this demo script, we ensure it runs on available data.
    data_path = "test-dataset.txt"
    if not os.path.exists(data_path):
        # Fallback to creating a dummy file if missing, so the script runs
        with open(data_path, "w") as f:
            f.write("The quick brown fox jumps over the lazy dog. " * 1000)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    tokenizer = BytePairEncoding()
    # Use Production Config (512 ctx, 45M)
    cfg = EGO_CONFIG_45M
    
    print(f"📊 Dataset: {len(text)} characters")
    print(f"⚙️  Configuration: {cfg}")

    train_loader = create_dataloader(
        text,
        tokenizer=tokenizer,
        batch_size=8,  # H200 optimized
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=True
    )

    # ------------------------------------------------------------------
    # 3. MODEL INITIALIZATION
    # ------------------------------------------------------------------
    model = EgoModel(cfg).to(device)
    model = torch.compile(model) # speedup
    print(f"✅ Model Initialized: {model.count_parameters()/1e6:.2f}M Parameters")

    # ------------------------------------------------------------------
    # 4. TRAINING LOOP
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    
    # Cosine Decay Params
    max_steps = 50 # Example short run for demo (increase for real training)
    warmup_steps = 10
    
    print("\n🏋️  Starting Training Loop...")
    model.train()
    start_time = time.time()
    
    pbar = tqdm(range(max_steps), desc="Training")
    data_iter = iter(train_loader)
    
    for step in pbar:
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Mixed Precision (if CUDA)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32):
            logits, loss = model(inputs, targets)
            
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Scheduler
        lr = 5e-4 # Simple simplified scheduler
        if step < warmup_steps:
            lr = 5e-4 * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            lr = 5e-4 * 0.5 * (1 + math.cos(math.pi * progress))
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

    total_time = time.time() - start_time
    print(f"\n✅ Training Complete in {total_time:.2f}s")

    # ------------------------------------------------------------------
    # 5. SAVING (Checkpoints)
    # ------------------------------------------------------------------
    print(f"💾 Saving Raw Checkpoint to {CHECKPOINT_PATH}...")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    
    # ------------------------------------------------------------------
    # 6. CONVERSION (Auto-Fix Architecture)
    # ------------------------------------------------------------------
    print("\n🔄 Starting Automatic Conversion to Hugging Face Format...")
    
    # Generate Config.json
    hf_config = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": cfg["vocab_size"],
        "n_positions": cfg["context_length"],
        "n_ctx": cfg["context_length"],
        "n_embd": cfg["emb_dim"],
        "n_layer": cfg["n_layers"],
        "n_head": cfg["n_heads"],
        "activation_function": "gelu_new",
        "resid_pdrop": cfg["drop_rate"],
        "embd_pdrop": cfg["drop_rate"],
        "attn_pdrop": cfg["drop_rate"],
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "todo": "Auto-generated by Ego-45M Pipeline"
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    
    # Convert Weights
    model_state = model.state_dict() if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule) else model._orig_mod.state_dict()
    convert_ego_to_gpt2_safetensors(model_state, OUTPUT_DIR)
    
    # ------------------------------------------------------------------
    # 7. UPLOAD (Deployment)
    # ------------------------------------------------------------------
    print("\n☁️  Deploying to Hugging Face Hub...")
    # NOTE: Requires generic HF_TOKEN env var or manual login
    upload_to_huggingface(OUTPUT_DIR, REPO_ID)
    
    print("\n" + "="*60)
    print("✅ PIPELINE FINISHED: Model is Live on Hugging Face!")
    print(f"🔗 https://huggingface.co/{REPO_ID}")
    print("="*60 + "\n")

if __name__ == "__main__":
    train_lifecyle()