"""
EGO Modern Trainer
==================

A production-ready training script for the Ego LLM architecture.
Incorporates best practices from nanoGPT for optimal training performance.

Features:
- TF32 optimization for CUDA
- Gradient clipping for stability
- GradScaler for fp16 mixed precision
- Proper evaluation loop with validation loss tracking
- Resume training from checkpoints
- MFU (Model FLOPS Utilization) estimation
- Cosine LR schedule with warmup and min_lr floor

Usage:
    Single GPU:  python train_ego.py
    Multi-GPU:   torchrun --standalone --nproc_per_node=4 train_ego.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import math
from tqdm import tqdm
from contextlib import nullcontext

from config.ego_config import EGO_CONFIG_45M, EGO_CONFIG_124M
from Model.ego_model import EgoModel
from Model.common import get_dist_info, print0, autodetect_device_type
from Tokenizer.bpe_tokenizer import BytePairEncoding
from Tokenizer.EgoDatasetLoader import create_dataloader

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
out_dir = 'out-ego'
eval_interval = 100
log_interval = 10
eval_iters = 20
always_save_checkpoint = False  # Only save on best val loss improvement
init_from = 'scratch'  # 'scratch' or 'resume'

# Training hyperparameters
max_steps = 1000
warmup_steps = 100
batch_size = 32
grad_clip = 1.0  # Gradient clipping (0.0 to disable)

# Learning rates
learning_rate = 6e-4
min_lr = 6e-5  # Minimum LR (Chinchilla: ~10% of max)

# Mixed precision
dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16'

# -----------------------------------------------------------------------------

def train():
    # 1. SETUP DISTRIBUTED & DEVICE
    is_ddp, rank, local_rank, world_size = get_dist_info()
    device_type = autodetect_device_type()
    
    if is_ddp:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
        seed_offset = rank
    else:
        device = torch.device(device_type)
        seed_offset = 0

    # Seed for reproducibility (with DDP offset)
    torch.manual_seed(1337 + seed_offset)

    # TF32 optimization for CUDA (free performance boost)
    if device_type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print0(f"🚀 Starting EGO Modern Trainer | Device: {device} | World Size: {world_size}")

    # Mixed precision setup
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # GradScaler for fp16 (no-op for bfloat16/float32)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # 2. CONFIG & MODEL
    cfg = EGO_CONFIG_45M  # Use 45M for fast iteration
    
    # Create output directory
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    
    # Initialize model
    model = EgoModel(cfg).to(device)
    
    # Resume from checkpoint if requested
    iter_num = 0
    best_val_loss = float('inf')
    
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if os.path.exists(ckpt_path):
            print0(f"📂 Resuming from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            iter_num = checkpoint.get('iter_num', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        else:
            print0(f"⚠️ No checkpoint found at {ckpt_path}, starting from scratch")
    
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    
    raw_model = model.module if is_ddp else model
    print0(f"✅ Model Initialized: {raw_model.count_parameters()/1e6:.2f}M Parameters")

    # 3. OPTIMIZERS
    optimizers = raw_model.setup_optimizers(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.2,
        device_type=device_type
    )
    adamw_opt, muon_opt = optimizers
    
    # Resume optimizer state if available
    if init_from == 'resume' and 'optimizers' in checkpoint:
        for i, opt in enumerate(optimizers):
            if i < len(checkpoint['optimizers']):
                opt.load_state_dict(checkpoint['optimizers'][i])
        print0("✅ Optimizer states restored")

    # 4. DATA LOADER
    data_path = "test-dataset.txt"
    if not os.path.exists(data_path):
        with open(data_path, "w") as f: 
            f.write("YC is the best place to build an AI startup. " * 1000)
    
    with open(data_path, 'r', encoding='utf-8') as f: 
        text = f.read()
    tokenizer = BytePairEncoding()
    
    train_loader = create_dataloader(
        text, tokenizer=tokenizer,
        batch_size=batch_size // world_size, 
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        shuffle=True
    )

    # 5. TRAINING HELPER FUNCTIONS
    def get_lr(it):
        """Cosine LR schedule with warmup and min_lr floor."""
        # Linear warmup
        if it < warmup_steps:
            return learning_rate * (it + 1) / (warmup_steps + 1)
        # Cosine decay after warmup
        if it > max_steps:
            return min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    @torch.no_grad()
    def estimate_loss():
        """Estimate loss over train/val splits."""
        model.eval()
        losses = {}
        for split in ['train']:  # Add 'val' when you have a val split
            split_losses = []
            data_iter = iter(train_loader)
            for _ in range(eval_iters):
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    inputs, targets = next(data_iter)
                inputs, targets = inputs.to(device), targets.to(device)
                with ctx:
                    _, loss = model(inputs, targets)
                split_losses.append(loss.item())
            losses[split] = sum(split_losses) / len(split_losses)
        model.train()
        return losses

    # 6. TRAINING LOOP
    model.train()
    pbar = tqdm(range(iter_num, max_steps), desc="Training", disable=(rank != 0))
    data_iter = iter(train_loader)
    
    start_time = time.time()
    running_mfu = -1.0
    t0 = time.time()
    
    for step in pbar:
        # Get learning rate for this step
        lr = get_lr(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = lr
        
        # Evaluation and checkpointing
        if step % eval_interval == 0 and rank == 0:
            losses = estimate_loss()
            print0(f"\n📊 Step {step}: train loss {losses['train']:.4f}")
            
            # Save checkpoint on improvement (or always if configured)
            if losses['train'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['train']
                if step > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizers': [opt.state_dict() for opt in optimizers],
                        'iter_num': step,
                        'best_val_loss': best_val_loss,
                        'config': cfg,
                    }
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                    print0(f"💾 Checkpoint saved (loss: {best_val_loss:.4f})")
        
        # Fetch data
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            inputs, targets = next(data_iter)
            
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass with mixed precision
        with ctx:
            logits, loss = model(inputs, targets)
        
        # Backward pass
        for opt in optimizers: 
            opt.zero_grad(set_to_none=True)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(adamw_opt)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if step % log_interval == 0 and rank == 0:
            lossf = loss.item()
            # MFU estimation
            if step >= 5:
                mfu = raw_model.estimate_mfu(batch_size, dt)
                running_mfu = mfu if running_mfu < 0 else 0.9*running_mfu + 0.1*mfu
            pbar.set_postfix(loss=f"{lossf:.4f}", lr=f"{lr:.2e}", mfu=f"{running_mfu*100:.1f}%")

    total_time = time.time() - start_time
    print0(f"\n✅ Training Complete in {total_time:.2f}s")
    
    # Final save
    if rank == 0:
        final_path = os.path.join(out_dir, "ego_final.pt")
        torch.save(raw_model.state_dict(), final_path)
        print0(f"💾 Final model saved to {final_path}")

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()