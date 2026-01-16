"""
Main Training Script for Ego
============================

This script implements the full training loop for the Ego model.
Features:
- Automatic device detection (CUDA, MPS, or CPU)
- 90/10 Train/Validation split of input text
- AdamW optimizer with weight decay
- Periodic loss logging and model evaluation
- Model checkpointing
"""

import os
import torch
import time
from config.ego_config import EGO_CONFIG_124M
from Model.ego_model import EgoModel
from Tokenizer.bpe_tokenizer import BytePairEncoding
from Tokenizer.EgoDatasetLoader import create_dataloader
from utils.training_utils import calc_loss_batch, evaluate_model


def train_model():
    print("="*60)
    print("EGO1 - Ego Training Pipeline")
    print("="*60)

    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # 2. Load Data and Split
    file_path = "test-dataset.txt"
    if not os.path.exists(file_path):
        # Fallback for local dev if file isn't in same dir
        file_path = os.path.join(os.path.dirname(__file__), "test-dataset.txt")
        
    if not os.path.exists(file_path):
        print(f"Error: test-dataset.txt not found.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    # Train/Val split (90% training)
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    print(f"Loaded dataset: {file_path}")
    print(f"Train set size: {len(train_data)} characters")
    print(f"Val set size: {len(val_data)} characters")

    # 3. Create Data Loaders
    tokenizer = BytePairEncoding()
    batch_size = 2
    max_length = 256 # Smaller context for demo training on small data
    stride = 128    # Sliding window for more batches
    
    train_loader = create_dataloader(
        train_data, 
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=True
    )

    val_loader = create_dataloader(
        val_data,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 4. Initialize Model
    # Adjust context length in config for this run
    cfg = EGO_CONFIG_124M.copy()
    cfg["context_length"] = max_length
    
    model = EgoModel(cfg).to(device)
    print(f"Model initialized ({model.count_parameters():,} parameters)")

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    

    # 6. Training Loop Configuration
    num_epochs = 1
    eval_freq = 2 # Frequency of evaluation
    eval_iter = 1 # Number of batches for evaluation estimate

    train_losses, val_losses = [], []
    start_time = time.time()

    model.train()
    
    print("\nStarting training loop...")
    for epoch in range(num_epochs):
        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Compute loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Log progress
            if i % eval_freq == 0:
                eval_results = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(eval_results["train_loss"])
                val_losses.append(eval_results["val_loss"])
                
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {i}/{len(train_loader)} | "
                      f"Train Loss: {eval_results['train_loss']:.4f} | "
                      f"Val Loss: {eval_results['val_loss']:.4f}")

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

    # 7. Save Model
    save_path = "ego_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    print("="*60)
    print("✅ Training sequence finished successfully!")
    print("="*60)


if __name__ == "__main__":
    train_model()