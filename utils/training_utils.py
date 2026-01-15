"""
Training Utilities for Ego
==========================

This module provide helper functions for training and evaluating the Ego model:
1. calc_loss_batch - Compute loss for a single batch
2. calc_loss_loader - Compute average loss over a data loader
3. evaluate_model - Periodic evaluation during training
"""

import torch
import torch.nn as nn


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Compute the cross-entropy loss for a single batch.
    
    Args:
        input_batch (torch.Tensor): Input token IDs (B, T)
        target_batch (torch.Tensor): Target token IDs (B, T)
        model (nn.Module): The GPT model
        device (torch.device): Device to run computation on
        
    Returns:
        torch.Tensor: The scalar loss value
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    
    # Forward pass
    # logits shape: (batch_size, seq_len, vocab_size)
    logits = model(input_batch)
    
    # Flatten logits and targets for CrossEntropyLoss
    # CrossEntropyLoss expects (N, C) for input and (N) for target
    # where N = batch_size * seq_len, C = vocab_size
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Compute the average loss over a data loader.
    
    Args:
        data_loader (DataLoader): The data loader to evaluate
        model (nn.Module): The GPT model
        device (torch.device): Device to run on
        num_batches (int, optional): Number of batches to sample. 
                                    If None, uses the whole loader.
                                    
    Returns:
        float: The average loss
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return 0.
        
    # Set model to evaluation mode
    model.eval()
    
    # Cap the number of batches if specified
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluate the model on both training and validation sets.
    
    Args:
        model (nn.Module): The GPT model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to run on
        eval_iter (int): Number of batches to use for evaluation
        
    Returns:
        dict: Dictionary containing 'train_loss' and 'val_loss'
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    
    model.train() # Set back to training mode
    return {
        "train_loss": train_loss,
        "val_loss": val_loss
    }
