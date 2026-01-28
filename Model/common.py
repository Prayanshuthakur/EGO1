import os
import torch
import torch.distributed as dist
import logging

def print0(s="", **kwargs):
    """Print only on rank 0."""
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        print(s, **kwargs)

def get_dist_info():
    """Get distributed training info."""
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        return True, rank, local_rank, world_size
    return False, 0, 0, 1

def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_peak_flops(device_name: str) -> float:
    # Simplified peak flops check
    name = device_name.lower()
    if "h100" in name or "h200" in name:
        return 989e12
    if "a100" in name:
        return 312e12
    if "4090" in name:
        return 165e12
    return float('inf')
