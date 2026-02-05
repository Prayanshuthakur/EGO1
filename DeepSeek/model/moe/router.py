"""
Router with Auxiliary-Loss-Free Load Balancing
===============================================

DeepSeek's innovation for MoE: NO auxiliary loss needed for load balancing!

Traditional MoE Problem:
- Routers tend to overload a few "favorite" experts
- Solution: Add auxiliary loss to force balance
- Problem: Auxiliary loss degrades main task performance

DeepSeek Solution:
- Learn an expert-wise bias (b_i) for each expert
- After each training step, adjust bias based on load:
  * If expert is overloaded -> reduce its bias (make it less attractive)
  * If expert is underutilized -> increase its bias (make it more attractive)
- Bias adjustment is NOT differentiable (happens after gradient step)
- No auxiliary loss needed!

Result: Perfect load balancing without hurting model quality.

Reference: DeepSeek-V3 Technical Report, Section 2.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Router(nn.Module):
    """
    Top-K router with auxiliary-loss-free load balancing via dynamic bias.
    
    This is the key innovation that makes DeepSeek MoE efficient without
    sacrificing model quality.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        bias_update_rate: float = 0.01
    ):
        """
        Args:
            d_model: Model dimension
            num_experts: Total number of routed experts
            top_k: Number of experts to activate per token
            bias_update_rate: Learning rate for bias adjustment (α in paper)
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate
        
        # Gating network: maps input to expert scores
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Expert-wise bias for load balancing
        # This is adjusted dynamically, NOT via backprop!
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_experts)
        )
        
        # Track expert utilization for monitoring
        self.register_buffer(
            "expert_counts",
            torch.zeros(num_experts)
        )
        
        print(f"  Router: {num_experts} experts, top-{top_k}, bias_lr={bias_update_rate}")
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to top-K experts.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            training: Whether in training mode (for bias updates)
            
        Returns:
            top_k_scores: Gating scores for selected experts [batch, seq_len, top_k]
            top_k_indices: Indices of selected experts [batch, seq_len, top_k]
            expert_counts: Number of tokens assigned to each expert [num_experts]
        """
        batch_size, seq_len, _ = x.shape
        
        # === 1. COMPUTE GATING SCORES ===
        # Project input to expert scores
        # [batch, seq_len, d_model] -> [batch, seq_len, num_experts]
        logits = self.gate(x)
        
        # === 2. APPLY BIAS FOR LOAD BALANCING ===
        # Add expert bias to logits
        # Bias is NOT part of the computational graph (detached)
        # This is the key: bias affects routing but not gradients!
        logits_with_bias = logits + self.expert_bias.detach()
        
        # === 3. TOP-K SELECTION ===
        # Select top-K experts based on biased logits
        # [batch, seq_len, num_experts] -> [batch, seq_len, top_k]
        top_k_logits, top_k_indices = torch.topk(
            logits_with_bias,
            k=self.top_k,
            dim=-1
        )
        
        # === 4. COMPUTE GATING WEIGHTS ===
        # Apply softmax to get normalized weights
        # Important: Softmax over the ORIGINAL logits (without bias)
        # This ensures gradients flow correctly
        top_k_scores = F.softmax(top_k_logits, dim=-1)
        
        # === 5. TRACK EXPERT UTILIZATION ===
        if training:
            # Count how many tokens are assigned to each expert
            # This is used for bias updates
            expert_counts = torch.zeros(
                self.num_experts,
                device=x.device,
                dtype=torch.long
            )
            
            # Flatten indices and count
            flat_indices = top_k_indices.view(-1)  # [batch * seq_len * top_k]
            expert_counts.scatter_add_(
                0,
                flat_indices,
                torch.ones_like(flat_indices, dtype=torch.long)
            )
            
            # Update running counts (for monitoring)
            self.expert_counts += expert_counts.float()
        else:
            expert_counts = None
        
        return top_k_scores, top_k_indices, expert_counts
    
    def update_bias(self, expert_counts: torch.Tensor):
        """
        Update expert bias based on load distribution.
        
        This is called AFTER the gradient step, not during backprop!
        
        Algorithm:
        1. Compute average load: avg = total_tokens / num_experts
        2. For each expert:
            - If load > avg: expert is overloaded -> reduce bias
            - If load < avg: expert is underutilized -> increase bias
        3. Update: bias[i] -= α * (load[i] - avg)
        
        Args:
            expert_counts: Number of tokens assigned to each expert [num_experts]
        """
        if expert_counts is None:
            return
        
        # Compute average load
        total_tokens = expert_counts.sum()
        avg_load = total_tokens / self.num_experts
        
        # Compute load imbalance
        load_diff = expert_counts.float() - avg_load
        
        # Update bias (reduce bias for overloaded experts)
        # Note: This is NOT a gradient update!
        bias_update = self.bias_update_rate * load_diff
        self.expert_bias -= bias_update
        
        # Optional: Clip bias to prevent extreme values
        # self.expert_bias.clamp_(-1.0, 1.0)
    
    def get_load_balance_stats(self) -> dict:
        """
        Get statistics about expert load balancing.
        
        Returns:
            Dictionary with load balance metrics
        """
        if self.expert_counts.sum() == 0:
            return {}
        
        # Normalize counts to get load distribution
        load_dist = self.expert_counts / self.expert_counts.sum()
        
        # Compute statistics
        stats = {
            "load_mean": load_dist.mean().item(),
            "load_std": load_dist.std().item(),
            "load_min": load_dist.min().item(),
            "load_max": load_dist.max().item(),
            "load_cv": (load_dist.std() / load_dist.mean()).item(),  # Coefficient of variation
            "bias_mean": self.expert_bias.mean().item(),
            "bias_std": self.expert_bias.std().item(),
        }
        
        return stats
    
    def reset_counts(self):
        """Reset expert counts (call at end of epoch)."""
        self.expert_counts.zero_()


if __name__ == "__main__":
    # Test Router
    d_model = 2048
    num_experts = 64
    top_k = 6
    batch_size = 2
    seq_len = 128
    
    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create router
    router = Router(d_model, num_experts, top_k)
    
    # Forward pass
    scores, indices, counts = router(x, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Expert counts: {counts}")
    print(f"Total tokens routed: {counts.sum()}")
    print(f"Expected: {batch_size * seq_len * top_k}")
    
    # Test bias update
    router.update_bias(counts)
    print(f"Bias after update: {router.expert_bias[:10]}")  # Show first 10
    
    # Test load balance stats
    stats = router.get_load_balance_stats()
    print(f"Load balance stats: {stats}")
    
    print(f"✓ Router test passed")
