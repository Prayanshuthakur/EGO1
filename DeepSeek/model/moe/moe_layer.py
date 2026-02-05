"""
Complete Mixture of Experts Layer
==================================

Combines routed experts + shared experts with efficient token dispatching.

Architecture:
1. Shared experts: Always activated for all tokens (capture common knowledge)
2. Routed experts: Top-K selection per token (capture specialized knowledge)
3. Token dispatching: Efficiently route tokens to selected experts
4. Output combination: Weighted sum of expert outputs

Key Features:
- Auxiliary-loss-free load balancing via dynamic bias
- Shared experts for common knowledge
- Efficient batched expert computation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .expert import Expert, SharedExpert
from .router import Router


class MoELayer(nn.Module):
    """
    Complete Mixture of Experts layer with shared and routed experts.
    
    This combines:
    - Shared experts (always active)
    - Routed experts (top-K selection)
    - Auxiliary-loss-free load balancing
    """
    
    def __init__(self, config):
        """
        Args:
            config: Model configuration with:
                - d_model: Model dimension
                - d_ff: FFN intermediate dimension
                - n_routed_experts: Number of routed experts
                - n_shared_experts: Number of shared experts
                - n_activated_experts: Top-K for routing
                - router_bias_update_rate: Bias learning rate
        """
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_activated_experts = config.n_activated_experts
        
        # === SHARED EXPERTS (always active) ===
        self.shared_experts = nn.ModuleList([
            SharedExpert(config.d_model, config.d_ff)
            for _ in range(config.n_shared_experts)
        ])
        
        # === ROUTED EXPERTS (top-K selection) ===
        self.routed_experts = nn.ModuleList([
            Expert(config.d_model, config.d_ff)
            for _ in range(config.n_routed_experts)
        ])
        
        # === ROUTER ===
        self.router = Router(
            d_model=config.d_model,
            num_experts=config.n_routed_experts,
            top_k=config.n_activated_experts,
            bias_update_rate=config.router_bias_update_rate
        )
        
        print(f"MoE Layer:")
        print(f"  - Shared experts: {config.n_shared_experts}")
        print(f"  - Routed experts: {config.n_routed_experts}")
        print(f"  - Active per token: {config.n_activated_experts + config.n_shared_experts}")
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through MoE layer.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            training: Whether in training mode
            
        Returns:
            output: MoE output [batch, seq_len, d_model]
            stats: Load balancing statistics (if training)
        """
        batch_size, seq_len, d_model = x.shape
        
        # === 1. SHARED EXPERTS (always active) ===
        # All tokens go through all shared experts
        shared_output = torch.zeros_like(x)
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(x)
        
        # Average shared expert outputs
        if self.n_shared_experts > 0:
            shared_output = shared_output / self.n_shared_experts
        
        # === 2. ROUTER: Select top-K experts ===
        # Get routing decisions
        top_k_scores, top_k_indices, expert_counts = self.router(x, training=training)
        # top_k_scores: [batch, seq_len, top_k]
        # top_k_indices: [batch, seq_len, top_k]
        
        # === 3. ROUTED EXPERTS ===
        # Flatten batch and sequence dimensions for efficient processing
        x_flat = x.view(-1, d_model)  # [batch * seq_len, d_model]
        top_k_scores_flat = top_k_scores.view(-1, self.n_activated_experts)  # [batch * seq_len, top_k]
        top_k_indices_flat = top_k_indices.view(-1, self.n_activated_experts)  # [batch * seq_len, top_k]
        
        # Initialize routed output
        routed_output_flat = torch.zeros_like(x_flat)  # [batch * seq_len, d_model]
        
        # Process each expert
        for expert_idx in range(self.n_routed_experts):
            # Find all tokens assigned to this expert
            # expert_mask: [batch * seq_len, top_k] - True where this expert is selected
            expert_mask = (top_k_indices_flat == expert_idx)
            
            # Get tokens for this expert
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                # No tokens assigned to this expert
                continue
            
            # Get input tokens for this expert
            expert_input = x_flat[token_indices]  # [num_tokens, d_model]
            
            # Forward through expert
            expert_output = self.routed_experts[expert_idx](expert_input)  # [num_tokens, d_model]
            
            # Get weights for this expert
            # For each token, find which position in top_k this expert is at
            expert_weights = torch.zeros(len(token_indices), device=x.device)
            for i, token_idx in enumerate(token_indices):
                # Find position of this expert in top_k for this token
                positions = (top_k_indices_flat[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    expert_weights[i] = top_k_scores_flat[token_idx, positions[0]]
            
            # Weight and accumulate expert output
            weighted_output = expert_output * expert_weights.unsqueeze(-1)
            routed_output_flat[token_indices] += weighted_output
        
        # Reshape back
        routed_output = routed_output_flat.view(batch_size, seq_len, d_model)
        
        # === 4. COMBINE SHARED AND ROUTED OUTPUTS ===
        output = shared_output + routed_output
        
        # === 5. COLLECT STATISTICS ===
        stats = None
        if training and expert_counts is not None:
            stats = {
                "expert_counts": expert_counts,
                "load_balance": self.router.get_load_balance_stats()
            }
        
        return output, stats
    
    def update_router_bias(self, expert_counts: torch.Tensor):
        """
        Update router bias based on expert utilization.
        
        This should be called after each training step.
        
        Args:
            expert_counts: Number of tokens assigned to each expert
        """
        self.router.update_bias(expert_counts)


if __name__ == "__main__":
    # Test MoE Layer
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        d_model: int = 2048
        d_ff: int = 5632
        n_routed_experts: int = 64
        n_shared_experts: int = 2
        n_activated_experts: int = 6
        router_bias_update_rate: float = 0.01
    
    config = TestConfig()
    batch_size = 2
    seq_len = 128
    
    # Create test input
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Create MoE layer
    moe = MoELayer(config)
    
    # Forward pass
    output, stats = moe(x, training=True)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expert counts: {stats['expert_counts']}")
    print(f"Load balance stats: {stats['load_balance']}")
    
    # Test bias update
    moe.update_router_bias(stats['expert_counts'])
    
    print(f"✓ MoE Layer test passed")
