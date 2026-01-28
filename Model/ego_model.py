import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect
from Model.layers import RMSNorm
from Model.transformer_block import TransformerBlock
from Model.common import print0

class EgoModel(nn.Module):
    """
    Ego SOTA - A Modern Transformer Architecture for 2026.
    
    Features:
    - RoPE (Rotary Positional Embeddings)
    - Per-layer Residual & X0 Scalars
    - Logit Softcapping (15.0)
    - RMSNorm (functional)
    - QK Norm in Attention
    - Multi-Optimizer support ready
    """
    
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        
        # 1. Embeddings (No absolute positional embeddings needed!)
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        
        # 2. Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg, i) for i in range(cfg["n_layers"])
        ])
        
        # Per-layer learnable scalars (inspired by nanochat/modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer
        # x0_lambdas: blends initial embedding back in at each layer
        self.resid_lambdas = nn.Parameter(torch.ones(cfg["n_layers"]))
        self.x0_lambdas = nn.Parameter(torch.zeros(cfg["n_layers"]))
        
        # Final Norm
        self.final_norm = RMSNorm(cfg["emb_dim"])
        
        # LM Head (Untied for better performance)
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
        # Precompute RoPE embeddings
        # We over-compute by 2X the context length to be safe
        self.rotary_seq_len = cfg["context_length"] * 2
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
        # Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Special scaled initialization for projections
            std = 0.02
            if hasattr(module, 'c_proj'): # Not quite right for nn.Linear
                 std *= (2 * self.cfg["n_layers"])**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Shape: (1, seq_len, 1, head_dim/2) for broadcasting
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, device_type='cuda'):
        """
        Setup dual-optimizer training (AdamW + Muon).
        
        Uses fused AdamW when available on CUDA for better performance.
        """
        # Separate parameters into groups for different optimizers
        matrix_params = []
        embedding_params = []
        lm_head_params = []
        scalar_params = [self.resid_lambdas, self.x0_lambdas]
        
        for name, param in self.named_parameters():
            if "token_embedding" in name:
                embedding_params.append(param)
            elif "lm_head" in name:
                lm_head_params.append(param)
            elif "resid_lambdas" in name or "x0_lambdas" in name:
                continue  # already in scalar_params
            elif param.ndim == 2:  # any other 2D parameter is a "matrix"
                matrix_params.append(param)
            else:  # any other parameter (e.g. 1D bias or norm scale)
                embedding_params.append(param)  # treat as small param for AdamW

        from Model.adamw import DistAdamW
        from Model.muon import Muon, DistMuon
        from Model.common import get_dist_info
        
        is_ddp, _, _, _ = get_dist_info()
        
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr),
            dict(params=embedding_params, lr=embedding_lr),
            dict(params=scalar_params, lr=0.005),  # sensitive
        ]
        
        # Use fused AdamW when available (significant speedup on CUDA)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        AdamWFactory = DistAdamW if is_ddp else torch.optim.AdamW
        adamw_optimizer = AdamWFactory(adam_groups, betas=(0.8, 0.95), eps=1e-10, **extra_args)
        
        if use_fused:
            print0(f"✅ Using fused AdamW optimizer")
        
        MuonFactory = DistMuon if is_ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        
        return [adamw_optimizer, muon_optimizer]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, kv_cache=None) -> torch.Tensor:
        B, T = idx.shape
        
        # 1. Embeddings
        x = self.token_embedding(idx)
        x0 = x # save for x0 residuals
        
        # 2. Get RoPE cos/sin for current seq len
        cos_sin = (self.cos[:, :T], self.sin[:, :T])
        
        # 3. Process Blocks
        for i, block in enumerate(self.transformer_blocks):
            # Blend in layer scalars
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, kv_cache=kv_cache)
            
        # 4. Final Norm
        x = self.final_norm(x)
        
        # 5. LM Head with Softcapping
        logits = self.lm_head(x)
        
        # Logit Softcapping (15.0) prevents training instabilities
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)
        
        # 6. Loss Calculation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return (logits, loss) if targets is not None else logits

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_mfu(self, batch_size, dt):
        """
        Estimate Model FLOPS Utilization (MFU) as ratio of A100 peak FLOPS.
        
        Based on PaLM paper Appendix B: https://arxiv.org/abs/2204.02311
        
        Args:
            batch_size: Number of samples per iteration
            dt: Time taken for one training iteration (seconds)
        
        Returns:
            MFU as a fraction (e.g., 0.5 means 50% utilization)
        """
        N = self.count_parameters()
        cfg = self.cfg
        L = cfg["n_layers"]
        H = cfg["n_heads"]
        Q = cfg["emb_dim"] // cfg["n_heads"]  # head dimension
        T = cfg["context_length"]
        
        # FLOPs per token (forward + backward = 2x, plus activations = 3x total)
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * batch_size
        
        # Express throughput as ratio of A100 bfloat16 peak FLOPS
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12  # A100 GPU bfloat16 peak is 312 TFLOPS
        
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Autoregressive generation
            logits = self(idx)
            logits = logits[:, -1, :] / (temperature + 1e-6)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
