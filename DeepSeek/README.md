# DeepSeek Implementation

A production-grade implementation of DeepSeek-style architecture with:
- **Multi-head Latent Attention (MLA)**: 90% KV cache reduction
- **Auxiliary-Loss-Free MoE**: Dynamic bias load balancing
- **FP8 Quantization**: Fine-grained mixed precision training
- **Industry-Standard Data Pipeline**: Complete preprocessing from raw text to tokens

## Architecture Highlights

### 1. Multi-head Latent Attention (MLA)
- Low-rank KV compression (32x compression ratio)
- Compressed latent dimension: d_c = 512
- Decoupled RoPE for positional encoding
- KV cache: 213GB → 7.6GB (28x smaller for 128K context)

### 2. Mixture of Experts (MoE)
- Fine-grained experts: 128-256 experts
- Top-K routing: K = 4-8 experts per token
- **No auxiliary loss** - dynamic bias adjustment for load balancing
- Shared experts for common knowledge

### 3. FP8 Quantization
- E4M3 format for forward pass
- Tile-wise activation scaling (1x128)
- Block-wise weight scaling (128x128)
- BF16/FP32 for critical operations

## Quick Start

```bash
# 1. Prepare data
python scripts/prepare_data.py --config config/data_config.py

# 2. Train model
python scripts/train.py --config config/training_config.py

# 3. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/model.pt
```

## Project Structure

```
DeepSeek/
├── config/              # Configuration files
├── data_pipeline/       # Complete data preprocessing
├── model/              # Model architecture
│   ├── attention/      # MLA implementation
│   ├── moe/           # MoE with dynamic bias
│   ├── quantization/  # FP8 quantization
│   └── layers/        # Core layers (RMSNorm, etc.)
├── training/          # Training infrastructure
├── inference/         # Inference optimizations
└── evaluation/        # Benchmarks and metrics
```

## Key Features

- ✅ 90% KV cache reduction via MLA
- ✅ Auxiliary-loss-free MoE load balancing
- ✅ FP8 mixed precision training
- ✅ Distributed training (FSDP/DeepSpeed)
- ✅ Comprehensive data pipeline
- ✅ Production-ready code with extensive comments

## References

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- Multi-head Latent Attention (MLA)
- Auxiliary-Loss-Free Load Balancing
- FP8 Mixed Precision Training
