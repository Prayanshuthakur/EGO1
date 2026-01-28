import torch
import torch.nn.functional as F
from types import SimpleNamespace

def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention fallback to SDPA.
    q, k, v: (B, T, H, D)
    """
    if window_size is None:
        window_size = (-1, -1)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]
    enable_gqa = q.size(1) != k.size(1)

    if enable_gqa:
        # Repeat K/V heads to match Q heads
        ratio = q.size(1) // k.size(1)
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)

    if (window < 0 or window >= Tq) and Tq == Tk:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    else:
        # Need explicit mask for sliding window
        device = q.device
        mask = torch.tril(torch.ones(Tq, Tk, device=device, dtype=torch.bool))
        if window > 0 and window < Tq:
            row_idx = torch.arange(Tq, device=device).unsqueeze(1)
            col_idx = torch.arange(Tk, device=device).unsqueeze(0)
            mask = mask & ((row_idx - col_idx) <= window)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    return y.transpose(1, 2)

def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    if window_size is None:
        window_size = (-1, -1)
        
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()

    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    q_sdpa = q.transpose(1, 2).contiguous()
    k_sdpa = k_full.transpose(1, 2).contiguous()
    v_sdpa = v_full.transpose(1, 2).contiguous()

    if q_sdpa.size(1) != k_sdpa.size(1):
        ratio = q_sdpa.size(1) // k_sdpa.size(1)
        k_sdpa = k_sdpa.repeat_interleave(ratio, dim=1)
        v_sdpa = v_sdpa.repeat_interleave(ratio, dim=1)

    y_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=causal)
    return y_sdpa.transpose(1, 2)

flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
