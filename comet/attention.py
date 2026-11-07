"""Comet attention: RoPE, QK-Norm, grouped-query attention.

Phase 3 finding (results/phase3.md): on this CPU (Ryzen 5 8640HS, BF16, context 256)
  - plain attention math is ~1.7x faster than the fused SDPA kernel, and
  - LayerNorm's fused kernel is ~4x faster than RMSNorm.
So both are switchable, and the defaults are the faster options.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Norm(nn.Module):
    """Scale-only normalization. kind="layer" (fast on CPU) or "rms"."""

    def __init__(self, dim, kind="layer", eps=1e-6):
        super().__init__()
        self.kind, self.eps = kind, eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.kind == "rms":
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        return F.layer_norm(x, (x.shape[-1],), self.weight, None, self.eps)


class Rotary(nn.Module):
    """Rotary position embeddings: position is encoded by rotating q and k."""

    def __init__(self, head_dim, ctx, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(ctx).float(), inv)  # (ctx, hd/2)
        self.register_buffer("cos", freqs.cos()[None, None], persistent=False)
        self.register_buffer("sin", freqs.sin()[None, None], persistent=False)

    def forward(self, x):  # x: (B, H, T, hd), rotate-half layout
        T = x.shape[2]
        cos, sin = self.cos[:, :, :T].to(x.dtype), self.sin[:, :, :T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.hd = cfg.d // cfg.heads
        self.impl = cfg.attn_impl
        self.q = nn.Linear(cfg.d, cfg.heads * self.hd, bias=False)
        self.kv = nn.Linear(cfg.d, 2 * cfg.kv_heads * self.hd, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.rope = Rotary(self.hd, cfg.ctx) if cfg.rope else None
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm, self.k_norm = Norm(self.hd, cfg.norm), Norm(self.hd, cfg.norm)
        self.register_buffer("mask", torch.ones(cfg.ctx, cfg.ctx, dtype=torch.bool).triu(1), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        H, KV, hd = self.heads, self.kv_heads, self.hd
        q = self.q(x).view(B, T, H, hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, KV, hd).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        if self.impl == "sdpa":
            if KV != H:
                k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # GQA without copying K/V: group query heads and broadcast over the group.
            g = H // KV
            q = q.view(B, KV, g, T, hd)
            att = (q @ k.unsqueeze(2).transpose(-2, -1)) / math.sqrt(hd)
            att = att.masked_fill(self.mask[:T, :T], float("-inf")).softmax(-1)
            y = (att @ v.unsqueeze(2)).view(B, H, T, hd)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))
