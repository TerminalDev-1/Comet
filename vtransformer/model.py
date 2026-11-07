"""VTransformer model (Phase 3 core).

Core upgrades over the baseline:
  - RoPE positions (no position table), QK-Norm, grouped-query attention (GQA)
  - SwiGLU MLP, scale-only norms, tied embeddings
  - attention kernel and norm type are switchable; defaults are the fastest on this CPU
    (plain attention math + LayerNorm; see results/phase3.md)
Later phases add parameter reuse (loops), EMOE, MoD, adaptive depth and MTP.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, Norm


@dataclass
class VConfig:
    vocab: int = 8192
    ctx: int = 256
    d: int = 256
    layers: int = 6
    heads: int = 4
    kv_heads: int = 2
    rope: bool = True
    qk_norm: bool = True
    mlp: str = "swiglu"  # "swiglu" or "gelu"
    norm: str = "layer"  # "layer" (fast on CPU) or "rms"
    attn_impl: str = "manual"  # "manual" (fast on CPU at ctx 256) or "sdpa"
    mlp_mult: float = 4.0  # hidden = mlp_mult * d (scaled by 2/3 for swiglu to keep params equal)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kind = cfg.mlp
        if cfg.mlp == "swiglu":
            h = int(cfg.mlp_mult * cfg.d * 2 / 3)
            h = (h + 31) // 32 * 32
            self.gate = nn.Linear(cfg.d, h, bias=False)
            self.up = nn.Linear(cfg.d, h, bias=False)
        else:
            h = int(cfg.mlp_mult * cfg.d)
            self.up = nn.Linear(cfg.d, h, bias=False)
        self.out = nn.Linear(h, cfg.d, bias=False)

    def forward(self, x):
        if self.kind == "swiglu":
            return self.out(F.silu(self.gate(x)) * self.up(x))
        return self.out(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = Norm(cfg.d, cfg.norm), Attention(cfg)
        self.n2, self.mlp = Norm(cfg.d, cfg.norm), MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class VTransformer(nn.Module):
    def __init__(self, cfg: VConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = Norm(cfg.d, cfg.norm)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.layers))

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.norm(x) @ self.tok.weight.T  # tied embeddings
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok.weight.numel()
        return {"total": total, "active": total, "unique": total, "non_embedding": total - emb}
