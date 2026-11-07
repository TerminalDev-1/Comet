"""Plain GPT baseline: the model VTransformer has to beat.

Standard GPT-2 style: learned position embeddings, LayerNorm, plain (unfused) causal
attention, GELU MLP, tied input/output embeddings. No sharing, no experts.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineConfig:
    vocab: int = 8192
    ctx: int = 256
    d: int = 256
    layers: int = 6
    heads: int = 4
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.proj = nn.Linear(cfg.d, cfg.d, bias=False)
        self.register_buffer("mask", torch.ones(cfg.ctx, cfg.ctx, dtype=torch.bool).triu(1), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.heads
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, hd).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.masked_fill(self.mask[:T, :T], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d, 4 * cfg.d, bias=False)
        self.out = nn.Linear(4 * cfg.d, cfg.d, bias=False)

    def forward(self, x):
        return self.out(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class BaselineGPT(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.ctx, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.d)
        self.apply(self._init)
        # GPT-2: scale residual output projections by depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.ln(x) @ self.tok.weight.T  # tied embeddings
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok.weight.numel() + self.pos.weight.numel()
        return {"total": total, "active": total, "unique": total, "non_embedding": total - emb}
