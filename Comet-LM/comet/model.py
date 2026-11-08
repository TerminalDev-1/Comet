"""Comet model (Phase 3 core).

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
from .emoe import EMOELayer, expert_params, make_pool


@dataclass
class CometConfig:
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
    # EMOE (Efficient Mixture of Experts). Off by default.
    emoe: bool = False
    n_experts: int = 64  # size of the ONE pool shared by all layers
    top_k: int = 4  # experts each token uses per layer
    expert_hidden: int = 128  # tiny experts (128 = smallest size that stays fast on this CPU)
    shared_hidden: int = 192  # always-on shared expert per layer (0 = off)
    balance_rate: float = 1e-3  # how fast router biases rebalance expert load
    capacity: float = 1.25  # expert slot size as a multiple of average load
    # Multi-token prediction (training only). 0 = off.
    mtp_depth: int = 0  # extra tokens ahead to predict (1 = also predict token t+2)
    mtp_weight: float = 0.3  # weight of the MTP loss
    mtp_frac: float = 0.5  # fraction of positions scored by the MTP head
    mtp_block: bool = True  # extra transformer block per MTP depth (False = light, cheaper)
    # Speedrun-style tricks (modded-nanoGPT). Off by default until tested.
    logit_softcap: float = 0.0  # >0: logits = cap * tanh(logits / cap)
    unet: bool = False  # U-Net skips: first-half block outputs feed the mirrored second-half blocks
    # Mixture of Depths. Off by default.
    mod: bool = False
    mod_capacity: float = 0.5  # target fraction of tokens processed in MoD layers
    mod_every: int = 2  # MoD on every 2nd layer (1, 3, 5)
    mod_aux_weight: float = 0.1  # weight of the router's side loss
    mod_bucket: int = 16  # round the processed-token count up to a multiple of this (0 = exact).
    # Exact counts change every step, which fragments the CPU heap: peak RAM crept from 1.2 to 2.3 GB.
    # Fixed bucket sizes keep RAM flat at the dense level (tools/membench.py).


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
    def __init__(self, cfg, pool=None, out_std=0.02):
        super().__init__()
        self.n1, self.attn = Norm(cfg.d, cfg.norm), Attention(cfg)
        self.n2 = Norm(cfg.d, cfg.norm)
        self.mlp = EMOELayer(cfg, pool, out_std) if pool is not None else MLP(cfg)

    def forward(self, x, pos=None, valid=None):
        x = x + self.attn(self.n1(x), pos, valid)
        return x + self.mlp(self.n2(x))


class MoDRouter(nn.Module):
    """Mixture of Depths: decide per token whether a block processes it or it skips ahead.

    Causal routing: a token is processed if its own router score > 0, so the decision never
    depends on future tokens (no leak into validation scores). A side loss (BCE) teaches the
    router to match "top `capacity` fraction of the sequence", which keeps ~capacity of tokens
    processed. The block's update is scaled by sigmoid(score), which trains the router.
    """

    def __init__(self, cfg):
        super().__init__()
        self.lin = nn.Linear(cfg.d, 1, bias=False)
        self.capacity = cfg.mod_capacity
        self.bucket = cfg.mod_bucket

    def forward(self, block, x):
        B, T, D = x.shape
        s = self.lin(x).squeeze(-1).float()  # (B, T)
        aux = None
        if self.training:
            k = max(1, int(T * self.capacity))
            with torch.no_grad():
                target = torch.zeros_like(s).scatter_(1, s.topk(k, dim=1).indices, 1.0)
            aux = F.binary_cross_entropy_with_logits(s, target)
        with torch.no_grad():
            sel = s > 0
            counts = sel.sum(1)
            C = max(1, int(counts.max()))
            if self.bucket:  # fixed shapes -> no heap fragmentation (padding slots are masked)
                C = min(T, -(-C // self.bucket) * self.bucket)
            ar = torch.arange(T, device=x.device)
            key = torch.where(sel, ar, ar + T)  # selected tokens first, in position order
            idx = key.argsort(dim=1)[:, :C]  # (B, C)
            valid = torch.arange(C, device=x.device)[None] < counts[:, None]
        xs = x.gather(1, idx[..., None].expand(B, C, D))
        delta = block(xs, pos=idx, valid=valid) - xs
        w = (torch.sigmoid(s).gather(1, idx) * valid).to(delta.dtype)
        x = x.scatter_add(1, idx[..., None].expand(B, C, D), delta * w[..., None])
        return x, aux, sel.float().mean()


class MTPModule(nn.Module):
    """Multi-token prediction, one extra depth (DeepSeek-V3 style).

    At position t it combines the previous depth's hidden state with the embedding of
    token t+j, runs one small transformer block, and predicts token t+j+1 with the
    shared (tied) output layer. Training only; generation ignores it.
    """

    def __init__(self, cfg):
        super().__init__()
        self.nh, self.ne = Norm(cfg.d, cfg.norm), Norm(cfg.d, cfg.norm)
        self.proj_in = nn.Linear(2 * cfg.d, cfg.d, bias=False)
        self.block = Block(cfg) if cfg.mtp_block else None  # light version: no extra block

    def forward(self, h, next_emb):
        z = self.proj_in(torch.cat([self.nh(h), self.ne(next_emb)], dim=-1))
        return self.block(z) if self.block is not None else z


class Comet(nn.Module):
    def __init__(self, cfg: CometConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.mtp = nn.ModuleList(MTPModule(cfg) for _ in range(cfg.mtp_depth))
        self.last_main_loss = None
        if cfg.emoe:
            self.pool, out_std = make_pool(cfg)  # stored once, used by every layer
            self.blocks = nn.ModuleList(Block(cfg, self.pool, out_std) for _ in range(cfg.layers))
        else:
            self.pool = None
            self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = Norm(cfg.d, cfg.norm)
        # Mixture of Depths on every `mod_every`-th layer (layers 1, 3, 5 for mod_every=2)
        self.mod_layers = [i for i in range(cfg.layers) if cfg.mod and i % cfg.mod_every == cfg.mod_every - 1]
        self.mod_routers = nn.ModuleDict({str(i): MoDRouter(cfg) for i in self.mod_layers})
        self.mod_frac = None  # fraction of tokens processed in MoD layers (last forward)
        if cfg.unet:  # one learnable weight per skip (layer i feeds layer L-1-i)
            self.skip_w = nn.Parameter(torch.ones(cfg.layers // 2))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.layers))

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        aux, fracs, skips = [], [], []
        L = len(self.blocks)
        for i, b in enumerate(self.blocks):
            if self.cfg.unet and i >= L - L // 2:
                x = x + self.skip_w[L - 1 - i] * skips[L - 1 - i]
            if str(i) in self.mod_routers:
                x, a, f = self.mod_routers[str(i)](b, x)
                fracs.append(f)
                if a is not None:
                    aux.append(a)
            else:
                x = b(x)
            if self.cfg.unet and i < L // 2:
                skips.append(x)
        if fracs:
            self.mod_frac = float(torch.stack(fracs).mean())
        logits = self.norm(x) @ self.tok.weight.T  # tied embeddings
        if self.cfg.logit_softcap:
            c = self.cfg.logit_softcap
            logits = c * torch.tanh(logits / c)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        self.last_main_loss = loss.detach()
        if self.training and self.mtp:
            loss = loss + self.cfg.mtp_weight * self._mtp_loss(x, targets)
        if aux:
            loss = loss + self.cfg.mod_aux_weight * torch.stack(aux).mean()
        return logits, loss

    def _mtp_loss(self, h, targets):
        """Extra losses for tokens 2..(1+mtp_depth) ahead, on a random sample of positions."""
        T = targets.shape[1]
        losses = []
        for j, mod in enumerate(self.mtp, start=1):
            # position t: hidden of depth j-1 + embedding of token t+j (= targets[:, t+j-1]) -> predict targets[:, t+j]
            h = mod(h[:, :T - j], self.tok(targets[:, j - 1:T - 1]))
            tgt = targets[:, j:]
            hs = self.norm(h).reshape(-1, h.shape[-1])
            tg = tgt.reshape(-1)
            if self.cfg.mtp_frac < 1.0:  # score only a sample of positions (output layer is costly)
                n = max(1, int(hs.shape[0] * self.cfg.mtp_frac))
                sel = torch.randperm(hs.shape[0], device=hs.device)[:n]
                hs, tg = hs[sel], tg[sel]
            lg = hs @ self.tok.weight.T
            losses.append(F.cross_entropy(lg.float(), tg))
        return torch.stack(losses).mean()

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok.weight.numel()
        active = total
        if self.pool is not None:
            pool_total = sum(p.numel() for p in self.pool.parameters())
            # each token uses top_k experts in each layer
            used = min(self.cfg.n_experts, self.cfg.top_k * self.cfg.layers) * expert_params(self.pool)
            touched = total - pool_total + min(used, pool_total)
            # "active" = parameters each token is multiplied through (sets compute cost)
            active = total - pool_total + self.cfg.top_k * self.cfg.layers * expert_params(self.pool)
            return {"total": total, "active": active, "unique": total, "non_embedding": total - emb,
                    "pool": pool_total, "touched_per_token": touched}
        return {"total": total, "active": active, "unique": total, "non_embedding": total - emb}
