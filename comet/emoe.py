"""EMOE: Efficient Mixture of Experts.

Three parts (see plan.md, Pillar 3):
  1. Fine-grained tiny experts (e.g. 64 experts of hidden size 128, each token uses k=4).
  2. One expert pool shared by ALL layers. Each layer has its own router, so layers can
     still specialize, but the expert weights are stored once.
  3. An always-on shared expert per layer for common basics.

Routing: sigmoid scores, top-k, weights renormalized over the chosen experts.
Load balancing: no extra loss. Each router keeps a per-expert bias that is nudged so
overloaded experts get picked less (bias only affects *which* experts are picked).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertPool(nn.Module):
    """E SwiGLU experts stored as stacked 3D weights (one matrix per expert)."""

    def __init__(self, n_experts, d, hidden, out_std, capacity=1.25):
        super().__init__()
        self.n, self.d, self.h = n_experts, d, hidden
        self.capacity = capacity  # expert slot size = capacity x average load
        self.w_gate = nn.Parameter(torch.randn(n_experts, d, hidden) * 0.02)
        self.w_up = nn.Parameter(torch.randn(n_experts, d, hidden) * 0.02)
        self.w_out = nn.Parameter(torch.randn(n_experts, hidden, d) * out_std)

    def run(self, xs, e_sorted, counts):
        """Run all experts at once with batched matmuls.

        xs: (M, d) token copies sorted by expert id; e_sorted: (M,) their expert ids;
        counts: (E,) tokens per expert. Each expert gets a padded slot of size max(counts),
        so there's no per-expert Python loop and no dropped tokens.
        """
        E, M = self.n, xs.shape[0]
        cap = min(int(counts.max()), math.ceil(self.capacity * M / E))
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(M, device=xs.device) - starts[e_sorted]
        keep = pos < cap  # overflow tokens (rare once balanced) only get the shared expert
        dt = torch.bfloat16 if torch.is_autocast_cpu_enabled() else xs.dtype
        buf = xs.new_zeros(E, cap, xs.shape[1], dtype=dt)
        buf = buf.index_put((e_sorted[keep], pos[keep]), xs[keep].to(dt))
        h = F.silu(torch.bmm(buf, self.w_gate.to(dt))) * torch.bmm(buf, self.w_up.to(dt))
        y = torch.bmm(h, self.w_out.to(dt))
        out = xs.new_zeros(M, xs.shape[1], dtype=y.dtype)
        out[keep] = y[e_sorted[keep], pos[keep]]
        return out


class SwiGLU(nn.Module):
    def __init__(self, d, hidden, out_std):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.out = nn.Linear(hidden, d, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=out_std)

    def forward(self, x):
        return self.out(F.silu(self.gate(x)) * self.up(x))


class EMOELayer(nn.Module):
    """One layer's view of the shared pool: its own router + bias + shared expert."""

    def __init__(self, cfg, pool: ExpertPool, out_std):
        super().__init__()
        self.pool = [pool]  # list so the pool isn't registered once per layer
        self.k = cfg.top_k
        self.router = nn.Linear(cfg.d, pool.n, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.register_buffer("bias", torch.zeros(pool.n))
        self.bias_lr = cfg.balance_rate
        self.shared = SwiGLU(cfg.d, cfg.shared_hidden, out_std) if cfg.shared_hidden > 0 else None
        self.last_load = None

    def forward(self, x):
        B, T, D = x.shape
        pool = self.pool[0]
        flat = x.reshape(-1, D)
        N = flat.shape[0]
        scores = torch.sigmoid(self.router(flat).float())  # (N, E)
        sel = torch.topk(scores + self.bias, self.k, dim=-1).indices  # bias only steers choice
        w = scores.gather(1, sel)
        w = (w / w.sum(-1, keepdim=True)).to(flat.dtype)

        flat_e = sel.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        tok = torch.arange(N, device=x.device).repeat_interleave(self.k)[order]
        counts_t = torch.bincount(flat_e, minlength=pool.n)
        ys = pool.run(flat[tok], flat_e[order], counts_t)
        ys = ys * w.reshape(-1)[order].unsqueeze(1)
        out = torch.zeros(N, D, dtype=ys.dtype, device=x.device).index_add_(0, tok, ys)

        if self.training:
            with torch.no_grad():  # aux-loss-free balancing
                load = counts_t.float()
                self.bias += self.bias_lr * torch.sign(load.mean() - load)
                self.last_load = load
        out = out.reshape(B, T, D)
        if self.shared is not None:
            out = out + self.shared(x)
        return out


def expert_params(pool: ExpertPool):
    return 3 * pool.d * pool.h


def make_pool(cfg):
    out_std = 0.02 / math.sqrt(2 * cfg.layers)
    return ExpertPool(cfg.n_experts, cfg.d, cfg.expert_hidden, out_std, cfg.capacity), out_std
