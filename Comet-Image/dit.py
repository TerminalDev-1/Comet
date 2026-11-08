"""Comet-Image: a small Diffusion Transformer (DiT, Peebles & Xie 2023) trained with flow matching
(rectified flow, as in Stable Diffusion 3).

The image is cut into patches (tokens), a plain Transformer processes them, and every block is told
the current noise level through adaptive LayerNorm (adaLN-Zero). The model predicts the "velocity"
that moves a noisy image toward a clean one; sampling follows that velocity from pure noise.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class TimeEmbed(nn.Module):
    def __init__(self, d, freq=256):
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):  # t in [0, 1]
        half = self.freq // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = (t[:, None] * 1000) * f[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], -1))


class Block(nn.Module):
    def __init__(self, d, heads, mlp_mult=4):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_mult * d), nn.GELU(approximate="tanh"), nn.Linear(mlp_mult * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, -1)
        B, T, D = x.shape
        q, k, v = self.qkv(modulate(self.n1(x), s1, sc1)).view(B, T, 3, self.heads, D // self.heads).unbind(2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + g1[:, None] * self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + g2[:, None] * self.mlp(modulate(self.n2(x), s2, sc2))


class DiT(nn.Module):
    def __init__(self, img=64, patch=4, d=384, layers=8, heads=6):
        super().__init__()
        self.img, self.patch = img, patch
        n = (img // patch) ** 2
        self.embed = nn.Conv2d(3, d, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.t = TimeEmbed(d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, patch * patch * 3)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t):
        B = x.shape[0]
        h = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        c = self.t(t)
        for b in self.blocks:
            h = b(h, c)
        sh, sc = self.ada(c).chunk(2, -1)
        h = self.out(modulate(self.norm(h), sh, sc))  # (B, n, p*p*3)
        g, p = self.img // self.patch, self.patch
        return h.view(B, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(B, 3, self.img, self.img)


def flow_loss(model, x1, t_shift=0.0):
    """Flow matching: mix noise and image at a random time t, predict the velocity (image - noise).
    t_shift > 0 spends more training on nearly-clean images (t near 1), where fine detail is learned."""
    x0 = torch.randn_like(x1)
    t = torch.sigmoid(torch.randn(x1.shape[0], device=x1.device) + t_shift)  # logit-normal, like SD3
    xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
    return F.mse_loss(model(xt, t).float(), (x1 - x0).float())


@torch.no_grad()
def sample(model, n, steps=40, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, 3, model.img, model.img, device=device, generator=g)
    for i in range(steps):
        t = torch.full((n,), i / steps, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device != "cpu"):
            v = model(x, t).float()
        x = x + v / steps
    return x.clamp(-1, 1)
