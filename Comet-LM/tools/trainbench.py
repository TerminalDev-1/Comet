"""Phase 0: time real training steps (forward + backward + optimizer) of a small GPT.

Usage:  python tools/trainbench.py --device cpu|dml [--threads 6] [--out results/train_cpu.json]
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, mask):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        # Plain attention math (works on every backend, including DirectML).
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.heads)
        att = att.masked_fill(mask[:T, :T], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.register_buffer("mask", torch.ones(ctx, ctx, dtype=torch.bool).triu(1))

    def forward(self, idx, targets):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x, self.mask)
        logits = self.ln(x) @ self.tok.weight.T  # tied embeddings
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))


def run(device, dtype_name, cfg, steps=12, warmup=3):
    torch.manual_seed(0)
    model = GPT(**cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    B, T, V = cfg["batch"], cfg["model"]["ctx"], cfg["model"]["vocab"]
    idx = torch.randint(0, V, (B, T), device=device)
    tgt = torch.randint(0, V, (B, T), device=device)

    use_bf16 = dtype_name == "bf16"
    if dtype_name == "fp16":
        model = model.half()
        # keep LayerNorm in fp32-friendly half; DirectML handles fp16 LN

    def step():
        if use_bf16:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(idx, tgt)
        else:
            loss = model(idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    for _ in range(warmup):
        loss = step()
    loss.item()
    t0 = time.perf_counter()
    for _ in range(steps):
        loss = step()
    loss.item()
    dt = (time.perf_counter() - t0) / steps
    tok_s = B * T / dt
    return {"params_M": round(n_params / 1e6, 2), "sec_per_step": round(dt, 4),
            "tokens_per_sec": round(tok_s), "tokens_in_25_min_M": round(tok_s * 1500 / 1e6, 1),
            "loss": round(float(loss), 3)}


CONFIGS = {
    "~5M (d=256, 6 layers)": {"model": dict(vocab=8192, d=256, layers=6, heads=4, ctx=256), "batch": 16},
    "~13M (d=384, 6 layers)": {"model": dict(vocab=8192, d=384, layers=6, heads=6, ctx=256), "batch": 16},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.device == "dml":
        import torch_directml
        device = torch_directml.device()
        dtypes = ["fp32", "fp16"]
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.threads)
        dtypes = ["fp32", "bf16"]

    results = {"device": args.device, "threads": args.threads if args.device == "cpu" else None,
               "torch": torch.__version__, "runs": {}}
    for name, cfg in CONFIGS.items():
        for dt in dtypes:
            key = f"{name} {dt}"
            try:
                r = run(device, dt, cfg)
            except Exception as e:
                r = {"error": f"{e.__class__.__name__}: {str(e)[:120]}"}
            results["runs"][key] = r
            print(key, r, flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
