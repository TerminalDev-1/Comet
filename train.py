"""Train any model in vtransformer.MODELS under a wall-clock time budget.

Example:
    python train.py --model baseline --name baseline_d256 --minutes 25 --d 256 --layers 6 --heads 4

The learning-rate schedule follows elapsed *training time* (not steps), so every run
uses its full budget and ends with a fully decayed learning rate.
Eval time is not counted against the budget.

Outputs:
    results/runs/<name>/log.jsonl     one line per eval (time, tokens, train/val loss, RAM)
    results/runs/<name>/summary.json  final numbers
    checkpoints/<name>.pt             final weights (not committed to git)
"""
import argparse
import dataclasses
import json
import math
import os
import time

import numpy as np
import psutil
import torch

from vtransformer import MODELS

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="baseline", choices=list(MODELS))
    p.add_argument("--name", required=True)
    p.add_argument("--minutes", type=float, default=25)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--eval_every_sec", type=float, default=60)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    # Any model-config field can be set with --set key=value (e.g. --set d=384)
    p.add_argument("--set", nargs="*", default=[], help="model config overrides key=value")
    # Shortcuts for common fields
    for f in ("d", "layers", "heads", "ctx"):
        p.add_argument(f"--{f}", type=int, default=None)
    return p.parse_args()


def build_config(args, cfg_cls):
    over = {}
    for f in ("d", "layers", "heads", "ctx"):
        v = getattr(args, f)
        if v is not None:
            over[f] = v
    fields = {f.name: f.type for f in dataclasses.fields(cfg_cls)}
    for kv in args.set:
        k, v = kv.split("=", 1)
        t = fields[k]
        t = {"int": int, "float": float, "bool": bool, "str": str}.get(t, t) if isinstance(t, str) else t
        over[k] = (v.lower() in ("1", "true", "yes")) if t is bool else t(v)
    return cfg_cls(**over)


class Data:
    def __init__(self, split, ctx, batch, seed):
        self.arr = np.memmap(os.path.join(ROOT, "data", f"{split}.bin"), dtype=np.uint16, mode="r")
        self.ctx, self.batch = ctx, batch
        self.rng = np.random.default_rng(seed)

    def get(self):
        ix = self.rng.integers(0, len(self.arr) - self.ctx - 1, self.batch)
        x = np.stack([self.arr[i:i + self.ctx] for i in ix]).astype(np.int64)
        y = np.stack([self.arr[i + 1:i + 1 + self.ctx] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def lr_at(frac, args):
    """Warmup, then cosine decay to min_lr_frac, as a function of elapsed time fraction."""
    if frac < args.warmup_frac:
        return args.lr * frac / args.warmup_frac
    p = min(1.0, (frac - args.warmup_frac) / (1 - args.warmup_frac))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def make_optimizer(model, args):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), fused=False)


def peak_ram_mb():
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**20)


@torch.no_grad()
def evaluate(model, val_batches, autocast):
    model.eval()
    losses = []
    for x, y in val_batches:
        with autocast():
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    cfg_cls, model_cls = MODELS[args.model]
    cfg = build_config(args, cfg_cls)
    model = model_cls(cfg)
    counts = model.param_counts()
    opt = make_optimizer(model, args)

    if args.dtype == "bf16":
        autocast = lambda: torch.autocast("cpu", dtype=torch.bfloat16)  # noqa: E731
    else:
        import contextlib
        autocast = contextlib.nullcontext

    train = Data("train", cfg.ctx, args.batch, args.seed)
    val = Data("val", cfg.ctx, args.batch, 0)
    val_batches = [val.get() for _ in range(args.eval_batches)]  # fixed val set

    out_dir = os.path.join(ROOT, "results", "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    log_path = os.path.join(out_dir, "log.jsonl")
    open(log_path, "w").close()

    budget = args.minutes * 60
    print(f"[{args.name}] {args.model} {dataclasses.asdict(cfg)}")
    print(f"params: total {counts['total']/1e6:.2f}M, active {counts['active']/1e6:.2f}M, "
          f"budget {args.minutes} min, {args.dtype}, {args.threads} threads", flush=True)

    def log(rec):
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    train_time, step, tokens = 0.0, 0, 0
    next_eval = 0.0
    wall_start = time.perf_counter()

    def wall_per_train():
        # Wall-clock seconds per training second (includes eval overhead), for the ETA.
        return (time.perf_counter() - wall_start) / train_time if train_time > 0 else 1.0
    recent = []
    while True:
        if train_time >= next_eval or train_time >= budget:
            vl = evaluate(model, val_batches, autocast)
            log({"step": step, "train_min": round(train_time / 60, 3), "tokens_M": round(tokens / 1e6, 3),
                 "train_loss": round(float(np.mean(recent)), 4) if recent else None,
                 "val_loss": round(vl, 4), "lr": round(lr_at(min(1, train_time / budget), args), 6),
                 "tok_per_sec": round(tokens / train_time) if train_time else 0, "peak_ram_mb": peak_ram_mb(),
                 "eta_min": round(max(0.0, budget - train_time) / 60 * wall_per_train(), 1)})
            recent = []
            next_eval += args.eval_every_sec
            if train_time >= budget:
                break

        t0 = time.perf_counter()
        lr = lr_at(train_time / budget, args)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train.get()
        with autocast():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        recent.append(loss.item())
        train_time += time.perf_counter() - t0
        step += 1
        tokens += x.numel()

    final_val = evaluate(model, val_batches, autocast)
    summary = {
        "name": args.name, "model": args.model, "config": dataclasses.asdict(cfg),
        "params": counts, "minutes": args.minutes, "steps": step, "tokens": tokens,
        "tok_per_sec": round(tokens / train_time), "final_val_loss": round(final_val, 4),
        "peak_ram_mb": peak_ram_mb(), "dtype": args.dtype, "threads": args.threads,
        "batch": args.batch, "lr": args.lr,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"model": args.model, "config": dataclasses.asdict(cfg), "state": model.state_dict()},
               os.path.join(ROOT, "checkpoints", f"{args.name}.pt"))
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
