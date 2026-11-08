"""Train any model in comet.MODELS under a wall-clock time budget.

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

from comet import MODELS
from comet.optim import build_optimizer

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="baseline", choices=list(MODELS))
    p.add_argument("--name", required=True)
    p.add_argument("--minutes", type=float, default=25)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "wsd"])
    p.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    p.add_argument("--muon_lr", type=float, default=0.02)
    p.add_argument("--expert_optim", default="muon", choices=["muon", "adamw"], help="optimizer for EMOE expert weights")
    p.add_argument("--decay_frac", type=float, default=0.25, help="wsd: fraction of time spent decaying")
    p.add_argument("--warmup_frac", type=float, default=0.03)
    p.add_argument("--ctx_warmup_frac", type=float, default=0.0,
                   help="short-context warmup: grow context from --ctx_start to full over this fraction of time "
                        "(batch grows to keep tokens per step equal)")
    p.add_argument("--ctx_start", type=int, default=64)
    p.add_argument("--ns_steps", type=int, default=5, help="Muon Newton-Schulz iterations")
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

    def get(self, ctx=None, batch=None):
        ctx, batch = ctx or self.ctx, batch or self.batch
        ix = self.rng.integers(0, len(self.arr) - ctx - 1, batch)
        x = np.stack([self.arr[i:i + ctx] for i in ix]).astype(np.int64)
        y = np.stack([self.arr[i + 1:i + 1 + ctx] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def lr_at(frac, args):
    """LR as a function of elapsed time fraction.
    cosine: warmup, then cosine decay to min_lr_frac.
    wsd:    warmup, flat (stable) LR, then linear decay to min_lr_frac over the last decay_frac.
    """
    if frac < args.warmup_frac:
        return args.lr * frac / args.warmup_frac
    if args.schedule == "wsd":
        start = 1 - args.decay_frac
        if frac < start:
            return args.lr
        p = min(1.0, (frac - start) / args.decay_frac)
        return args.lr * (1 - (1 - args.min_lr_frac) * p)
    p = min(1.0, (frac - args.warmup_frac) / (1 - args.warmup_frac))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def make_optimizer(model, args):
    return build_optimizer(model, args.optim, args.lr, args.weight_decay, args.muon_lr, args.expert_optim,
                           ns_steps=args.ns_steps)


def ctx_batch_at(frac, args, full_ctx):
    """Short-context warmup: context grows in steps of ctx_start; tokens per step stay constant."""
    if frac >= args.ctx_warmup_frac:
        return full_ctx, args.batch
    c = args.ctx_start + (full_ctx - args.ctx_start) * frac / args.ctx_warmup_frac
    c = max(args.ctx_start, int(c) // args.ctx_start * args.ctx_start)
    return c, args.batch * full_ctx // c


def peak_ram_mb():
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2**20)


@torch.no_grad()
def evaluate(model, val_batches, autocast):
    model.eval()
    losses = []
    correct, total = 0, 0
    for x, y in val_batches:
        with autocast():
            logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    return float(np.mean(losses)), correct / total  # loss, next-token accuracy


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
            vl, va = evaluate(model, val_batches, autocast)
            log({"step": step, "train_min": round(train_time / 60, 3), "tokens_M": round(tokens / 1e6, 3),
                 "train_loss": round(float(np.mean(recent)), 4) if recent else None,
                 "val_loss": round(vl, 4), "val_acc": round(va, 4), "lr": round(lr_at(min(1, train_time / budget), args), 6),
                 "tok_per_sec": round(tokens / train_time) if train_time else 0, "peak_ram_mb": peak_ram_mb(),
                 "eta_min": round(max(0.0, budget - train_time) / 60 * wall_per_train(), 1),
                 "mod_frac": round(model.mod_frac, 3) if getattr(model, "mod_frac", None) is not None else None})
            recent = []
            next_eval += args.eval_every_sec
            if train_time >= budget:
                break

        t0 = time.perf_counter()
        mult = lr_at(train_time / budget, args) / args.lr  # schedule multiplier
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        x, y = train.get(*ctx_batch_at(train_time / budget, args, cfg.ctx))
        with autocast():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        main = getattr(model, "last_main_loss", None)  # log next-token loss only (not MTP extras)
        recent.append((main if main is not None else loss).item())
        train_time += time.perf_counter() - t0
        step += 1
        tokens += x.numel()

    final_val, final_acc = evaluate(model, val_batches, autocast)
    summary = {
        "name": args.name, "model": args.model, "config": dataclasses.asdict(cfg),
        "params": counts, "minutes": args.minutes, "steps": step, "tokens": tokens,
        "tok_per_sec": round(tokens / train_time), "final_val_loss": round(final_val, 4),
        "final_val_acc": round(final_acc, 4),
        "peak_ram_mb": peak_ram_mb(), "dtype": args.dtype, "threads": args.threads,
        "batch": args.batch, "lr": args.lr, "schedule": args.schedule,
        "ctx_warmup_frac": args.ctx_warmup_frac, "ctx_start": args.ctx_start, "ns_steps": args.ns_steps,
        "optim": args.optim, "expert_optim": args.expert_optim, "muon_lr": args.muon_lr if args.optim == "muon" else None,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    torch.save({"model": args.model, "config": dataclasses.asdict(cfg), "state": model.state_dict()},
               os.path.join(ROOT, "checkpoints", f"{args.name}.pt"))
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
