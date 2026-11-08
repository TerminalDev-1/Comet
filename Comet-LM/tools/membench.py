"""Peak training RAM and speed for one model setup, measured cleanly.

Each setup runs in its own fresh process, on random tokens (no data file pages in the number),
for a fixed number of real training steps (forward + backward + Muon/AdamW step).

    python tools/membench.py                         # dense vs MoD, side by side
    python tools/membench.py --one --set mod=true    # a single setup (used internally)

Reports: peak RAM above the idle process, steady-state tokens/sec, and RAM per million parameters.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def one(args):
    import psutil
    import torch

    from comet import MODELS
    from comet.optim import build_optimizer
    from train import build_config

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cfg_cls, model_cls = MODELS["comet"]
    ns = argparse.Namespace(d=None, layers=None, heads=None, ctx=None, set=args.set)
    cfg = build_config(ns, cfg_cls)
    proc = psutil.Process()
    base = proc.memory_info().rss
    model = model_cls(cfg)
    opt = build_optimizer(model, "muon", 0.003, 0.1, 0.02, "adamw")
    params = sum(p.numel() for p in model.parameters())
    gen = torch.Generator().manual_seed(0)
    times = []
    for step in range(args.steps):
        x = torch.randint(0, cfg.vocab, (args.batch, cfg.ctx), generator=gen)
        y = torch.randint(0, cfg.vocab, (args.batch, cfg.ctx), generator=gen)
        t0 = time.perf_counter()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
    mi = proc.memory_info()
    peak = getattr(mi, "peak_wset", mi.rss)
    steady = sorted(times[args.steps // 3:])[len(times[args.steps // 3:]) // 2]  # median, skip warmup
    print(json.dumps({
        "set": args.set, "params_M": round(params / 1e6, 2),
        "peak_ram_mb": round((peak - base) / 2**20), "process_peak_mb": round(peak / 2**20),
        "tok_per_sec": round(args.batch * cfg.ctx / steady),
        "mb_per_M_params": round((peak - base) / 2**20 / (params / 1e6), 1),
    }))


SETUPS = {
    "dense": [],
    "mod_exact": ["mod=true", "mod_bucket=0"],
    "mod": ["mod=true"],
    "mod_b32": ["mod=true", "mod_bucket=32"],
    "mod_b16": ["mod=true", "mod_bucket=16"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--one", action="store_true")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--setups", nargs="*", default=list(SETUPS))
    args = p.parse_args()
    if args.one:
        return one(args)
    for name in args.setups:
        cmd = [sys.executable, os.path.abspath(__file__), "--one", "--steps", str(args.steps),
               "--batch", str(args.batch), "--threads", str(args.threads), "--set", *SETUPS[name]]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        if out.returncode:
            print(name, "FAILED\n", out.stderr[-2000:])
            continue
        r = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"{name:12s} params {r['params_M']:5.2f}M  peak RAM +{r['peak_ram_mb']:5d} MB  "
              f"({r['mb_per_M_params']:6.1f} MB per M params)  {r['tok_per_sec']:6d} tok/s", flush=True)


if __name__ == "__main__":
    main()
