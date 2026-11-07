"""Phase 0 hardware benchmark: matrix-multiply speed on CPU (and DirectML if present).

Usage:  python tools/hwbench.py [--device cpu|dml] [--out results/hw_cpu.json]
"""
import argparse
import json
import os
import platform
import time

import torch


def bench_matmul(m, k, n, dtype, device, iters=None, min_time=0.6):
    a = torch.randn(m, k, device=device).to(dtype)
    b = torch.randn(k, n, device=device).to(dtype)
    # warmup
    for _ in range(3):
        c = a @ b
    _sync(c)
    flops = 2 * m * k * n
    if iters is None:
        t0 = time.perf_counter()
        c = a @ b
        _sync(c)
        one = max(time.perf_counter() - t0, 1e-5)
        iters = max(3, int(min_time / one))
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    _sync(c)
    dt = time.perf_counter() - t0
    return flops * iters / dt / 1e9  # GFLOPS


def _sync(t):
    # Pull one value back to force the device to finish.
    t.reshape(-1)[0].item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.device == "dml":
        import torch_directml
        device = torch_directml.device()
        thread_options = [None]
    else:
        device = torch.device("cpu")
        thread_options = [4, 6, 12]

    if args.device == "dml":
        # DirectML hard-crashes on BF16, so test FP16 instead.
        dtypes = {"fp32": torch.float32, "fp16": torch.float16}
    else:
        dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16}

    results = {
        "device": args.device,
        "cpu": platform.processor(),
        "torch": torch.__version__,
        "big_matmul": {},     # 2048^3: best-case speed
        "model_matmul": {},   # shapes like a small transformer layer
        "expert_matmul": {},  # EMOE tiny-expert shapes
    }

    for threads in thread_options:
        if threads:
            torch.set_num_threads(threads)
        tag = f"threads={threads}" if threads else "gpu"
        for dname, dt in dtypes.items():
            try:
                g = bench_matmul(2048, 2048, 2048, dt, device)
            except Exception as e:  # dtype not supported on device
                g = f"error: {e.__class__.__name__}"
            results["big_matmul"][f"{tag} {dname}"] = g
            print(f"[big 2048^3]      {tag:10s} {dname}: {g if isinstance(g, str) else f'{g:8.1f} GFLOPS'}")

    # Pick best thread count for the remaining tests.
    if args.device != "dml":
        best = max(
            (k for k, v in results["big_matmul"].items() if "bf16" in k and not isinstance(v, str)),
            key=lambda k: results["big_matmul"][k],
        )
        best_threads = int(best.split()[0].split("=")[1])
        torch.set_num_threads(best_threads)
        results["best_threads"] = best_threads
        print(f"best threads for bf16: {best_threads}")

    # Small-transformer shapes: tokens x d_model x hidden
    tokens = 8192
    for d in (256, 384, 512):
        for dname, dt in dtypes.items():
            try:
                g = bench_matmul(tokens, d, 4 * d, dt, device)
            except Exception as e:
                g = f"error: {e.__class__.__name__}"
            results["model_matmul"][f"d={d} {dname}"] = g
            print(f"[model d={d}]     {dname}: {g if isinstance(g, str) else f'{g:8.1f} GFLOPS'}")

    # EMOE tiny experts: each expert gets ~tokens*k/E tokens, d_model=384
    for h in (64, 128, 256, 512):
        for dname, dt in dtypes.items():
            try:
                g = bench_matmul(1024, 384, h, dt, device)
            except Exception as e:
                g = f"error: {e.__class__.__name__}"
            results["expert_matmul"][f"hidden={h} {dname}"] = g
            print(f"[expert h={h}]    {dname}: {g if isinstance(g, str) else f'{g:8.1f} GFLOPS'}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
