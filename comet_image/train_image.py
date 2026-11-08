"""Train Comet-Image (small DiT, flow matching) on a crop dataset made by prep_images.py.

    python comet_image/train_image.py --data <crops.npy> --out <private run dir> --name <run> --minutes 20

Logs one line per minute (loss, images/sec, ETA), saves an 8x8 sample grid every --grid_every minutes,
and at the end: EMA weights, a final grid, and a "nearest training crop" grid to check for copying.
Outputs go to --out, which should be OUTSIDE the git repo (the data is private).
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from dit import DiT, flow_loss, sample


def to_img(x):  # (B,3,H,W) in [-1,1] -> uint8 HWC
    return ((x.float() + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()


def save_grid(imgs, path, cols=8, scale=2):
    n, h, w, _ = imgs.shape
    rows = (n + cols - 1) // cols
    g = np.full((rows * (h + 2) + 2, cols * (w + 2) + 2, 3), 20, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        g[2 + r * (h + 2):2 + r * (h + 2) + h, 2 + c * (w + 2):2 + c * (w + 2) + w] = im
    Image.fromarray(g).resize((g.shape[1] * scale, g.shape[0] * scale), Image.NEAREST).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--patch", type=int, default=4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grid_every", type=float, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="continue from a previous run's ema.pt (same model size)")
    p.add_argument("--warmup", type=int, default=500, help="LR warmup steps")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    data = torch.from_numpy(np.load(a.data)).to(dev)  # uint8 (N,H,W,3), lives on the GPU
    N, H = data.shape[0], data.shape[1]
    model = DiT(img=H, patch=a.patch, d=a.d, layers=a.layers, heads=a.heads).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev)["state"])
        print(f"continuing from {a.init}", flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    print(f"[{a.name}] {N:,} crops {H}x{H}, DiT {params / 1e6:.1f}M params, {a.minutes} min", flush=True)

    budget, t_train, step, seen = a.minutes * 60, 0.0, 0, 0
    next_log, next_grid, losses = 60.0, a.grid_every * 60, []
    wall0 = time.perf_counter()
    while t_train < budget:
        t0 = time.perf_counter()
        frac = t_train / budget
        lr = a.lr * min(1.0, step / max(1, a.warmup)) * (1.0 if frac < 0.8 else 1 - (frac - 0.8) / 0.2 * 0.9)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, N, (a.batch,), device=dev)
        x = data[idx].permute(0, 3, 1, 2).float() / 127.5 - 1
        flip = torch.rand(a.batch, device=dev) < 0.5  # mirror half the batch: more variety
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        step += 1
        seen += a.batch
        losses.append(loss.item())
        if t_train >= next_log:
            wall_per = (time.perf_counter() - wall0) / t_train
            print(json.dumps({"min": round(t_train / 60, 2), "step": step, "loss": round(float(np.mean(losses)), 4),
                              "img_per_sec": round(seen / t_train), "vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                              "eta_min": round((budget - t_train) / 60 * wall_per, 1)}), flush=True)
            losses, next_log = [], next_log + 60
        if t_train >= next_grid:
            save_grid(to_img(sample(ema, 64, seed=1)), os.path.join(out, f"grid_{t_train / 60:04.1f}min.png"))
            print(f"GRID {t_train / 60:.1f} min", flush=True)
            next_grid += a.grid_every * 60

    final = to_img(sample(ema, 64, steps=60, seed=1))
    save_grid(final, os.path.join(out, "grid_final.png"))
    # Copying check: for each of the first 16 samples, its nearest training crop (pixel distance)
    s = torch.from_numpy(final[:16]).to(dev).float().flatten(1)
    best = torch.full((16,), float("inf"), device=dev)
    arg = torch.zeros(16, dtype=torch.long, device=dev)
    for i in range(0, N, 8192):
        chunk = data[i:i + 8192].float().flatten(1)
        dist = torch.cdist(s, chunk)
        m, j = dist.min(1)
        upd = m < best
        best, arg = torch.where(upd, m, best), torch.where(upd, j + i, arg)
    pairs = np.concatenate([np.stack([final[k], data[arg[k]].cpu().numpy()]) for k in range(16)])
    save_grid(pairs, os.path.join(out, "nearest_pairs.png"), cols=8)
    rms = (best / (H * H * 3) ** 0.5).cpu().numpy()
    torch.save({"config": vars(a), "img": H, "state": ema.state_dict()}, os.path.join(out, "ema.pt"))
    summary = {"name": a.name, "params": params, "crops": N, "steps": step, "minutes": a.minutes,
               "img_per_sec": round(seen / t_train), "nearest_rms_pixel_dist": [round(float(v), 1) for v in rms]}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
