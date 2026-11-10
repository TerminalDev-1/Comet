"""Train a latent Diffusion Transformer (flow matching) on SD VAE latents from encode_latents.py.

    python train_latent.py --data <latents.npy> --out <runs dir> --name Comet-Image-v4-Flash-2026-11-08 --minutes 15

Latent diffusion: the DiT never sees pixels. A 256x256 image is a 4x32x32 latent, i.e. 256 tokens with
patch 2, so training is about as cheap as the old 64x64 pixel models while producing 256x256 images.
Sample grids use --flash_steps Euler steps (default 30; distill_flash.py makes 8-step models). Checkpoints every --ckpt_every
minutes; a file named STOP in the run folder ends training cleanly. --init <ema.pt> continues training from
an earlier run's weights (Preview models keep training this way).
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image

from dit import DiT, flow_loss, sample

SCALE = 0.18215
VAE_ID = "stabilityai/sd-vae-ft-mse"


@torch.no_grad()
def decode(vae, z):
    x = vae.decode((z / SCALE).half()).sample.float()
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def save_grid(imgs, path, cols=4):
    n, h, w, _ = imgs.shape
    rows = (n + cols - 1) // cols
    g = np.full((rows * (h + 4) + 4, cols * (w + 4) + 4, 3), 20, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        g[4 + r * (h + 4):4 + r * (h + 4) + h, 4 + c * (w + 4):4 + c * (w + 4) + w] = im
    Image.fromarray(g).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--minutes", type=float, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--grid_every", type=float, default=5)
    p.add_argument("--grid_n", type=int, default=16)
    p.add_argument("--flash_steps", type=int, default=30)
    p.add_argument("--ckpt_every", type=float, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_data", default=None,
                   help="held-out latents (never trained on, from encode_latents.py --holdout): also report their loss")
    p.add_argument("--init", default=None, help="ema.pt of an earlier run to continue from")
    p.add_argument("--t_shift", type=float, default=0.0, help=">0 trains more on nearly-clean images (fine detail)")
    p.add_argument("--grow_layers", type=int, default=0,
                   help="with --init: deepen the model to this many blocks. Each trained block is followed by a new one "
                        "whose adaLN-Zero gates start at zero, so the grown model starts out exactly like the old one")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    data = torch.from_numpy(np.load(a.data)).to(dev)  # float16 (N, 4, S, S), small enough for the GPU
    N, C, S = data.shape[0], data.shape[1], data.shape[2]
    if a.init:
        ck = torch.load(a.init, map_location=dev)
        old_layers = 1 + max(int(k.split(".")[1]) for k in ck["state"] if k.startswith("blocks."))
        state = ck["state"]
        if a.grow_layers:
            assert a.grow_layers % old_layers == 0, "--grow_layers must be a multiple of the old block count"
            every = a.grow_layers // old_layers  # old block i goes to slot i*every; the slots after it are new
            a.layers = a.grow_layers
            state = {(f"blocks.{int(k.split('.')[1]) * every}." + k.split(".", 2)[2]) if k.startswith("blocks.") else k: v
                     for k, v in state.items()}
        else:
            a.layers = old_layers
    model = DiT(img=S, patch=a.patch, d=a.d, layers=a.layers, heads=a.heads, in_ch=C).to(dev)
    if a.init:
        missing, unexpected = model.load_state_dict(state, strict=not a.grow_layers)
        assert not unexpected, unexpected
        grown = f", grown {old_layers} -> {a.layers} blocks" if a.grow_layers else ""
        print(f"continuing from {a.init} (step {ck['step']}, {ck['train_min']} min){grown}", flush=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16).to(dev).eval()
    params = sum(x.numel() for x in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    print(f"[{a.name}] {N:,} latents {C}x{S}x{S} (= {S * 8}x{S * 8} images), latent DiT {params / 1e6:.1f}M params, "
          f"{a.minutes} min", flush=True)

    budget, t_train, step, seen = a.minutes * 60, 0.0, 0, 0
    next_log, next_grid, next_ckpt, losses = 60.0, a.grid_every * 60, a.ckpt_every * 60, []
    stop_file = os.path.join(out, "STOP")

    def save_ckpt():
        torch.save({"config": vars(a), "img": S, "in_ch": C, "step": step, "train_min": round(t_train / 60, 2),
                    "state": ema.state_dict()}, os.path.join(out, "ema.pt"))

    # Fixed eval: same 1,024 latents, noise and times every call (t_shift 0), so runs with different
    # --t_shift settings stay comparable. Measured on the EMA model.
    eg = torch.Generator(device=dev).manual_seed(123)
    e_idx = torch.randperm(N, generator=eg, device=dev)[:1024]
    e_noise = torch.randn(1024, C, S, S, generator=eg, device=dev)
    e_t = torch.sigmoid(torch.randn(1024, generator=eg, device=dev))

    held = torch.from_numpy(np.load(a.eval_data)).to(dev)[:1024] if a.eval_data else None

    @torch.no_grad()
    def loss_on(src, n):
        tot = 0.0
        for i in range(0, n, 256):
            j = min(i + 256, n)
            x1, x0, t = src(i, j).float(), e_noise[i:j], e_t[i:j]
            xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = ema(xt, t).float()
            tot += torch.nn.functional.mse_loss(v, x1 - x0).item() * (j - i)
        return round(tot / n, 4)

    def eval_loss():
        """Loss on 1,024 fixed training latents; with --eval_data, a dict that adds the held-out loss
        (same noise and times). A held-out loss that rises while the training one falls means overfitting."""
        tr = loss_on(lambda i, j: data[e_idx[i:j]], 1024)
        if held is None:
            return tr
        return {"train": tr, "heldout": loss_on(lambda i, j: held[i:j], len(held))}

    print(json.dumps({"min": 0.0, "eval_loss": eval_loss()}), flush=True)

    def grid(tag, steps):
        z = sample(ema, a.grid_n, steps=steps, seed=1)
        save_grid(decode(vae, z), os.path.join(out, f"grid_{tag}.png"))

    while t_train < budget:
        t0 = time.perf_counter()
        frac = t_train / budget
        lr = a.lr * min(1.0, step / max(1, a.warmup)) * (1.0 if frac < 0.8 else 1 - (frac - 0.8) / 0.2 * 0.9)
        for g in opt.param_groups:
            g["lr"] = lr
        x = data[torch.randint(0, N, (a.batch,), device=dev)].float()
        flip = torch.rand(a.batch, device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = flow_loss(model, x, t_shift=a.t_shift)
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
            print(json.dumps({"min": round(t_train / 60, 2), "step": step, "loss": round(float(np.mean(losses)), 4),
                              "img_per_sec": round(seen / t_train), "vram_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                              "train_min_left": round((budget - t_train) / 60, 1)}), flush=True)
            losses, next_log = [], next_log + 60
        if t_train >= next_ckpt:
            save_ckpt()
            next_ckpt += a.ckpt_every * 60
        if os.path.exists(stop_file):
            print(f"STOP file found at {t_train / 60:.1f} min: finishing now", flush=True)
            break
        if t_train >= next_grid:
            grid(f"{t_train / 60:04.1f}min_{a.flash_steps}steps", a.flash_steps)
            print(f"GRID {t_train / 60:.1f} min", json.dumps({"eval_loss": eval_loss()}), flush=True)
            next_grid += a.grid_every * 60

    save_ckpt()
    t0 = time.perf_counter()
    grid(f"final_{a.flash_steps}steps", a.flash_steps)
    flash_sec = time.perf_counter() - t0
    grid("final_30steps", 30)
    summary = {"name": a.name, "eval_loss": eval_loss(), "params": params, "latents": N, "image_size": S * 8, "steps": step,
               "train_minutes": round(t_train / 60, 2), "img_per_sec": round(seen / t_train),
               "flash_steps": a.flash_steps, "sec_per_16_images_flash": round(flash_sec, 2)}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
