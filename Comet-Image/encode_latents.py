"""Encode images into Stable Diffusion VAE latents (for latent-diffusion training, e.g. Comet-Image-v4-Flash).

    python encode_latents.py --src <dir> [<dir> ...] --out <latents.npy> [--size 256 --crops 5 --min_frac 0.6]

Each image gives one centered view plus `--crops` random square crops (from --min_frac of the short side up
to all of it), resized to size x size, then encoded by the VAE (stabilityai/sd-vae-ft-mse, downloaded on
first use). Saves float16 latents (N, 4, size/8, size/8), already multiplied by the SD scaling factor 0.18215.
"""
import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image

SCALE = 0.18215
VAE_ID = "stabilityai/sd-vae-ft-mse"


def views(path, size, crops, min_frac, rng):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return []
    W, H = im.size
    short = min(W, H)
    if short < 64:
        return []
    out = []
    x0, y0 = (W - short) // 2, (H - short) // 2
    out.append(im.crop((x0, y0, x0 + short, y0 + short)).resize((size, size), Image.BICUBIC))
    lo = min(max(64, int(short * min_frac)), short)
    for _ in range(crops):
        s = int(rng.integers(lo, short + 1))
        x, y = int(rng.integers(0, W - s + 1)), int(rng.integers(0, H - s + 1))
        out.append(im.crop((x, y, x + s, y + s)).resize((size, size), Image.BICUBIC))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--crops", type=int, default=5)
    p.add_argument("--min_frac", type=float, default=0.6)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8, help="threads that load and crop images")
    a = p.parse_args()
    dev = "cuda"
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16).to(dev).eval()
    print(f"VAE loaded: {VAE_ID}, {sum(x.numel() for x in vae.parameters()) / 1e6:.0f}M params", flush=True)
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = sorted(f for d in a.src for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                   if f.lower().endswith(exts))
    print(f"{len(files)} images -> up to {len(files) * (a.crops + 1):,} latents at {a.size // 8}x{a.size // 8}", flush=True)
    lat, buf = [], []

    def flush():
        x = torch.from_numpy(np.stack([np.asarray(v) for v in buf])).to(dev).permute(0, 3, 1, 2).half() / 127.5 - 1
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mean * SCALE
        lat.append(z.cpu().numpy().astype(np.float16))
        buf.clear()

    # Images are loaded and cropped by a thread pool (one seeded RNG per image), in chunks so memory stays small.
    load = lambda i: views(files[i], a.size, a.crops, a.min_frac, np.random.default_rng(i))
    chunk = 1000
    with ThreadPoolExecutor(a.workers) as ex:
        for c in range(0, len(files), chunk):
            for vs in ex.map(load, range(c, min(c + chunk, len(files)))):
                buf += vs
                while len(buf) >= a.batch:
                    rest = buf[a.batch:]
                    del buf[a.batch:]
                    flush()
                    buf += rest
            print(f"  {min(c + chunk, len(files))}/{len(files)} images, {sum(len(l) for l in lat):,} latents", flush=True)
    if buf:
        flush()
    arr = np.concatenate(lat)
    np.save(a.out, arr)
    print(f"DONE: {a.out} {arr.shape}, {arr.nbytes / 2**20:.0f} MB, latent std {arr.astype(np.float32).std():.3f}")


if __name__ == "__main__":
    main()
