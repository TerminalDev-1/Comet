"""Copy check for latent models: are generated images new, or copies of training images?

    python copy_check_latent.py <ema.pt> --data <latents.npy> --out <png> [--n 8 --steps 30]

Samples n images, finds the nearest training latent for each (L2 distance in latent space), and saves a
grid: top row generated, bottom row its nearest training image. Prints the distances next to a reference:
how far two crops of *different* training images typically are from each other's nearest neighbour.
A generated image much closer to a training image than that reference would be a copy.
"""
import argparse
import json

import numpy as np
import torch
from diffusers import AutoencoderKL

from dit import sample
from generate_flash import load_model
from train_latent import VAE_ID, decode, save_grid


@torch.no_grad()
def nearest(q, data, skip_self=False):
    best_d = torch.full((q.shape[0],), float("inf"), device=q.device)
    best_i = torch.zeros(q.shape[0], dtype=torch.long, device=q.device)
    qf = q.flatten(1)
    for s in range(0, data.shape[0], 4096):
        chunk = data[s:s + 4096].float().flatten(1)
        d = torch.cdist(qf, chunk)
        if skip_self:  # ignore the 6 crops of the same source image (stored next to each other)
            for r in range(q.shape[0]):
                lo = (skip_self[r] // 6) * 6 - s
                d[r, max(lo, 0):max(lo + 6, 0)] = float("inf")
        m, i = d.min(1)
        upd = m < best_d
        best_d[upd], best_i[upd] = m[upd], i[upd] + s
    return best_d, best_i


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    dev = "cuda"
    model, _ = load_model(a.weights, dev)
    data = torch.from_numpy(np.load(a.data)).to(dev)
    z = sample(model, a.n, steps=a.steps, seed=a.seed)
    gd, gi = nearest(z, data)
    ridx = torch.randint(0, data.shape[0], (64,), generator=torch.Generator().manual_seed(0)).tolist()
    rd, _ = nearest(data[ridx].float(), data, skip_self=ridx)
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16).to(dev).eval()
    save_grid(np.concatenate([decode(vae, z), decode(vae, data[gi].float())]), a.out, cols=a.n)
    print(json.dumps({"generated_to_nearest_train": [round(x, 1) for x in gd.tolist()],
                      "generated_min": round(gd.min().item(), 1), "generated_median": round(gd.median().item(), 1),
                      "train_to_nearest_other_image_median": round(rd.median().item(), 1),
                      "train_to_nearest_other_image_min": round(rd.min().item(), 1)}))


if __name__ == "__main__":
    main()
