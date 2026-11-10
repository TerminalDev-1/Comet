"""Remove duplicate and near-duplicate latents from a training set (and anything that duplicates a held-out set).

    python dedup_latents.py --data <latents.npy> --out <dedup.npy> [--holdout <heldout.npy> ...] [--thresh 5]
    python dedup_latents.py --data <latents.npy> --report        # only print how many there are

Compares every latent with every other one on the GPU (L2 distance over all 4x32x32 values). For reference, a latent's
nearest *different* image is typically 45-55 away; the same picture saved twice (e.g. a NASA image listed under two
IDs) is ~0. Pairs closer than --thresh are duplicates: the first copy is kept, later ones are dropped. Training latents
closer than --thresh to any --holdout latent are dropped too, so the held-out check stays honest.
"""
import argparse

import numpy as np
import torch


def nearest_earlier(x, chunk=1024):
    """For each row i: distance to the closest row j < i (inf for row 0)."""
    n = x.shape[0]
    sq = (x * x).sum(1)
    best = torch.full((n,), float("inf"), device=x.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = sq[s:e, None] + sq[None, :e] - 2 * x[s:e] @ x[:e].T  # only compare with rows up to e
        rows = torch.arange(s, e, device=x.device)[:, None]
        d2.masked_fill_(torch.arange(e, device=x.device)[None, :] >= rows, float("inf"))  # j < i only
        best[s:e] = d2.min(1).values.clamp_min(0).sqrt()
    return best


def nearest_in(x, y, chunk=1024):
    """For each row of x: distance to the closest row of y."""
    sy = (y * y).sum(1)
    out = []
    for s in range(0, x.shape[0], chunk):
        xs = x[s:s + chunk]
        d2 = (xs * xs).sum(1)[:, None] + sy[None, :] - 2 * xs @ y.T
        out.append(d2.min(1).values.clamp_min(0).sqrt())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--holdout", nargs="*", default=[])
    p.add_argument("--thresh", type=float, default=5.0)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    raw = np.load(a.data)
    x = torch.from_numpy(raw).cuda().flatten(1).float()
    n = x.shape[0]
    d = nearest_earlier(x)
    for t in (0.5, 2, 5, 10, 20):
        print(f"  closer than {t:>4} to an earlier latent: {(d < t).sum().item():,}", flush=True)
    drop = d < a.thresh
    print(f"{a.data}: {n:,} latents, {drop.sum().item():,} duplicates (< {a.thresh})", flush=True)
    for h in a.holdout:
        hx = torch.from_numpy(np.load(h)).cuda().flatten(1).float()
        leak = nearest_in(x, hx) < a.thresh
        print(f"  {leak.sum().item():,} training latents duplicate a held-out latent in {h}", flush=True)
        drop |= leak
    if a.report or not a.out:
        return
    keep = (~drop).cpu().numpy()
    np.save(a.out, raw[keep])
    print(f"DONE: {a.out} {keep.sum():,} latents (dropped {n - keep.sum():,})", flush=True)


if __name__ == "__main__":
    main()
