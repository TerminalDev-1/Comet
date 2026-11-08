"""Turn a folder of screenshots into a small-image training set of random crops.

    python comet_image/prep_images.py --src <dir> [<dir> ...] --out <file.npy> [--size 64 --crops 32]

Each image gives `--crops` random square crops (from --min_frac of the short side up to all of it),
plus one centered full view, all resized to size x size. Reads png/jpg/jpeg/webp, recursively. Output: uint8 array (N, size, size, 3).
Keep the source folders and the output OUT of the git repo (they're private).
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

ARGS = None


def crops_of(path):
    size, n = ARGS.size, ARGS.crops
    rng = np.random.default_rng(abs(hash(os.path.basename(path))) % 2**32)
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return np.zeros((0, size, size, 3), np.uint8)
    W, H = im.size
    short = min(W, H)
    if short < 32:  # tiny snips (a few pixels tall) can't make useful crops
        return np.zeros((0, size, size, 3), np.uint8)
    out = []
    # one centered full view (square of the short side)
    x0, y0 = (W - short) // 2, (H - short) // 2
    out.append(np.asarray(im.crop((x0, y0, x0 + short, y0 + short)).resize((size, size), Image.BOX)))
    lo = min(max(size, int(short * ARGS.min_frac)), short)  # small screenshots: crops can't exceed the short side
    for _ in range(n):
        s = int(rng.integers(lo, short + 1))
        x, y = int(rng.integers(0, W - s + 1)), int(rng.integers(0, H - s + 1))
        out.append(np.asarray(im.crop((x, y, x + s, y + s)).resize((size, size), Image.BOX)))
    return np.stack(out)


def init(a):
    global ARGS
    ARGS = a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--crops", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min_frac", type=float, default=0.12,
                   help="smallest crop as a fraction of the short side (0.12 = details for screenshots; ~0.6 for photos)")
    a = p.parse_args()
    exts = (".png", ".jpg", ".jpeg", ".webp")
    files = sorted(f for d in a.src for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                   if f.lower().endswith(exts))
    print(f"{len(files)} images -> {len(files) * (a.crops + 1):,} crops at {a.size}x{a.size}", flush=True)
    with Pool(a.workers, initializer=init, initargs=(a,)) as pool:
        parts = pool.map(crops_of, files, chunksize=8)
    arr = np.concatenate(parts)
    np.save(a.out, arr)
    print(f"saved {a.out}: {arr.shape}, {arr.nbytes / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
