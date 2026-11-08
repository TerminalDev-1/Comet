"""Download the Microcosmos dataset (CC0 general images, Pixel-Dust/Microcosmos on Hugging Face).

    python fetch_microcosmos.py --out <folder>

Only the image files are downloaded (jpg/png), not the precomputed .npz/.zip extras.
"""
import argparse
import os

from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    path = snapshot_download("Pixel-Dust/Microcosmos", repo_type="dataset", local_dir=a.out,
                             allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp", "README.md"], max_workers=8)
    n, size = 0, 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                n += 1
                size += os.path.getsize(os.path.join(root, f))
    print(f"DONE: {n} images, {size / 2**30:.2f} GB in {path}")


if __name__ == "__main__":
    main()
