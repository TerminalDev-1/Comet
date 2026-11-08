"""Generate single images from a Comet-Image checkpoint (one PNG per image, no grid).

    python comet_image/generate.py <run_dir>/ema.pt --out <folder> [--n 1 --seed 42 --steps 60 --scale 4]

Each image is saved on its own, upscaled `--scale`x (64 -> 256 px by default) so it's easy to look at.
"""
import argparse
import os

import torch
from PIL import Image

from dit import DiT, sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--smooth", action="store_true", help="smooth (bicubic) upscaling instead of sharp pixels")
    p.add_argument("--vivid", type=float, default=1.0, help="color saturation multiplier, e.g. 1.3 (1 = as generated)")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    c = ck["config"]
    model = DiT(img=ck["img"], patch=c["patch"], d=c["d"], layers=c["layers"], heads=c["heads"]).to(dev).eval()
    model.load_state_dict(ck["state"])
    os.makedirs(a.out, exist_ok=True)
    x = sample(model, a.n, steps=a.steps, device=dev, seed=a.seed)
    imgs = ((x.float() + 1) * 127.5).round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    name = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    for i, im in enumerate(imgs):
        img = Image.fromarray(im)
        if a.vivid != 1.0:
            from PIL import ImageEnhance
            img = ImageEnhance.Color(img).enhance(a.vivid)
        size = (img.width * a.scale, img.height * a.scale)
        img = img.resize(size, Image.BICUBIC if a.smooth else Image.NEAREST)
        path = os.path.join(a.out, f"{name}_seed{a.seed}_{i + 1}.png" if a.n > 1 else f"{name}_seed{a.seed}.png")
        img.save(path)
        print("saved", path)


if __name__ == "__main__":
    main()
