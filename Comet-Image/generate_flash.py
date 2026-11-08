"""Generate images with a Comet-Image latent model (e.g. Comet-Image-v4-Flash).

    python generate_flash.py <weights.safetensors or ema.pt> --out <folder> [--n 4 --seed 7 --steps 8 --scale 1]

Loads the latent DiT, samples latents in --steps Euler steps (8 = Flash; 30 = slower, sharper), decodes them
with Stable Diffusion's VAE (stabilityai/sd-vae-ft-mse, downloaded on first use) and saves one PNG per image.
For .safetensors weights, the matching <name>.json config must sit next to the file.

Export a training checkpoint for sharing:
    python generate_flash.py <run>/ema.pt --export <name>.safetensors
"""
import argparse
import json
import os

import torch
from PIL import Image

from dit import DiT, sample

SCALE = 0.18215
VAE_ID = "stabilityai/sd-vae-ft-mse"
ARCH_KEYS = ("img", "in_ch", "patch", "d", "layers", "heads")


def load_model(path, dev):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        cfg = json.load(open(os.path.splitext(path)[0] + ".json"))
        state = {k: v.float() for k, v in load_file(path).items()}
    else:
        ck = torch.load(path, map_location="cpu")
        c = ck["config"]
        cfg = {"img": ck["img"], "in_ch": ck.get("in_ch", 3), "patch": c["patch"], "d": c["d"],
               "layers": c["layers"], "heads": c["heads"]}
        state = ck["state"]
    model = DiT(**{k: cfg[k] for k in ARCH_KEYS}).to(dev).eval()
    model.load_state_dict(state)
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--out", default="flash_out")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--scale", type=int, default=1, help="upscale the saved PNGs by this factor")
    p.add_argument("--export", default=None, help="save fp16 .safetensors + .json config here, then exit")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(a.weights, dev)

    if a.export:
        from safetensors.torch import save_file
        save_file({k: v.half().contiguous() for k, v in model.state_dict().items()}, a.export)
        cfg = {k: cfg[k] for k in ARCH_KEYS} | {"vae": VAE_ID, "latent_scale": SCALE, "image_size": cfg["img"] * 8,
                                                 "flash_steps": 8}
        json.dump(cfg, open(os.path.splitext(a.export)[0] + ".json", "w"), indent=2)
        print(f"exported {a.export} ({os.path.getsize(a.export) / 2**20:.0f} MB) + config")
        return

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    z = sample(model, a.n, steps=a.steps, device=dev, seed=a.seed)
    with torch.no_grad():
        x = vae.decode((z / SCALE).to(vae.dtype)).sample.float()
    imgs = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.weights))[0]
    for i, im in enumerate(imgs):
        img = Image.fromarray(im)
        if a.scale > 1:
            img = img.resize((img.width * a.scale, img.height * a.scale), Image.LANCZOS)
        path = os.path.join(a.out, f"{name}_seed{a.seed}_{i + 1}_{a.steps}steps.png")
        img.save(path)
        print("saved", path)


if __name__ == "__main__":
    main()
