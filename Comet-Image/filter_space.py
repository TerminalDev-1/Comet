"""Keep only real space imagery in a NASA download, using CLIP zero-shot scoring.

    python filter_space.py --images <nasa>/images [--threshold 0.5]

Each image is compared with "space" descriptions (sun, moon, planets, galaxies, Earth from orbit...)
and "not space" descriptions (people, crowds, lab equipment, diagrams...). Images whose space
probability is below --threshold are MOVED to <images>/../rejected (never deleted), so a wrong call
can be undone by moving the file back. Writes filter_scores.jsonl next to the images folder.
"""
import argparse
import json
import os
import shutil

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

SPACE = ["a photo of outer space", "an astronomy image of a galaxy", "a colorful nebula in space",
         "a photo of the sun", "a close-up of the solar surface", "a photo of the moon", "craters on the moon",
         "a photo of a planet", "Jupiter or Saturn", "the surface of Mars", "Earth seen from space",
         "clouds seen from orbit", "a starry night sky", "an aurora seen from space", "a comet in the sky",
         "a rocket launching into the sky"]
NOT_SPACE = ["a photo of people", "a group of people at an event", "a portrait of a person", "people in hard hats",
             "laboratory equipment", "engineers working in a clean room", "a machine part", "a diagram or chart",
             "text on a slide", "an office or building interior", "a person giving a speech", "an airplane on a runway"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--model", default="openai/clip-vit-base-patch32")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(a.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(a.model)
    with torch.no_grad():
        t = proc(text=SPACE + NOT_SPACE, return_tensors="pt", padding=True).to(dev)
        text = model.get_text_features(**t)
        text = text / text.norm(dim=-1, keepdim=True)
    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rej_dir = os.path.join(os.path.dirname(os.path.abspath(a.images)), "rejected")
    os.makedirs(rej_dir, exist_ok=True)
    scores, kept, rejected = [], 0, 0
    for i in range(0, len(files), a.batch):
        names, ims = [], []
        for f in files[i:i + a.batch]:
            try:
                ims.append(Image.open(os.path.join(a.images, f)).convert("RGB"))
                names.append(f)
            except Exception:
                pass
        with torch.no_grad():
            x = proc(images=ims, return_tensors="pt").to(dev)
            img = model.get_image_features(**x)
            img = img / img.norm(dim=-1, keepdim=True)
            probs = (100 * img @ text.T).softmax(-1)
            p_space = probs[:, :len(SPACE)].sum(-1).tolist()
        for f, ps in zip(names, p_space):
            keep = ps >= a.threshold
            scores.append({"file": f, "p_space": round(ps, 3), "kept": keep})
            if keep:
                kept += 1
            else:
                shutil.move(os.path.join(a.images, f), os.path.join(rej_dir, f))
                rejected += 1
        if (i // a.batch) % 20 == 0:
            print(f"  {i + len(names)}/{len(files)}  kept {kept}  rejected {rejected}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(a.images)), "filter_scores.jsonl"), "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")
    print(f"DONE: kept {kept}, moved {rejected} to {rej_dir}")


if __name__ == "__main__":
    main()
