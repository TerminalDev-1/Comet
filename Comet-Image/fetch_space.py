"""Download space images for Comet-Image's space model: the Sun (SDO via Helioviewer) and Mars (rover cameras).

    python fetch_space.py --source sun      --out <folder> --n 6000 [--workers 6]
    python fetch_space.py --source mars2020 --out <folder> --n 7000 [--workers 16]
    python fetch_space.py --source msl      --out <folder> --n 7000 [--workers 16]

sun:      SDO images rendered by the Helioviewer API (no watermark) at random dates from 2010-06 to 2025-12, in 10
          wavelengths (AIA EUV/UV and HMI visible light). About 30% are the full disk, 70% are close-ups (zoomed in on a
          random spot of the disk or limb). Credit: NASA/SDO and the AIA and HMI science teams, via Helioviewer.org.
mars2020: Perseverance Mastcam-Z left-camera color images (filter 0, full frames), a few from each of many random pages of the raw-image API.
          Credit: NASA/JPL-Caltech/ASU.
msl:      Curiosity Mastcam (left, wide) and MAHLI (rock close-ups) images. Credit: NASA/JPL-Caltech/MSSS.

Images are shrunk so the short side is at most 384 px and saved as JPEG; images smaller than 256 px or grayscale are
skipped. Writes credits.jsonl. Safe to re-run with another --seed: existing files are skipped.
"""
import argparse
import io
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import requests
from PIL import Image

PER_PAGE = 6  # images taken from each 100-image API page (spreads the sample over many sols)
UA = {"User-Agent": "comet-image-dataset/1.0"}
HV = "https://api.helioviewer.org/v2/takeScreenshot/"
# (layer, weight): SDO wavelengths that look most different from each other get more weight
SUN_LAYERS = [("SDO,AIA,AIA,171", 3), ("SDO,AIA,AIA,193", 3), ("SDO,AIA,AIA,304", 3), ("SDO,AIA,AIA,211", 2),
              ("SDO,AIA,AIA,131", 1), ("SDO,AIA,AIA,335", 1), ("SDO,AIA,AIA,94", 1), ("SDO,AIA,AIA,1600", 1),
              ("SDO,AIA,AIA,1700", 1), ("SDO,HMI,HMI,continuum", 3)]


def save_image(content, path, min_side=256, max_short=384):
    im = Image.open(io.BytesIO(content)).convert("RGB")
    if min(im.size) < min_side:
        return False
    small = np.asarray(im.resize((64, 64)), dtype=np.float32)
    if np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean() < 4:
        return False  # grayscale (e.g. navigation or narrow-band filter frames)
    s = max_short / min(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    im.save(path, quality=92)
    return True


def sun_jobs(n, rng):
    start, days = datetime(2010, 6, 1), (datetime(2025, 12, 31) - datetime(2010, 6, 1)).days
    layers, weights = zip(*SUN_LAYERS)
    for i in range(int(n * 1.2)):
        date = start + timedelta(days=rng.randrange(days), minutes=rng.randrange(24 * 60))
        layer = rng.choices(layers, weights)[0]
        if rng.random() < 0.3:  # full disk, about 90% of the frame
            scale, x0, y0 = rng.uniform(4.1, 4.6), 0, 0
        else:  # close-up on a random spot of the disk, sometimes past the limb (prominences)
            scale = rng.uniform(0.6, 2.4)
            r, ang = 1000 * rng.random() ** 0.5, rng.uniform(0, 2 * np.pi)
            x0, y0 = round(r * np.cos(ang)), round(r * np.sin(ang))
        params = {"date": date.strftime("%Y-%m-%dT%H:%M:%SZ"), "imageScale": round(scale, 3),
                  "layers": f"[{layer},1,100]", "x0": x0, "y0": y0, "width": 512, "height": 512,
                  "display": "true", "watermark": "false"}
        name = f"sun_{i:06d}_{layer.split(',')[-1]}.jpg"
        yield name, (HV, params), {"source": "NASA/SDO via Helioviewer.org", "layer": layer, **params}


def pages_parallel(fetch, pages, workers=16):
    """Fetch API listing pages 16 at a time (one at a time was the bottleneck), yielding each page's items."""
    with ThreadPoolExecutor(workers) as ex:
        for c in range(0, len(pages), workers * 2):
            for items in ex.map(fetch, pages[c:c + workers * 2]):
                yield items


def mars2020_jobs(n, rng):
    api = "https://mars.nasa.gov/rss/api/"
    base = {"feed": "raw_images", "category": "mars2020", "feedtype": "json", "num": 100, "order": "sol desc",
            "search": "|MCZ_LEFT|MCZ_RIGHT", "extended": "sample_type::full,"}
    total = requests.get(api, params={**base, "num": 1, "page": 0}, headers=UA, timeout=30).json()["total_results"]
    pages = list(range(total // 100))
    rng.shuffle(pages)
    def fetch(pg):
        try:
            return requests.get(api, params={**base, "page": pg}, headers=UA, timeout=30).json()["images"]
        except Exception:
            return []

    for items in pages_parallel(fetch, pages):
        items = [it for it in items if it["image_files"].get("full_res", "").rsplit("/", 1)[-1][:3] == "ZL0"]
        rng.shuffle(items)
        for it in items[:PER_PAGE]:  # left camera, filter 0 = RGB color; few per page so sequences don't repeat
            url = it["image_files"]["full_res"]
            fname = url.rsplit("/", 1)[-1]
            yield "m20_" + fname.rsplit(".", 1)[0] + ".jpg", (url, None), {
                "source": "NASA/JPL-Caltech/ASU", "sol": it.get("sol"), "url": url}


def msl_jobs(n, rng):
    api = "https://mars.nasa.gov/api/v1/raw_image_items/"
    for inst, share in (("MAST_LEFT", 0.8), ("MAHLI", 0.2)):
        base = {"order": "sol desc", "per_page": 100, "condition_1": "msl:mission", "condition_2": f"{inst}:instrument"}
        total = requests.get(api, params={**base, "per_page": 1, "page": 0}, headers=UA, timeout=30).json()["total"]
        pages = list(range(total // 100))
        rng.shuffle(pages)
        quota = int(n * 2.5 * share)

        def fetch(pg, base=base):
            try:
                return requests.get(api, params={**base, "page": pg}, headers=UA, timeout=30).json()["items"]
            except Exception:
                return []

        for items in pages_parallel(fetch, pages):
            if quota <= 0:
                break
            items = [it for it in items if (it.get("https_url") or "").endswith((".jpg", ".JPG", ".png"))]
            rng.shuffle(items)
            for it in items[:PER_PAGE]:  # few per page: neighbours are often focus stacks of the same rock
                url = it["https_url"]
                quota -= 1
                yield "msl_" + url.rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".jpg", (url, None), {
                    "source": "NASA/JPL-Caltech/MSSS", "instrument": inst, "sol": it.get("sol"), "url": url}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["sun", "mars2020", "msl"])
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    jobs = {"sun": sun_jobs, "mars2020": mars2020_jobs, "msl": msl_jobs}[a.source](a.n, rng)
    sess = requests.Session()
    sess.headers.update(UA)
    lock, done, credits = threading.Lock(), [0], []

    def get(job):
        name, (url, params), credit = job
        path = os.path.join(a.out, name)
        if done[0] >= a.n:
            return
        if os.path.exists(path):
            with lock:
                done[0] += 1
            return
        try:
            r = sess.get(url, params=params, timeout=60)
            r.raise_for_status()
            if save_image(r.content, path):
                with lock:
                    done[0] += 1
                    credits.append({"file": name, **credit})
                    if done[0] % 250 == 0:
                        print(f"{done[0]} images", flush=True)
        except Exception:
            pass

    with ThreadPoolExecutor(a.workers) as ex:
        pending = []
        for job in jobs:
            if done[0] >= a.n:
                break
            pending.append(ex.submit(get, job))
            if len(pending) >= a.workers * 4:  # keep the queue short so we stop soon after reaching --n
                pending.pop(0).result()
        for f in pending:
            f.result()
    with open(os.path.join(a.out, f"credits_{a.source}_{a.seed}.jsonl"), "a", encoding="utf-8") as f:
        for c in credits:
            f.write(json.dumps(c) + "\n")
    print(f"DONE: {done[0]} {a.source} images in {a.out}", flush=True)


if __name__ == "__main__":
    main()
