"""Download a random sample of PD12M (Spawning/PD12M: public-domain and CC0 photos, all kinds of subjects).

    python fetch_pd12m.py --out <folder> --n 6000 [--workers 64]

Reads one metadata parquet (image URLs), shuffles it, downloads in parallel, shrinks each image so its short
side is at most 384 px (enough for 256x256 training) and saves JPEGs. Failed downloads are skipped.
"""
import argparse
import io
import os
import random
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = [f for f in HfApi().list_repo_files("Spawning/PD12M", repo_type="dataset") if f.endswith(".parquet")]
    df = pd.read_parquet(hf_hub_download("Spawning/PD12M", files[0], repo_type="dataset"))
    col = next(c for c in df.columns if c.lower() in ("url", "image_url"))
    urls = df[col].tolist()
    random.Random(a.seed).shuffle(urls)
    urls = urls[: int(a.n * 1.3)]  # some downloads fail
    print(f"{files[0]}: {len(df):,} rows, trying {len(urls):,}", flush=True)
    sess = requests.Session()
    sess.headers["User-Agent"] = "Comet-Image dataset fetcher (research)"
    done = [0]

    def get(i_url):
        i, url = i_url
        if done[0] >= a.n:
            return
        try:
            r = sess.get(url, timeout=15)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            if min(im.size) < 256:
                return
            s = 384 / min(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
            im.save(os.path.join(a.out, f"pd{i:06d}.jpg"), quality=92)
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"{done[0]} images", flush=True)
        except Exception:
            pass

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(get, enumerate(urls)))
    print(f"DONE: {done[0]} images in {a.out}", flush=True)


if __name__ == "__main__":
    main()
