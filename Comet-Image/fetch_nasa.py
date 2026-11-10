"""Download space images from NASA's Image and Video Library (images-api.nasa.gov, no API key needed).

    python fetch_nasa.py --out <folder> [--per_query 450 --size medium --workers 8] [--more]

Searches a list of space topics (sun, moon, galaxies, nebulae, planets, Earth from space...), keeps
each image once (by NASA id), and downloads the medium-size JPEG. Writes credits.jsonl with the
NASA id, title, center and source URL of every image (NASA media is generally not copyrighted;
credit NASA). Safe to re-run: existing files are skipped.
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://images-api.nasa.gov"
QUERIES = ["sun", "solar flare", "solar eclipse", "moon", "full moon", "lunar surface", "galaxy", "nebula",
           "star cluster", "milky way", "planet", "jupiter", "saturn", "mars surface", "earth from space",
           "aurora", "comet", "hubble", "james webb", "supernova", "deep field", "venus", "neptune", "black hole"]
MORE_QUERIES = ["sunspot", "solar prominence", "coronal mass ejection", "solar dynamics observatory", "solar corona",
                "sun ultraviolet", "jupiter juno", "great red spot", "saturn rings", "cassini saturn", "uranus", "pluto",
                "mercury messenger", "asteroid", "lunar crater", "apollo lunar surface", "earthrise", "lunar reconnaissance",
                "orion nebula", "andromeda galaxy", "spiral galaxy", "globular cluster", "planetary nebula",
                "star forming region", "chandra x-ray", "spitzer infrared", "mars reconnaissance orbiter", "mars crater",
                "martian landscape", "io volcano", "europa", "titan", "earth at night", "hurricane from space"]
UA = {"User-Agent": "comet-image-dataset/1.0"}


def get_json(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def search(q, limit):
    items, page = [], 1
    while len(items) < limit:
        url = f"{API}/search?" + urllib.parse.urlencode({"q": q, "media_type": "image", "page": page, "page_size": 100})
        j = get_json(url)
        got = (j or {}).get("collection", {}).get("items", [])
        if not got:
            break
        items += got
        page += 1
    return items[:limit]


def pick_url(item, size):
    for link in item.get("links", []):
        href = link.get("href", "")
        if href.endswith(f"~{size}.jpg"):
            return href
    href = (item.get("links") or [{}])[0].get("href", "")  # fall back to the preview image
    return href if href.endswith(".jpg") else None


def download(job):
    path, url = job
    if os.path.exists(path):
        return True
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception:
            time.sleep(2 * (i + 1))
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--per_query", type=int, default=450)
    p.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--more", action="store_true", help="search MORE_QUERIES instead of the original list")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    seen, jobs, credits = set(), [], []
    rejected = os.path.join(a.out, "rejected")  # filter_space.py moves non-space images here; don't fetch them again
    for q in (MORE_QUERIES if a.more else QUERIES):
        n_new = 0
        for it in search(q, a.per_query):
            d = (it.get("data") or [{}])[0]
            nid = d.get("nasa_id")
            if not nid or nid in seen:
                continue
            url = pick_url(it, a.size)
            if not url:
                continue
            seen.add(nid)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in nid)[:120]
            if os.path.exists(os.path.join(rejected, safe + ".jpg")) or os.path.exists(os.path.join(a.out, "images", safe + ".jpg")):
                continue
            n_new += 1
            jobs.append((os.path.join(a.out, "images", safe + ".jpg"), url))
            credits.append({"nasa_id": nid, "title": d.get("title"), "center": d.get("center"),
                            "query": q, "url": url})
        print(f"{q!r}: +{n_new} (total {len(jobs)})", flush=True)
    with open(os.path.join(a.out, "credits_more.jsonl" if a.more else "credits.jsonl"), "w", encoding="utf-8") as f:
        for c in credits:
            f.write(json.dumps(c) + "\n")
    print(f"downloading {len(jobs)} images...", flush=True)
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, ok in enumerate(ex.map(download, jobs), 1):
            done += ok
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    size = sum(os.path.getsize(os.path.join(a.out, "images", f)) for f in os.listdir(os.path.join(a.out, "images")))
    print(f"DONE: {done}/{len(jobs)} images, {size / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
