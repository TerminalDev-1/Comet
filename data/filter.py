"""Data filtering: clean text, quality filters, exact + near-duplicate removal (MinHash LSH).

Used by data/prepare.py. Can also run alone to print stats:
    python data/filter.py data/raw/train-00000.parquet
"""
import hashlib
import re
import sys
from collections import Counter, defaultdict

import numpy as np

# Common broken-encoding sequences (mojibake) seen in web/LLM text.
MOJIBAKE = {
    "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€”": "-", "â€“": "-",
    "â€¦": "...", "Ã©": "é", "Â": "",
}
FANCY = {"‘": "'", "’": "'", "“": '"', "”": '"',
         "—": "-", "–": "-", "…": "...", " ": " ",
         # In TinyStories the replacement char is almost always a broken closing quote.
         "�": '"'}

_ws = re.compile(r"[ \t]+")
_many_newlines = re.compile(r"\n{3,}")
_word = re.compile(r"[a-z']+")


def clean(text: str) -> str:
    for bad, good in MOJIBAKE.items():
        if bad in text:
            text = text.replace(bad, good)
    for bad, good in FANCY.items():
        if bad in text:
            text = text.replace(bad, good)
    text = _ws.sub(" ", text)
    text = _many_newlines.sub("\n\n", text)
    return text.strip()


def quality_reason(text: str):
    """Return why a document should be dropped, or None to keep it."""
    words = text.split()
    n = len(words)
    if n < 30:
        return "too_short"
    if n > 1500:
        return "too_long"
    letters = sum(c.isalpha() for c in text)
    if letters / len(text) < 0.70:
        return "low_letter_ratio"
    non_ascii = sum(ord(c) > 127 for c in text)
    if non_ascii / len(text) > 0.01:
        return "non_english_chars"
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.7:
        return "repeated_lines"
    # One word dominating the story = looping/garbled generation.
    top = Counter(w.lower() for w in words).most_common(1)[0][1]
    if n >= 50 and top / n > 0.20:
        return "repetitive_words"
    if text[-1] not in '.!?"\')':
        return "cut_off"
    return None


# ---- MinHash near-duplicate detection ----
NUM_PERM = 32
BANDS = 8          # 8 bands x 4 rows: catches pairs with Jaccard >~ 0.7
ROWS = NUM_PERM // BANDS
SHINGLE = 5
_PRIME = (1 << 61) - 1
_rng = np.random.default_rng(1234)
_A = _rng.integers(1, _PRIME, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, _PRIME, NUM_PERM, dtype=np.uint64)


def _shingle_hashes(text: str) -> np.ndarray:
    w = _word.findall(text.lower())
    if len(w) < SHINGLE:
        w = w + [""] * (SHINGLE - len(w))
    grams = {" ".join(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1)}
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") >> 3 for g in grams),
        dtype=np.uint64, count=len(grams))


def minhash(text: str) -> np.ndarray:
    h = _shingle_hashes(text)
    # (a*h + b) mod 2^64 — fast universal-ish hashing; overflow wrap is fine here.
    with np.errstate(over="ignore"):
        v = _A[:, None] * h[None, :] + _B[:, None]
    return v.min(axis=1)


def dedup(texts):
    """Return (kept_indices, n_exact, n_near)."""
    seen_exact = set()
    buckets = [defaultdict(list) for _ in range(BANDS)]
    sigs = {}
    kept, n_exact, n_near = [], 0, 0
    for i, t in enumerate(texts):
        norm = " ".join(_word.findall(t.lower()))
        key = hashlib.blake2b(norm.encode(), digest_size=16).digest()
        if key in seen_exact:
            n_exact += 1
            continue
        seen_exact.add(key)
        sig = minhash(t)
        dup = False
        cands = set()
        for b in range(BANDS):
            band = sig[b * ROWS:(b + 1) * ROWS].tobytes()
            cands.update(buckets[b].get(band, ()))
        for j in cands:
            if np.mean(sigs[j] == sig) >= 0.8:  # estimated Jaccard >= 0.8
                dup = True
                break
        if dup:
            n_near += 1
            continue
        sigs[i] = sig
        for b in range(BANDS):
            buckets[b][sig[b * ROWS:(b + 1) * ROWS].tobytes()].append(i)
        kept.append(i)
    return kept, n_exact, n_near


def filter_texts(texts, log=print):
    cleaned = [clean(t) for t in texts if t]
    reasons = Counter()
    good = []
    for t in cleaned:
        r = quality_reason(t) if t else "empty"
        if r:
            reasons[r] += 1
        else:
            good.append(t)
    kept_idx, n_exact, n_near = dedup(good)
    out = [good[i] for i in kept_idx]
    stats = {
        "input_docs": len(texts),
        "dropped_quality": dict(reasons),
        "dropped_exact_dup": n_exact,
        "dropped_near_dup": n_near,
        "kept_docs": len(out),
    }
    log(stats)
    return out, stats


if __name__ == "__main__":
    import pyarrow.parquet as pq
    texts = pq.read_table(sys.argv[1]).column("text").to_pylist()
    filter_texts(texts)
