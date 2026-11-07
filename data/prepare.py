"""Phase 1: filter TinyStories, train an 8k BPE tokenizer, write train.bin / val.bin.

Downloads (if missing) into data/raw/ from Hugging Face:
    data/raw/train-00000.parquet   (one of 4 TinyStories train shards, ~530k stories, ~249 MB)
    data/raw/validation.parquet    (~10 MB)

Usage:  python data/prepare.py
Outputs: data/tokenizer.json, data/train.bin, data/val.bin (uint16 token ids), data/prepare_stats.json
"""
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, os.path.dirname(__file__))
from filter import filter_texts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
VOCAB = 8192
EOT = "<|endoftext|>"


HF = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/"
FILES = {
    "train-00000.parquet": "train-00000-of-00004-2d5a1467fff1081b.parquet",  # ~249 MB
    "validation.parquet": "validation-00000-of-00001-869c898b519ad725.parquet",  # ~10 MB
}


def download():
    import urllib.request
    os.makedirs(RAW, exist_ok=True)
    for local, remote in FILES.items():
        path = os.path.join(RAW, local)
        if not os.path.exists(path):
            print("downloading", remote)
            urllib.request.urlretrieve(HF + remote, path)


def load(name):
    return pq.read_table(os.path.join(RAW, name)).column("text").to_pylist()


def train_tokenizer(texts):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB, special_tokens=[EOT], min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode_to_bin(tok, texts, path):
    eot = tok.token_to_id(EOT)
    chunks = []
    for i in range(0, len(texts), 10000):
        for enc in tok.encode_batch(texts[i:i + 10000]):
            chunks.append(np.asarray(enc.ids, dtype=np.uint16))
            chunks.append(np.asarray([eot], dtype=np.uint16))  # document separator (packing)
    arr = np.concatenate(chunks)
    arr.tofile(path)
    return int(arr.size)


def main():
    t0 = time.time()
    download()  # skips files that already exist
    stats = {}
    print("filtering train...")
    train, stats["train_filter"] = filter_texts(load("train-00000.parquet"))
    print("filtering val...")
    val, stats["val_filter"] = filter_texts(load("validation.parquet"))

    print("training tokenizer...")
    tok = train_tokenizer(train)
    tok.save(os.path.join(HERE, "tokenizer.json"))

    print("encoding...")
    stats["train_tokens"] = encode_to_bin(tok, train, os.path.join(HERE, "train.bin"))
    stats["val_tokens"] = encode_to_bin(tok, val, os.path.join(HERE, "val.bin"))
    stats["vocab_size"] = tok.get_vocab_size()
    stats["avg_tokens_per_story"] = round(stats["train_tokens"] / len(train), 1)
    stats["chars_per_token"] = round(sum(map(len, val)) / stats["val_tokens"], 2)
    stats["seconds"] = round(time.time() - t0, 1)

    sample = val[0][:200]
    ids = tok.encode(sample).ids
    stats["roundtrip_ok"] = tok.decode(ids) == sample
    stats["example"] = {"text": sample[:80], "tokens": [tok.id_to_token(i) for i in ids[:20]]}

    with open(os.path.join(HERE, "prepare_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
