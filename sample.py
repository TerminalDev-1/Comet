"""Generate text from a checkpoint.

    python sample.py checkpoints/baseline_d256.pt --prompt "Once upon a time" --n 3
"""
import argparse
import os

import torch
from tokenizers import Tokenizer

from comet import MODELS

ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPTS = ["Once upon a time", "Tom and his dog", "The little girl was sad because", "One day, a big bird"]


def load(path):
    ck = torch.load(path, map_location="cpu")
    cfg_cls, model_cls = MODELS[ck["model"]]
    model = model_cls(cfg_cls(**ck["config"]))
    model.load_state_dict(ck["state"])
    return model.eval()


@torch.no_grad()
def generate(model, tok, prompt, max_new=150, temperature=0.8, top_k=40, seed=0):
    torch.manual_seed(seed)
    eot = tok.token_to_id("<|endoftext|>")
    ids = torch.tensor([tok.encode(prompt).ids])
    ctx = model.cfg.ctx
    for _ in range(max_new):
        logits = model(ids[:, -ctx:])[:, -1, :].float() / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(logits.softmax(-1), 1)
        if nxt.item() == eot:
            break
        ids = torch.cat([ids, nxt], 1)
    return tok.decode(ids[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--prompt", default=None)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--max_new", type=int, default=150)
    args = p.parse_args()
    model = load(args.ckpt)
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    prompts = [args.prompt] if args.prompt else PROMPTS
    for pr in prompts:
        for i in range(args.n):
            print(f"--- {pr!r} (sample {i + 1})")
            print(generate(model, tok, pr, args.max_new, seed=i))


if __name__ == "__main__":
    main()
