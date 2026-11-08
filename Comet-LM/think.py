"""Watch a model think: give it NEW stories (names and objects it never saw in the reasoning data),
let it write its own <think>...</think>, then check its Answer.

    python think.py checkpoints/<run>.pt --device cuda [--show 8]
"""
import argparse
import json
import os
import re
from collections import defaultdict

import torch
from tokenizers import Tokenizer

from sample import load

ROOT = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def answer(model, tok, prompt, dev, max_new=80):
    eot = tok.token_to_id("<|endoftext|>")
    ids = tok.encode(prompt).ids
    start = len(ids)
    for _ in range(max_new):
        x = torch.tensor([ids[-model.cfg.ctx:]], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            nxt = int(model(x)[0, -1].argmax())
        if nxt == eot:
            break
        ids.append(nxt)
        text = tok.decode(ids[start:])
        if re.search(r"Answer: .*\n", text):
            break
    text = tok.decode(ids[start:])
    m = re.search(r"Answer:\s*(.*)", text)
    return text.split("\n\n")[0].strip(), (m.group(1).strip().rstrip(".") if m else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--show", type=int, default=8)
    p.add_argument("--n", type=int, default=400)
    args = p.parse_args()
    dev = torch.device(args.device)
    model = load(args.ckpt).to(dev)
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    tests = [json.loads(l) for l in open(os.path.join(ROOT, "data", "think_test.jsonl"))][:args.n]
    right, by = 0, defaultdict(lambda: [0, 0])
    shown = set()
    for t in tests:
        out, ans = answer(model, tok, t["prompt"], dev)
        ok = ans.lower() == t["answer"].lower()
        right += ok
        by[t["kind"]][0] += ok
        by[t["kind"]][1] += 1
        if t["kind"] not in shown and len(shown) < args.show:
            shown.add(t["kind"])
            story = t["prompt"].replace("\n<think>", "")
            print(f"\n{story}\n<think>{out}\n   {'RIGHT' if ok else 'WRONG'} (expected: {t['answer']})")
    print(f"\nHeld-out reasoning accuracy: {right}/{len(tests)} = {right / len(tests) * 100:.1f}%")
    print("  " + "  ".join(f"{k} {a}/{n}" for k, (a, n) in by.items()))


if __name__ == "__main__":
    main()
