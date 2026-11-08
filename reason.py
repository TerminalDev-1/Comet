"""Story reasoning test: watch a model reason in English.

Each probe is a short story prompt plus a right and a wrong one-word answer. The model "picks"
whichever answer it finds more likely. Many probes come in mirrored pairs (Tom gives to Sue /
Sue gives to Tom), so always guessing the more common word scores only ~50%.

    python reason.py checkpoints/<run>.pt [--device cuda]

During training, train.py --reason calls score() and watch() at every eval.
"""
import argparse
import contextlib
import os

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))

# (category, prompt, right, wrong)
PROBES = [
    # who has the object
    ("tracking", "Tom had a red ball. He gave the ball to Sue. Now the ball is with", " Sue", " Tom"),
    ("tracking", "Sue had a red ball. She gave the ball to Tom. Now the ball is with", " Tom", " Sue"),
    ("tracking", "Lily put her doll in the box and her ball in the bag. The doll was in the", " box", " bag"),
    ("tracking", "Lily put her doll in the box and her ball in the bag. The ball was in the", " bag", " box"),
    ("tracking", "Ben had a cat and Mia had a dog. The dog belonged to", " Mia", " Ben"),
    ("tracking", "Ben had a dog and Mia had a cat. The dog belonged to", " Ben", " Mia"),
    # cause and effect
    ("cause", "Mia played all day and was very tired. So she went to", " bed", " play"),
    ("cause", "The ice cream sat in the hot sun. Soon it", " melted", " froze"),
    ("cause", "The glass fell off the table and", " broke", " smiled"),
    ("cause", "It was cold outside, so Tom put on his", " coat", " shorts"),
    ("cause", "The sun went down and the sky became", " dark", " bright"),
    ("cause", "Sam was very hungry, so he ate a big", " sandwich", " rock"),
    # feelings that follow from events
    ("feeling", "Sam lost his favorite toy. He felt very", " sad", " happy"),
    ("feeling", "Anna got a big present for her birthday. She felt very", " happy", " sad"),
    ("feeling", "The big dog barked at the little cat. The cat was", " scared", " happy"),
    ("feeling", "Tom found his lost puppy. He was so", " happy", " sad"),
    # who is who
    ("who", "Max is a dog and Lily is a girl. The one who barked was", " Max", " Lily"),
    ("who", "Lily is a dog and Max is a boy. The one who barked was", " Lily", " Max"),
    ("who", "Ben is big and Tim is small. The small boy was", " Tim", " Ben"),
    ("who", "Tim is big and Ben is small. The small boy was", " Ben", " Tim"),
    # small numbers
    ("numbers", "Lily had two apples. She ate one apple. Now she has", " one", " two"),
    ("numbers", "Tom had one cookie. Mom gave him one more. Now he has", " two", " one"),
    # opposites
    ("opposite", "The box was not heavy. It was very", " light", " heavy"),
    ("opposite", "The room was not dark. It was", " bright", " dark"),
]

# Shown in English at every eval: the model's own continuation (greedy).
WATCH = [
    "Tom had a red ball. He gave the ball to Sue. Now the ball is with",
    "Max is a dog and Lily is a girl. The one who barked was",
    "Sam lost his favorite toy. He felt",
    "Lily had two apples. She ate one apple. Now she has",
]


def _autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def _logprob(model, ids, n_answer, device):
    x = torch.tensor([ids], device=device)
    with _autocast(device):
        logits = model(x)
    lp = F.log_softmax(logits[0].float(), -1)
    T = len(ids)
    return sum(lp[t - 1, ids[t]].item() for t in range(T - n_answer, T))


@torch.no_grad()
def score(model, tok, device):
    """Fraction of probes where the right answer beats the wrong one, overall and per category."""
    was = model.training
    model.eval()
    hits, cats = 0, {}
    for cat, prompt, good, bad in PROBES:
        p = tok.encode(prompt).ids
        g, b = tok.encode(good).ids, tok.encode(bad).ids
        ok = _logprob(model, p + g, len(g), device) > _logprob(model, p + b, len(b), device)
        hits += ok
        c = cats.setdefault(cat, [0, 0])
        c[0] += ok
        c[1] += 1
    model.train(was)
    return hits / len(PROBES), {k: f"{a}/{n}" for k, (a, n) in cats.items()}


@torch.no_grad()
def watch(model, tok, device, max_new=6):
    """Greedy continuations of the WATCH prompts, so a person can see what it 'thinks'."""
    was = model.training
    model.eval()
    out = []
    for prompt in WATCH:
        ids = tok.encode(prompt).ids
        for _ in range(max_new):
            x = torch.tensor([ids[-model.cfg.ctx:]], device=device)
            with _autocast(device):
                nxt = int(model(x)[0, -1].argmax())
            ids.append(nxt)
        out.append((prompt, tok.decode(ids[len(tok.encode(prompt).ids):]).strip().split("\n")[0]))
    model.train(was)
    return out


def main():
    from tokenizers import Tokenizer

    from sample import load
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    dev = torch.device(args.device)
    model = load(args.ckpt).to(dev)
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    acc, cats = score(model, tok, dev)
    print(f"Reasoning score: {acc * 100:.0f}% ({round(acc * len(PROBES))}/{len(PROBES)}; chance = 50%)  {cats}")
    for prompt, ans in watch(model, tok, dev):
        print(f"  {prompt} -> {ans}")


if __name__ == "__main__":
    main()
