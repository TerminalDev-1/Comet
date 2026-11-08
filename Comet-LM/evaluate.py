"""Score a checkpoint on the fixed validation set (same batches train.py uses).

    python evaluate.py baseline_d256_l6 vt_v1_muon_12m [--write]

Prints val loss and next-token accuracy. --write adds final_val_acc to the run's summary.json.
"""
import argparse
import json
import os

import torch

from sample import load
from train import Data, ROOT, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--write", action="store_true")
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(6)
    for name in args.runs:
        model = load(os.path.join(ROOT, "checkpoints", f"{name}.pt"))
        val = Data("val", model.cfg.ctx, args.batch, 0)
        batches = [val.get() for _ in range(args.eval_batches)]
        loss, acc = evaluate(model, batches, lambda: torch.autocast("cpu", dtype=torch.bfloat16))
        print(f"{name}: val loss {loss:.4f}, next-token accuracy {acc * 100:.2f}%")
        if args.write:
            path = os.path.join(ROOT, "results", "runs", name, "summary.json")
            s = json.load(open(path))
            s["final_val_acc"] = round(acc, 4)
            json.dump(s, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
