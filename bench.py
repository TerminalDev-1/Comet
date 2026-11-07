"""Build the benchmark image and table from training runs.

    python bench.py                       # default: baseline vs latest Comet race
    python bench.py --runs baseline_d256_l6 vt_v1_muon_12m

Outputs results/benchmark.jpg and results/benchmark.md (shown in the README).
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "results", "runs")
COLORS = {"baseline": "#8a8f98", "comet": "#2f6fed", "vtransformer": "#2f6fed"}  # vtransformer = old name


def load(name):
    d = os.path.join(RUNS, name)
    log = [json.loads(l) for l in open(os.path.join(d, "log.jsonl"))]
    summary = json.load(open(os.path.join(d, "summary.json")))
    return log, summary


def label(s):
    if s["model"] == "baseline":
        return f"Baseline GPT ({s['minutes']:.0f} min)"
    extras = [n for n, on in (("EMOE", s["config"].get("emoe")), ("MoD", s["config"].get("mod")),
                              ("MTP", s["config"].get("mtp_depth"))) if on]
    return f"Comet{' + ' + ' + '.join(extras) if extras else ''} ({s['minutes']:.0f} min)"


def color(s):
    if s["model"] == "baseline":
        return COLORS["baseline"]
    return "#e8743b" if s["config"].get("mod") or s["config"].get("emoe") else COLORS["comet"]


def first_time_reaching(log, target):
    """Interpolated training minute where val loss first reaches target, or None."""
    for a, b in zip(log, log[1:]):
        if b["val_loss"] <= target < a["val_loss"]:
            f = (a["val_loss"] - target) / (a["val_loss"] - b["val_loss"])
            return a["train_min"] + f * (b["train_min"] - a["train_min"])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["baseline_d256_l6", "vt_v1_muon_12m"])
    p.add_argument("--out", default=os.path.join(ROOT, "results", "benchmark.jpg"))
    p.add_argument("--md", default=os.path.join(ROOT, "results", "benchmark.md"))
    p.add_argument("--title", default="Comet vs. Baseline GPT")
    args = p.parse_args()

    runs = [(n, *load(n)) for n in args.runs]
    base = next(r for r in runs if r[2]["model"] == "baseline")
    target = base[2]["final_val_loss"]

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), dpi=130)
    fig.suptitle(f"{args.title} — same laptop (Ryzen 5 8640HS, CPU, BF16)",
                 fontsize=14, fontweight="bold")

    # 1. Val loss vs training time
    a = ax[0, 0]
    for name, log, s in runs:
        c = color(s)
        a.plot([r["train_min"] for r in log], [r["val_loss"] for r in log], color=c, lw=2.2, label=label(s))
    a.axhline(target, color="#d64545", ls="--", lw=1.2)
    a.text(0.3, target - 0.04, f"Baseline final: {target}", color="#d64545", va="top", fontsize=9)
    k = 0
    for name, log, s in runs:
        if s["model"] != "baseline":
            t = first_time_reaching(log, target)
            if t:
                c = color(s)
                a.plot([t], [target], "o", color=c, ms=7)
                a.annotate(f"reached in {t:.1f} min ({base[2]['minutes'] / t:.1f}× faster)", (t, target),
                           xytext=(t + 1.2, target + 0.45 + 0.3 * k), fontsize=9, color=c,
                           arrowprops=dict(arrowstyle="->", color=c))
                k += 1
    a.set_ylim(2.2, 4.6)
    a.set_xlabel("Training time (minutes)")
    a.set_ylabel("Validation loss (lower is better)")
    a.set_title("Quality vs. training time")
    a.legend(frameon=False)

    # 2. Val loss vs tokens seen
    a = ax[0, 1]
    for name, log, s in runs:
        a.plot([r["tokens_M"] for r in log], [r["val_loss"] for r in log], color=color(s),
               lw=2.2, label=label(s))
    a.axhline(target, color="#d64545", ls="--", lw=1.2)
    a.set_ylim(2.2, 4.6)
    a.set_xlabel("Training tokens seen (millions)")
    a.set_ylabel("Validation loss")
    a.set_title("Quality vs. data used (learning per token)")
    a.legend(frameon=False)

    # 3. Final loss + speed
    a = ax[1, 0]
    names = [label(s).replace(" (", "\n(") for _, _, s in runs]
    cols = [color(s) for _, _, s in runs]
    bars = a.bar(names, [s["final_val_loss"] for _, _, s in runs], color=cols, width=0.55)
    for b, (_, _, s) in zip(bars, runs):
        a.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
               f"{s['final_val_loss']:.3f}\n{s['tok_per_sec']:,} tok/s", ha="center", va="bottom", fontsize=9)
    a.set_ylim(0, max(s["final_val_loss"] for _, _, s in runs) * 1.3)
    a.set_ylabel("Final validation loss")
    a.set_title("Final score (and training speed)")

    # 4. Params, RAM, tokens
    a = ax[1, 1]
    metrics = ["Total params (M)", "Active params (M)", "Peak RAM (GB)", "Tokens seen (M)"]
    width = 0.8 / len(runs)
    for i, (_, _, s) in enumerate(runs):
        vals = [s["params"]["total"] / 1e6, s["params"]["active"] / 1e6, s["peak_ram_mb"] / 1024, s["tokens"] / 1e6]
        xs = [j + i * width for j in range(len(metrics))]
        bs = a.bar(xs, vals, width=width, color=color(s), label=label(s))
        for b, v in zip(bs, vals):
            a.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks([j + width * (len(runs) - 1) / 2 for j in range(len(metrics))])
    a.set_xticklabels(metrics, fontsize=9)
    a.set_title("Size, memory and data")
    a.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, format="jpeg", pil_kwargs={"quality": 92})
    print("saved", args.out)

    # Markdown table
    md = ["# Benchmark", "", f"![benchmark]({os.path.basename(args.out)})", "",
          "| Model | Train time | Final val loss | Time to reach baseline's final loss | Tokens/sec | Tokens seen | Params (total / active) | Peak RAM |",
          "|---|---|---|---|---|---|---|---|"]
    for name, log, s in runs:
        t = first_time_reaching(log, target) if s["model"] != "baseline" else s["minutes"]
        md.append(f"| {label(s)} | {s['minutes']:.0f} min | **{s['final_val_loss']:.3f}** | "
                  f"{f'{t:.1f} min' if t else '—'} | {s['tok_per_sec']:,} | {s['tokens'] / 1e6:.1f}M | "
                  f"{s['params']['total'] / 1e6:.2f}M / {s['params']['active'] / 1e6:.2f}M | {s['peak_ram_mb']:,} MB |")
    with open(args.md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("saved", args.md)


if __name__ == "__main__":
    main()
