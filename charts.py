"""Lab-style bar charts for the README.

    python charts.py

Outputs:
  results/benchmark.jpg   headline: accuracy, time to baseline quality, speed-up (12-min runs)
  results/ablations.jpg   what each part adds (3-minute runs)
Numbers come from results/runs/*/summary.json and log.jsonl.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "results", "runs")

INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
GRAY, COMET, COMET_LIGHT, EXPER = "#b8bec7", "#2f6fed", "#8fb0f5", "#e8743b"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left",
})


def summary(name):
    return json.load(open(os.path.join(RUNS, name, "summary.json")))


def log(name):
    return [json.loads(l) for l in open(os.path.join(RUNS, name, "log.jsonl"))]


def time_to(name, target):
    rows = log(name)
    for a, b in zip(rows, rows[1:]):
        if b["val_loss"] <= target < a["val_loss"]:
            f = (a["val_loss"] - target) / (a["val_loss"] - b["val_loss"])
            return a["train_min"] + f * (b["train_min"] - a["train_min"])
    return summary(name)["minutes"]


def bars(ax, labels, values, colors, fmt, title, subtitle, ymax=None, better="higher", hatches=None):
    xs = range(len(values))
    bs = ax.bar(xs, values, color=colors, width=0.62, zorder=3)
    if hatches:
        for b, h in zip(bs, hatches):
            if h:
                b.set_hatch(h)
                b.set_edgecolor("white")
    top = ymax or max(values) * 1.18
    for b, v, c in zip(bs, values, colors):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.015, fmt(v), ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=INK if c != GRAY else MUTED)
    ax.set_ylim(0, top)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=10.5, color=INK)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, labelcolor=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=26, color=INK)
    arrow = "↑ higher is better" if better == "higher" else "↓ lower is better"
    ax.text(0, 1.035, f"{subtitle}   ·   {arrow}", transform=ax.transAxes, fontsize=9.5, color=MUTED)


def headline():
    base = summary("baseline_d256_l6")
    dense, mod = summary("race12_dense"), summary("race12_mod")
    target = base["final_val_loss"]
    labels = ["Baseline GPT\n25 min", "Comet\n12 min", "Comet + MoD\n12 min"]
    colors = [GRAY, COMET, COMET_LIGHT]

    acc = [base["final_val_acc"] * 100, dense["final_val_acc"] * 100, mod["final_val_acc"] * 100]
    t = [base["minutes"], time_to("race12_dense", target), time_to("race12_mod", target)]
    speed = [base["minutes"] / x for x in t]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6.2), dpi=150)
    fig.patch.set_facecolor("white")
    bars(axs[0], labels, acc, colors, lambda v: f"{v:.1f}%", "Next-token accuracy",
         "Held-out TinyStories", ymax=56)
    bars(axs[1], labels, t, colors, lambda v: f"{v:.1f} min", "Time to baseline quality",
         f"Minutes to reach val loss {target}", ymax=29, better="lower")
    bars(axs[2], labels, speed, colors, lambda v: f"{v:.1f}×", "Training speed-up",
         "vs. Baseline GPT", ymax=3.0)

    fig.text(0.012, 0.965, "Comet", fontsize=24, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.895, "A Transformer variant that trains 2.5× faster on a laptop CPU  "
             "(AMD Ryzen 5 8640HS, no GPU, BF16)", fontsize=12, color=MUTED, va="top")
    fig.text(0.012, 0.02,
             f"Same data (TinyStories, 8k BPE), ~6.5–6.9M parameters each. Comet bars: one session, run back to back. "
             f"Best single Comet run: {summary('vt_v1_muon_12m')['final_val_acc'] * 100:.1f}% accuracy, "
             f"{time_to('vt_v1_muon_12m', target):.1f} min to baseline quality. One run per setting.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.83), w_pad=3)
    out = os.path.join(ROOT, "results", "benchmark.jpg")
    fig.savefig(out, format="jpeg", pil_kwargs={"quality": 93})
    print("saved", out)


def ablations():
    rows = [  # (label, run, color, hatch)
        ("Baseline\nGPT", "ab3_baseline", GRAY, None),
        ("+ Comet\narchitecture", "ab3_vt_adamw", COMET_LIGHT, None),
        ("+ Muon\n= Comet", "ab3_vt_muon", COMET, None),
        ("Comet\n(control*)", "ab3_dense_control", COMET, None),
        ("+ MoD", "ab3_mod_every2", EXPER, None),
        ("+ MTP\nlight", "ab3_mtp_d1_light", EXPER, None),
        ("+ EMOE\nlite", "ab3_emoe2_lite_e128k2h128", EXPER, None),
        ("+ EMOE", "ab3_emoe2_e64k2h256", EXPER, None),
        ("+ MTP\nfull", "ab3_mtp_d1", EXPER, None),
        ("+ MoD\nevery layer", "ab3_mod_every1", EXPER, None),
    ]
    vals = [summary(r)["final_val_acc"] * 100 for _, r, _, _ in rows]
    fig, ax = plt.subplots(figsize=(15, 6.4), dpi=150)
    fig.patch.set_facecolor("white")
    bars(ax, [r[0] for r in rows], vals, [r[2] for r in rows], lambda v: f"{v:.1f}%",
         "What each part adds", "Next-token accuracy after a 3-minute training run", ymax=46,
         hatches=[r[3] for r in rows])
    ax.axvline(2.5, color=GRID, lw=1.5, ls="--", zorder=1)
    ax.text(1.0, 44.2, "Core build-up", ha="center", fontsize=10, color=MUTED, fontweight="bold")
    ax.text(6.0, 44.2, "Experimental options (same session as control*)", ha="center", fontsize=10,
            color=MUTED, fontweight="bold")
    fig.text(0.012, 0.965, "Comet — ablations", fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.9, "Each bar is one 3-minute run with its own full learning-rate schedule, "
             "same data and laptop", fontsize=12, color=MUTED, va="top")
    fig.text(0.012, 0.02,
             "*The laptop ran slower in the later session, so Comet was re-run as a control; experimental options "
             "are compared to that bar. One run per setting; differences under ~0.5 points may be noise.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.84))
    out = os.path.join(ROOT, "results", "ablations.jpg")
    fig.savefig(out, format="jpeg", pil_kwargs={"quality": 93})
    print("saved", out)


if __name__ == "__main__":
    headline()
    ablations()
