"""Frontier Comet: one launch-style benchmark chart for the whole Comet family.

    python benchmarks/frontier_comet.py      ->  benchmarks/frontier_comet.jpg

Like a frontier-lab launch chart, but every bar is a Comet model (or the baseline it was built to beat).
Each panel is its own task, so bars are only compared within a panel. Numbers come from the write-ups:
Comet-LM/results/benchmark.md, Comet-LM/results/comet_preview.md, Comet-Image/README.md.
"""
import os
import sys

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "Comet-LM"))
from charts import BG, COMET, COMET_LIGHT, EXPER, GRAY, MUTED, bars  # noqa: E402  (shared neon style)

# (label, value, color) per panel
PANELS = [
    dict(title="Training speed-up", subtitle="To baseline quality, CPU", better="higher",
         fmt=lambda v: f"{v:.2f}×", ymax=3.0,
         rows=[("Baseline\nGPT", 1.0, GRAY), ("Comet\nv1", 25.0 / 9.8, COMET)]),
    dict(title="Language modeling", subtitle="TinyStories val loss", better="lower",
         fmt=lambda v: f"{v:.2f}", ymax=3.0,
         rows=[("Baseline\n25 min CPU", 2.607, GRAY), ("Comet v1\n12 min CPU", 2.441, COMET_LIGHT),
               ("Preview\n10 min GPU", 1.74, COMET)]),
    dict(title="Reasoning", subtitle="400 held-out stories", better="higher",
         fmt=lambda v: f"{v:.1f}%", ymax=85,
         rows=[("Comet-Preview\nv1", 50.5, COMET_LIGHT), ("Comet-Preview\nv2", 68.0, COMET)]),
    dict(title="Image training speed", subtitle="Images/sec, 4060 Ti", better="higher",
         fmt=lambda v: f"{v:.0f}", ymax=450,
         rows=[("Pixel DiT\n73.6M", 371 / 8, GRAY), ("v4-Flash\nlatent 32.6M", 371, COMET)]),
    dict(title="Image generation", subtitle="Fixed eval loss", better="lower",
         fmt=lambda v: f"{v:.3f}", ymax=0.85,
         rows=[("v4-Flash\n15 min", 0.7182, GRAY), ("v4.1-Flash\n120 min", 0.6644, COMET_LIGHT),
               ("v4.2-Preview\n180 min", 0.6574, COMET)]),
    dict(title="8-step distillation", subtitle="Gap to 30-step images", better="lower",
         fmt=lambda v: f"{v:.4f}", ymax=0.095,
         rows=[("v4.1-Flash\n8 steps", 0.0794, GRAY), ("v4.1-Flash-Lite\n8 steps", 0.0057, COMET)]),
]


def main():
    fig, axs = plt.subplots(1, len(PANELS), figsize=(25, 7), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax, p in zip(axs, PANELS):
        rows = [r for r in p["rows"] if r[1] is not None]
        bars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], p["fmt"],
             p["title"], p["subtitle"], ymax=p["ymax"], better=p["better"])
    fig.text(0.010, 0.965, "Frontier Comet", fontsize=28, fontweight="bold", color=COMET, va="top",
             path_effects=[pe.withStroke(linewidth=7, foreground=COMET, alpha=0.15)])
    fig.text(0.010, 0.885, "The whole Comet family on one chart: language, reasoning and image models by Talmeez Ahmad, "
             "trained in 10–180 minutes on a laptop CPU and an RTX 4060 Ti", fontsize=12.5, color=MUTED, va="top")
    fig.text(0.010, 0.02,
             "Each panel is a different task; compare bars within a panel only. One run per setting. Pixel DiT speed "
             "estimated from the 8× speed-up measured against it. Image loss: flow-matching loss on 1,024 fixed latents with fixed noise and times (EMA weights). Distillation gap: MSE between 8-step and 30-step latents from 128 fixed seeds.",
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.82), w_pad=3)
    out = os.path.join(HERE, "frontier_comet.jpg")
    fig.savefig(out, format="jpeg", pil_kwargs={"quality": 93})
    print("saved", out)


if __name__ == "__main__":
    main()
