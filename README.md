# Comet

**Comet is a family of models by Talmeez Ahmad, made with Claude (Anthropic).** This repo is the hub: every Comet model lives in its own folder here, and they all share the same training setup (a home GPU driven from a laptop over SSH).

*Coming next: **Comet-Image-v4-Preview**, an image model trained on public general and space images.*

## The family

| Model | Folder | What it is | Best result |
|---|---|---|---|
| **Comet v1** (sunset) | [Comet-LM/](Comet-LM/) | Efficient Transformer trained on a laptop CPU | Baseline GPT quality **2.55× faster** (9.8 vs 25 min) |
| **Comet-Preview** | [Comet-LM/](Comet-LM/) | 15.7M language model that **thinks in English** in `<think>` tags | **68%** on held-out story reasoning (v2), 13 min of GPU training |
| **Comet-Image-v4-Preview** | [Comet-Image/](Comet-Image/) | Image model on public general + NASA space images | In preparation |
| Comet-Image-Preview v1, v2, v3 | [Comet-Image/](Comet-Image/) | 🗄️ Legacy, discontinued (screenshot-only) | |

Write-ups: [Comet-LM/README.md](Comet-LM/README.md) (v1), [Comet-LM/results/comet_preview.md](Comet-LM/results/comet_preview.md), [Comet-Image/README.md](Comet-Image/README.md).

## How the hub works

- **One folder per model line.** New Comet models (a classifier, a tiny specialist, a looped Transformer...) get their own folder: `Comet-<Name>/`.
- **Naming:** experimental models are `Comet-<Variant>-YYYY-MM-DD`, e.g. `Comet-Image-v4-Preview-2026-11-08`. **Preview** = the first training pass is released, then the model keeps training until it's right. Discontinued models are marked 🗄️ Legacy.
- **Training hardware:** an RTX 4060 Ti PC, controlled from the laptop over SSH, so nobody has to sit at it. Setup: [SSH_SETUP.md](SSH_SETUP.md) and [pc_setup.ps1](pc_setup.ps1).
- **One shared Python environment** at `Comet/.venv` (not in git).
- **Private data stays private:** anything made from personal files (screenshots, crops, weights trained on them) lives in a `private/` folder that git ignores.
- Ideas for future architectures: [research.md](research.md).

## License

Comet is by **Talmeez Ahmad**, made with Claude (Anthropic). You can use and fork it freely, but you must credit Talmeez Ahmad and include the attribution notice in your README. See [LICENSE.md](LICENSE.md).
