# Comet (hub)

> Last queued task (big PD12M dataset run) is done: see [next-session.md](next-session.md).

**Comet is Talmeez Ahmad's model brand: an umbrella of models, not one architecture.** This repo (`C:\Users\gamer\Desktop\Comet`, GitHub TerminalDev-1/Comet) is the **hub for training every Comet model**. Each model line gets its own folder inside it. Any model can join: language models, image models, image classifiers, specialized and very tiny models, looped Transformers. When Talmeez says "train a model in the Comet family", make a `Comet-<Name>/` folder here (or use an existing one) and train it.

## Layout

| Folder | Contents |
|---|---|
| `Comet-LM/` | Language models: Comet v1 (sunset efficient Transformer) and Comet-Preview (`<think>` reasoning). Code: `comet/`, `train.py` (`--device cuda`, `--reason`, `--data think_`, `--init`), `reason.py`, `think.py`, `data/gen_think.py`. Data `.bin` files and `checkpoints/` are untracked. |
| `Comet-Space/` | Space-only image model (Comet-Space-Preview): fine-tuned from Comet-Image on Sun (SDO/Helioviewer), Mars rover and NASA images. Code is shared with `Comet-Image/` (`fetch_space.py`, `fetch_nasa.py --more/--queries`). |
| `Comet-Image/` | Image models: v1–v3 (Legacy, screenshot-trained), v4-Flash-Preview (public data, keeps training). `prep_images.py`, `dit.py`, `train_image.py`, `generate.py`. `private/` holds grids and single images (gitignored). |
| `benchmarks/` | `frontier_comet.py` → `frontier_comet.jpg`: the "Frontier Comet" family chart at the top of the README (update it when a model gets a new result). |
| root | `README.md` (family index; keep it updated), `LICENSE.md`, `SSH_SETUP.md`, `pc_setup.ps1`, `research.md`. |

Shared Python environment: `Comet/.venv` (run e.g. `../.venv/Scripts/python train.py` from a model folder).

## Rules

- **Naming:** `Comet-<Variant>-YYYY-MM-DD-HHMM` for experimental models: date + **UK time (Europe/London, 24h) when the training run finished**, e.g. `Comet-Image-v4-Flash-Preview-2026-11-09-1604`. Every trained checkpoint/release gets its own name; date-only names from before 2026-11-09 stay as they are. **GA releases** drop "Preview" and the date/time suffix (e.g. `Comet-Image-v4.1-Flash`, `-Lite` for the distilled variant); their checkpoint date+time goes in the release notes. **"Preview" + date** means: the first training pass gets released, then the model keeps being trained further until it's perfect. Discontinued models are labeled 🗄️ Legacy (weights kept privately unless Talmeez says delete).
- **Image line:** Comet-Image-Preview v1/v2/v3 (screenshot-trained) are Legacy. Current GA: `Comet-Image-v4.1-Flash` (30 steps); latest preview `Comet-Image-v4.2-Flash-Preview-2026-11-10-2038` (1855 + 30 min on `latents256_big2.npy` = 202,114 latents: earlier mix + 49,127 PD12M shard-2 photos + 44,082 space latents; held-out eval with `pd2_holdout.npy`; train further on `latents256_big2_dedup.npy`, 195,217 latents after `dedup_latents.py`); space line `Comet-Space-Preview-2026-11-10-2109` (1855 + 30 min on `space_latents.npy`, held-out `space_holdout.npy`; train further on `space_latents_dedup.npy`, 43,905 latents). Distill only GA checkpoints, never previews and `Comet-Image-v4.1-Flash-Lite` (8-step distillation); earlier preview line `Comet-Image-v4-Flash-Preview-2026-11-08` → `-2026-11-09-1641` (latent DiT on SD-VAE latents, 32.6M, 256×256, 8-step sampling; public general + NASA space images; `encode_latents.py` → `train_latent.py`; data in `C:\Users\User\comet_data`). A 73.6M v4-Preview attempt was stopped early with no checkpoint and is treated as never released. A later model may mix screenshots + public data (name TBD).
- **Architecture:** Transformer first, looped Transformer second. **Image models: latent DiT** (flow matching on SD-VAE latents, as in v4-Flash) is the preferred architecture: it trains efficiently with low RAM/VRAM. Don't modify architectures for efficiency unless asked; improve training (recipe, data, schedule, hardware) instead.
- **Where training runs:** the RTX 4060 Ti PC over SSH (`ssh gpupc`). Its clone of this repo is at `C:\Users\User\Comet` with its own `.venv`; untracked data and checkpoints sit inside the model folders there too (e.g. `Comet-LM\data`). Private image data lives in `C:\Users\User\comet_private`. Post ETA updates every 1–2 minutes.
- **Private data stays private:** screenshot-based data, its crops, weights trained on it and copy-check images never get pushed. Only code, text write-ups and generated sample grids that Talmeez approved get published.
- When a new model is added, add it to the family table in `README.md`.
- **Every release shows samples of the latest checkpoints:** put a sample grid of each newly released checkpoint in its model README and in the release notes, and update the "Latest checkpoints" section near the top of the hub `README.md` (one grid per line, same seed across lines, e.g. seed 42, 30 steps). Generated grids only; never private data.
- **Preview lines keep a "Development status" table** at the top of their model README (status, total training time, checkpoints, data, latest eval, why not GA yet, what GA needs). Update it with every preview release.
