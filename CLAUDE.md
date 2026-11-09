# Comet (hub)

**Comet is Talmeez Ahmad's model brand: an umbrella of models, not one architecture.** This repo (`C:\Users\gamer\Desktop\Comet`, GitHub TerminalDev-1/Comet) is the **hub for training every Comet model**. Each model line gets its own folder inside it. Any model can join: language models, image models, image classifiers, specialized and very tiny models, looped Transformers. When Talmeez says "train a model in the Comet family", make a `Comet-<Name>/` folder here (or use an existing one) and train it.

## Layout

| Folder | Contents |
|---|---|
| `Comet-LM/` | Language models: Comet v1 (sunset efficient Transformer) and Comet-Preview (`<think>` reasoning). Code: `comet/`, `train.py` (`--device cuda`, `--reason`, `--data think_`, `--init`), `reason.py`, `think.py`, `data/gen_think.py`. Data `.bin` files and `checkpoints/` are untracked. |
| `Comet-Image/` | Image models: v1–v3 (Legacy, screenshot-trained), v4-Flash-Preview (public data, keeps training). `prep_images.py`, `dit.py`, `train_image.py`, `generate.py`. `private/` holds grids and single images (gitignored). |
| `benchmarks/` | `frontier_comet.py` → `frontier_comet.jpg`: the "Frontier Comet" family chart at the top of the README (update it when a model gets a new result). |
| root | `README.md` (family index; keep it updated), `LICENSE.md`, `SSH_SETUP.md`, `pc_setup.ps1`, `research.md`. |

Shared Python environment: `Comet/.venv` (run e.g. `../.venv/Scripts/python train.py` from a model folder).

## Rules

- **Naming:** `Comet-<Variant>-YYYY-MM-DD` for experimental models. **"Preview" + date** means: the first training pass gets released, then the model keeps being trained further until it's perfect. Discontinued models are labeled 🗄️ Legacy (weights kept privately unless Talmeez says delete).
- **Image line:** Comet-Image-Preview v1/v2/v3 (screenshot-trained) are Legacy. Current: `Comet-Image-v4-Flash-Preview-2026-11-08` (latent DiT on SD-VAE latents, 32.6M, 256×256, 8-step sampling; public general + NASA space images; `encode_latents.py` → `train_latent.py`; data in `C:\Users\User\comet_data`). A 73.6M v4-Preview attempt was stopped early with no checkpoint and is treated as never released. A later model may mix screenshots + public data (name TBD).
- **Architecture:** Transformer first, looped Transformer second. Don't modify architectures for efficiency unless asked; improve training (recipe, data, schedule, hardware) instead.
- **Where training runs:** the RTX 4060 Ti PC over SSH (`ssh gpupc`). Its clone of this repo is at `C:\Users\User\Comet` with its own `.venv`; untracked data and checkpoints sit inside the model folders there too (e.g. `Comet-LM\data`). Private image data lives in `C:\Users\User\comet_private`. Post ETA updates every 1–2 minutes.
- **Private data stays private:** screenshot-based data, its crops, weights trained on it and copy-check images never get pushed. Only code, text write-ups and generated sample grids that Talmeez approved get published.
- When a new model is added, add it to the family table in `README.md`.
