# Comet (hub)

**Comet is Talmeez Ahmad's model brand: an umbrella of models, not one architecture.** This repo (`C:\Users\gamer\Desktop\Comet`, GitHub TerminalDev-1/Comet) is the **hub for training every Comet model**. Each model line gets its own folder inside it. Any model can join: language models, image models, image classifiers, specialized and very tiny models, looped Transformers. When Talmeez says "train a model in the Comet family", make a `Comet-<Name>/` folder here (or use an existing one) and train it.

## Layout

| Folder | Contents |
|---|---|
| `Comet-LM/` | Language models: Comet v1 (sunset efficient Transformer) and Comet-Preview (`<think>` reasoning). Code: `comet/`, `train.py` (`--device cuda`, `--reason`, `--data think_`, `--init`), `reason.py`, `think.py`, `data/gen_think.py`. Data `.bin` files and `checkpoints/` are untracked. |
| `Comet-Image/` | Image models: v1–v3 (Legacy, screenshot-trained), v4-Preview (public data), v5-Flash (planned). `prep_images.py`, `dit.py`, `train_image.py`, `generate.py`. `private/` holds grids and single images (gitignored). |
| root | `README.md` (family index; keep it updated), `LICENSE.md`, `SSH_SETUP.md`, `pc_setup.ps1`, `research.md`. |

Shared Python environment: `Comet/.venv` (run e.g. `../.venv/Scripts/python train.py` from a model folder).

## Rules

- **Naming:** `Comet-<Variant>-YYYY-MM-DD` for experimental models. **"Preview" + date** means: the first training pass gets released, then the model keeps being trained further until it's perfect. Discontinued models are labeled 🗄️ Legacy (weights kept privately unless Talmeez says delete).
- **Image line:** Comet-Image-Preview v1/v2/v3 (screenshot-trained) are Legacy. `Comet-Image-v4-Preview` (73.6M) was stopped early and is Legacy/experimental. Current: `Comet-Image-v4.1-Preview` (smaller, faster, public general + NASA space images), then `Comet-Image-v5-Flash` (fast few-step generation, trained on screenshots + public data).
- **Architecture:** Transformer first, looped Transformer second. Don't modify architectures for efficiency unless asked; improve training (recipe, data, schedule, hardware) instead.
- **Where training runs:** the RTX 4060 Ti PC over SSH (`ssh gpupc`). Its clone of this repo is at `C:\Users\User\Comet` with its own `.venv`; untracked data and checkpoints sit inside the model folders there too (e.g. `Comet-LM\data`). Private image data lives in `C:\Users\User\comet_private`. Post ETA updates every 1–2 minutes.
- **Private data stays private:** screenshot-based data, its crops, weights trained on it and copy-check images never get pushed. Only code, text write-ups and generated sample grids that Talmeez approved get published.
- When a new model is added, add it to the family table in `README.md`.
