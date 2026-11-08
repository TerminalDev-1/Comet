# Comet

**Comet is Talmeez Ahmad's model brand: an umbrella of models, not one architecture.** Any model can join the family: language models, image models, image classifiers, specialized and very tiny models, looped Transformers. When Talmeez says "train a model in the Comet family", build and train it here under the Comet name.

- **Naming:** `Comet-<Variant>-YYYY-MM-DD` for experimental models (e.g. `Comet-Preview-2026-11-08`, `Comet-Image-Preview-2026-11-08`).
- **Architecture:** Transformer first, looped Transformer second. Don't modify architectures for efficiency unless asked; improve training (recipe, data, schedule, hardware) instead.
- **Where training runs:** the RTX 4060 Ti PC, driven from the laptop over SSH (`ssh gpupc`, project at `C:\Users\User\Comet`). See SSH_SETUP.md. Use `--device cuda`. Post ETA updates every 1–2 minutes.
- **Private data stays private:** screenshot-based data, its crops, the weights and the copy-check images live in `private/` (gitignored) on the laptop and `C:\Users\User\comet_private` on the PC. Only code, text write-ups and generated sample grids that Talmeez approved get published.

## Family so far

| Model | What | Write-up |
|---|---|---|
| Comet v1 (chapter 1, sunset) | Efficient Transformer on a laptop CPU: baseline quality 2.55× faster | README.md, results/ |
| Comet-Preview | 15.7M language model that thinks in `<think>` tags; 68% on new stories (v2) | results/comet_preview.md |
| Comet-Image-Preview | 22M Diffusion Transformer trained on screenshots | results/comet_image.md |

## Main scripts

- `train.py` (language models; `--device cuda`, `--reason`, `--data think_`, `--init`), `reason.py`, `think.py`, `data/gen_think.py`
- `comet_image/` (`prep_images.py`, `dit.py`, `train_image.py`)
