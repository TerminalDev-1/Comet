# Comet-Image

*Part of the [Comet hub](../README.md).*

| Model | Status |
|---|---|
| Comet-Image-v4-Preview-2026-11-08 | **Next:** first public-data model (general images + NASA space images). In preparation. |
| Comet-Image-Preview-v3-2026-11-08 | 🗄️ **Legacy, discontinued** (2026-11-08) |
| Comet-Image-Preview-v2-2026-11-08 | 🗄️ **Legacy, discontinued** (2026-11-08) |
| Comet-Image-Preview-2026-11-08 (v1) | 🗄️ **Legacy, discontinued** (2026-11-08) |

*v1, v2 and v3 were trained only on Talmeez's screenshots and are discontinued. Their write-ups stay below as a record. Next: **Comet-Image-v4-Preview**, trained on public general images (space, the sun and moon, everyday photos), not screenshots. "Preview" means the first training pass is released, then the model keeps training until it's right.*

## Legacy: Comet-Image-Preview v1, v2 and v3 (discontinued)

![Comet-Image-Preview-v2 samples (legacy)](results/comet_image_v2_grid.png)

*Legacy: 64 images from **Comet-Image-Preview-v2-2026-11-08** (shown at 2× size). None of them is a real screenshot.*

The first **image model** in the Comet family: a small **Diffusion Transformer (DiT)** trained with flow matching (the Stable Diffusion 3 recipe) on Talmeez's own screenshots, on the RTX 4060 Ti over SSH.

*The screenshots, the training crops, the copy-check images and the weights are private: they stay out of this repo (in `Comet-Image/private/` locally, `~/comet_private` on the PC).*

### Setup (v1)

| | |
|---|---|
| Data | 2,647 screenshots (laptop + PC) → **87,186 random square crops** at 64×64 (`prep_images.py`), mirrored at random during training |
| Model | DiT, 22.0M params: 4×4 patches (256 tokens), d=384, 8 layers, 6 heads, adaLN-Zero time conditioning (`dit.py`) |
| Training | 20 min, batch 128, AdamW 3e-4 with warmup and cooldown, bf16, EMA 0.999. 5,532 steps, ~590 images/sec, 6.4 GB VRAM (`train_image.py`) |
| Sampling | 60 Euler steps from pure noise |
| Loss | 0.90 (min 1) → 0.11 (min 2) → 0.061 (min 10) → **0.053** (min 20) |

### What v1 makes

Small images that clearly *look like screenshots*, without being any particular one:
- **Document and web pages:** white backgrounds, a bolder heading line, rows of tiny "text" (fake squiggles, not readable), paragraph gaps, the odd blue link.
- **Dark-mode windows:** black or charcoal backgrounds with faint text rows and small colored accents (green, orange, blue).
- **App layouts:** horizontal bands, like a toolbar, then content, then a bottom bar; blue desktop-like backgrounds.
- **Weak spots:** busy or photo-like crops come out as blocky mosaics (the 4×4 patch grid shows). No readable text, which is expected at 64×64 and 20 minutes.

Progress across grids (min 4 → 8 → 12 → 16 → 20): static with the right colors → text rows and regions → headings and paragraphs → app-window bands → sharper, with smoother flat areas.

### Copy check (v1)

For 16 samples, the nearest of all 87k training crops (pixel distance) was found and compared side by side. The nearest real crops have readable UI text and sharp edges; the generated ones don't match any of them. Very small distances (10–12) only happen for near-blank tiles (all dark or all white). **The model makes new screenshot-like images; it isn't copying training images.**

### v2: 20 more minutes (Comet-Image-Preview-v2-2026-11-08, legacy)

Continued from v1's weights for another 20 minutes (lr 2e-4), with 8 new screenshots added (87,450 crops). Loss **0.053 → 0.046**. Compared with v1: crisper text pages (evenly spaced lines, indents, bold headings, blue link lines), smoother gradients (blue and gray backgrounds no longer blocky), and fewer chaotic mosaics. Busy photo-like crops are still the weak spot.

**Copy check (v2):** the closest matches (pixel distance 3–13) are all near-black tiles next to other near-black tiles (an empty dark screen next to an empty dark screen). Every tile with content differs clearly from its nearest real crop. It still makes new images rather than copying.

### v3: 128×128, sharper and more colorful (Comet-Image-Preview-v3-2026-11-08, legacy)

![Comet-Image-Preview-v3 samples](results/comet_image_v3_grid.png)

Continued from v2, but at **128×128** (4× the pixels): v2's 16×16 position map was stretched to 32×32 so its weights carry over. Two training-recipe changes:
- `--t_shift 0.5`: more practice on nearly-clean images, where fine detail is learned
- `--color_boost 2`: colorful and detailed crops are picked up to ~2.3× more often

Data: 45,050 crops at 128×128 (kept in RAM; only batches go to the GPU). Batch 32 (batch 64 needed 10 GB and overflowed the 8 GB card), lr 1e-4, 20 min, 4,602 steps, 123 images/sec, 5.4 GB VRAM. Loss 0.055 → **0.048**.

![v2 vs v3, same seeds](results/v2_vs_v3.png)

*Left: v2 (64×64, upscaled). Right: v3 (128×128). Same seeds.*

- ✅ **Less blurry:** text lines are crisp and word-shaped instead of a soft smear.
- ✅ **More color:** cyan, blue, orange, yellow and green accents show up far more often.
- ⚠️ **Messier layouts:** more collage-like compositions and noise in dark areas. 22M parameters now fill 4× the pixels with only 20 minutes of practice at this size.

**Copy check (v3):** nearest-real distances 22–86 (higher than v2's), and every sample clearly differs from its nearest training crop.

## Usage (on the GPU PC, from `Comet-Image/`)

```
python prep_images.py --src <screenshot folders> --out <private>/screens64.npy
python train_image.py --data <private>/screens64.npy --out <private>/runs --name Comet-Image-Preview-2026-11-08 --minutes 20
python generate.py <private>/runs/<run>/ema.pt --out <folder> --n 1 --seed 7 --smooth
```

## Next ideas

- Longer training and/or 128×128 resolution (needs about 4× compute)
- Patch size 2 for finer detail (4× more tokens, slower)
- Muon instead of AdamW, as in the language models
