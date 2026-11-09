# Next session: big dataset for Comet-Image (PD12M)

**Claude: this is a queued task. At the start of the next session, tell Talmeez it's waiting and ask if he's ready. When he says he's ready, run it end to end. Target: done on 2026-11-10.** When it's finished, mark it done at the bottom of this file (or delete the file) and commit.

## Why

Comet-Image has plateaued: 3 hours of training on the same ~9.5k images (NASA space + Microcosmos) only moves eval loss ~0.007 per hour, and images stay mostly abstract. A 64.5M grown model wasn't better either (dropped). The fix is **much more, much more varied data**. A first test added 1,700 PD12M photos (`latents256_plus.npy`), too few and too short to show gains.

## Steps (all on the GPU PC over `ssh gpupc`, from `C:\Users\User\Comet\Comet-Image`)

1. **Fetch 20,000–50,000 PD12M images** (public domain / CC0, everyday subjects) with `fetch_pd12m.py` into `C:\Users\User\comet_data\pd12m` (1,700 are already there; use a different `--seed` or skip existing files).
   - Speed seen: ~500 images/min at 64 workers, so 20k ≈ 40 min, 50k ≈ 100 min. Try `--workers 128`. No GPU needed; can run while nothing else trains.
   - One metadata parquet has 100k rows (~75% download OK). For more than ~60k, add a `--shard` option to read more parquet files.
   - Launch detached (`Invoke-CimMethod Win32_Process Create`, see SSH_SETUP.md); post progress every ~60 s.
2. **Encode** with `encode_latents.py --src C:\Users\User\comet_data\pd12m --out C:\Users\User\comet_data\pd_latents256_big.npy` (6 views per image; 20k images ≈ 120k latents ≈ 940 MB fp16). Check it still fits on the 8 GB GPU next to training (old latents are 467 MB); if not, use `--crops 2`.
3. **Merge** with `C:\Users\User\merge.py` (old `latents256.npy` + new) into `latents256_big.npy`.
4. **Train** from the **latest base checkpoint**: `Comet-Image-v4.2-Flash-Preview-2026-11-09-2134` (`runs\Comet-Image-v4.2-Flash-Preview-data-run2\ema.pt`). Check `comet-image-checkpoints` memory first and call out anything older. Recipe: `train_latent.py --lr 2e-4 --warmup 100 --t_shift 0.5 --ckpt_every 5 --grid_every 10`, 32.6M model (no `--grow_layers`). Ask Talmeez how long (he prefers short: 15–30 min); ETA every minute.
5. **Check:** fixed eval loss, a side-by-side grid vs 2134 (same seeds, 30 steps), and `copy_check_latent.py`.
6. **Release** as `Comet-Image-v4.2-Flash-Preview-YYYY-MM-DD-HHMM` (UK time when training finished), GitHub pre-release with notes (credit PD12M / public domain + CC0). Update both READMEs, `benchmarks/frontier_comet.py`, the checkpoint memory, and the hub CLAUDE.md. Optionally distill an 8-step Lite from it with `distill_flash.py` (15 min).

## Watch out for

- Power cuts have hit twice: checkpoints every 5 min, and re-copy launch scripts after a cut (one got corrupted with null bytes).
- PowerShell over SSH eats `$N`-style variables inside double quotes: write file names out literally.
- Public text (commits, READMEs, release notes) stays professional.

## Status

- [ ] Not started (queued 2026-11-09).
