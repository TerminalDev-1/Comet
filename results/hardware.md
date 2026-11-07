# Phase 0: Hardware results

Machine: AMD Ryzen 5 8640HS (Zen 4, 6 cores / 12 threads), Radeon 760M, 15 GB RAM, Windows 11.
Software: Python 3.13, PyTorch 2.15.0 (CPU build). Raw numbers: [hw_cpu.json](hw_cpu.json). Script: `tools/hwbench.py`.

## CPU matrix multiply (2048 × 2048 × 2048)

| Threads | FP32 GFLOPS | BF16 GFLOPS |
|---|---|---|
| 4 | 266 | 765 |
| 6 | 343 | 757 |
| 12 | 386 | 823 |

## Small-transformer shapes (8192 tokens, d_model → 4·d_model)

| d_model | FP32 | BF16 |
|---|---|---|
| 256 | 283 | 705 |
| 384 | 333 | 842 |
| 512 | 285 | 725 |

## EMOE tiny-expert shapes (1024 tokens, 384 → hidden)

| Expert hidden size | FP32 | BF16 |
|---|---|---|
| 64 | 172 | 435 |
| 128 | 317 | 652 |
| 256 | 360 | 752 |
| 512 | 382 | 804 |

## Findings

1. **BF16 works and is ~2.2–2.5× faster than FP32.** Use BF16 everywhere we can.
2. **4 threads get ~93% of 12-thread BF16 speed.** Default to **4–6 threads** for cooler, quieter runs; use 12 only when chasing max speed.
3. **EMOE expert size:** hidden 64 loses ~45% efficiency; **128 is the smallest good size, 256+ is near full speed.** Start EMOE with expert hidden size 128–256.
4. **Training speed estimate:** BF16 peak is ~0.75–0.85 TFLOPS. Real training usually reaches ~40–50% of peak, so ~300–400 GFLOPS. That matches the plan's time-budget table: a ~5M-active model sees ~15–20M tokens in 25 minutes.

## Radeon 760M (DirectML)

**Not tested yet.** `torch-directml` has no build for Python 3.13 (it needs Python 3.12 or older). See plan.md, Phase 0.
