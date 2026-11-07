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

Setup: separate Python 3.12 environment (`.venv-dml`), `torch-directml` with PyTorch 2.4.1. Raw numbers: [hw_dml.json](hw_dml.json).
**BF16 is not supported** (DirectML hard-crashes), so FP16 was tested instead.

| Matmul test | FP32 GFLOPS | FP16 GFLOPS |
|---|---|---|
| 2048³ | 732 | 1178 |
| Model shapes (d=256–512) | 127–1039 (noisy) | 809–938 |
| Expert shapes (hidden 64–512) | 755–1010 | 670–907 |

On raw matrix math, the GPU is only ~1.1–1.4× the CPU's BF16 speed.

## Real training steps (forward + backward + AdamW)

Small GPT, context 256, batch 16, vocab 8192. Script: `tools/trainbench.py`. Raw numbers: [train_cpu.json](train_cpu.json), [train_dml.json](train_dml.json).

| Model | CPU FP32 (6 thr) | **CPU BF16 (6 thr)** | GPU FP32 | GPU FP16 |
|---|---|---|---|---|
| 6.9M params (d=256, 6 layers) | 3,299 tok/s | **7,151 tok/s** | 4,191 tok/s | 5,978 tok/s (loss = NaN) |
| 13.9M params (d=384, 6 layers) | 1,824 tok/s | **3,521 tok/s** | 2,831 tok/s | 4,163 tok/s (loss = NaN) |
| Tokens in 25 min (6.9M model) | 4.9M | **10.7M** | 6.3M | 9.0M |

## Verdict

- **CPU + BF16 is the training device.** It beats the GPU in FP32 by ~25–70%, and it's numerically stable.
- GPU FP16 is about equal to the CPU, but pure FP16 training gives NaN. DirectML has no working mixed-precision (autocast) support, and some optimizer ops fall back to the CPU. Not worth the complexity.
- The CPU and GPU share one power, heat and memory budget, so running both at once wouldn't add much.
- **Measured speed is lower than the first estimate:** a ~7M model sees ~11M tokens in 25 minutes, not 15–20M. Phase 3 upgrades (fused attention kernel, `torch.compile`, GQA) should push this up.
