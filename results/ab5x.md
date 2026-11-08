# 3-minute A/B tests toward the 5× goal (2026-11-08)

Each run: Comet v1, 3 minutes, Muon + WSD, one change. A plain control ran first and last.
Controls: **3.056** and **3.051**, so the start and end agree, but the laptop slowed down in the middle (softcap, U-Net, batch 8 ran at 3,900–4,500 tok/s instead of ~5,200), which handicaps those three runs.

| Run | Change | Val loss | vs control (~3.054) | Accuracy | Tokens/sec | Tokens seen |
|---|---|---|---|---|---|---|
| control | none | 3.056 | | 39.7% | 5,162 | 0.93M |
| **ctxwarm** | **context 64 → 256 over the first 30% (same tokens per step)** | **2.993** | **−0.061 ✅** | **40.5%** | 5,036 | 0.91M |
| ns3 | Muon with 3 Newton-Schulz steps instead of 5 | 3.079 | +0.025 ❌ | 39.5% | 5,025 | 0.91M |
| softcap | logit softcap 30 | 3.123 | +0.069 ❌* | 38.9% | 4,514 | 0.82M |
| unet | U-Net skips (layer 0→5, 1→4, 2→3) | 3.146 | +0.092 ❌* | 38.8% | 4,236 | 0.77M |
| batch8 | batch 8 instead of 16 (2× more steps) | 3.118 | +0.064 ❌* | 39.1% | 3,878 | 0.70M |
| muonlr03 | Muon LR 0.03 instead of 0.02 | 3.060 | +0.006 🟰 | 39.6% | 5,209 | 0.94M |
| control2 | none | 3.051 | | 39.7% | 5,239 | 0.95M |

\*Ran during the slowdown and saw 12–25% fewer tokens, so part of the loss gap is the laptop, not the idea. Not conclusive, but none of them looked like a win either.

**Winner: short-context warmup**, a clear 0.06 gain at equal tokens, the biggest single gain since Muon. It's a training-recipe trick, not an architecture change, so it carries over to any future architecture (including Comet 2, see research.md).

These were the last Transformer tweak experiments; the project is moving to a non-Transformer architecture (research.md).
