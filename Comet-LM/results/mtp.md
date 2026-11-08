# Multi-token prediction (MTP)

## What was built (`comet/model.py`, `MTPModule`)

DeepSeek-V3 style, training only:
- At each position, combine the model's final hidden state with the embedding of the *next* token, run it through one extra transformer block, and predict the token **after** that (t+2), using the shared output layer.
- The MTP loss is added with weight 0.3. The logged training loss and all validation numbers are **next-token only**, so they compare directly with other runs.
- The output layer is costly (8,192-word vocabulary), so the MTP head scores only a random sample of positions (`mtp_frac`).
- **Light version** (`mtp_block=false`): skips the extra block (just a projection), scores 25% of positions.

Turn on with `--set mtp_depth=1` (add `mtp_block=false mtp_frac=0.25` for light).

## 3-minute A/B (same session, plus a fresh dense control)

The laptop was slower in this session than when the original dense run was made (4,641 vs 5,072 tokens/sec), so dense Comet was **rerun as a control** right after. All rows below are from the same session.

| Model | Val loss | Next-token acc | Tokens/sec |
|---|---|---|---|
| **Dense Comet (control)** | **3.115** | **39.1%** | 4,641 |
| MTP light (no block, 25% positions) | 3.172 | 38.5% | 4,108 |
| EMOE lite (128×128, top-2) | 3.202 | 38.1% | 4,173 |
| EMOE tuned (64×256) | 3.230 | 37.8% | 3,861 |
| MTP full (extra block, 50% positions) | 3.273 | 37.4% | 3,381 |

(Original dense run from earlier, faster session: 3.061, 39.7%.)

## Verdict

- **Neither MTP nor EMOE beats dense Comet in a 3-minute run.** Both cost speed (MTP light −11%, MTP full −27%), and neither learns enough extra per token at this tiny scale to make up for it.
- **This matches published results:** MTP helps large models (billions of parameters) and hurts or does nothing for small ones. MoE pays off with much more data per expert.
- **Light MTP is the closest** (+0.057). With a smaller weight or a longer run it might break even, but it's not a clear win.
- The 12-minute races for EMOE and MTP were **not run**: the 3-minute results say they'd lose, so they'd waste time.

## What this tells us about Comet's direction

On this laptop, within 10–15 minutes, the wins so far come from **making each step cheaper or smarter at no extra cost** (Muon, QK-Norm, RoPE, CPU-tuned kernels). Features that **add compute or parameters per step** (EMOE, MTP) don't pay off in such short runs. The remaining plan items that **cut compute** (Mixture of Depths, adaptive loop depth) fit the time budget better.
