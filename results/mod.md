# Mixture of Depths (MoD)

## What was built (`comet/model.py`, `MoDRouter`; `comet/attention.py`)

- In selected layers, a tiny router (one linear layer) scores every token. **Tokens scoring > 0 go through the block; the rest skip it** and keep their value.
- **Causal routing:** each token decides from its own state only. The common "top 50% of the sequence" rule peeks at future tokens, which would make validation scores look better than they are. Instead, a side loss (BCE, weight 0.1) teaches the router to *predict* the top-50% pattern, and that keeps about 50% of tokens processed.
- **Verified:** changing a future token leaves earlier predictions exactly unchanged (max difference 0.0).
- Processed tokens run as a smaller sequence with their **original positions** (RoPE) and a causal mask, padded per batch. The block's update is scaled by sigmoid(score), which trains the router.

Turn on with `--set mod=true mod_every=2` (MoD on layers 1, 3, 5).

## 3-minute A/B (same session as the dense control)

| Model | Val loss | Next-token acc | Tokens/sec | Tokens processed in MoD layers |
|---|---|---|---|---|
| Dense Comet (control) | 3.115 | 39.1% | 4,641 | — |
| **MoD, every 2nd layer** | **3.106** | 39.0% | **5,037 (+9%)** | ~50% |
| MoD, every layer | 3.293 | 36.5% | 5,490 (+18%) | ~53% |

## Verdict

- **MoD on every other layer: a tie on quality (−0.009, within noise), 9% faster.** It's the first feature since Phase 3 that doesn't lose to dense Comet, and it does less work per token.
- **MoD on every layer loses clearly:** skipping in all layers throws away too much, more than the 18% speed-up buys back.
- The router held the 50% target well (49–54%) throughout training.

## Why the gain is small

At context 256 with d=256, attention is cheap, and the routing overhead (gather, scatter, padding, masks) eats part of the saving. MoD's gain should grow with longer contexts and wider models, where skipped work is a bigger share.
