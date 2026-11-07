# Phase 6 (first pass): EMOE

## What was built (`comet/emoe.py`)

- **Fine-grained tiny experts** in **one pool shared by all 6 layers**; each layer has its own router.
- **Always-on shared expert** per layer (hidden 192).
- Sigmoid routing, top-k, **no-extra-loss load balancing** (per-expert router bias).
- Batched expert compute: one batched matmul for all experts, each with a padded slot capped at 1.25× average load. Overflow tokens only get the shared expert.
- Muon handles the stacked 3D expert weights (one orthogonalization per expert matrix); routers use AdamW.

Turn on with `--set emoe=true n_experts=128 top_k=4 expert_hidden=128`.

## CPU speed (fwd + bwd + Muon step, tokens/sec)

| Version | Speed vs. dense |
|---|---|
| First try (Python loop over experts) | 15% (688 vs 4,660). Every expert slice created a full-size gradient copy. |
| Batched matmul | 45% |
| + capacity cap + BF16 buffer | **55–70%** |

## 3-minute A/B (each run with its own full schedule)

| Model | Total params | Active per token | Val loss | Next-token acc | Tokens/sec |
|---|---|---|---|---|---|
| **Dense Comet v1** | 6.5M | 6.5M | **3.061** | — | 5,072 |
| EMOE 64 experts × 256, top-2 | **16.8M** | 6.6M | 3.419 | 35.7% | 2,726 |
| EMOE 128 experts × 128, top-4 | **16.9M** | 6.7M | 3.419 | 35.8% | 2,801 |

**Per token** (same number of tokens seen), EMOE is about even with dense: 3.42 vs. ~3.46–3.48. That slightly flatters EMOE, because its final point had a fully cooled learning rate.

## Verdict

- **EMOE works and holds 2.6× the parameters for the same compute per token.**
- **It doesn't win yet within a 10–15 minute budget:** it runs at ~60% of dense speed, and the extra parameters don't learn faster per token at this tiny data scale (~0.5M tokens). MoE gains normally show up with much more data per expert.
- Not run: the 12-minute EMOE race. It would see ~40% fewer tokens than dense and lose, so it would have wasted the time.

## Ideas to make EMOE pay off (next pass)

1. **Speed:** bigger batches (more tokens per expert per step), fused routing, fewer copies. Target ≥85% of dense.
2. **Per-layer pools vs. one shared pool:** test whether sharing experts across layers costs quality.
3. **Fewer, bigger experts at the start** (e.g. 16×512, top-2): less routing overhead, less padding.
4. **Pair with multi-token prediction:** more learning signal per token helps the extra parameters learn faster.
