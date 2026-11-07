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

## Tuning pass (second round)

**Speed fixes:**
1. **Flat row indexing** (`index_copy` / `index_select` into one flat buffer) replaced 2D advanced indexing. Forward + backward went from 1.7× dense to 1.4× dense.
2. **AdamW for expert weights** (`--expert_optim adamw`). Muon's cost grows with parameter count, and the expert pool is most of the parameters: optimizer step 385 ms → 38 ms.
3. **"Lite" EMOE:** smaller always-on expert (128) + 2 experts of 128 per token, so *less* math per token than dense while holding 2.6× the parameters.

| EMOE version | Total params | Speed vs. dense |
|---|---|---|
| First pass (64×256, Muon experts) | 16.8M | 55–60% |
| Tuned (64×256, AdamW experts) | 16.8M | **79%** |
| Tuned lite (128×128, top-2, shared 128) | 16.7M | **91%** |

**3-minute A/B after tuning:**

| Model | Total params | Val loss | Next-token acc | Tokens/sec |
|---|---|---|---|---|
| **Dense Comet v1** | 6.5M | **3.061** | **39.7%** | 5,072 |
| EMOE tuned (64×256) | 16.8M | 3.230 | 37.8% | 3,861 |
| **EMOE tuned lite (128×128)** | 16.7M | 3.202 | 38.1% | 4,173 |
| EMOE first pass (for reference) | 16.9M | 3.419 | 35.8% | 2,801 |

**Result:** tuning cut the gap to dense from 0.36 to 0.14 and nearly closed the speed gap. EMOE still doesn't beat dense in short runs. Per token seen, it's about equal to dense, so the extra 10M parameters aren't adding learning yet at ~1M training tokens. This matches what's known about mixture-of-experts: the payoff comes when each expert sees much more data.

## Ideas to make EMOE pay off (next pass)

1. **Speed:** bigger batches (more tokens per expert per step), fused routing, fewer copies. Target ≥85% of dense.
2. **Per-layer pools vs. one shared pool:** test whether sharing experts across layers costs quality.
3. **Fewer, bigger experts at the start** (e.g. 16×512, top-2): less routing overhead, less padding.
4. **Pair with multi-token prediction:** more learning signal per token helps the extra parameters learn faster.
