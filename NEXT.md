# Comet — where we left off (2026-11-07, 11 PM)

Read this first tomorrow. Full detail lives in [plan.md](plan.md) and the `results/*.md` write-ups.

## Where things stand

| | Status |
|---|---|
| Phase 0 hardware | ✅ CPU + BF16 wins; Radeon 760M (DirectML) loses |
| Phase 1 data | ✅ TinyStories, filtered + deduplicated, 8k BPE tokenizer, 110M tokens |
| Phase 2 baseline | ✅ Plain GPT, 25 min → val loss **2.607**, accuracy **45.2%** |
| Phase 3 Comet v1 | ✅ **Reaches baseline quality in ~10 min (2.5×)**; 12 min → 2.42–2.45, 47.3–47.6% |
| EMOE | ✅ built + tuned (91% of dense speed, 2.6× params). ❌ doesn't beat dense in 3 min |
| Multi-token prediction | ✅ built (full + light). ❌ doesn't beat dense in 3 min |
| Mixture of Depths | ✅ built (causal, leak-free). 🟰 ties dense at 12 min, +24% tokens/sec |
| Looped blocks + adaptive depth | ⏳ not started |
| Reusable package (Phase 11) | ⏳ not started |
| Save to global CLAUDE.md (Phase 12) | ⏳ after the architecture is final |
| Research paper (Phase 13) | ⏳ not started |

Charts: [results/benchmark.jpg](results/benchmark.jpg) (headline bars), [results/ablations.jpg](results/ablations.jpg) (what each part adds), [results/curves.jpg](results/curves.jpg) (loss curves).

## What Comet is (settled)

**Comet is an original, efficiency-improved Transformer architecture.** It's still in the Transformer family (as are Llama, Mistral and nearly every frontier model), and its specific design genuinely trains 2.5× faster than a standard GPT on the same laptop. That speed-up didn't exist before Comet.

The published components it builds on, all credited:

| Part | Where it comes from |
|---|---|
| Rotary positions (RoPE) | Su et al., 2021 |
| QK-Norm | Dehghani et al., 2023 |
| Grouped-query attention | Ainslie et al., 2023 |
| SwiGLU | Shazeer, 2020 |
| Muon optimizer | Keller Jordan et al., 2024 (modded-nanoGPT speedrun) |
| Warmup-stable-decay schedule | MiniCPM, 2024 |

Comet's **own** contributions so far:
1. **EMOE's single expert pool shared by all layers** (each layer has its own router). This is the most original idea. It works, but hasn't beaten dense yet.
2. **CPU-specific findings:** on this CPU, "fused" attention is 1.7× slower and RMSNorm ~4× slower than the plain versions. These are useful, but they're engineering, not architecture.
3. **Causal MoD with RoPE on token subsets.** Close to the original MoD paper's predictor idea; it ties.

**How the paper says it:** "Comet: an efficiency-focused Transformer architecture that trains 2.5× faster on a laptop CPU." Claim the design and the measured result as Comet's; credit each component's source (as the Llama paper did); cite MoEUT (Csordás et al., 2024) next to EMOE. Making EMOE or looped blocks win at equal memory would make Comet stronger still, but the architecture claim doesn't depend on it.

## Ways to make Comet genuinely its own (pick tomorrow)

1. **Make EMOE's shared pool the headline, tested where MoE should win.** Compare at **equal memory** (not equal time), or give runs more tokens. If one shared pool beats per-layer experts at the same parameter count, that's a publishable, original result.
2. **Hybrid layers: replace some attention layers with a cheap recurrent / state-space layer** (the Griffin / Mamba idea family). Then Comet stops being a pure Transformer. These layers are cheap on CPUs, and combined with EMOE that mix would be Comet's own.
3. **Looped blocks + adaptive depth** (next in the plan): shares weights across depth, and tokens decide how many loops they need. Saves memory; it may tie like MoD.
4. **"Absorb more" on the data side:** ordering/curriculum (easy stories first) or dedup-aware sampling. Cheap to test, and it fits the 10–15 minute goal.

My recommendation: **fix the MoD memory problem first** (see "Open issues" below, short task), then **1** (it's already built and is Comet's most original idea), then **2**.

## Open issues from an outside review (address tomorrow)

A separate review of a copy of this repo flagged these. My take and the fix for each:

### 1. MoD works against Goal 2 ("more parameters per GB") — ✅ valid, top priority
- **Finding:** MoD raises peak RAM from **1.4 GB to 2.3 GB (+64%)** while keeping the same parameter count. Comet's goals include fitting *more* parameters per GB, so MoD as built moves the wrong way. I listed the RAM number but didn't call out the conflict; I should have.
- **Likely causes:** per-layer gather/scatter copies, a full (batch × 256 × 256) mask per MoD layer, and padding every sequence to the longest selected count.
- **Fix to try:** build the mask once and reuse it across the 3 MoD layers, drop the separate eye/valid masks (use a single additive bias), skip padding by grouping sequences, and measure peak RAM with a dedicated memory test, not just the process working set (which also includes data pages).
- **Rule from now on:** every feature reports **peak RAM next to loss and speed**, and anything that raises RAM per parameter needs a reason.

### 2. MoD tokens learn less each — ✅ valid (already in results/mod.md)
- Skipped tokens bypass 3 of 6 layers, so each token teaches less. At 12 minutes the +24% speed and the per-token loss canceled out (2.458 vs 2.453).
- Small correction to the review's wording: it's per-token learning that drops, not "learning-rate degradation."

### 3. MoD complicates fast text generation (KV cache) — ✅ valid for Phase 11 (package)
- Today `sample.py` recomputes the whole context every step (no KV cache), so MoD generation works but is slow.
- A real KV cache must handle tokens that skipped a layer (they have no key/value entry there). That's solvable (store KV only for processed tokens, plus their positions) but it's extra work, and it matters for the reusable package. Until then, MoD stays **off by default** (it already is).

### 4. EMOE's single shared pool may strain representations — ⚠️ a hypothesis to test
- **Claim:** early layers (low-level token patterns) and late layers (meaning, story structure) may not be able to reuse the same experts well. One shared pool could force a compromise.
- **This is exactly what tomorrow's option 1 tests:** shared pool vs. per-layer pools at **equal memory**. If the shared pool loses, a middle option is **group-shared pools** (layers 0–2 share one pool, layers 3–5 share another).
### 5. EMOE experts are starved of data at this scale — ✅ valid (matches results/emoe.md)
- At 1–3M training tokens, each of 64–128 experts sees only a small slice, so they stay under-trained. MoE normally needs tens of millions of tokens or more.
- **Implication for tomorrow's test:** compare at equal memory *and* try fewer, larger experts (e.g. 16–32) so each one gets more data within 12 minutes.

### 6. Muon on stacked expert weights is ~10× slower — ✅ valid, already worked around
- Muon's optimizer step on the expert pool took **385 ms vs 38 ms** with AdamW (that's why `--expert_optim adamw` exists). Muon's cost grows with parameter count, and experts are most of the parameters.
- **Open idea:** keep Muon's quality for experts at lower cost: fewer Newton-Schulz steps (3 instead of 5), or orthogonalize the expert weights only every few steps.

### 7. CPU-tuned defaults are wrong for GPUs and long contexts — ✅ valid, matters for the package (Phase 11)
- Comet's defaults (plain attention math, LayerNorm) were measured fastest **on this Ryzen at context 256**. On GPUs, fused attention (SDPA / FlashAttention) and RMSNorm are the faster choices.
- Plain attention builds a full T×T score table, so at contexts of 2,048+ it gets slow and memory-hungry; fused attention would be much faster there.
- **Fix:** make the defaults **`auto`**: pick fused attention on GPUs or for contexts ≥1,024, plain math on CPU at short contexts; RMSNorm on GPU, LayerNorm on CPU. Optionally run a quick startup benchmark to choose. Document this as "Comet adapts its kernels to the hardware" rather than one fixed choice.

### 8. Multi-token prediction "capacity deficit" — ✅ valid (matches results/mtp.md)
- DeepSeek-V3 showed MTP helps at 671B parameters. At ~6.5M, making the same hidden states also predict token t+2 acts as a harmful extra constraint: val loss 3.17 vs 3.115 for dense.
- **Action:** keep MTP off for small models; only retest it if Comet scales up a lot (it may also help later for faster generation via speculative decoding).

### 9. Novelty: every part of the 2.5× win was invented elsewhere — ✅ factually right, framing noted
- RoPE, QK-Norm, SwiGLU, GQA, Muon and WSD are all published work, and Comet's own additions (EMOE shared pool, MoD, light MTP) haven't beaten the dense core within 10–15 minutes yet.
- **Our stance on the name:** Comet is an architecture in the same sense Llama is (Llama also combined published parts: RMSNorm, SwiGLU, RoPE). The paper credits every source and presents EMOE as Comet's own addition, with honest results.
- **Review's verdict for making it clearly original:** make the **shared EMOE pool** or **looped-block parameter sharing** win at **equal parameter or memory budgets**. That's the same as tomorrow's options 1 and 3, which confirms the order.

## How to resume

```bash
cd "C:\Users\gamer\Desktop\$Transformer"
.venv/Scripts/python data/prepare.py      # only if data/train.bin is missing (~4 min)
# Comet v1 (the winner), 12 minutes:
.venv/Scripts/python train.py --model comet --name my_run --minutes 12 --lr 0.003 --optim muon --schedule wsd
# Options: --set mod=true | --set emoe=true n_experts=128 top_k=2 expert_hidden=128 shared_hidden=128 (+ --expert_optim adamw) | --set mtp_depth=1 mtp_block=false mtp_frac=0.25
.venv/Scripts/python evaluate.py my_run --write   # val loss + accuracy
.venv/Scripts/python charts.py                     # lab-style bar charts
.venv/Scripts/python bench.py --runs baseline_d256_l6 my_run   # loss curves + table
```

## Rules we work by

- Runs are **10–15 minutes** (3-minute A/B tests first). Never hours.
- **ETA updates every 1–2 minutes** during training.
- **Same-session controls:** the laptop's speed changes with heat (dense Comet ran at 4,500–5,100 tok/s on different occasions), so compare against a dense run from the same session.
- Commit and push after every change; `results/benchmark.jpg` stays at the top of the README.
- PC only (CPU, BF16, 6 threads).
