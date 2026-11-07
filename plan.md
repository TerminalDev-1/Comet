# VTransformer — Plan

**Status:** planning. No code yet.
**Code name:** `vtransformer`
**Its Mixture of Experts design:** **EMOE** (Efficient Mixture of Experts)

## Goals

Build a Transformer variant that, compared with a standard Transformer on the same hardware:

1. **Trains faster:** reaches the same validation loss in less wall-clock time (target 1.5–3×).
2. **Fits more parameters per GB of memory** (target 2–4×), by reusing parameters, sharing experts and cutting optimizer memory.
3. **Runs on a PC:** small models that write English trained in minutes to under an hour. PC only; no Android target.

Success is measured, not assumed: every idea gets tested against a plain baseline Transformer (see "How we prove it works").

## Target hardware

| Part | Spec | What it means |
|---|---|---|
| CPU | AMD Ryzen 5 8640HS (Zen 4, 6 cores / 12 threads) | Main training device. Has AVX-512 with BF16 support, so low-precision math is fast. |
| GPU | Radeon 760M (integrated, shares system RAM) | No CUDA. Optional: try `torch-directml`. CPU is the safe default. |
| RAM | 15 GB (shared with GPU) | Memory is the main limit, which is why "parameters per GB" is a core goal. |
| Python | 3.13 | PyTorch must support 3.13 (current releases do). |

Realistic scale: **dense models of 1–20M parameters**, and **EMOE models with several times more total parameters** for similar compute. Trained on **~50–500M tokens**.

---

## Pillar 1: Smarter attention

Attention is how each token decides which earlier tokens matter. We make it both smarter (better at picking the right tokens) and cheaper.

### 1a. Core upgrades (proven, low risk)
- **RoPE (rotary position embeddings):** positions are built into the attention math, so there's no learned position table.
- **QK-Norm:** normalize queries and keys before the dot product. Keeps attention scores from blowing up, which allows a **higher learning rate**.
- **Grouped-Query Attention (GQA):** many query heads share fewer key/value heads (e.g. 8 query heads, 2 KV heads). Fewer parameters and less memory.
- **Fused attention kernel:** `torch.nn.functional.scaled_dot_product_attention`.

### 1b. Smarter attention experiments
- **Gated attention output:** a learned sigmoid gate on each head's output lets the model switch off heads that aren't useful for a token.
- **Differential attention:** each head computes two attention maps and subtracts one, which cancels "noise" attention to irrelevant tokens.
- **Hybrid local + global layers:** most layers use **sliding-window attention** (the last ~128 tokens). Every few layers, one **global** layer sees the full context.

Each is a config switch, so we can measure each one separately.

---

## Pillar 2: Reusing parameters

### 2a. Looped shared blocks
- Store **K unique blocks** (e.g. 4) and run them **R times** (e.g. 3 loops), giving **12 effective layers from 4 layers' worth of weights**.

### 2b. Make each loop a little different
- **Per-loop norm / scale parameters**, which are tiny.
- **Loop index embedding:** a learned vector added each loop.
- **Per-loop low-rank adapters (LoRA-style, rank ~8):** about 1–3% extra parameters, recovering most of the quality of unshared layers.

### 2c. Other sharing
- **Tied embeddings:** input embedding and output layer share one matrix.
- **Small custom tokenizer:** 8k-vocab BPE trained on our data, instead of GPT-2's 50k.

**Note:** reuse saves **memory**, not **compute**. 12 effective layers still cost 12 layers of math. The win is that the same RAM fits a wider or bigger model.

---

## Pillar 3: EMOE (Efficient Mixture of Experts)

Normal MoE: each layer has a few big feed-forward "experts" (e.g. 8), and each token uses 1–2 of them. You get more parameters for the same compute, but every layer stores its own experts.

**EMOE has three parts:**

### 3a. Fine-grained tiny experts
- Split experts into many small ones, e.g. **64 small experts instead of 8 big ones**, and each token picks ~8.
- Many more possible combinations, so the model learns more from the same parameters, and the same quality is reachable with **fewer total parameters**.
- Smaller experts alone don't save memory (64 small = 8 big in total size). The saving comes from needing a smaller total pool, plus 3b.

### 3b. One expert pool shared across all layers (main memory saver)
- Instead of every layer owning its own experts, **all layers and all loops draw from one shared pool** of tiny experts.
- Each layer (and each loop) has its own small **router** that picks which experts to use, so layers can still specialize.
- Fits naturally with Pillar 2's looped blocks.

### 3c. Always-on shared expert
- One small expert that every token always uses, for common basics (grammar, frequent words), so the tiny experts can specialize.

### 3d. Training details
- **Load-balancing loss** (or bias-based balancing without an extra loss) so all experts get used.
- **Top-k routing** with softmax or sigmoid scores. Compare both.
- Batched expert compute (group tokens by expert) to keep CPU math efficient.

### Caveat
Many tiny experts means many small matrix operations, which can be slower on a CPU than a few big ones. We'll **benchmark expert size** (hidden size 64 / 128 / 256) to find the sweet spot.

### EMOE tests
Race three versions with the same compute per token:
1. Dense VTransformer (no experts)
2. Normal MoE (8 big experts per layer)
3. EMOE (tiny experts, shared pool, shared expert)

Measure quality **per GB of memory** and **per minute of training**.

---

## Pillar 4: Efficient compute and memory during training

### 4a. Memory per parameter (more parameters on the same hardware)
A normal AdamW setup uses about **16 bytes per parameter** (FP32 weight, gradient and two optimizer values). We cut that:
- **BF16 weights and gradients**, with an FP32 master copy only where needed.
- **Muon optimizer** for 2D weight matrices: 1 extra value per weight instead of AdamW's 2.
- **8-bit optimizer states** for the parts that still use AdamW.
- **Activation checkpointing** (optional): recompute activations instead of storing them.
- **Gradient accumulation:** large effective batch from small micro-batches.

Target: **roughly half the memory per parameter** of the baseline setup, before counting reuse and EMOE.

### 4b. Faster math
- **BF16 autocast on CPU** (Zen 4 supports it natively).
- **`torch.compile`** (optional on Windows, needs MSVC; fall back if it fails).
- **Thread tuning:** `torch.set_num_threads` = physical cores (6), then benchmark.

### 4c. Fewer steps to the same loss
- **Muon optimizer** (also a speed win, as shown in public speedruns).
- **Warmup → stable → decay (WSD) learning-rate schedule.**
- **Higher learning rates**, made safe by QK-Norm and good initialization.

### 4d. Data handling
- **Sequence packing:** no padding waste.
- **Pre-tokenized memory-mapped `.bin` files.**
- **Sequence-length curriculum:** start at 64 tokens, grow to 256–512.

---

## Pillar 5: Less compute per token

### 5a. Adaptive loop depth
- Each token gets a small **halting score** after each pass through the looped blocks (Pillar 2). Once it's confident, the token exits early, and its state is carried forward unchanged.
- Easy tokens ("the", punctuation) take 1 pass; hard tokens take up to the full R loops.
- A small **ponder cost** in the loss discourages wasting loops.
- Target: average loops per token well below R, with the same quality.

### 5b. Mixture of Depths (MoD)
- In selected layers, a router scores every token and only the **top ~50%** (capacity, tunable) go through the layer. The rest skip it through the residual connection.
- Those layers cost about half the attention and feed-forward compute.
- Combine with EMOE: MoD picks *which tokens* get computed, and EMOE picks *which experts* they use.
- Use **expert-choice style top-k per sequence** in training. For generation, a small predictor decides skip/keep, so the model never needs to look ahead.

### 5c. Multi-token prediction (MTP)
- Besides the next token, small extra heads predict tokens **+2 and +3**, sharing the main model trunk and the tied output embedding.
- More learning signal per step means fewer steps to the same loss.
- The extra heads are training-only (they can also be used later for faster generation by speculative decoding).
- Loss weight for the extra heads is tunable (start ~0.3).

### 5d. Data filtering
- **Deduplication:** exact and near-duplicate removal (MinHash) before tokenizing.
- **Quality filters:** drop very short, garbled, or non-English documents (simple heuristics first, e.g. symbol ratio and repeated lines).
- Later option: score documents with a small trained model and keep the best.
- Goal: reach the same loss from fewer tokens, which means less training time.

---

## Benchmark: the parameter ceiling challenge

**Scenario:** on this laptop, a standard Transformer tops out at about **30M parameters** before training becomes impractical (too slow, or doesn't fit). Phase 2 measures the real ceiling.

**Goal:** VTransformer (EMOE + MoD + loops + lean memory) trains **250M–500M total parameters** on the same laptop, at about the same cost per token as the 30M baseline.

**Why it's possible:**
- **Compute** depends on *active* parameters per token, not total. EMOE keeps active parameters around the baseline's size (~20–30M), and MoD plus adaptive depth cut compute further.
- **Memory** depends on *total* parameters. At ~8 bytes per parameter (Pillar 4a), 500M parameters is ~4 GB for weights, gradients and optimizer state, which fits in 15 GB with room for activations.

**Caveats:**
- A 500M EMOE model is not as smart as a 500M dense model. A rough rule of thumb puts an MoE between its active and total size (around the geometric mean, so ~120M dense-equivalent for 30M active / 500M total). It should still clearly beat the 30M baseline.
- More experts need more training tokens before all of them learn something useful, so this benchmark uses longer runs.

**Scoreboard (same laptop, same time budget):**

| Model | Total params | Active per token | Peak RAM | Val loss | Tokens/sec |
|---|---|---|---|---|---|
| Baseline Transformer | ~30M (ceiling) | ~30M | measure | measure | measure |
| VTransformer + EMOE + MoD | **250M–500M** | ~20–30M | measure | measure | measure |

Win condition: VTransformer fits 250M+ total parameters **and** gets a lower validation loss than the baseline in the same training time.

---

## How we prove it works

Every claim gets checked against a **baseline**: a plain GPT-style Transformer (learned positions, standard attention, no sharing, no experts, FP32 AdamW, cosine schedule).

**Fair comparison rules:**
- Same data, same tokenizer.
- Compare at **equal training time** and at **equal memory**.

**What we measure:**
- Validation loss vs. **wall-clock time** (main speed chart).
- Validation loss vs. **peak memory** (main "parameters per GB" chart).
- Tokens per second.
- Parameter counts: total, unique, and active per token.
- Sample text quality on fixed prompts.

**Ablations:** turn each feature on by itself, and keep only what helps.

---

## Dataset

- **Primary: TinyStories.** Simple English stories. Small models learn fluent English from it quickly.
- **Later: a slice of FineWeb-Edu** for harder, more varied English.
- Downloads will be confirmed before they happen.

---

## Project layout (planned)

```
VTransformer/
  plan.md              # this file
  README.md
  requirements.txt     # torch, numpy, tokenizers, datasets, matplotlib
  vtransformer/
    config.py          # all switches (attention type, loops, experts, etc.)
    attention.py       # RoPE, QK-Norm, GQA, gating, differential, sliding window
    emoe.py            # EMOE: tiny experts, shared pool, routers, shared expert
    mod.py             # Mixture of Depths token routing
    halting.py         # adaptive loop depth (halting scores, ponder cost)
    mtp.py             # multi-token prediction heads
    block.py           # transformer block + per-loop adapters
    model.py           # VTransformer model
    baseline.py        # plain GPT baseline (+ normal MoE variant)
    optim.py           # Muon + AdamW (8-bit) setup, WSD schedule
  data/
    prepare.py         # download, train tokenizer, write token .bin files
    filter.py          # dedup + quality filters
  train.py             # one training script for all models
  sample.py            # generate text from a checkpoint
  bench.py             # speed/memory benchmarks
  results/             # logs, charts, comparison notes
```

---

## Phases

### Phase 0: Setup
- Install PyTorch (CPU build), plus `tokenizers`, `numpy`, `matplotlib`.
- Check BF16 on this CPU; benchmark matrix multiply at various sizes (also tells us good EMOE expert sizes).
- Try `torch-directml` on the Radeon 760M; keep it only if it beats the CPU.

### Phase 1: Data
- Download TinyStories, run dedup + quality filters, train an 8k BPE tokenizer, write `train.bin` / `val.bin`.

### Phase 2: Baseline
- Plain GPT baseline (~10M params). Train for 30 minutes and record loss and memory. This is the number to beat.
- Find the baseline's real parameter ceiling on this laptop (expected ~30M).

### Phase 3: VTransformer v1 (core)
- RoPE + QK-Norm + GQA + fused attention + tied embeddings + BF16 + WSD schedule.

### Phase 4: Memory per parameter
- BF16 weights, Muon, 8-bit optimizer states, checkpointing. Measure bytes per parameter.

### Phase 5: Parameter reuse
- Looped shared blocks, per-loop norms and loop embeddings, then LoRA adapters. Compare at equal RAM.

### Phase 6: EMOE
- Normal MoE first (as a reference), then fine-grained experts, then the shared pool across layers, then the shared expert. Benchmark expert sizes.

### Phase 6b: Less compute per token
- Multi-token prediction, then Mixture of Depths, then adaptive loop depth. Ablation for each.

### Phase 7: Smarter attention experiments
- Gated attention, differential attention, hybrid local/global layers. Run an ablation for each.

### Phase 8: Training speed-ups
- Sequence-length curriculum, `torch.compile`, thread tuning.

### Phase 9: Results
- Final head-to-head: baseline vs. normal MoE vs. VTransformer + EMOE + MoD. Charts and a write-up of what worked and what didn't.
- Run the **parameter ceiling challenge** (30M baseline vs. 250M–500M VTransformer).

### Phase 10 (optional): Scale up
- Best config on FineWeb-Edu, at the biggest size this PC can handle.

### Phase 11: Save it globally
- Once VTransformer and EMOE are finished and tested, save a summary to the global CLAUDE.md (name, how it works, when to offer it), so every future AI project can use it as an option.

---

## Target numbers (to check, not promises)

| Metric | Baseline | VTransformer goal |
|---|---|---|
| Time to reach baseline's final val loss | 30 min | **≤ 15 min** |
| Memory per parameter during training | ~16 bytes | **~8 bytes** |
| Total parameters that fit in the same RAM | 1× | **2–4×** (reuse + EMOE + lean optimizer) |
| Trainable total parameters on this laptop | ~30M | **250M–500M** (EMOE + MoD) |
| Average loops per token (adaptive depth) | — | **well below max** |
| Unique parameters for same depth (dense) | 1× | **~0.35×** |

---

## Open questions

1. **End use:** finish sentences / write stories, chat, or something else? Decides the data after Phase 9.
2. **Radeon GPU:** worth the time to get DirectML working, or stay on the CPU?
3. **Context length:** is 256–512 tokens enough, or do you need longer?

## Decisions

- **Name:** VTransformer (no hyphens or spaces). Was "$Transformer" / "STransformer".
- **MoE design name:** EMOE (Efficient Mixture of Experts). Was "SMOE".
- **PC only.** Everything runs on the PC. No Android export or phone target unless it becomes absolutely necessary.
