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
    block.py           # transformer block + per-loop adapters
    model.py           # VTransformer model
    baseline.py        # plain GPT baseline (+ normal MoE variant)
    optim.py           # Muon + AdamW (8-bit) setup, WSD schedule
  data/
    prepare.py         # download, train tokenizer, write token .bin files
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
- Download TinyStories, train an 8k BPE tokenizer, write `train.bin` / `val.bin`.

### Phase 2: Baseline
- Plain GPT baseline (~10M params). Train for 30 minutes and record loss and memory. This is the number to beat.

### Phase 3: VTransformer v1 (core)
- RoPE + QK-Norm + GQA + fused attention + tied embeddings + BF16 + WSD schedule.

### Phase 4: Memory per parameter
- BF16 weights, Muon, 8-bit optimizer states, checkpointing. Measure bytes per parameter.

### Phase 5: Parameter reuse
- Looped shared blocks, per-loop norms and loop embeddings, then LoRA adapters. Compare at equal RAM.

### Phase 6: EMOE
- Normal MoE first (as a reference), then fine-grained experts, then the shared pool across layers, then the shared expert. Benchmark expert sizes.

### Phase 7: Smarter attention experiments
- Gated attention, differential attention, hybrid local/global layers. Run an ablation for each.

### Phase 8: Training speed-ups
- Sequence-length curriculum, `torch.compile`, thread tuning.

### Phase 9: Results
- Final head-to-head: baseline vs. normal MoE vs. VTransformer + EMOE. Charts and a write-up of what worked and what didn't.

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
