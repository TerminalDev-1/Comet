# VTransformer — Plan

**Status:** planning. No code yet.
**Code name:** `vtransformer`
**Its Mixture of Experts design:** **EMOE** (Efficient Mixture of Experts)

## Goals

Build a Transformer variant that, compared with a standard Transformer on the same hardware:

1. **Trains faster:** reaches the same validation loss in less wall-clock time (target 1.5–3×).
2. **Fits more parameters per GB of memory** (target 2–4×), by reusing parameters, sharing experts and cutting optimizer memory.
3. **Runs on a PC in 20–30 minutes:** every training run fits a **hard 20–30 minute budget** (never hours). PC only; no Android target.

Success is measured, not assumed: every idea gets tested against a plain baseline Transformer (see "How we prove it works").

## Target hardware

| Part | Spec | What it means |
|---|---|---|
| CPU | AMD Ryzen 5 8640HS (Zen 4, 6 cores / 12 threads) | Main training device. Has AVX-512 with BF16 support, so low-precision math is fast. |
| GPU | Radeon 760M (integrated, shares system RAM) | No CUDA. Optional: try `torch-directml`. CPU is the safe default. |
| RAM | 15 GB (shared with GPU) | Memory is the main limit, which is why "parameters per GB" is a core goal. |
| Python | 3.13 | PyTorch must support 3.13 (current releases do). |

### Time budget math (measured in Phase 0, CPU BF16, 6 threads)
- Training costs about **6 × active parameters** operations per token. The CPU really delivers ~300 GFLOPS in training.
- In a **25-minute run on the CPU**:

| Model (plain GPT) | Tokens/sec | Tokens in 25 min |
|---|---|---|
| 6.9M params | 7,151 (measured) | ~11M |
| 13.9M params | 3,521 (measured) | ~5M (too few for good English) |
| 30M params | ~1,500 (est.) | ~2M |

Realistic scale on the **CPU**: **~3–7M active parameters**, with **EMOE total parameters around 50–150M**, trained on **~10–15M tokens** per run. Phase 3 speed-ups (fused attention, `torch.compile`, GQA) should raise tokens/sec.
The **Radeon 760M** was tested and is slower than the CPU for training (see `results/hardware.md`), so everything trains on the CPU.

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

**Rule:** both models get the same **25-minute training budget** (20–30 min range).

**Scenario:** within 25 minutes, a standard Transformer's useful ceiling is its best-performing size. On the CPU, that's expected to be **~5–10M parameters**. Bigger dense models see too few tokens to learn well. Phase 2 measures the real ceiling.

**Goal:** in the same 25 minutes, VTransformer (EMOE + MoD + loops + lean memory) trains **~10–20× more total parameters** than the baseline's ceiling: **~50–150M total parameters on the CPU**.
- **Stretch goal: 250M+.** The Radeon GPU can't help (tested in Phase 0), so this depends on MoD, adaptive depth and Phase 3/8 speed-ups raising tokens/sec enough to train more experts in 25 minutes.

**Why it's possible:**
- **Compute** depends on *active* parameters per token, not total. EMOE keeps active parameters at the baseline's size, and MoD plus adaptive depth cut compute further.
- **Memory** depends on *total* parameters. At ~8 bytes per parameter (Pillar 4a), even 500M parameters is ~4 GB, so memory is not the limit. Time is.

**Caveats:**
- An EMOE model is not as smart as a dense model of the same total size. A rough rule of thumb puts an MoE around the geometric mean of its active and total size.
- Each expert only sees a slice of the data. Within 25 minutes, too many experts means undertrained experts, so we tune total size to what the time budget can actually train, not just what fits in memory.

**Scoreboard (same laptop, 25 minutes each):**

| Model | Total params | Active per token | Peak RAM | Val loss | Tokens/sec |
|---|---|---|---|---|---|
| Baseline Transformer | ~5–10M (ceiling) | same | measure | measure | measure |
| VTransformer + EMOE + MoD | **~50–150M (stretch 250M+)** | ~baseline | measure | measure | measure |

Win condition: VTransformer trains 10×+ more total parameters **and** gets a lower validation loss than the baseline in the same 25 minutes.

---

## How we prove it works

Every claim gets checked against a **baseline**: a plain GPT-style Transformer (learned positions, standard attention, no sharing, no experts, FP32 AdamW, cosine schedule).

**Fair comparison rules:**
- Same data, same tokenizer.
- Every run is **20–30 minutes** (default 25).
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
  results/             # logs, benchmark.jpg, benchmark.md
```

---

## Phases

### Phase 0: Setup
- ✅ Install PyTorch 2.15 (CPU build), plus `tokenizers`, `numpy`, `matplotlib`, in `.venv`.
- ✅ BF16 works: ~0.75–0.85 TFLOPS, ~2.3× faster than FP32. 4–6 threads get ~93% of 12-thread speed.
- ✅ EMOE expert hidden size: 128 minimum, 256+ near full speed.
- ✅ `torch-directml` on the Radeon 760M (Python 3.12 venv `.venv-dml`): **loses to the CPU**. No BF16, FP16 training gives NaN, and FP32 is 25–70% slower than CPU BF16. **Decision: train on the CPU.**
- ✅ Real training speed: a 6.9M-param GPT does **7,151 tokens/sec** on CPU BF16 (6 threads), so ~11M tokens per 25-minute run.
- Results: `results/hardware.md`.

### Phase 1: Data ✅
- Run: `python data/prepare.py` (downloads, filters, trains tokenizer, writes `.bin` files; ~4 minutes).
- ✅ Downloaded 1 of 4 TinyStories train shards (530k stories) + validation. That's ~10× what one 25-minute run uses.
- ✅ Filters (`data/filter.py`): repaired broken quotes, dropped 10,380 low-quality stories (cut-off endings, broken characters, garbled text) and **19,849 exact duplicates (3.7%)**. MinHash near-duplicate check found none at ≥80% similarity (verified it catches edited copies).
- ✅ Kept **499,646 stories → 109.9M train tokens**; validation 21,569 stories → 4.6M tokens.
- ✅ 8192-token byte-level BPE tokenizer (`data/tokenizer.json`): ~4.1 characters per token, ~220 tokens per story, exact round-trip.

### Phase 2: Baseline
- Plain GPT baseline. Train for 25 minutes and record loss and memory. This is the number to beat.
- Find the baseline's real ceiling: the size that gets the best loss in 25 minutes (expected ~5–10M on the CPU).

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
- Run the **parameter ceiling challenge** (baseline ceiling vs. VTransformer, 25 minutes each).
- **Publish the benchmark (required):**
  - `bench.py` makes **`results/benchmark.jpg`**, one image with 4 charts: validation loss vs. training time, validation loss vs. peak RAM, total vs. active parameters, and tokens/sec.
  - Also save a **`results/benchmark.md`** table with the exact numbers, plus sample text from each model on the same prompts.
  - Show `benchmark.jpg` at the top of the **README** on GitHub.
  - Commit and push both files, and regenerate them after every major training run so the repo always shows the latest results.

### Phase 10 (optional): Scale up
- Best config on FineWeb-Edu, at the biggest size this PC can handle.

### Phase 11: Save it globally
- Once VTransformer and EMOE are finished and tested, save a summary to the global CLAUDE.md (name, how it works, when to offer it), so every future AI project can use it as an option.

---

## Target numbers (to check, not promises)

| Metric | Baseline | VTransformer goal |
|---|---|---|
| Time to reach baseline's final val loss | 25 min | **≤ 12 min** |
| Memory per parameter during training | ~16 bytes | **~8 bytes** |
| Total parameters that fit in the same RAM | 1× | **2–4×** (reuse + EMOE + lean optimizer) |
| Total parameters trained in 25 min | ~5–10M | **~50–150M, stretch 250M+** (EMOE + MoD) |
| Average loops per token (adaptive depth) | — | **well below max** |
| Unique parameters for same depth (dense) | 1× | **~0.35×** |

---

## Open questions

1. **End use:** finish sentences / write stories, chat, or something else? Decides the data after Phase 9.
2. **Context length:** is 256–512 tokens enough, or do you need longer?

## Decisions

- **Name:** VTransformer (no hyphens or spaces). Was "$Transformer" / "STransformer".
- **MoE design name:** EMOE (Efficient Mixture of Experts). Was "SMOE".
- **Time budget:** every training run is 20–30 minutes (default 25). Never hours.
- **Benchmark image:** when training is done, `results/benchmark.jpg` is pushed to GitHub and shown in the README.
- **Training device:** CPU with BF16, 6 threads by default. The Radeon 760M was tested in Phase 0 and lost.
- **PC only.** Everything runs on the PC. No Android export or phone target unless it becomes absolutely necessary.
