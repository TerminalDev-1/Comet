# $Transformer — Plan

**Status:** planning. No code yet.
**Code name:** `stransformer` (Python names can't contain `$`).

## Goal

Build a Transformer variant that, compared with a standard Transformer of the same training budget:

1. **Learns faster:** reaches a lower validation loss in less wall-clock time.
2. **Fits more parameters into the same memory**, by reusing them.
3. **Trains on a laptop:** English-capable small models in minutes to under an hour, not days.

Success is measured, not assumed: every idea gets tested against a plain baseline Transformer (see "How we prove it works").

## Target hardware

| Part | Spec | What it means |
|---|---|---|
| CPU | AMD Ryzen 5 8640HS (Zen 4, 6 cores / 12 threads) | Main training device. Has AVX-512 with BF16 support, so low-precision math is fast. |
| GPU | Radeon 760M (integrated, shares system RAM) | No CUDA. Optional: try `torch-directml`. CPU is the safe default. |
| RAM | 15 GB (shared with GPU) | Keep models around 5–50M parameters and batch memory small. |
| Python | 3.13 | PyTorch must support 3.13 (current releases do). |

Realistic scale: **1–20M parameters**, trained on **~50–500M tokens**. That's enough to write fluent, simple English (like the TinyStories models).

---

## Pillar 1: Smarter attention

Attention is how each token decides which earlier tokens matter. We make it both smarter (better at picking the right tokens) and cheaper.

### 1a. Core upgrades (proven, low risk)
- **RoPE (rotary position embeddings):** positions are built into the attention math, so there's no learned position table. Generalizes better to longer text.
- **QK-Norm:** normalize queries and keys before the dot product. Keeps attention scores from blowing up, which allows a **higher learning rate**. That means faster training.
- **Grouped-Query Attention (GQA):** many query heads share fewer key/value heads (e.g. 8 query heads, 2 KV heads). Fewer parameters and less memory, with almost no quality loss.
- **Fused attention kernel:** use `torch.nn.functional.scaled_dot_product_attention`, which is much faster than writing the softmax by hand.

### 1b. "Smarter" attention experiments (the $Transformer's own ideas)
- **Gated attention output:** multiply each head's output by a learned sigmoid gate based on the input. Lets the model switch off heads that aren't useful for a given token. Cheap, and recent research shows it improves quality and stability.
- **Differential attention:** each head computes two attention maps and subtracts one from the other, which cancels "noise" attention to irrelevant tokens. Sharper focus on what matters.
- **Hybrid local + global layers:** most layers use **sliding-window attention** (each token looks at the last ~128 tokens, which is cheap). Every few layers, one **global** layer sees the full context. Most language structure is local, so this saves compute without losing long-range ability.

Each of these is a switch in the config, so we can turn them on and off and measure each one separately.

---

## Pillar 2: Reusing parameters

A normal 12-layer Transformer has 12 separate sets of weights. The $Transformer stores fewer unique layers and runs them more than once.

### 2a. Looped shared blocks
- Store **K unique blocks** (e.g. 4) and run them **R times** (e.g. 3 loops), giving **12 effective layers from 4 layers' worth of weights**.
- Same depth of "thinking", about a **third of the parameter memory**.

### 2b. Make each loop a little different (so reuse doesn't hurt quality)
- **Per-loop LayerNorm / scale parameters:** tiny, cheap, and they let each pass behave differently.
- **Loop index embedding:** a small learned vector added each loop, so the block knows which pass it's on.
- **Per-loop low-rank adapters (LoRA-style):** a small rank-8 "delta" on the shared weights for each loop. Adds about 1–3% parameters and recovers most of the quality of unshared layers.

### 2c. Other sharing
- **Tied embeddings:** the input embedding and output layer share one matrix. Big saving for small models.
- **Small custom tokenizer:** a BPE tokenizer with an 8k vocabulary, trained on our data, instead of GPT-2's 50k. Shrinks the embedding table a lot and frees parameters for the actual layers.

### Important note
Reusing parameters saves **memory**, not **compute**. 12 effective layers still cost 12 layers of math per token. The win: for the same RAM, we can afford a **wider** model (bigger hidden size) than an unshared Transformer. That's how we "train more parameters than the original Transformer" on this laptop.

---

## Pillar 3: Efficient compute during training

How we get the most learning out of every second of CPU time.

### 3a. Number formats and kernels
- **BF16 autocast on CPU:** Zen 4 has native BF16 instructions, so roughly 2× faster matrix math at the same quality.
- **`torch.compile`:** fuses operations together. Optional on Windows, because it needs a C++ compiler (MSVC). We'll try it, and fall back to normal mode if it fails.
- **Thread tuning:** set `torch.set_num_threads` to the physical core count (6) and benchmark.

### 3b. Smarter optimization (fewer steps to the same loss)
- **Muon optimizer** for the 2D weight matrices, with AdamW for embeddings and norms. Muon has repeatedly cut training time in public speed-run competitions.
- **Warmup → stable → decay (WSD) learning-rate schedule:** a short warmup, a long flat high learning rate, then a fast decay at the end. Lets us stop at any point and still get a good model.
- **Higher learning rates**, made safe by QK-Norm and good initialization.

### 3c. Data handling
- **Sequence packing:** glue documents together with separator tokens, so no compute is wasted on padding.
- **Pre-tokenize once** to a `.bin` file of token IDs that's memory-mapped, so data loading costs nearly nothing.
- **Sequence-length curriculum:** start training on short sequences (64 tokens), grow to full length (256–512). Early learning is cheap and fast.

### 3d. Memory savers (so we can fit bigger models)
- **Gradient accumulation:** big effective batch from small micro-batches.
- **Activation checkpointing** (optional): recompute some activations instead of storing them. Trades a little speed for a lot of memory.

---

## How we prove it works

Every claim gets checked against a **baseline**: a plain GPT-style Transformer (learned positions, standard attention, no sharing, AdamW, cosine schedule).

**Fair comparison rules:**
- Same data, same tokenizer, same wall-clock time budget (e.g. 30 minutes).
- Compare at **equal training time** and at **equal memory**.

**What we measure:**
- Validation loss vs. **wall-clock time** (the main chart).
- Tokens per second.
- Peak RAM.
- Parameter count (unique and effective).
- Sample text quality, judged by eye on fixed prompts.

**Ablations:** turn each feature on by itself, so we know which ideas actually help and drop the ones that don't.

---

## Dataset

- **Primary: TinyStories.** Simple English stories made for training small models. Small models learn fluent English from it very quickly. Ideal for proving the idea.
- **Later: a slice of FineWeb-Edu** (general educational web text), to test on harder, more varied English.
- Downloads will be confirmed before they happen.

---

## Project layout (planned)

```
$Transformer/
  plan.md              # this file
  README.md
  requirements.txt     # torch, numpy, tokenizers, datasets, matplotlib
  stransformer/
    config.py          # all switches (attention type, loops, sharing, etc.)
    attention.py       # RoPE, QK-Norm, GQA, gating, differential, sliding window
    block.py           # transformer block + per-loop adapters
    model.py           # $Transformer model
    baseline.py        # plain GPT baseline
    optim.py           # Muon + AdamW setup, WSD schedule
  data/
    prepare.py         # download, train tokenizer, write token .bin files
  train.py             # one training script for both models
  sample.py            # generate text from a checkpoint
  bench.py             # speed/memory benchmarks
  results/             # logs, charts, comparison notes
```

---

## Phases

### Phase 0: Setup
- Install PyTorch (CPU build), plus `tokenizers`, `numpy`, `matplotlib`.
- Check that BF16 works on this CPU, and benchmark matrix-multiply speed.
- Try `torch-directml` on the Radeon 760M. Keep it only if it beats the CPU.

### Phase 1: Data
- Download TinyStories, train an 8k BPE tokenizer, write `train.bin` / `val.bin`.

### Phase 2: Baseline
- Build the plain GPT baseline (~10M params). Train for 30 minutes and record the loss curve. This is the number to beat.

### Phase 3: $Transformer v1 (core)
- RoPE + QK-Norm + GQA + fused attention + tied embeddings + BF16 + WSD schedule.
- Compare with the baseline.

### Phase 4: Parameter reuse
- Add looped shared blocks, per-loop norms and loop embeddings, then LoRA adapters.
- Use the saved memory to make the model wider. Compare at equal RAM.

### Phase 5: Smarter attention experiments
- Gated attention, differential attention, hybrid local/global layers. Run an ablation for each.

### Phase 6: Training speed-ups
- Muon optimizer, sequence-length curriculum, `torch.compile`, thread tuning.

### Phase 7: Results
- Final head-to-head run, charts, and a write-up of what worked and what didn't.

### Phase 8 (optional): Scale up
- Train the best config on FineWeb-Edu, and push to the biggest model this laptop can handle.

---

## Target numbers (to check, not promises)

| Metric | Baseline | $Transformer goal |
|---|---|---|
| Time to reach baseline's final val loss | 30 min | **≤ 15 min** |
| Unique parameters for same depth | 1× | **~0.35×** |
| Width that fits in the same RAM | 1× | **~1.5×** |
| Fluent TinyStories-style text | ~30 min | **~10–15 min** |

---

## Open questions

1. **End use:** finish sentences / write stories, chat, or something else? This decides the data after Phase 7.
2. **Radeon GPU:** worth the time to get DirectML working, or stay on the CPU?
3. **Context length:** is 256–512 tokens enough, or do you need longer?
4. **Running it on Android later:** should we plan to export the trained model to run on a phone (e.g. ExecuTorch or ONNX)?
