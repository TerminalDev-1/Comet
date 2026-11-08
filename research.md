# Beyond the Transformer: architecture options

*Researched 2026-11-08. Goal: stop bolting parts onto a Transformer and build Comet's next model on a different foundation, trained by a C++ stack instead of Python.*

## What the new architecture has to do

Same project goals as before, so the options are judged against them:

1. **Train fast on this laptop** (Ryzen 5 8640HS, CPU only, BF16), in 10–15 minute runs.
2. **Write more coherent stories** than today's Comet in the same time.
3. **More parameters per GB** of RAM.
4. **Simple enough to hand-write in C++**, including the backward pass (the math for learning).

The bar to beat is Comet v1: **val loss 2.441 in 12 minutes, baseline quality in 9.8 minutes.**

---

## The families

### 1. Linear recurrent models (the "RNN comeback")

**What they are:** instead of every word looking back at every earlier word (attention), the model reads left to right and keeps a fixed-size memory state that it updates at each word. Cost grows linearly with text length, and generation needs constant memory per word.

**Main 2025–2026 work:**
- **Gated DeltaNet** (NVIDIA, ICLR 2025): the memory is updated with the "delta rule", which fixes what it got wrong instead of just adding more. It beat Mamba-2, RetNet and other recurrent models at 1.3B parameters.
- **RWKV-7 "Goose"** (2025): a similar delta-style update, with near state-of-the-art English results. Its authors say it can track state in ways Transformers formally can't.
- **Mamba-3** (ICLR 2026): a state-space model built for fast generation. A state half the size of Mamba-2's matches Mamba-2's quality.
- **Kimi Linear / KDA** (Moonshot, 2025): Gated DeltaNet with finer gates. Its 3:1 hybrid (3 recurrent layers per attention layer) beat full attention while cutting cache use by 75%.

**Fit for Comet:** ⭐⭐⭐⭐⭐ for the CPU (no T×T attention table, small memory). Quality is proven at scale. The catch: it's a crowded field, so Comet needs its own twist to be original.

### 2. Minimal RNNs (minGRU / minLSTM)

**What they are:** "Were RNNs All We Needed?" (2024) stripped GRUs and LSTMs down until the gates depend only on the current word, not the previous state. That makes them trainable in parallel (a "parallel scan") instead of one word at a time, and they still compete with Mamba on small tasks.

**Fit for Comet:** ⭐⭐⭐⭐⭐ for C++. It's the simplest model that works: a few matrix multiplies and one running sum, with a backward pass you can write by hand. Its quality ceiling is lower than delta-rule models, since its memory is a single vector, not a matrix.

### 3. Test-time memory (Titans, TTT layers)

**What they are:** part of the model is a small network that keeps learning *while it reads*. Surprising words update its weights more strongly (Google's Titans, 2025). It's a memory that literally trains itself on the story so far.

**Fit for Comet:** ⭐⭐⭐ It directly targets coherence (remembering who did what across a story). But it's harder to build: the inner learning step needs its own gradient math, and that's much harder to hand-write in C++. Delta-rule models (family 1) are a simple, one-step version of the same idea.

### 4. Tiny recursive networks (HRM, TRM)

**What they are:** one very small network applied over and over, refining its answer each pass. TRM (Oct 2025) used **one 2-layer, 7M-parameter network** to score 45% on ARC-AGI-1, beating huge LLMs on those puzzles.

**Fit for Comet:** ⭐⭐⭐ Excellent for parameters per GB (one block reused many times). It's proven on puzzles, though, not story writing. Applying it to language is open research: that makes it riskier, but more original if it works.

### 5. Tokenizer-free hierarchical models (H-Net)

**What they are:** the model reads raw bytes and learns its own word boundaries ("dynamic chunking"), then runs a bigger network on those chunks. In equal-compute tests, a byte-level H-Net beat a BPE-token Transformer (ICLR 2026). It's also more robust to typos and odd text.

**Fit for Comet:** ⭐⭐⭐ It removes the tokenizer step entirely, a nice fit for a reusable training kit ("give it any text"). But bytes mean ~4–5× more steps per story at the bottom level, which hurts in 10-minute CPU runs unless that level is a cheap recurrent layer.

### 6. Diffusion language models (LLaDA, Mercury)

**What they are:** instead of writing word by word, they start from fully masked text and fill it in over several rounds. LLaDA 8B matched LLaMA-3 8B; Mercury generates 1,000+ tokens/sec.

**Fit for Comet:** ⭐⭐ Their win is fast *generation*, not fast training. They usually need more training compute than left-to-right models, and they still use a Transformer inside.

### 7. Energy-based models (EBTs)

**What they are:** the model learns to score how good an answer is, then "thinks" by improving its answer to lower that score. It scaled better than Transformer++ along several axes (2025), but had slightly worse pretraining perplexity.

**Fit for Comet:** ⭐ Training needs gradients of gradients, which is expensive and hard in hand-written C++. It's a poor fit for 10-minute CPU runs.

### 8. Wild cards (high risk)

- **Backprop-free training** (Forward-Forward and similar): no backward pass at all. That would be very C++-friendly, but nothing close to Transformer quality on language yet.
- **Associative memory / modern Hopfield networks:** closely related to family 3.

---

## Options for Comet 2

### Option A: Recurrent Comet (delta-rule core) ⭐ recommended

A from-scratch recurrent language model: a Gated-DeltaNet/RWKV-7 style memory update, a minGRU-simple design where possible, and **no attention at all**.
- **Why:** best fit for a CPU (linear cost, small fixed memory), proven quality, and simple enough to hand-write forward *and* backward in C++.
- **Comet's own twist** (pick one or more to test):
  1. **EMOE's shared expert pool** carried over (one pool, every layer has its own router). Still Comet's most original idea.
  2. **Self-reset memory:** the model learns when to wipe or keep its state at story and sentence boundaries, for coherence.
  3. **Two-speed memory:** a fast state updated every word, plus a slow state updated once per sentence (the HRM idea, applied to language).
- **Risk:** low to medium.

### Option B: Recursive tiny network for language

One small recurrent block (Option A's core) applied several times per word, with the model deciding how many passes each word needs (TRM-style).
- **Why:** the most parameters per GB, and the most original. Nobody has shown TRM-style recursion working for story writing.
- **Risk:** high. It might train slower per minute than Option A.

### Option C: Byte-level hierarchical recurrent model

H-Net's dynamic chunking with cheap recurrent layers on bytes and a bigger recurrent core on chunks. No tokenizer.
- **Why:** fits the reusable training kit ("feed it any text"). It's original as an all-recurrent H-Net.
- **Risk:** medium-high. More bytes per story means fewer stories seen in 10 minutes.

### Option D: Start with A, grow into B or C

Build Option A first, since every other option reuses its core. Once it beats Comet v1, add recursion (B) or byte chunking (C) as the next research step.

**My pick: D.** Option A first is the fastest path to a working model in C++ that's genuinely not a Transformer, and it keeps B and C open.

---

## The C++ training stack

| Approach | What it means | Pros | Cons |
|---|---|---|---|
| **From scratch** (llm.c style) | Hand-written C++: matrix math with AVX-512 + BF16 (the Ryzen's Zen 4 cores support both), forward and backward for every layer | Full control, no Python anywhere, small and fast to start | Every new layer needs its backward pass written by hand and checked. llm.c's plain CPU code ran ~6× slower than PyTorch, so fast matrix kernels are essential |
| **LibTorch** (PyTorch's C++ API) | C++ code calling PyTorch's engine | Automatic backward pass, same fast kernels as today | Still PyTorch underneath, just without Python |
| **Hybrid** | From-scratch C++, with a library like oneDNN or OpenBLAS for the matrix multiplies only | Fast matrix math without writing it ourselves; the rest stays ours | One outside dependency |

**My pick: hybrid.** Recurrent models are simple enough that hand-writing forward and backward is very doable. Using a tuned library for matrix multiplies avoids llm.c's slowdown.

**Fairness rule:** the plain GPT baseline gets rebuilt in the same C++ stack. Then any speed-up is clearly from the architecture, not from C++ vs Python.

---

## Sources

- [Gated Delta Networks (arXiv 2412.06464)](https://arxiv.org/pdf/2412.06464v3)
- [RWKV-7 "Goose" (arXiv 2503.14456)](https://arxiv.org/pdf/2503.14456)
- [Mamba-3 (arXiv 2603.15569)](https://arxiv.org/pdf/2603.15569)
- [Kimi Linear / KDA (arXiv 2510.26692)](https://arxiv.org/abs/2510.26692)
- [Were RNNs All We Needed? (arXiv 2410.01201)](https://arxiv.org/pdf/2410.01201)
- [Titans: Learning to Memorize at Test Time (alphaXiv 2501.00663)](https://www.alphaxiv.org/overview/2501.00663v1)
- [Less is More: Recursive Reasoning with Tiny Networks (arXiv 2510.04871)](https://arxiv.org/abs/2510.04871)
- [H-Net: Dynamic Chunking (arXiv 2507.07955)](https://arxiv.org/html/2507.07955v2)
- [Mercury: diffusion LLMs (arXiv 2506.17298)](https://arxiv.org/html/2506.17298v1)
- [Energy-Based Transformers (alphaXiv 2507.02092)](https://www.alphaxiv.org/overview/2507.02092v1)
- [Beyond Transformers survey (Springer, 2026)](https://link.springer.com/chapter/10.1007/978-3-032-17187-0_22)
- [llm.c: GPT-2 training in pure C (Hackaday)](https://hackaday.com/2024/04/28/train-a-gpt-2-llm-using-only-pure-c-code)
