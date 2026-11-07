# Comet

A faster-to-train Transformer variant, built and benchmarked on a laptop CPU (AMD Ryzen 5 8640HS, no GPU).

![Comet benchmark](results/benchmark.jpg)

## Latest result (Phase 3)

| Model | Train time | Final val loss | Time to reach 2.607 | Tokens seen |
|---|---|---|---|---|
| Baseline GPT | 25 min | 2.607 | 25.0 min | 9.1M |
| **Comet v1** | **12 min** | **2.420** | **9.7 min (2.6× faster)** | **3.6M** |

Same data (TinyStories), same tokenizer, same size (~6.5–6.9M parameters), same laptop.
Comet beats the baseline's 25-minute score in under 10 minutes, while seeing 2.5× fewer tokens.
Full numbers: [results/benchmark.md](results/benchmark.md) · details: [results/phase3.md](results/phase3.md).

## What's in Comet so far

- **Attention:** rotary positions (RoPE), QK-Norm, grouped-query attention
- **MLP:** SwiGLU
- **Training:** Muon optimizer (hidden matrices) + AdamW (embeddings), warmup-stable-decay schedule, BF16
- **CPU-tuned:** plain attention math and LayerNorm, which measured faster than the "fused" kernel and RMSNorm on this CPU

## Experimental options (built and tested, off by default)

| Option | Turn on | 3-min result vs dense Comet | Notes |
|---|---|---|---|
| **EMOE** (Efficient Mixture of Experts) | `--set emoe=true` | 3.20 vs 3.115 | 2.6× the parameters at the same compute; 91% of dense speed. [results/emoe.md](results/emoe.md) |
| **Multi-token prediction** | `--set mtp_depth=1` | 3.17 vs 3.115 | Light version shown. [results/mtp.md](results/mtp.md) |
| **Mixture of Depths** | `--set mod=true` | 3.106 vs 3.115 | 12-min race: tie (2.458 vs 2.453), +24% tokens/sec. [results/mod.md](results/mod.md) |

Coming next (see [plan.md](plan.md)): parameter reuse (looped blocks), adaptive loop depth, reusable package, research paper.

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python data/prepare.py          # download + filter TinyStories, train tokenizer (~4 min)
.venv/Scripts/python train.py --model comet --name my_run --minutes 12 --lr 0.003 --optim muon --schedule wsd
.venv/Scripts/python sample.py checkpoints/my_run.pt
.venv/Scripts/python bench.py --runs baseline_d256_l6 my_run
```
