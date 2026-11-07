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

## The honest picture: is Comet still "just a Transformer"?

**Yes, right now it is.** Comet v1 is a Transformer with a modern recipe. Every part that produced the 2.5× win was invented elsewhere and is public:

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

**Is that bad?** No. Nearly every frontier model is a Transformer with changes. The 2.5× result is real and reproducible. But for a paper to say "a new architecture," Comet needs **at least one original part that measurably wins**. Otherwise the honest title is "a fast training recipe for small models on CPUs."

## Ways to make Comet genuinely its own (pick tomorrow)

1. **Make EMOE's shared pool the headline, tested where MoE should win.** Compare at **equal memory** (not equal time), or give runs more tokens. If one shared pool beats per-layer experts at the same parameter count, that's a publishable, original result.
2. **Hybrid layers: replace some attention layers with a cheap recurrent / state-space layer** (the Griffin / Mamba idea family). Then Comet stops being a pure Transformer. These layers are cheap on CPUs, and combined with EMOE that mix would be Comet's own.
3. **Looped blocks + adaptive depth** (next in the plan): shares weights across depth, and tokens decide how many loops they need. Saves memory; it may tie like MoD.
4. **"Absorb more" on the data side:** ordering/curriculum (easy stories first) or dedup-aware sampling. Cheap to test, and it fits the 10–15 minute goal.

My recommendation: **1 first** (it's already built and is Comet's most original idea), then **2**.

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
