# Phase 3: Comet v1

## Result

**Comet reached the baseline's final 25-minute loss (2.607) in 9.7 minutes (2.6× faster), and finished its 12-minute run at 2.420.** It saw only 3.6M tokens vs. the baseline's 9.1M, so it learns far more per token.

| Minute | Baseline (25-min run) | Comet (12-min run) |
|---|---|---|
| 1 | 4.384 | 3.817 |
| 3 | 3.763 | 3.163 |
| 6 | 3.358 | 2.846 |
| 9 | 3.143 | 2.666 |
| 10 | 3.078 | **2.578** ✅ below 2.607 |
| 12 | 2.971 | **2.420** |
| 25 | **2.607** | — |

Run: `python train.py --model comet --name vt_v1_muon_12m --minutes 12 --lr 0.003 --optim muon --muon_lr 0.02 --schedule wsd`
Config: d=256, 6 layers, 4 query heads / 2 KV heads, RoPE, QK-Norm, SwiGLU, tied embeddings, 6.52M params.

## Fair 3-minute A/B (each run with its own full schedule)

| Run | Val loss @ 3 min | Tokens/sec |
|---|---|---|
| Baseline GPT (AdamW, cosine) | 3.980 | 6,102 |
| Comet v1 (AdamW, WSD) | 3.473 | 6,021 |
| **Comet v1 (Muon, WSD)** | **3.061** | 5,072 |

- Architecture alone (same optimizer family) is a clear win: 3.98 → 3.47.
- **Muon is the biggest single win:** 3.47 → 3.06, despite costing ~16% speed (Newton-Schulz steps on the CPU).

## CPU findings (important for later phases)

Measured on this Ryzen with BF16, context 256:

| Component | Result |
|---|---|
| Fused attention kernel (SDPA) | **1.7× slower** than plain attention math (23.5 ms vs 14.2 ms fwd+bwd) |
| RMSNorm | **~4× slower** than LayerNorm (2.7 ms vs 0.7 ms), since LayerNorm has a fused CPU kernel |
| QK-Norm | ~3% speed cost |
| GQA (2 KV heads) | small speed gain, fewer params |

Defaults are set to the faster options (`attn_impl="manual"`, `norm="layer"`); the other options stay as switches.

## Sample (prompt "Once upon a time", same settings as the baseline sample)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the grass with her friends. One day, they found a big box in the park. It was very tall and pretty. Lily's friend, Timmy, said, "Wow, they are very pretty!"
>
> Lily said, "Wow, the box is so cool! Can we play with it?"
>
> "Sure," said Timmy. "But you have to be careful."

More coherent than the baseline (consistent characters, sensible dialogue), but still makes logic mistakes in other prompts.

## Caveats

- One run per setting (no repeats), so small differences (~0.02) could be noise. The big gaps here are well beyond that.
- Muon was planned for Phase 4 but pulled forward because it's the largest win.
