# Phase 2: Baseline result (the number to beat)

Run: `python train.py --model baseline --name baseline_d256_l6 --minutes 25`
Raw data: [runs/baseline_d256_l6/](runs/baseline_d256_l6/)

| | |
|---|---|
| Model | Plain GPT: d=256, 6 layers, 4 heads, context 256, tied embeddings |
| Parameters | 6.89M total (4.73M non-embedding), all active |
| Training | 25 min, BF16 autocast, AdamW (FP32 states), 6 threads, batch 16×256 |
| Steps / tokens | 2,221 steps, 9.10M tokens |
| Speed | 6,064 tokens/sec average |
| Peak RAM | 1,420 MB (process working set, includes data pages) |
| **Final val loss** | **2.607** |

## Loss curve

| Train minutes | Val loss | Tokens/sec |
|---|---|---|
| 0 | 9.091 | — |
| 3 | 3.763 | 6,132 |
| 6 | 3.358 | 5,726 |
| 9 | 3.143 | 5,562 |
| 12 | 2.971 | 5,532 |
| 15 | 2.839 | 5,619 |
| 18 | 2.740 | 5,679 |
| 21 | 2.662 | 5,836 |
| 25 | **2.607** | 6,064 |

## Sample (prompt "Once upon a time", temperature 0.8)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the grass. One day, she saw a big, tall tree with a big nose. She walked over to the tree and saw it was a very small hole. She quickly put the tree in a tree and saw that was sad.

Grammar and story structure are good. Logic and consistency (who is who, what objects do) are weak. That's typical for this loss level.

## Notes

- **Still improving at the end:** loss was dropping until the last minute. The model is limited by the 25-minute budget, not by its size.
- **Speed was ~15–20% below the Phase 0 benchmark** (7,151 tok/s), dipping to ~5,500 mid-run. Likely causes: thermal throttling over a long run, plus gradient clipping and data loading. This is the realistic number for long runs.
- **Ceiling sweep not done yet:** this is one size (6.9M). Finding the baseline's best size in 25 minutes needs 2 more runs (~3M and ~14M).
