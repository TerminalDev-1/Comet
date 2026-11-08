# Benchmark

![benchmark](curves.jpg)

| Model | Train time | Final val loss | Time to reach baseline's final loss | Tokens/sec | Tokens seen | Params (total / active) | Peak RAM |
|---|---|---|---|---|---|---|---|
| Baseline GPT (25 min) | 25 min | **2.607** | 25.0 min | 6,064 | 9.1M | 6.89M / 6.89M | 1,420 MB |
| Comet (12 min) | 12 min | **2.441** | 9.8 min | 4,658 | 3.4M | 6.52M / 6.52M | 1,403 MB |
| Comet + MoD (12 min) | 12 min | **2.475** | 10.3 min | 5,169 | 3.7M | 6.53M / 6.53M | 1,485 MB |
