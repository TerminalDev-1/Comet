# Benchmark

![benchmark](benchmark.jpg)

| Model | Train time | Final val loss | Time to reach baseline's final loss | Tokens/sec | Tokens seen | Params (total / active) | Peak RAM |
|---|---|---|---|---|---|---|---|
| Baseline GPT (25 min) | 25 min | **2.607** | 25.0 min | 6,064 | 9.1M | 6.89M / 6.89M | 1,420 MB |
| Comet (12 min) | 12 min | **2.420** | 9.7 min | 5,029 | 3.6M | 6.52M / 6.52M | 1,415 MB |
