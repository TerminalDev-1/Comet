# Comet-Preview-2026-11-08

A ~15.7M-parameter Comet trained on the **RTX 4060 Ti** (driven from the laptop over SSH), then taught to **think in English** with `<think>` tags.

## Recipe (13 minutes of training)

1. **Base, 10 min:** TinyStories, d=384, 8 layers, 6 heads, batch 32, Muon + WSD, short-context warmup (64 → 256 over the first 30%), fused attention + RMSNorm. Run: `Comet-Preview-Base-2026-11-08`. Final val loss **1.74** (laptop CPU's best in 12 min: 2.44).
2. **Thinking, 3 min:** fine-tuned on 120k generated reasoning examples (`data/gen_think.py`), mixed 50/50 with TinyStories. Format: story → `Question:` → `<think>steps</think>` → `Answer:`. Run: `Comet-Preview-2026-11-08`.
3. **Test:** 400 stories with names and objects never seen in the reasoning data (`think.py`). Full output: [comet_preview_think_test.txt](comet_preview_think_test.txt).

## Result: 50.5% on new stories (202/400)

| Type | Right | |
|---|---|---|
| Feelings ("won the race" → proud) | 47/50 | ✅ |
| Object moved once | 42/50 | ✅ |
| Counting (add / take away, in words) | 39/50 | ✅ |
| Who has it (one handoff) | 24/50 | ⚠️ |
| Who has it (two handoffs) | 23/50 | ⚠️ |
| Who barked (animal vs person) | 10/50 | ❌ |
| Who's taller / bigger | 10/50 | ❌ |
| Where is it (two places) | 7/50 | ❌ |

**Diagnosis:** it learned the *format* and the rules (counting, feelings), but it **replaces new names and objects with memorized ones**. "Ollie is taller than Finn" became `<think>Zoe is taller than Ben…</think>`. The reasoning data used only 20 names and 10 objects, so memorizing them was easier than copying from the story. "Where is it" was also confused with the "moved" template.

**Next (v2):** hundreds of names and objects in the reasoning data, so copying from the story is the only way to score; keep the test names unseen. Maybe also a longer thinking phase.

## GPU size sweep (4 min each, before choosing the size)

| Size | Params | Val loss | Tokens/sec | VRAM |
|---|---|---|---|---|
| S (d256, 6L) | 6.5M | 1.820 | 146k | 1.7 GB |
| **M (d384, 8L)** | **15.7M** | **1.823** | 85k | 2.5 GB |
| L (d512, 8L) | 26.4M | 1.844 | 65k | 3.0 GB |
| XL (d640, 10L) | 48.3M | 1.930 | 39k | 4.3 GB |

The story-reasoning probe (`reason.py`, 24 two-choice questions) stayed at 58–71% for all sizes during plain story training. Plain stories teach cause and feelings, but not tracking who has what.
