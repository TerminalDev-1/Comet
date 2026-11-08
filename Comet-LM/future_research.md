# Comet — future research ideas

Ideas that, **to my knowledge, nobody has published yet**, chosen to fit Comet's world: small models, laptop CPUs, 10–15 minute training runs.

> ⚠️ **Before claiming any of these as "first":** my knowledge has a cutoff and this field moves fast. Each idea needs a literature search (arXiv, Google Scholar, Papers with Code) before it goes into the paper as novel. Closest known work is listed for each, so we know what to cite and how we differ.

> ⚠️ **Correction about EMOE:** sharing experts across layers is *not* new. **MoEUT** (Csordás et al., 2024) shares MoE layers across depth in a looped "universal" Transformer. EMOE's version (no looping, a router per layer, an always-on shared expert, bias-based balancing) is a variant, and the paper must cite MoEUT.

---

## 1. Thermal-aware training scheduler
**Idea:** laptops slow down when hot (we saw dense Comet swing between 4,500 and 5,100 tok/s). A scheduler reads CPU temperature and clock speed while training, and adapts threads, batch size and the learning-rate clock to get the most learning per minute (or per joule) as the chip heats up.
**Why it may be new:** training research assumes steady datacenter hardware; consumer-laptop thermals are basically ignored.
**Closest known work:** power capping and DVFS studies for datacenter GPUs; thermal-aware *inference* on phones.
**First experiment (≤15 min):** log temperature + tok/s across one 12-min run, then compare fixed 6 threads against an adaptive 4/6/12-thread policy.

## 2. Time-budget scaling laws for laptop CPUs
**Idea:** "Given 10 minutes on a laptop CPU, what model size, batch size, vocabulary size and learning rate give the best model?" A formula fitted from many short runs, like the Chinchilla scaling laws, but for wall-clock time on consumer hardware.
**Why it may be new:** scaling laws are fitted for GPU clusters and FLOPs; nobody fits them for minutes on a laptop.
**Closest known work:** Chinchilla (Hoffmann et al., 2022); "Cramming" (Geiping & Goldstein, 2022: BERT in one GPU-day).
**First experiment:** 6 sizes × 3-minute runs (~20 min total), fit loss vs. size, then predict and verify a 12-minute run.

## 3. Expert mitosis: growing EMOE during training
**Idea:** fixes the "experts are starved of data" problem. Start with a few large experts (each gets lots of data early), then split each into two smaller experts partway through, copying weights plus a little noise so the children specialize. The expert count grows as the model has learned enough to use it.
**Why it may be new:** known methods turn a dense model into MoE once ("sparse upcycling") or grow width/depth; splitting experts repeatedly *inside* one short run doesn't appear to be studied.
**Closest known work:** Sparse Upcycling (Komatsuzaki et al., 2022); Net2Net growing (Chen et al., 2015).
**First experiment:** 3-min runs: 16 experts fixed vs. 8 → 16 → 32 via mitosis at minutes 1 and 2.

## 4. Cheap Muon for expert weights
**Idea:** Muon is Comet's biggest single win, but it's ~10× too slow on EMOE's expert pool. Make it cheap: orthogonalize each expert only every k steps (round-robin), reuse orthogonalization work across experts, or orthogonalize only the experts that were used this step.
**Why it may be new:** Muon is from 2024; how to use it efficiently on MoE weights is still mostly open.
**Closest known work:** Muon (Jordan et al., 2024); Moonshot AI's "Muon is scalable" (2025) for large dense models.
**First experiment:** time + 3-min quality for round-robin Muon (k = 4) vs. AdamW experts vs. full Muon.

## 5. One router for all of a token's compute
**Idea:** today MoD (skip a layer?), EMOE (which experts?) and adaptive loops (how many passes?) each have their own router. A single "compute budget" router per token decides all three from one small score vector, so easy tokens get the cheap path everywhere and hard tokens get the expensive one.
**Why it may be new:** these mechanisms are usually studied one at a time.
**Closest known work:** Mixture-of-Recursions (2025) combines routing with recursive depth; MoD (Raposo et al., 2024); MoEUT.
**First experiment:** after looped blocks exist, compare separate routers vs. one shared router at equal speed.

## 6. Group-shared expert pools
**Idea:** in between "every layer has its own experts" and "one pool for all layers": early layers share one pool (token-level patterns), late layers share another (meaning, story structure). This tests the review's concern about early and late layers needing different features.
**Why it may be new:** sharing granularity across depth (all vs. groups vs. none) at equal memory hasn't been mapped systematically, as far as I know.
**Closest known work:** MoEUT; ALBERT-style cross-layer sharing (Lan et al., 2019).
**First experiment:** at equal parameter count: per-layer vs. 2 groups vs. 1 shared pool, 3 minutes each.

## 7. Hardware-adaptive architecture
**Idea:** Comet measured that the "fast" choices (fused attention, RMSNorm) were slower on this CPU. Turn that into a feature: at startup, Comet runs a 10-second micro-benchmark and picks the attention kernel, norm type, expert size and batch size that give the best *quality per second* on that exact machine.
**Why it may be new:** kernel autotuning exists, but picking *architecture* choices by measured speed on the user's own device, during setup, isn't standard.
**Closest known work:** TVM / Triton autotuning (kernels only); hardware-aware neural architecture search (e.g. ProxylessNAS, 2019), which searches offline, not per device at setup.
**First experiment:** build the `auto` defaults from review point 7, then check they pick the right options on CPU vs. a GPU machine.

## 8. Energy per quality on consumer hardware
**Idea:** report **watt-hours to reach a given loss** as a headline metric, measured on an ordinary laptop. Combine with idea 1 to train "greener," not just faster.
**Why it may be new:** energy is reported for big datacenter runs, rarely for small models on personal machines, and almost never as the optimization target.
**Closest known work:** "Energy and Policy Considerations for Deep Learning in NLP" (Strubell et al., 2019); the ML CO2 Impact calculator.
**First experiment:** log CPU package power (HWiNFO or AMD's tools) during baseline vs. Comet runs, then add it to the bar charts.

## 9. Tokenizer chosen by the time budget
**Idea:** vocabulary size changes both quality per token and speed (the output layer is a big cost on CPU). Pick the vocabulary size that gives the best loss *per minute*, not per token, and maybe grow it during training.
**Why it may be new:** vocabulary-size studies optimize per FLOP or per token, not per wall-clock minute on a CPU.
**Closest known work:** "Scaling Laws with Vocabulary" (Tao et al., 2024).
**First experiment:** retokenize with 4k / 8k / 16k vocab, 3-min runs each, compare (scored per character, not per token, so it's fair).

---

## Suggested order
1. **#4 cheap Muon for experts** and **#6 group-shared pools**: build directly on EMOE, cheap to test, and could make EMOE win.
2. **#3 expert mitosis**: the most original EMOE idea, and it targets the data-starvation problem directly.
3. **#1 thermal-aware training** and **#8 energy per quality**: unique to laptops, a strong angle for the paper.
4. **#2 time-budget scaling laws**: turns all our short runs into a reusable result.
