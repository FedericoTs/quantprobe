# Open checks and tasks

Live debt, ordered by what it would cost a user. Every item names its evidence; nothing here is a
hunch. Items are removed when closed by measurement, not when they stop feeling urgent.

## A. Contradicted by external evidence — fix before the next release

**A1. GLM-family `kvp` is 30–60× too large.** `MODELS["glm-744b"]["kvp"]=188416` is `[est]` and
contradicted by an external 4×-cluster run whose decode is *flat* across 32× of context. At our
value the KV read at 262K would be 3× the weight bytes and decode would collapse; it doesn't. We
therefore predict a configuration that demonstrably runs **cannot fit**, and fall to disk-streaming
at 12.6 tok/s against a measured 103. Close it by reading GLM-5.2's attention config, **not** by
back-solving from the datapoint. → `weights/data/external_glm52_4x6000.md`

**A2. "Evict KV" advice sits on a VRAM cliff.** Same command, 193–195 tok/s with ~713 MiB of
desktop VRAM held, 437–438 with ~465 MiB. Four runs each side, error bars <0.5%. We ship this as
the long-prompt recommendation quoting 391.72 — a user with a browser open may get a value *worse
than every other frontier point*. Find the cliff edge by occupying VRAM in steps, then gate or
withdraw. → `weights/data/prereg22_asymmetric_topk.log`

## B. Unvalidated foundations

**B1. 82% of shipped presets have never been run.** 2 of 17 machines are `[measured]`, 1 is
validated against published figures, **14 are `[est]`**. 4 of 10 models have a verified layer
count. Contributed datapoints from other people's hardware: **1**. Every placement finding from
2026-07-26/27 rests on one 2016 desktop.

**B2. The multi-device aggregation factor is untested.** `agg_bw(v, 0.85)` applies one constant
regardless of interconnect. ds4 publishes a 5× spread from the link alone on identical hardware
(TB5 582/25.1, WiFi 250/10.7, VPN 114/3.6). Our two 4-device datapoints do **not** show the
constant is wrong — one is +12%, the other unscoreable — so this is untested, not broken.
→ `weights/data/external_glm52_three_clusters.md`

**B3. All-in-VRAM decode has no point prediction — only a measured floor.** Efficiency varies
0.32–0.56 across 8 models; six candidate explanations refuted (fixed overhead, clock state,
bytes-per-token, monotone-in-bits, per-format constant, bytes-weighted tensor mix — the last
forces a negative efficiency for Q5_K when solved). Shipped as a one-sided bound: real ≥ 0.90×
predicted, 13/13, falsifiable by any single measurement below it. Within one architecture the
pattern is monotone in the dominant tensor type; across architectures it does not transfer.
Needs a second card.

**B4. In-VRAM anchors were measured at inconsistent GPU temperatures.** The same file measured
17.53 → 17.03 → 16.89 as the card warmed to 72 °C, error bars ±0.02. Re-measure the `VRAM_GAPS`
set thermally settled before any of them justifies a constant.

## C. Open experiments

**C1. Asymmetric top-k, Stage 2.** Stage 1 passed: k=4 gives +21% to +63% prefill on all three
recommended placements, clearing the 14.3% frontier ceiling. Stage 2 asks whether the **+20.6%
perplexity cost attaches to ingestion or only to generation**, via slot save/restore between a k=4
and a k=8 server. Kill if within 5 points of the all-k=4 arm. If it survives it ships as an
**upstream PR against a pinned SHA**, never a fork. → `preregistrations/2026-07-27-asymmetric-topk-prefill.md`

**C2. Dense partial offload has never been measured.** Declared in `audit.py:UNMEASURED_PLACEMENTS`.
Needs a dense model that overflows 6 GB.

**C3. Streaming-tier prefetch gap, now quantified.** Our disk tier models naive LRU and is ~7×
pessimistic against ds4's prefetching engine (0.7 vs ~4.8 tok/s). Belongs in the README
limitations, not only in a deep-dive.

## D. Scope corrections to published claims

**D1. The "don't fork" verdict is regime-scoped.** The 1–6% ceiling (and its 14.3% closed form)
was computed entirely in the VRAM- and host-resident regimes. On the **streaming** tier a
purpose-built engine looks worth ~7×. The arithmetic stands; stating it as a general claim about
custom runtimes did not.

**D2. Task-level evals are absent.** Perplexity and KL divergence only — no MMLU/HellaSwag. Already
in the README limitations; listed here because every quality claim we make inherits it.

## E. Owed to the community

**E1. TurboQuant on Pascal**, promised to u/MoneroApe. Never run.
**E2. Make `bench --contribute` frictionless.** One contributed datapoint in the project's life is
the single clearest signal that B1 will not fix itself.

---

## The one measurement we cannot make: C-06 on a modern GPU

Batched decode on our GTX 1060 saturates at ~2× by 4 slots, identically across MoE/dense and
every placement — and no mechanism we model explains it. We own exactly one GPU, so this is the
single highest-value contribution a user can make. One command, any post-2020 card:

```bash
llama-batched-bench -m your-model.gguf -ngl 99 -c 8192 -npp 512 -ntg 128 -npl 1,2,4,8
```

Send the printed table (plus your GPU model) via `quantprobe bench --contribute` or a GitHub
issue. If your `S_TG` column scales past ~2.3× from `npl 1` to `npl 8`, our ceiling is a museum
piece of Pascal and we will publish that. If it does not, it is a law, and we will publish that
instead. Either answer is valuable; we cannot produce either one ourselves.

---

## The one upstream PR worth writing: ggml CPU `MUL_MAT_ID`

Measured (pre-registrations #27/#31/#32): the CPU expert path delivers 17.1 GB/s where the dense
kernel delivers 28.4 on the same box, same format, same threads - and the memory system is proven
indifferent to the access pattern (shuffled 2 MB slabs: +2.9%). The ~40% is code: ~1,150 small
per-expert GEMVs per token against dense's ~140 large ones, with sub-linear thread scaling
(1x/1.64x/2.17x). Prize if fixed: **+65% on the host share, raw decode 22.25 -> ~30 tok/s
realistic** on 2016-class hardware - the novel-generation regime where speculation cannot help.
Vehicle: an upstream PR (batched/fused expert GEMV, per-token expert grouping), never a fork -
every quantprobe user runs stock llama.cpp, and that property is worth more than any local gain.
