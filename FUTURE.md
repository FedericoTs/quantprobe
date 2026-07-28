# Open checks and tasks

Live debt, ordered by what it would cost a user. Every item names its evidence; nothing here is a
hunch. Items are removed when closed by measurement, not when they stop feeling urgent.

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

**B3. All-in-VRAM point predictions exist only on the calibrated path.** The mechanism behind the
0.32–0.56 efficiency spread is no longer open: format unpack instruction cost plus metadata
application density (L-15/L-16, preregs #52/#53/#56/#57). v1.20.1 prices GPU decode per format on
the calibrated path (`spec.FORMAT_EBW` scaled by the anchor ratio), and the ladder's
self-consistent column lands at ~12% median with every big-model miss an under-promise
(v1.20.2 correction, `MACHINE_LADDER.md`). What remains: the uncalibrated preset path still ships
only the one-sided floor (real ≥ 0.90× predicted, 13/13 — the all-in-VRAM ratchet gates format
efficiency off presets deliberately), and every efficiency number in the ladder comes from one
Pascal card. Needs a second card for that half.

**B4. The boost-state check has been exercised on exactly one card.** The anchor drift that used
to sit here (17.53 → 17.03 → 16.89 on the same file) closed by measurement, and not the way the
item predicted: the driver was a stuck GPU boost state — SM at 1506 MHz on a cool, quiet box
against 1835+ healthy — not temperature, cleared by reboot, with the original calibration
reproducing to 0.5% afterwards (preregs #60/#61, C-10). `calibrate` now measures sustained clocks
(tg128, 1 s sampling, 3-sample minimum) and flags the stuck state. What remains: one consumer
Pascal card is the entire evidence base — whether other cards drift this way, or in ways the
check misses, is unmeasured.

## C. Open experiments

**C1. Asymmetric top-k, Stage 2.** Stage 1 passed: k=4 gives +21% to +63% prefill on all three
recommended placements, clearing the 14.3% frontier ceiling. Stage 2 asks whether the **+20.6%
perplexity cost attaches to ingestion or only to generation**, via slot save/restore between a k=4
and a k=8 server. Kill if within 5 points of the all-k=4 arm. If it survives it ships as an
**upstream PR against a pinned SHA**, never a fork. → `preregistrations/2026-07-27-asymmetric-topk-prefill.md`

**C2. Dense partial offload: measured in #66, and the fit math at depth is wrong.** The tool's
emitted 26/28-layer split on the 7B predicted 15.0 tok/s at 16k depth and measured 6.34 — −58%,
the one hard miss of that program. The 4k arm on the same model measured −0.9%, so the KV *term*
is right; the placement fit math over-commits VRAM at depth (compute buffer and KV not counted
jointly). Opened as C-11, fix queued; the rows live in `MACHINE_LADDER.md`. The
`audit.py:UNMEASURED_PLACEMENTS` reason string still says never-measured and needs the same
update.

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
Vehicle: TWO upstream deliverables, never a fork. #1 (measured 2026-07-27, prereg #34): build
guidance - GGML_OPENMP=ON on Windows/mingw routes ~2,070 barriers/token through kernel
semaphores and costs ~40% of CPU decode; GGML_OPENMP=OFF recovers it with ggml's own spin
barrier (11.92 -> 16.64 tok/s, one flag). #2: elementwise-chain fusion under the existing
ggml_cpu_try_fuse_ops hook against the residual 10.8 ms/token. Vehicle for both -
every quantprobe user runs stock llama.cpp, and that property is worth more than any local gain.
