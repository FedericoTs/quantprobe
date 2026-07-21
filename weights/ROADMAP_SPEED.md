# ROADMAP_SPEED.md — Campaign 2: 7B near-lossless, minimal memory, maximal tok/s on the GTX 1060

Consolidated 2026-06-11 from 4 scout briefs + 24 hostile-verified ideas (scores 4–7.5).
Duplicates merged; every verdict improvement is folded into the spec it amends.
Methodology unchanged from Campaign 1: every claim measured on THIS hardware, honest
bytes = decoded containers, bit-exact gates, pre-registered kill criteria, 0.5B-first
scale ladder (0.5/1.5/3/7B all local).

---

## 0. Context and consolidated facts

**What we have.** Per-group-128 signed-Hadamard + AWQ scaling + 64-level ECVQ with
entropy-coded iid indices (D_idx=0 proven → static-table rANS optimal AND GPU-friendly)
+ 0.5% fp16 outliers. Honest storage ~3.0 b/w; current packed runtime 6.6 b/w resident.
Verified direct-execution runtime (evoq) runs Qwen2.5-7B on the 1060 at 0.02 tok/s
(Python overhead, not kernel time). 4-point scale law confirmed by pre-registered
prediction; lambda=.003-class at 7B is +0.06 ppl (near-lossless) at ~3.9 b/w.

**Load-bearing facts established by the scouts (each changed the plan):**

1. **RE-BASELINED TARGET.** The brief's "Q4_K_M 12–18 tok/s" is stale. Community
   llama-bench: GTX 1060 6GB does **27.79 tok/s tg128 on 7B Q4_0** (~52% of 192 GB/s
   effective), pp512 417–446. Q4_K_M scales by bytes to **~24–26 tok/s**. The target
   to beat is ~2x harder than briefed. (llama.cpp discussion #15013; re-measure locally in A0.)
2. **THE KERNEL IS DECODE-COMPUTE-BOUND, NOT BANDWIDTH-BOUND.** ~6.5e9 non-embedding
   symbols/token × ~12–20 int ops/symbol rANS ≈ 80–130 Gops/token vs a ~45–58 tok/s
   bandwidth ceiling at 3.3GB resident. This INVERTS the memory-bound rationale that
   makes ZipServ/dtANS win on modern GPUs. Decode throughput in **Gweights/s** is the
   binding constraint and the single number the campaign hinges on.
3. **Required decode rate, derived (use this everywhere, not "Gsym/s" loosely):**
   `required Gweights/s (in-situ, fused) = tok/s_target × 6.5`.
   12 tok/s → ~80; 20 tok/s → ~130; Q4_K_M parity (24) → ~155.
   Scaled prior-art extrapolations (DietGPU/Recoil/nvCOMP per-SM rates) predict only
   ~8–40 Gweights/s generic on 10 Pascal SMs. **The gap is 2–4x even after
   specialization — dynamic masking, pair-decode, and spec-decode are not nice-to-haves;
   at least one must land or full-decode loses to llama.cpp.**
4. **sm_61 fast path is dp4a int8 (~17.6 TOPS), not fp32 (4.4 TFLOPS).** The "2–4 Tops"
   budget was ~4x pessimistic IF inner loops are int8/dp4a. fp16 ARITHMETIC is banned
   (1/64 rate on cc 610, llama.cpp blacklists it explicitly); half is storage-only.
5. **FWHT leaves the weight kernel.** Per-group-128 rotation along the contraction dim
   commutes through the dot product: apply signed-FWHT-128 to the ACTIVATION vector
   once per layer (fp32, microseconds). In-kernel iFWHT would cost ~7 add-ops/weight
   and dominate. The weight inner loop collapses to: rANS-decode → int8 LUT → dp4a —
   shape-identical to llama.cpp MMVQ.
6. **The novelty niche is real but thin and time-limited.** No published system gates
   VARIABLE-LENGTH entropy decode with contextual sparsity (DFloat11/dtANS/EntroLLM
   decode densely; ZipServ fused but abandoned variable-length; PowerInfer/LLM-in-a-flash
   skip fetch of fixed-length blocks). BUT arXiv 2511.04477 already does sparsity-skipped
   dequant of fixed-length quantized groups, arXiv 2512.21911 already does sparse
   verification in spec-decode, and ZipServ walked away from variable-length-in-the-hot-loop.
   Cite PowerInfer, EIE, dtANS, ZipServ, 2511.04477 as nearest prior art; the window is months.
7. **SiLU reality.** Qwen2.5 has weak natural sparsity. Training-free near-lossless
   operating point is ~25–30% unstructured (TEAL +0.07–0.09 ppl); 40–50% costs
   +0.3–0.8 ppl and risks Sirius-style reasoning collapse. Our format needs
   **128-block-aligned** sparsity, which is unmeasured anywhere and plausibly worse —
   that measurement (B1) gates the entire masking track.
8. **Spec decode is the highest-confidence multiplier** (published numbers on this exact
   model family: 0.5B→7B ~62% acceptance, 1.67–2.36x) and is unusually favorable for a
   decode-bound runtime (decode paid once per K verified tokens). It partially COMPETES
   with masking (union-mask erosion ~1−(1−s)^K) — B3 arbitrates with a measurement.
9. **Memory budget closes with discipline.** 3.3GB weights + 448MB KV fp16@8k + <5MB
   activations + CUDA context ≈ 4.05GB on ~5.0–5.6GB usable (WDDM audit in A0 confirms).
   KV quant / eviction / PagedAttention: consciously rejected at ≤8k batch-1 (recorded in §8).
10. **Toolchain landmine:** Pascal dropped from CUDA 13.x AND from torch 2.8+ cu128/cu129
    wheels. Pin CUDA ≤12.x toolkit + a Pascal-supporting torch wheel NOW (A0), record in
    the verifier manifest.

**Standing engineering rules (banned in code review if violated):**
- Never materialize decoded weight tiles to DRAM (one fp16 FFN matrix = 136MB; decode
  flows rANS → LUT → dp4a in registers/smem only).
- No fp16 arithmetic anywhere; fp16/int8 storage only; fp32 accumulate.
- No host-side per-token decisions (breaks CUDA-graph capture, the known ~1.2x); all
  masking/gating decisions on-device.
- Honest bytes = full decoded container: payload + ALL split states, offsets, tables,
  scales, LUTs, per-group metadata, outliers. 16-bit states per group-128 alone are
  0.125 b/w = ~4% at 3.0 b/w — header accounting is material and must be itemized.
- Every published tok/s is paired with end-to-end quality from the SAME binary at the
  SAME operating point (no mask-ON speed with mask-OFF ppl).

---

## 1. Targets table — baselines to beat (community anchors; ALL re-measured locally in A0)

| System | Storage | Resident (est) | tg128 tok/s | pp512 | Quality note |
|---|---|---|---|---|---|
| llama.cpp Q4_0 7B | ~3.6GB file | ~4.0GB | **27.8** (28.2 w/FA) anchor | 417–446 | reference speed point |
| **llama.cpp Q4_K_M 7B (PRIMARY)** | 4.08GB file (4.85 b/w avg) | ~4.4–4.5GB | **~24–26 est → MEASURE** | ~400 est | community-standard quality |
| llama.cpp IQ3_XXS 7B (SOFT target) | ~3.06 b/w | ~3.3GB | slower than Q4_K_M on Pascal (grid-LUT tax) → MEASURE | MEASURE | worse quality than ours at matched bits |
| llama.cpp IQ3_S 7B | ~3.44 b/w | ~3.6GB | MEASURE | MEASURE | |
| evoq today | ~3.0 b/w honest | 6.6 b/w packed | 0.02 (Python) | — | +0.06 ppl @ 3.9 b/w proven |
| evoq Campaign-2 goal | 3.1–3.9 b/w | **3.3–3.9GB** | dense fused 10–20 est; ×masking ×spec | MMQ-tile path | near-lossless (+0.06–0.15) |
| Physical ceilings (1060) | — | — | bandwidth ~45–58; decode-bound 10–20 dense | — | |

**Claim ladder (pre-registered):** (i) guaranteed-claimable first win = beat IQ3-class on
ALL THREE axes (bytes, quality, tok/s); (ii) primary win = bytes+quality win vs Q4_K_M
with tok/s ≥ parity, dense vs dense; (iii) stretch = beat Q4_K_M tok/s outright;
(iv) any spec-decode headline must give llama.cpp its own draft (spec-vs-spec fair fight).

---

## 2. PHASE A — Foundations: convert every load-bearing extrapolation into a measured number

*Everything in A is measurement or infrastructure. No GA. Budget: ~3–5 weeks total,
A0→A1→A2 strictly ordered, A3 parallel, A4 conditional on A1.*

### A0. gate-pack-1060 — baselines, VRAM audit, toolchain pin, free VRAM win  [score 7.5; 2–3 days]
- **Mechanism:** (1) llama-bench Qwen2.5-7B Q4_K_M + IQ3_XXS + IQ3_S tg128/pp512 on this
  card — **pre-registered validity condition: full offload (-ngl 99) verified, no silent
  partial offload on the 6GB/WDDM card**; (2) VRAM audit: empty-CUDA-context +
  WDDM display reservation via **cudaMemGetInfo deltas + DXGI budget** (nvidia-smi is
  not honest under WDDM paging); (3) pin CUDA ≤12.x toolkit AND Pascal-supporting torch
  wheel; record driver/clocks in `weights/baselines.json` (the verifier manifest);
  (4) move the untied embedding table to CPU RAM (7KB row-gather/token, ~390MB VRAM
  freed at 6-bit) — gated on bit-identical logits vs GPU-resident embedding.
- **Verifier:** third-party llama-bench on fixed prompts; cudaMemGetInfo deltas;
  bit-identical-logits gate for the embed move. (The "tok/s within 2%" embed gate is
  vacuous at 0.02 tok/s — re-run it once the C++ loop exists.)
- **Kill criterion:** usable VRAM < 4.5GB → re-budget the 3.3GB-resident + draft plan
  before any kernel work. All downstream thresholds are DERIVED from the measured
  Q4_K_M number, not from the stale 12–18 estimate.
- **Prior art note:** pure measurement; embed-to-CPU is standard llama.cpp practice.

### A1. decode-rate gate — the campaign's single decisive number  [scores 6.5/6.5/7.5 merged; staged: 1–2 days + 1.5–2 weeks]
- **Mechanism, staged per verdict improvements:**
  - **Stage 0 (1–2 days, before ANY custom kernel):** build stock DietGPU for sm_61
    under CUDA 12.x; measure generic byte-rANS decode Gsym/s on the REAL 0.5B container,
    streamed cold end-to-end (container ≫ 1.5MB L2, so hot-cache inflation is structurally
    avoided). This anchors the 8–25 extrapolation for ~20% of the effort.
  - **Stage 1 (the specialized microbench):** static compile-time table, 64-symbol
    alphabet; **pair-symbol decode as the PRIMARY candidate, not a knob** (4096-cell
    product alphabet; use empirical-support pruning + top-K-pairs-with-escape; direct
    slot tables at L=2^12 are NOT zero-rate-loss — alias-method or L=2^13 swept);
    32-lane word-interleaved substreams (dtANS pattern: ballot_sync + 2×popc);
    Recoil-style 16-bit post-renorm split states. All intrinsics confirmed on sm_61.
  - **Two numbers reported, both SHA-gated on the GPU-written buffer:** (a) isolated
    decode Gweights/s; (b) a **fused-proxy variant** — decode + int8-LUT + dp4a
    dummy-accumulate (+ optional 7-stage FWHT arm) compiled at the planned fused kernel's
    register/smem budget — because isolated decode systematically overstates fused throughput.
  - **Anti-overfit:** sweep knobs (table slots 2^10–2^13, renorm width, ILP segment
    length, streams/block) on 3 dev containers; **report the headline on a 4th held-out
    container generated after the sweep.**
- **Verifier:** bit-exact hash match vs the reference Python decoder on real 0.5B
  containers; median-of-5, CUDA events, clocks logged.
- **Kill criteria — RE-DENOMINATED in Gweights/s with thresholds derived from A0's
  measured baseline (the old 5/15 Gsym/s gates were miscalibrated by ~5–30x and are void):**
  - **< 25 Gweights/s fused-proxy:** fused full-decode architecture DEAD. Pivot to
    hybrid: packed-6-bit resident hot layers + rANS as storage/transport only.
  - **25–80:** full decode loses to Q4_K_M by itself → **masking and/or spec-decode are
    MANDATORY, not optional**; flagship becomes masked/spec-amortized by construction.
  - **≥ 80 (≈ 0.8 × measured-Q4_K_M × 6.5):** dense fused full-decode is viable;
    greenlight A4 as the headline path.
  - Sub-gates: pair-table measured rate loss ≤ 0.3% vs single-symbol; container header
    scheme must keep total non-payload overhead < 1% at chunk granularity (see A2).
- **Prior art note:** DietGPU, Recoil (16-bit split states), dtANS arXiv 2603.01915
  (warp-interleave + fused-SpMVM tricks, decodes everything), multians (Pascal existence
  proof, tens of Gsym/s on a 20-SM GTX 1080 — the outcome is half-predictable: expect
  the 25–80 branch), pair-decode is FSE/Oodle folklore. Component novelty zero; the
  measured sm_61 number is the deliverable.

### A2. container v2 — pair-tANS + warp-interleaved random-access substreams  [score 6; day-0 script + 1–1.5 weeks, overlaps A1]
- **Mechanism:**
  - **Day-0, BEFORE any CUDA (30-min numpy script, binding):** exact pair-tANS coded
    rate from the real per-tensor index histograms; run the actual slot-assignment at
    L ∈ {2^12, 2^13, 2^14}; include pruned-alphabet + escape variants.
    **Binding rate gate: honest b/w(pair container, ALL headers+states+tables) ≤
    honest b/w(single-symbol baseline) + 0.03 b/w.** Kills or re-scopes the pair table
    in hours instead of after a week of kernel work.
  - Then: encoder-side container v2 — pair-tANS, 32-lane word-interleaving with one
    base pointer per warp-chunk, 16-bit split states. **State the random-access
    granularity honestly: interleaving gives chunk-level (~4096-weight) entry, NOT
    per-group-128 (lane offsets are data-dependent; per-group-128 entry costs ~4–6%
    rate).** Co-design chunk size with the masking unit from B1/C2 (e.g. chunk = one
    FFN row = 28 groups) rather than fixing 32 lanes blindly.
  - LLM-in-a-flash row-column bundling: each FFN neuron's up/gate rows + down column
    substreams contiguous, so one future gating decision skips one contiguous region.
- **Verifier:** SHA-256 round-trip of every substream vs original index arrays; honest
  bytes itemized (payload / states / offsets / tables / bundling metadata) vs the 3.0 b/w
  baseline; decode rate measured on the A1 harness **under the fused kernel's actual
  smem/register budget**, GPU clocks recorded.
- **Kill criterion:** pair-decode < 1.4x single-symbol in fused-proxy context → drop the
  pair table, keep interleave+split-state layout (it gates C2's random access regardless).
  Total non-payload overhead that cannot get under 1% of honest bytes at chunk
  granularity → redesign headers before anything ships.
- **Prior art note:** dtANS (near-verbatim for interleaving), Recoil (split states),
  Collet/Giesen multi-symbol decode, US6956511. Composition unclaimed; field converging.

### A3. evoq-arena — the joint verifier everything else evolves against  [score 7; 1–2 weeks, parallel; extends weights/quant_arena.py + evocompress/evaluator.py]
- See §6 for the full joint-verifier design (it incorporates all verdict improvements:
  resampled eval pools, host-RSS axis, two-tier eval, thermal protocol).
- **Kill criterion (on the arena itself):** timing CV > 3% across 5×5 repeats after
  countermeasures, OR 0.5B↔7B Spearman of format candidates across the existing 4-point
  ladder < 0.8 → the arena cannot discriminate; halt all evolution until redesigned.

### A4. evoq-mmvq — dense fused kernel (CONDITIONAL on A1 ≥ 25, headline if ≥ 80)  [score 6; 2–4 weeks]
- **Mechanism:** llama.cpp MMVQ skeleton verbatim (1–2 rows/threadblock, register-direct
  weight streaming, NO smem weight staging, Q8_1 activation pre-quantization with
  precomputed sums so ECVQ offsets cost zero per-weight work, warp-shuffle reduction);
  activation-side signed-FWHT-128 pre-pass (fp32, once per layer); 64-level ECVQ
  codebook **snapped to int8 × per-group fp scale** (IQ4_NL's kvalues trick — 64B table
  in registers/cmem) feeding dp4a; **pair-decode in the foundation, not the GA**
  (it is the difference between compute-bound 12–20 and bandwidth-bound 20–30 tok/s);
  container v2 substreams; 6-bit dp4a lm_head kernel; C++ token loop + CUDA-graph
  capture (~1.2x) replaces the Python loop. Prefill: MMQ-style decode-once-into-smem-int8-tile
  (decode amortizes over ncols — the format's CHEAPEST regime; copy +16B padding,
  stream-k, need_check verbatim). Fallback path if fusion disappoints: DFloat11-style
  per-block decode into smem/L2 tile then dense dp4a.
- **Verifier (fixed per verdicts — the old "bit-exact vs evoq fp32" gate was impossible
  and is void):**
  1. **Offline quality gate first:** int8-snap + Q8_1-of-rotated-activations are new
     error sources → held-out ppl within pre-registered budget vs the fp32-LUT arena
     reference **at 0.5B AND 1.5B** before the snapped pipeline is frozen as reference.
  2. **Bit-exactness at the int32-accumulator level:** every candidate/kernel-variant
     must produce per-(row, group-128) int32 dot products bit-identical to a slow torch/CPU
     golden of the SAME integer pipeline (order-free: integer addition is associative);
     the fp32 scale-and-sum runs in ONE canonical fixed-order epilogue shared verbatim
     by all variants (excluded from evolution, included in timing). This keeps the gate
     un-gameable while freeing geometry/reduction mutations in C1.
  3. tok/s median-of-5, fixed prompts, clocks logged; profiler decode-ops/token and
     DRAM bytes/token published alongside.
- **Kill criterion (recalibrated):** dense fused on 0.5B < 40% of same-card
  byte-scaled Q4_K_M-equivalent throughput after a bounded tuning budget → kill dense-fused
  as headline; pivot to masked-decode-only (C2) over packed-6-bit hot layout, per the A1 branch.
- **Prior art note:** QTIP (activation-side Hadamard + fast in-kernel decode — closest
  full-stack neighbor), ZipServ (fused but fixed-length; their decoupled baselines show
  decode-into-buffer is a 3–5x tax at batch-1 — our negative control), DFloat11
  (decoupled, batch-1 overhead), FLUTE, MLX issue #3043 (concept circulating). Composition
  novelty on a neglected target (variable-length rANS in a dp4a GEMV on sm_61).

---

## 3. PHASE B — Science oracles: pure torch, no CUDA, runs PARALLEL to A1/A2/A4

*These produce the numbers that decide which Phase C tracks exist. All thresholds
pre-registered before measurement. Budget: ~2 weeks wall-clock, mostly unattended.*

### B1. 128-block structured-sparsity quality law  [merged ideas 2+12+19, scores 6/7/6; ~1 week]
- **Question:** what does 128-block-structured gating (the ONLY granularity that commutes
  with the codec) cost vs unstructured TEAL at matched MEASURED decoded fraction, on
  Qwen2.5 + this codec?
- **Mechanism:** forward-hook mask simulation in the existing arena harness. Grid:
  decoded-fraction targets {10,20,25,30,40,50}% × statistic {128-block energy,
  WINA-weighted block energy (per-group weight norms precomputed, charged to honest
  bytes), unstructured TEAL control, CATS FFN-only} × basis {Hadamard, PCA/LaRoSA
  control}. Greedy per-layer allocation (TEAL recipe) applied to BOTH structured and
  unstructured arms (fairness). Log per-token masks → decoded-symbols/token counted
  exactly; adjacent-token Jaccard; union growth K=2..8 (feeds B3).
- **Verdict improvements folded in (all mandatory):**
  - **Granularity stratification:** 0.5B hidden=896 = only 7 blocks of 128 (vs 28 at 7B)
    — a raw 0.5B kill would fire on a granularity artifact. Report penalties separately
    per matrix class (7-group residual-stream inputs vs 38-group down_proj); **the kill
    decision binds ONLY on the ≥32-group stratum**, plus a pre-registered 2–3-cell
    confirmation slice at 1.5B (12/70 groups) verifying the penalty shrinks with group
    count; fit the penalty trend vs blocks-per-hidden-dim across the ladder (exactly the
    4-point-scale-law methodology).
  - **Statistically powered gates:** absolute ppl deltas measured at 40% sparsity (where
    both arms clear the 0.07-ppl seed-noise band), not a ratio whose denominator equals
    the noise; reasoning gate = **GSM8K ≥ 500 samples with binomial power calc, or
    (cheaper, deterministic) ppl over gold GSM8K solution traces**; escrow slice held out;
    paired per-item McNemar if accuracy is used.
  - **Re-verify mask invariance WITH AWQ scaling in the loop** (block energy must be
    computed on scaled activations; pure-Hadamard invariance does not automatically survive
    per-channel scaling).
  - Read arXiv 2511.04477 first; reuse their layout/index-gather overhead numbers.
- **Kill criterion (binding on C2 and the fusion-novelty claim):** on the ≥32-group
  stratum, confirmed at 1.5B: structured penalty > 2× unstructured at matched measured
  decoded fraction, OR reasoning gate fails at the near-lossless threshold, OR no arm
  reaches ≥ 20% FLOP-weighted block sparsity at ≤ +0.05 ppl → **skip-decode fusion is
  dead as a quality-adjusted win**; demote to appendix/negative result; spec-decode (C3)
  takes the flagship multiplier slot.
- **Prior art note:** TEAL/WINA/CATS/LaRoSA anchor the unstructured numbers (none
  measure 128-block, none sub-7B); BlockFFN predicts chunk-sparsity is expensive
  training-free (a pass here is genuinely informative); Sirius mandates the reasoning gate.

### B2. co-activation permutation oracle + analytic joint-deadness kill  [idea 3 reduced per verdict, score 4.5→salvaged core; 1–2 days]
- **Mechanism:** from cached activation masks (one calibration pass), compute per-layer
  the joint-dead frequency of the single best greedy/spectral 128-cluster of FFN
  intermediate neurons (permutation is exact: permute up/gate rows + down columns before
  rotation; model function unchanged). This is the analytic ceiling for down-proj
  group-skip — hours of popcount work, NO GA, NO re-encode.
- **Pre-registered controls:** identity, random permutation, greedy clustering. Threshold
  tau set by a ppl-with-gating-ON gate on CROSS-DOMAIN held-out data (trace on wikitext,
  evaluate on chat), never by skip fraction.
- **Also run the dominating control the verdict demanded:** re-encode down_proj only
  with rotation group 16/32 (or unrotated, per-channel scaled) and measure the b/w
  penalty — if < 0.3 b/w on down-proj it enables native unstructured CATS row-skip at
  ~50% and makes any permutation GA moot.
- **Kill criterion:** best greedy cluster < 25% jointly dead at the quality-anchored
  threshold → NO permutation can average ≥10% group-skip → kill the down-proj-gating
  branch entirely (up/gate output-row gating needs no alignment and survives independently).
  GA on permutations only if greedy passes AND the GA beats greedy by ≥1.5x in a bounded spike.
- **Prior art note:** Neuralink (arXiv 2410.19274), Pool & Yu channel permutations,
  MoEfication clustering. The 128-all-dead requirement (P~s^128 under independence) makes
  the expected outcome a kill — that negative is cheap and load-bearing.

### B3. amortization duel — spec-decode vs masking, measured before either gets kernel budget  [ideas 7/15/20 merged, scores 6/6/5; ~1 week]
- **Question:** do the two decode-amortization levers compose, and what does each
  actually buy on this stack?
- **Mechanism (offline, torch + existing evoq runtimes):**
  - Measure adjacent-token 128-block mask overlap and **union active fraction at the
    REALIZED accepted-K from actual evoq-0.5B draft acceptance traces on chat+code
    prompts** (not synthetic K, not the independence formula): thresholds frozen at a
    pre-registered held-out quality bound (≤ +0.05 ppl vs dense) BEFORE measuring unions.
  - **Draft-speed gate (the failure mode prior verdicts caught):** cycle time =
    K·t_draft + t_verify. The "0 VRAM CPU draft" arithmetic fails on its own numbers
    (draft 20–40 tok/s vs target 15–25 → slowdown). Primary arm = **resident ~300MB GPU
    draft, required ≥ ~6× the target's measured dense tok/s standalone** (needs
    CUDA-graph/persistent-kernel dispatch); Rust AVX2 CPU draft is fallback only and must
    demonstrate ≥ 60 tok/s sustained INCLUDING attention/KV on real prompts.
  - **Pre-register the transferable cost model, then validate it:**
    `speedup = E[accepted+1] / (K·c + c_verify)` with c, c_verify MEASURED on-device —
    the experiment validates a model, not a replication of llama.cpp's acceptance number.
  - Masking arm gets the same teeth: any masked configuration must pass its ppl budget
    BEFORE its tok/s counts (the duel must not be structurally rigged toward the lossy lever).
- **Verifier:** greedy spec output token-identical to the frozen reference runtime
  (reference = the SAME pipeline at K=1, with a pre-registered fp-determinism policy so
  reduction-order noise can't cause false kills); acceptance length and union fraction
  logged per suite.
- **Kill criteria / binding fork:** union active fraction > 85% at accepted-K ≥ 3 →
  levers don't compose; ship spec-decode DENSE and record it. Acceptance speedup model
  predicts < 1.2x end-to-end on chat → drop the draft path. **Whichever lever wins gets
  the Phase C/D slot; pre-registration prevents double-counting in all campaign projections.**
- **Prior art note:** arXiv 2512.21911 (sparse verification — overlap >0.8 in most
  layers, so composition has a fighting chance), Polar Sparsity (union erosion),
  SpecExec/Sequoia (weight-fetch amortization principle), ML-SpecQD + llama.cpp #10466
  (this exact draft/target family). The rANS-decode-skip-under-union angle is the only
  unpublished sliver.

---

## 4. PHASE C — Co-evolution: GA only where Phase B proved a non-empty search space

*Each track opens ONLY if its B-gate passed. Fitness always via the §6 arena. The GA's
burden of proof: beat the hand-built literature recipe by a pre-registered margin or freeze.*

### C1. kernel × container co-evolution  [ideas 14+23 merged, scores 6.5/6.5; 2–4 weeks, gated on A4 landing]
- **Mechanism:** EVOLVE-BLOCKs in the fused kernel (tile/launch geometry, ILP segment
  length, LUT placement, renorm scheduling, pair-vs-single decode, smem table layout,
  prefill tiling) + container genes (interleave width, table radix, pair order, chunk
  size, lambda class, hot-layout policy). LLM-as-mutator, committed diffs, ~20-step
  honest budget (not "thousands of evals" — that contradicted the cadence).
  Container re-interleaving computed once per geometry CLASS so repacking can't blow
  the 2-min eval budget.
- **Verdict improvements folded in:**
  - **3-arm matched-budget ablation, pre-registered:** (A) kernel-genes-only at frozen
    container; (B) container-genes-only at frozen kernel; (C) joint. The "co-evolution"
    claim stands ONLY if C strictly Pareto-dominates the union of A and B frontiers —
    otherwise the honest result is "evolution polished a fresh kernel."
  - **Quadruple objective** (storage bytes, RESIDENT bytes, ppl, tok/s) — resident as a
    scored axis closes the loophole where a cache/placement gene buys tok/s with spare VRAM.
  - Correctness via the A4 int32-accumulator gate + canonical epilogue (geometry
    mutations stay legal; numeric cheating stays impossible).
- **Kill criterion:** 20 committed diffs yield < 10% tok/s at iso-quality-iso-bytes →
  downgrade claim to "evolution polished constants," revert to hand engineering.
  Run-to-run tok/s noise > 5% that interleaving can't tame → fix the rig first.
- **Prior art note:** EvoEngineer/KernelFoundry/AlphaEvolve/Sakana (kernel evolution,
  including the reward-hacking cautionary tale — hence the bit-exact gate); KernelFoundry
  evolves kernels over a FIXED format; joint format+kernel search under a measured-throughput
  verifier is the unclaimed sliver. AQLM/QuIP#/QTIP already publish (bits, ppl, tok/s)
  triples — do not claim "first joint frontier"; claim the joint SEARCH.

### C2. skip-decode masked kernel + gating-policy evolution  [ideas 13+22+4 merged, scores 6/5/4; 1–2 weeks after A4; GATED BY B1 PASS + B2 verdict]
- **Mechanism:** on-device pre-kernel computes per-128-block energies of the (AWQ-scaled,
  rotated) activations vs calibrated thresholds → compacted active-chunk list (~100µs
  exclusive scan) → fused kernel iterates active chunks only; random access via A2's
  split-state container at the chunk granularity co-designed in A2. Skipped chunk =
  skipped bytes + ~15 decode ops/weight + MACs — the economy llama.cpp structurally
  cannot copy. CUDA-graph-safe (no host sync).
  **Evolution escalation ladder (per verdict):** grid-searched static per-layer
  thresholds (TEAL/WINA/CATS seeds, no recompile) → per-layer parameter vector in
  constant memory → AST/expression evolution ONLY if parameters plateau. Co-activation
  permutation from B2 absorbed into the container at encode time if it passed (it
  converts unstructured into block-aligned sparsity — likely a bigger lever than any
  evolved gate).
- **Verifier:** (1) threshold=0 → bit-identical (int32-gate) to the dense kernel —
  machinery adds zero drift; (2) decoded-fraction instrumented by in-kernel atomics, not
  inferred; (3) fitness = measured tok/s (policy overhead included) subject to sealed
  held-out ppl ≤ +0.1 and powered reasoning gate (GSM8K ≥ 100 screen / ≥ 500 confirm,
  final never-touched set evaluated once through real free-running decode);
  (4) **absolute tok/s vs same-day Q4_K_M reported alongside relative speedup** —
  a 1.3x over a non-competitive base is not a win.
- **Kill criterion:** no policy ≥ 1.25x measured tok/s at ≤ +0.1 ppl-equivalent on 0.5B
  within budget (or B1's kill already fired) → masking dropped from the flagship, kernel
  hooks removed, counters published as the honest negative ("decode-ops scaled, latency
  didn't" is real sm_61 scheduling information).
- **Prior art note:** arXiv 2511.04477 (fixed-length analog — the delta is variable-length
  decode-skip), TEAL/WINA/CATS/LaRoSA (element-level numbers DO NOT transfer to 128-block;
  that is exactly what B1 measured), DejaVu/ShadowLLM (predictor-based — we need none:
  the mask is exact and pre-GEMV).

### C3. spec-amortize — GEMM(K≤8) verify path  [ideas 7+15 merged; 1–2 weeks after A4; GATED BY B3 draft-speed + cost-model gates]
- **Mechanism:** extend the fused kernel from GEMV to GEMM(K≤8) (same MMVQ batch regime):
  decode each chunk once into registers, dp4a against K Q8_1 activation columns —
  decode paid once, extra MACs nearly free on a decode-bound device. Draft per B3's
  verdict (resident GPU draft primary). If B3 found composition viable: exact union
  mask over the K columns (max of block energies — no prediction, bit-exact), decode
  each substream at most once.
- **Verifier:** greedy spec output token-identical to the frozen masked-dense reference
  (mask thresholds IMMUTABLE once set by the quality gate — evolution may touch K and
  acceptance policy, never thresholds: closes the quality-drift exploit); tok/s on
  pre-registered chat+code+reasoning suites vs THREE controls: mask-only, spec-only,
  **llama.cpp Q4_K_M with --model-draft Qwen2.5-0.5B (the fair fight)**.
- **Kill criterion:** mean accepted < 1.3 on chat, or per-token overhead eats > 30% of
  theoretical amortization → kill resident-draft variant, then CPU-draft, then the track.
  B3's union verdict is binding on whether the masked-verify variant is built at all.
- **Prior art note:** arXiv 2512.21911 published the sparse-verification headline;
  our citable delta is union-gated VARIABLE-LENGTH decode + the DFloat11 weakness
  (entropy-decode amortization) it never paired with speculation.

### C4. Gated micro-spikes (days each, strict entry gates, else skipped)
- **C4a. Transform screen** [idea 21 reduced per verdict]: pre-registered **6-arm
  factorial** — {identity, signed-perm only, H128 champion, block-diag H64, PCA-128
  per layer-class, PCA-128 + signed-perm} — full honest pipeline (real rANS bytes incl.
  metadata, FIXED 128-block masking policy, fixed seeds, one fresh confirmation slice
  used once on the winner). Run AFTER A4 so the sparsity term is calibrated in measured
  tok/s-per-decoded-fraction. **GA escalation only if a non-Hadamard arm achieves ≥ 5pp
  measured block sparsity at ≤ +0.02 b/w and ≤ equal ppl.** Otherwise freeze FWHT forever.
- **C4b. Fisher-weighted ECVQ rate reallocation** [idea 5 decoupled per verdict]:
  stage-1 UNGATED ablation first — K ∈ {2,4} lambda-classes driven by Fisher/E[x²]
  sensitivity (SqueezeLLM-style) in the existing encoder, same honest-bytes equality
  constraint, 2-bit table-select (0.016 b/w) charged. **Kill: < 0.02 ppl or < 0.05 b/w
  gain → kill the whole branch before any gating co-evolution.** If gating later lands,
  add the matched-decode constraint (candidate expected decoded bytes/token ≤ baseline)
  so co-evolved gating can't buy ppl with unmeasured bandwidth.
- **C4c. Static mixed-format hot layout** [idea 6 replaced per verdict]: ONLY if C2's
  gating produced measured Zipf skew in decode frequency (dense decode is uniform —
  without gating there is no skew, full stop). Pack-time per-group format flag: hot
  groups as fixed-width 6-bit packed indices (~1 op/weight, 25% less bandwidth tax and
  33% more coverage/GB than the rejected int8 runtime cache), cold groups rANS. No
  residency bitmap, no LRU, no GA (uniform group cost degenerates the knapsack to a
  frequency sort). Hot-set from training traces; **hit-rate and the ≥ 1.1x end-to-end
  gate measured on held-out prompts.** Resident bytes WITH layout reported separately
  from the storage headline.

---

## 5. PHASE D — Flagship: the pre-registered head-to-head  [ideas 8+16+24 merged; 4–6 weeks realistic]

- **Entry condition (the 2-day kernel kill gate, pre-registered):** before any 7B
  assembly, the fused kernel on ONE 7B FFN matrix (3584×18944) must sustain the rate
  implied by the A1 branch taken (e.g. ≥ ~100 Gweights/s end-to-end for a 15 tok/s dense
  headline; ≥ ~50 with the masking/spec multipliers B/C actually delivered). Below the
  floor → fall back to packed-6.6 b/w or hybrid BEFORE spending weeks on runtime assembly.
- **Composition (assembly, not invention — every piece individually gated upstream):**
  C++ token loop, CUDA-graph captured; dense fused path + (if alive) masked path +
  (if alive) GEMM(K) verify path; embed on CPU, 6-bit dp4a lm_head on GPU; preallocated
  contiguous fp16 KV at capped 8k (448MB); hot layout if C4c landed; MMQ-tile prefill;
  operating point (3.1 vs 3.9 b/w) **fixed in the pre-registration, not post-hoc**;
  7B quality pre-registered as a 5th scale-law prediction BEFORE the run.
- **Verifier protocol (comparison-parity clause, all same card, same day, same driver,
  interleaved A/B):**
  1. **PRIMARY tok/s claim = dense evoq vs llama-bench Q4_K_M tg128 AND pp512** (prefill
     reported — it is where on-the-fly decode is weakest and where our format is cheapest;
     hiding either direction is banned). Masking and speculation reported as separately
     measured multipliers; any spec-enabled headline gives llama.cpp its own draft.
  2. Honest storage bytes (SHA-256 decoded-container accounting, itemized) AND resident
     bytes via **cudaMemGetInfo + DXGI budget deltas applied identically to both runtimes**
     (nvidia-smi is not honest under WDDM) + host RSS/pinned bytes.
  3. Quality: held-out ppl + top-50 logit KL vs fp16 reference + **GSM7K/GSM8K-200**
     (n=25 cannot resolve a 2-point delta) + small lm-eval battery — **measured from the
     same binary at the same operating point as the published tok/s.**
  4. Chat liveness: logged interactive session. Spec runs token-identical to target-greedy.
- **Claim ladder & kill (pre-registered, no silent goalpost moves):** beat IQ3-class on
  all three axes simultaneously = guaranteed claim; bytes+quality vs Q4_K_M with tok/s
  parity = primary claim; Q4_K_M tok/s outright = stretch. tok/s < same-day Q4_K_M →
  retreat to the IQ3-domination claim exactly as measured. Cannot beat IQ3_XXS tok/s at
  equal-or-better bytes+quality → **runtime thesis FAILED; publish the negative with
  decode-ops counters and the (bits, ppl, tok/s) surface** — honest-verifier science
  either way.
- **Prior art note for the writeup:** DFloat11 (resident entropy-coded, dense, loses
  batch-1 speed — our negative control), arXiv 2511.04477 (the delta is variable-length),
  ZipServ (fused, abandoned variable-length), PowerInfer + EIE (ancestors), dtANS
  (fused tANS+SpMVM, dense). Claim precisely: "contextual sparsity gating variable-length
  entropy decode on GPU, enabled by per-substream random access" — and only if C2 survived.

---

## 6. Joint-verifier design — the evoq-arena (built in A3, used by everything)

**Measured axes per candidate (the quadruple + provenance):**
1. **Honest storage bytes** — full decoded container, itemized (payload/states/offsets/
   tables/LUTs/scales/outliers/bundling), SHA-256'd; -inf on round-trip failure
   (evocompress/evaluator.py culture).
2. **Resident bytes** — cudaMemGetInfo deltas + DXGI budget (WDDM-honest), PLUS
   **host RSS + pinned-allocation bytes as a scored axis** (a candidate must not hide
   working set in 16GB system RAM and stream over PCIe).
3. **tok/s** — CUDA-event-timed 64-token greedy generation, warmup discarded,
   median-of-5, **interleaved with a frozen reference candidate** to cancel
   thermal/clock drift (no clock-locking on GeForce Pascal); clocks/driver logged.
4. **Quality battery** — held-out ppl + top-50 logit KL vs fp16 reference + reasoning set.

**Anti-Goodhart (the structural fixes from the verdicts):**
- **Seeded resampling, not frozen sets:** 256k-token ppl pool, 200-item GSM8K pool,
  timing-prompt pool; each candidate's screen draws subsets with seed = SHA-256(container);
  the frozen reference is re-scored on the SAME draw (scores stay comparable across
  thermal conditions). Cell-elite/champion promotion must pass a never-touched
  confirmation slice. **Non-negotiable for mask-policy candidates, whose tok/s is
  input-dependent and trivially overfit to fixed prompts.**
- **Bit-exactness gates:** container SHA round-trip; mask=all ⇒ reference-logit-exact;
  kernel variants ⇒ int32-accumulator-exact + canonical epilogue. Red-team acceptance
  test: the harness must catch a deliberately corrupted container and a deliberately
  cheating kernel before any evolution run is trusted.
- **Two-tier eval:** screen = cached pre-encoded 0.5B container, 16k-token ppl +
  64-token timing (~90s/candidate); promotion = full battery (~10 min). Encode once,
  evolve many.
- **Calibration/eval discipline:** three-way split (fitness / selection-validation /
  final untouched test); thresholds calibrated on data disjoint from all eval pools.
- **Archive:** Pareto-front logger with full provenance over the quadruple (full
  MAP-Elites machinery is over-engineering at ~dozens of meaningful cells on a 6GB card;
  keep the cell keying as metadata).
- **Arena self-gates (from A3):** timing CV ≤ 3%; 0.5B↔7B Spearman ≥ 0.8 on the existing
  4-point ladder, else evolution halts.

---

## 7. First 3 actions (this week)

1. **A0 gate-pack:** run llama-bench Qwen2.5-7B **Q4_K_M + IQ3_XXS + IQ3_S** tg128/pp512
   on this 1060 with full-offload validity check (-ngl 99, watch for silent partial
   offload); cudaMemGetInfo/DXGI + WDDM empty-context VRAM audit; pin CUDA ≤ 12.x toolkit
   AND Pascal-supporting torch wheel; commit `weights/baselines.json`. Every downstream
   kill threshold is derived from these numbers.
2. **A1-stage-0 + A2-day-0 (parallel, ~1 day each):** build stock DietGPU for sm_61 and
   measure generic rANS Gsym/s on the real 0.5B container, cold-streamed, SHA-gated; AND
   run the 30-minute numpy pair-tANS exact-rate script on real per-tensor index
   histograms at L ∈ {2^12, 2^13, 2^14} incl. pruned+escape variants — binding gate:
   pair container ≤ single-symbol + 0.03 b/w all-in.
3. **Launch the Phase B torch oracles on 0.5B (unattended while kernel work proceeds):**
   B1 granularity-stratified 128-block sparsity quality ladder (with AWQ-scaling
   invariance recheck and the co-activation-permutation arm from B2) + B3 adjacent-token
   mask overlap / union-at-realized-K measurement from real evoq-0.5B draft traces.

---

## 8. Consciously rejected routes (recorded so they stay rejected)

- **KV quantization (q8_0/q4_0/KIVI/KVQuant/QuaRot-on-KV):** at ≤ 8k batch-1 chat, KV is
  224–448MB fp16 and not the problem. Re-open ONLY if a 16k+ context goal appears
  (the Hadamard+ECVQ-on-KV spike is pre-scoped in scout 2 with kill criteria).
- **Token eviction (H2O/SnapKV):** permanent-loss, ≥16k-serving-oriented, breaks the
  bit-exact verifier culture. Skip.
- **PagedAttention:** irrelevant at batch 1; preallocate one contiguous capped-n_ctx region.
- **ReLUfication (ProSparse/TurboSparse), Mixture-of-depths/LayerSkip:** require training;
  break bit-exact correspondence with the shipped checkpoint.
- **SparseInfer sign-bit prediction:** needs uncoded sign planes (~1 b/w) — defeats the codec.
- **Deja Vu/PowerInfer predictor MLPs + 85–90% sparsity assumptions:** ReLU-family only;
  borrow the scheduling pattern, never the sparsity numbers.
- **Runtime LRU int8 decoded-group cache (idea 6 original):** hits cost 2.67× bandwidth,
  dense decode frequency is uniform without gating, and a full cache forfeits the
  resident-memory headline. Replaced by C4c static mixed-format layout, itself gated on
  measured gating skew.
- **multians-style bit-serial tANS, generic 256-symbol chunked decoders with per-chunk
  table builds:** wrong specialization; ours is one static compile-time 64-symbol
  (or pruned-pair) table.
- **fp16 arithmetic on cc 610, CUDA 13.x, torch ≥ 2.8 cu12.8+ wheels:** hardware/toolchain
  landmines, banned in A0's manifest.
- **Decode-into-DRAM-scratch architectures:** ZipServ measured them at 0.17–0.34× cuBLAS
  at batch-1 — our standing negative control, banned in code review.

---

## 9. Dependency graph (one screen)

```
A0 baselines ──┬─> derives ALL kill thresholds
               ├─> A1 decode-rate gate (stage 0 → stage 1) ──┬─ <25 ──> hybrid pivot (packed-6-bit + rANS transport)
A2 day-0 rate ─┘                                             ├─ 25–80 → masking/spec MANDATORY branch
A2 container v2 <───────────────────────────────────────────┴─ ≥80 ──> A4 dense fused headline
A3 arena (parallel; self-gates: CV≤3%, Spearman≥0.8)
A4 dense fused kernel (needs A1-pass, A2, A3)

B1 block-sparsity law ──fail──> kill C2 + fusion-novelty claim → C3 takes the slot
B2 permutation oracle ──fail──> kill down-proj gating branch (up/gate row-skip survives)
B3 amortization duel ──> binding fork: compose / spec-dense-only / mask-only

C1 kernel×container co-evo (needs A4; 3-arm ablation decides the claim)
C2 skip-decode + policy (needs A4 + B1-pass + B2 verdict)
C3 spec-amortize (needs A4 + B3 gates)
C4a/b/c micro-spikes (each behind its own entry gate)

D flagship (needs: 2-day single-matrix kill gate + whatever survived B/C)
```

Total realistic wall-clock: ~3–4 months. The cheapest week (A0+A1-stage-0+A2-day-0+B
oracles) can kill or redirect ~80% of the planned engineering — run it first, exactly
in the order of §7.
