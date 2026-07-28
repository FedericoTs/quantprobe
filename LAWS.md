# The Four Placement Laws

*Every law below states its claim, the measurement that established it, and a falsifiable prediction
anyone can test. All numbers from one 2016 desktop (GTX 1060 6 GB · 16 GB DDR4 · SATA). Full logs in-tree.*

---

## Law 1 — Rotation is rank-conditional
**Incoherence rotation — the foundation of modern quantization codecs (QuIP#, QTIP, QuaRot) — helps
full-rank tensors and destroys low-rank bottlenecks.**

- *The measurement:* the same orthogonal rotation costs **+0.006 ppl** on a full-rank MLP
  (eff. rank 1168) and **+1623 ppl** on the low-rank KV-latent (eff. rank 394) — a **~270,000×** swing
  on effective rank alone. Every gauge tried on the bottleneck (Hadamard, SVD, diagonal) made it worse;
  only native-basis precision repairs it.
- *The prediction:* any architecture that manufactures low-rank structure (MLA latents, LoRA merges,
  GQA projections, SSM states) will be damaged, not helped, by incoherence processing at low bits.

## Law 2 — Trained networks are dense everywhere
**Load-balanced training fills every axis a data-free method can reach: there is no free lunch left in
the weights, the routing, or the activations.**

- *The measurement:* routed experts sit exactly at the Gaussian rate-distortion floor
  (rel-MSE 0.069 = D(R=2), identical across all 64); 1-bit collapses (+253 ppl) under every codec;
  routing is flat (a token needs ~5.3 of its top-6 experts); activations are diffuse (72–84% of neurons
  carry 90% of energy); and expert usage is **domain-flat** — prose and code use *identical* expert sets
  (Jaccard 1.00). Thirty candidate levers, twenty-nine measured dead.
- *The prediction:* 2-bit is the data-free floor for any load-balance-trained model; task-trimming
  experts and semantic "brain-region" paging will fail on any of them.

### Law 2 — scope boundary, measured (2026-07-25)

The ~2-bit floor is a **post-training, data-free** claim. Its boundary now has a number: Bonsai-27B
(qwen3.5-family hybrid, **natively trained at ~1.13 bits**, 3.8 GB total) scores **10.87 ± 0.34** on
our WikiText-2 gate — statistical parity with our best post-training depth-aware 2.5-bit Qwen3-30B
(11.08) at 2.2× fewer bits per weight. Training-time quantization sidesteps the PTQ floor entirely;
Law 2 never bounded it and does not now. Two systems-side measurements from the same session, both
error-barred: (1) the Pascal low-bit decode collapse (gl = 0.04) is **dequant-format-dependent, not
bit-width-dependent** — Q1_0's trivial dequant runs the 27B all-in-VRAM at 11.94 ± 0.04 tok/s on the
GTX 1060 where the gl model predicted ~1.8 — **superseded 2026-07-26 by pre-registration #16, which
went further: no low-bit decode collapse for the three formats it measured.** A matched triple
(same 7B, same card, all in VRAM) decodes 20.03 / 19.17 / 18.11 at Q4_K_M / Q2_K / IQ3_XS — a 10%
band across 2.8–4.5 bits and across those formats. The collapse is real but lives entirely in
**prefill**, where the same IQ3_XS pays **6.8×**. Dequantization is compute; prefill is
compute-bound and decode is not. Re-scoped 2026-07-28: wider format ladders then found decode
collapse that is format-dependent, not bit-dependent — Q2_K all-in-VRAM measures 45% below the
byte model and slower in absolute tok/s than Q4_0 while 32% smaller (V-17, prereg #53), and IQ
formats decode 2.7× slower than K-quants on CPU tiers (V-11, prereg #31). Decode still ignores
bit-width; it does not ignore format — the unpack cost sets the tax (L-15).
The gate has been removed from the planner and `gl` no longer touches decode; (2) linear-attention hybrids (48 gated-delta layers)
carry a CPU compute tax — measured η ≈ 0.30 vs the dense-GQA 0.62 class, and my same-day staked
CPU estimate missed by 2×, published here per house rules. Raw logs:
`weights/data/bonsai_bench.log`, `weights/data/bonsai_ppl.log`.

## Law 3 — Fragility is measurable, not predictable
**Where a model breaks at low bits is model-specific: no configuration flag, architecture family, or
weight statistic predicts it — but a 30-minute functional probe measures it exactly.**

- *The measurement:* the depth-fragility atlas — Gemma-4-12B **late**-fragile (~4×), Qwen2.5-7B late
  (~2–3×), Qwen3-30B-MoE late (~2.3×), **Mistral-7B early-fragile (~25×)** despite being Qwen's
  architectural near-twin. Weight kurtosis points the *wrong way* on Gemma. Placement by the probe:
  byte-identical GGUF files **2.25 ppl apart** (10.02 vs 12.27); the depth-aware recipe halves Gemma's
  2-bit gap (1.91× → 1.45×) and, data-free, edges an imatrix-calibrated community quant at 30B scale.
- *The prediction:* for any new model, the band probe (`quant_probe.py`) beats every static allocation
  rule; guessing the fragile end without probing risks forfeiting up to a 25× fragility differential.

### Law 3 refinement — structural and statistical allocation are orthogonal (2026-07-26)

*Prompted by testing against [apex-quant](https://github.com/localai-org/apex-quant) (mudler),
whose MoE-aware design supplied the always-active/heavy-tailed argument, and surfaced by
u/MoneroApe.*

Measured in [pre-registration #12](preregistrations/2026-07-25-recipe-upgrade-shexp-imatrix.md) on
Qwen3.5-35B-A3B at ~3 bits, each lever isolated against an identical band:

| lever | kind | ppl effect |
|---|---|---|
| importance-matrix calibration | statistical (*which weights* carry signal) | **−8.5%** |
| always-active tensor protection | structural (*which roles* are load-bearing) | **−3.2%** |
| both | | **−9.1%** (not additive) |

**Law 3 stands unchanged and is reinforced** — *where* a model is fragile still has to be measured,
and no calibration substitutes for that. What is new: fragility-by-depth (structural) and
importance-by-activation (statistical) are **different axes that stack with diminishing returns**.
Calibration already routes precision toward the heavy-tailed always-active tensors that role rules
protect by name, which is why 8.5 + 3.2 yields 9.1 rather than 11.7. Practical consequence: after a
good imatrix, additional hand-written role rules buy little — the remaining headroom is in
*structural* protection-by-default, not more named exceptions.

Two scope limits, stated plainly: this is one model, and the calibration was **in-domain**
(wikitext train → wikitext test), the most favourable case for a wikitext metric. The size of the
lever is established; its generality across corpora and metrics is not.

**Quality-curve caveat.** `qual_of(bits)` in the planner maps bits → quality cost on the assumption
of a *particular* recipe quality. These results show ~9% spread at fixed bits depending on recipe,
so treat printed quality costs as recipe-conditional. The curve is not re-fitted on one model.

## Law 4 — The tiered decode law
**Decode speed is a placement identity: `tok/s = η(tier) × bandwidth ÷ active-bytes-per-token`, with
the utilization constant η collapsing per memory tier — amended 2026-07-28 (L-15): within a tier,
η is a function of the format's unpack instruction cost, not the tier alone.**

- *The measurement:* η = 0.56 (VRAM, format-averaged — the per-format band is ~0.31–0.62, see the
  amendment below) · 0.29–0.68 (RAM: dense ≈0.65, MoE ≈0.35 — the scatter penalty)
  · 0.88–1.0 (disk), across 7B→744B **including colibri's independently published tiers** (his 0.48 and
  0.88 sit inside our bands). Pre-registered hits: a 110B model streamed from SATA at **0.19 tok/s**
  (predicted 0.2–0.3); a RAM overclock (2133→3000) delivered **×1.52** on dense (predicted ×1.41+);
  and when bandwidth rose, the 30B's bottleneck *migrated* to RAM capacity — exactly as a law-governed
  system should behave.
- *Corollaries, each measured:* on poor-decode GPUs, experts belong on the CPU (+54%, one flag);
  batch-union returns scaling, but less than first quoted — the early 4.5× at batch 8 was superseded
  by pre-registration #26: aggregate decode saturates at roughly 2× by about 4 slots, near-identically
  across placements and architectures (C-06/V-09); **speculative decoding is antagonistic
  to MoE sparsity** on bandwidth-bound tiers (verify-batches union ~40 experts vs 8 — measured 2.3×
  *slower* with a draft); and the MoE scatter penalty is NOT a memory-system property — corrected
  2026-07-27 (D-05, preregs #32/#33): shuffled 2MB expert-slab reads measure 24.56 GB/s vs 23.88
  sequential (+2.9%, noise — the memory system is indifferent to the scatter), and the profiled
  deficit is ~32 ms/token of fixed per-op machinery in the CPU graph executor.
  **Strengthened 2026-07-26 (Law 6 arms S-a/S-b/S-e):** THREE independent speculation mechanisms
  now collapse on offloaded MoE — draft-model (2.3x slower), MTP (0.76x), and ngram prompt-lookup
  (+3% where the same mode gives a dense model **+110%**). The expert-union tax is a property of
  speculative decoding on offloaded MoE, not a quirk of any one mechanism. Corollary for users,
  re-scoped 2026-07-28: with experts in RAM, draft-model and MTP speculation will not pay, and
  nothing accelerates novel output (D-09/D-10). The measured exception: free n-gram drafting pays
  2.4–5× on copy-regime output — edits, refactors, quoting — on the same experts-in-RAM split
  (preregs #28/#36/#37, V-12); the +3% above was the untuned default on novel text, not the
  mechanism's ceiling.
- *The prediction:* measure any machine's tier bandwidths and any model's active bytes, and this
  equation prices its decode speed before you download a single weight.
- *Scope limit (2026-07-26), stated before it was measured:* the law prices **one token per
  weight read**. Multi-token prediction and speculative decoding break that assumption by design —
  several accepted tokens per read — so the law reads LOW for them by the acceptance multiplier
  (a community datapoint: ~1.8x on Qwen3.6-35B-A3B with MTP). This is a missing factor, not a
  refutation: the bytes-and-bandwidth term is unchanged and the multiplier sits on top. Measuring
  it is Law 6's job. **Measured 2026-07-26 (arm S-e): the sign FLIPS with placement.** Same
  model, same file, MTP toggled on the same server. Full 2x2 across architecture and tier:
  dense GPU-resident **1.17x**, dense CPU **1.046x** (+-0.01, reproducible), MoE experts-in-RAM
  **0.76x**, MoE spilling VRAM **1.94x**. Architecture decides the SIGN on a slow tier: dense
  gains where MoE loses, consistent with a dense verify batch re-reading the same weights while
  a MoE batch unions extra experts. The 1.94x is MTP rescuing a model from paging, not a
  general property - no other cell came close. MTP trades more work per forward pass for fewer
  passes, so it pays only where an extra *pass* costs more than an extra *batch*. MTP does
  NOT escape the expert-union tax that makes draft speculation 2.3x slower on MoE. There is
  therefore no single multiplier to add to this law - a joint term is required, and the
  compute-bound regime (model comfortably GPU-resident) remains untested on this hardware.
- *Scope confirmation (2026-07-26):* pre-registration #12 changed **which** weights carry precision
  (importance-matrix calibration) at identical format and identical file size, and measured CPU
  prefill **unchanged** (44.80 vs 43.67 tok/s, inside run-to-run noise — staked at ±5% and hit).
  Law 4 is a function of bytes and bandwidth, not of weight *content*. Confirmed, not assumed.

### Law 4 amendment (2026-07-28) — η is a function of format, not tier alone (L-15/L-16)

The per-tier η values above are format-averaged. Measured same tier, same card, same model, same
session (preregs #52/#53): Q4_0 η **0.619** · Q4_K_M **0.553** · Q2_K **0.340** — the all-in-VRAM
band runs ~0.31–0.62 by format. The mechanism was isolated at the metal with a standalone CUDA
harness (zero llama.cpp): a matvec with no unpacking runs at 95% of the streaming ceiling, the same
bytes with a naive unpack at 42%, and dp4a recovers ~80% — on an ALU-weak GPU the format's unpack
instruction cost sets decode speed as much as the byte count does (L-15). The K-quant deficit has a
named cause: **metadata application density** (L-16) — Q2_K's definition forces a scale+min chain
every 4 bytes at 2 bits, 4× Q4_K's density per byte; a confirmation arm with identical loads and
identical dp4a count gained +23% (preregs #56/#57), and the full decomposition lands within 9–12%
of measurement. Shipped consequence: prefer Q4_0 over Q4_K_M for all-in-VRAM decode on pre-Ampere
(+19% end-to-end measured, 26.87 vs 22.72 tok/s — speed-only, Q4_K_M is higher quality per byte),
and the planner prices decode per-format (`spec.FORMAT_EBW`). Scope: one Pascal card (cc 6.1); on
Ampere+ the ALU-to-bandwidth ratio flips and the ranking may invert — unverified, replication asked.

### Law 4, general form (v1.3–v1.4, 2026-07-24) — a restatement, not a revision

Every formulation below NESTS: the v1.0 statement is the single-dominant-tier special case, v2 adds
the KV term, and the general form covers any tier set. **No measured anchor moved at any step** —
the anchor suite re-proves all of them on every commit.

**General form.** For a machine described as memory tiers *i* (aggregated devices count as one tier:
multi-GPU bandwidth sums x ~0.85 tensor-parallel efficiency [est], striped disks x ~0.75 [est,
from the RAID-0 eta 0.66 datapoint]), and a placement assigning bytes to tiers:

**tok/s = [ Σ_i  bytes-read-per-token from tier i ÷ (η_i · BW_i) ]⁻¹**

where bytes-per-token = always-active weights + routed-expert reads (hit-rate = resident fraction,
by routing flatness, Law 2) + ctx · kv-bytes/pos on KV's tier. Fit is checked per tier with KV
counted. The v1.0 form is recovered when one tier dominates the sum.

**Corollary (tier boundaries).** Speed is a step function of placement, so the marginal value of a
gigabyte is ~zero mid-tier and enormous at a boundary (measured: a one-quant-step shave across our
RAM boundary is worth x4-6). All size levers should be priced by boundary distance.

**Corollary (lever gates).** Lever validity is configuration-conditional — Law 1's shape recurring
at the systems level. Measured example, corrected 2026-07-27 (V-08, prereg #25): quantized K-cache
costs -83% at 16k depth with flash attention OFF (per-token dequant tax); with -fa 1 the SAME
Pascal card measures q8_0 KV as a depth win — +37% vs f16 at d16384, 3.04x vs KV-eviction, at a
~6% cost at short context. The gate is a property of the FA configuration, not the hardware class —
we had conditioned it on the wrong variable. Optimizers over the law must carry measured gates,
not assume levers are universal.

**Corollary (levers share one budget, and the budget is fungible).** Measured 2026-07-27 on
Qwen3-30B-A3B, one session. Weights, KV cache and the compute buffer draw on the same VRAM, so the
placement dimensions cannot be optimised one at a time:

- `-ub 2048` is worth **+73% prefill** where weights are host-resident and **−39%** where they are
  not (pre-registration #19) — a double dissociation, so the mechanism is host-to-device
  amortisation, not generic batching.
- Adding that lever **inverted which placement is fastest at prefill**: the expert split wins at
  the default ubatch (279 vs 200) and loses at `-ub 2048` (162 vs 350), because the split spends
  the VRAM the compute buffer needs (#20).
- Evicting KV with `-nkvo` recovers **2.42×** of prefill on the split, and **nothing** on
  all-experts-to-CPU (345 → 336) — fungibility is placement-specific: only a starved
  configuration can spend the refund (#21).

Consequence — **refuted 2026-07-27 (L-07, pre-registration #25).** This section originally
concluded there is no single best placement, only a Pareto frontier selected by the
prompt-to-generation ratio, with the wrong point costing up to 2.25×. Re-measured with every cell
in ONE session, ONE configuration (split + `-ub 1024` + KV-in-VRAM) wins at every ratio. The 2.25×
shrank to nothing as three of our own measurement errors were corrected — the headline cell had
been measured past the compute-buffer cliff — so the frontier was an artefact of the sweep, not a
property of the machine. The lever measurements above stand; the frontier conclusion drawn from
them does not.

**Dead end (measured, 2026-07-27): expert-count reduction.** Halving MoE top-k (8 → 4) behaves
exactly as the byte model predicts — active parameters 3.3B → 2.25B, decode **20.32 → 27.13
tok/s (×1.335)**, within 2% of the arithmetic prediction. It still should not be used: measured
WikiText-2 perplexity rises **9.24 → 11.14 (+20.6%)**, while simply quantizing 2.95 → 2.0 bits on
the same placement buys **more** speed (×1.424) for **four times less** quality (×1.048). The
lever is real, correctly modelled, and strictly dominated (pre-registration #18). Recorded so it
is not re-derived.

### Law 4 v2 — the context term (v1.1, 2026-07-23)

The formulation above is the **short-context law**. u/RogerAI--fyi (Reddit) correctly observed it
omits per-token KV reads: every generated token re-reads the entire KV cache, so at depth the byte
budget gains a second term —

**`tok/s = η(tier) × BW ÷ (active-bytes + ctx × kv-bytes/pos ÷ η_kv-adjusted)`**, where the KV term is
served by **whichever tier KV lives on** (VRAM in hybrid placements, RAM on CPU-only boxes — placement
matters for context too, i.e. the law recurses).

- *The measurement (same 2016 box, warm-up-controlled):* tg32 clean baseline **20.02 ± 0.02** →
  at depth 16384 **16.12 ± 0.06** (**−19.5%**). Pre-registered −8…−15% — scored honestly as a
  **near-miss**: the pure bandwidth term (16384 × 98 KB/pos off the 192 GB/s tier) explains ~70% of
  the slope; the residual is depth-dependent attention *compute* on Pascal. Shipped calibration:
  η_kv ≈ 0.70 (single point — falsify or refine it: `quantprobe bench --depth N --contribute`).
- *Architecture matters:* kv-bytes/pos is per-model (Qwen3-30B 98 KB exact; MLA models ~10× smaller —
  DeepSeek-V2-Lite 31 KB; SWA models slope on global layers only [est]). KV also **consumes capacity**
  on its tier: at 16k the 30B no longer fits a 16 GB box as pure-CPU — the fit-checker knows.
- *The prediction:* CPU-only boxes (KV in RAM at ~45 GB/s) degrade **steeper** with context than
  hybrid placements (KV in VRAM at 192): the law says **−29% at 16k** for 30B-A3B Q4 pure-CPU on
  DDR4-45 (8.0 → 5.7 tok/s) vs −19.5% for the hybrid — pre-registered, unmeasured as of v1.1;
  band ±10 points. A `bench --depth 16384` on any CPU-only box settles it.

---

**The umbrella claim:** at low bits on commodity hardware, *placement beats budget* — which layers get
the bits (Law 3), which tier serves the bytes (Law 4), where rotation is applied (Law 1) — because the
budget itself has no slack left to give (Law 2).
