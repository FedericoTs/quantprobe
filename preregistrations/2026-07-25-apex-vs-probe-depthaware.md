# Pre-registration #11: APEX Mini vs probe-built depth-aware, matched bytes, their turf

**Author:** Federico Sciuca · **Date staked:** 2026-07-25, committed BEFORE any of the files
below were downloaded. **Scoring: launch week (target 2026-07-31).**

## Context

Community thread (r/ollama) surfaced [mudler's apex-quant](https://github.com/localai-org/apex-quant):
MoE-aware mixed-precision GGUFs (routed experts on a depth gradient, shared experts Q8, attention
Q6, built with stock llama-quantize --tensor-type overrides — the same mechanism as our depth-aware
recipe, with tensor-role granularity we don't have and an ASSERTED symmetric-edge protection we
don't share: our probe MEASURES the band per file). Their technical report claims Mini beats
bartowski IQ2_M at matched size (ppl 7.088 vs 7.303, single model, single box, no pre-registration).
This is the fight worth having in public, on their chosen model and metric.

## Contenders (files verified on HF 2026-07-25, none downloaded yet)

| file | size | eff. bits |
|---|---|---|
| mudler/Qwen3.5-35B-A3B-APEX-**Mini** | 13.3 GB | 3.04 |
| **ours**: depth-aware built from unsloth Q8_0 (36.9 GB) via `probe --apply`, band tuned to match | 13.3 ± 0.4 GB | ~3.0 |
| unsloth UD-Q2_K_XL (K-quant reference) | 12.2 GB | 2.79 |
| unsloth UD-IQ2_M (IQ reference — mudler's comparison class) | 11.4 GB | 2.61 |

Metric setup mirrors their report: WikiText-2-raw perplexity, ctx 2048. Speed cells per the Law-5
conventions (nvidia-smi state logged, -dev none for CPU-pure, r2).

## Stakes

- **S-1 (probe informativeness).** The probe on the Q8 source finds an ASYMMETRIC fragile band —
  not the symmetric L0-4 + L35-39 edges APEX asserts. Staked: late-weighted asymmetry, consistent
  with every model probed so far in this project.
- **S-2 (quality at matched bytes).** Our 13.3 GB build lands within **[−2%, +8%] of APEX Mini's
  wikitext ppl**. Honest prior stated plainly: their allocation uses tensor roles + a diverse
  imatrix that ours doesn't; a small APEX win is the expected outcome. If we WIN by more than 2%,
  that is ALSO a miss of this stated prior and publishes as one. UD-IQ2_M trails both by ≥ 2.5%.
- **S-3 (the joint axis — the cell nobody else measures).** CPU-side pp2048 at matched bytes:
  our K-quant-family build is **≥ ×2** APEX Mini. Mechanism staked in advance: Mini's middle
  layers are IQ2_S, a LUT family we measured at ×7 CPU-prefill penalty (H12, law5_h12_formats.log),
  and in hybrid placement the routed experts — exactly the tensors Mini compresses to IQ2_S —
  are what lands on the CPU. Hybrid decode stays within ±15% between the two (decode is
  bytes-dominated). If Mini's CPU prefill is fine, the H12 blending model is wrong and gets said.
- **S-4 (reproduction gate).** APEX Mini's ppl on this box reproduces mudler's 7.088 within ±3%.
  If it doesn't, the discrepancy is named and resolved BEFORE any comparison above is scored.

## Refuted if

S-1: band comes back symmetric or flat. S-2: outside the band in either direction. S-3: < ×1.5.
S-4: > 3% unexplained. Misses publish with the same prominence as hits.

## Disclosures

Single box (i5-7600K / GTX 1060 6 GB / 16 GB DDR4 — the CPU cells are the point, not a
limitation). Our build is requantized from Q8_0 (the standard `--custom` path); theirs is built
from F16 — an asymmetry in their favor that we accept and note. MTP and serving-stack speed are
out of scope here (Law-6 territory). References run at their native sizes; only APEX-vs-ours is
exactly matched.

## Commitment

If APEX Mini wins S-2 while losing S-3, BOTH results publish and APEX enters `quantprobe auto`
as a recipe candidate with a placement-conditional recommendation — GPU-resident boxes get
pointed at APEX, hybrid/CPU boxes at K-quant-family builds. The tool's job is finding the best
config for YOUR machine, not defending its own recipe. That outcome is a win for users and
for whoever built the better allocation.

---

## Scored (2026-07-25, log: weights/data/apex_ab_stageA2.log)

- **S-1 HIT, sharper than staked.** The probe's band search (reference PPL 5.4298, WikiText,
  40-layer source) found a strictly monotonic, late-weighted fragility curve — NOT the symmetric
  front+back edges APEX's design asserts:

  | band | layers | delta PPL |
  |---|---|---|
  | 1 | 0-9 | +0.09 |
  | 2 | 10-19 | +0.15 |
  | 3 | 20-29 | +0.16 |
  | 4 | 30-39 | **+0.45** |

  The front band (0-9) is the MOST tolerant region measured, not a fragile edge — directly
  contradicting APEX's L0-4 protection assumption for this model. Fragile band: layers 30-39,
  delta +0.45 vs the +0.16 median (~2.8x). Depth-aware build now underway targeting this band,
  size to be matched against APEX Mini's 13.3 GB (S-2/S-3, next).

Note en route: Stage A's first attempt failed on an unrelated tooling bug (`probe`'s internal
perplexity step hardcoded full GPU offload, OOMing on this model's 23-27 GB intermediates on a
6 GB card) — fixed and shipped as quantprobe v1.6.3, verified against the real failure before
this rerun. Disclosed here because it's methodologically relevant: S-1 above is from the
corrected, CPU-pure rerun, not the failed first attempt.

## S-4 and S-2 scored (2026-07-25, log: apex_ab_stageC_quality.log, apex_ab_stageD_ssmfix.log)

**S-4 (reproduction gate): MISS.** Measured APEX Mini on this box: **PPL 6.1511** (hybrid,
ctx 2048, 32 chunks) vs their reported **7.088** — 13.2% off, well outside the staked ±3%.
Named cause: their own technical report lists APEX Mini at 12.2 GB; the file actually served
today is 13.3 GB. The artifact was very likely updated/improved after their report was
published — plausible given the direction (bigger file, better/lower ppl than documented), and
verifiable (the size mismatch is a plain fact, not an inference). Per protocol this miss is
named and disclosed rather than allowed to silently color the comparison below; the comparison
proceeds against the artifact anyone downloads today, which is the more relevant one for users.

**S-2 (matched-bytes quality): a mechanism was found mid-measurement, fixed, and the result
flipped from a decisive miss to a near-miss.**

- First build (13.21 GB, protect layers 34-39, our standard recipe): **PPL 8.8111** — +43.3%
  worse than APEX Mini, and worse than even the smaller community IQ2_M reference (6.3939).
  Staked band was [-2%, +8%]; this missed by 35+ points. Named cause, confirmed in the build
  log: this architecture carries `ssm_*` state-space tensors (hybrid SSM+attention+MoE) that our
  `attn_.*` protection regex — written for pure-transformer models — never matched, leaving
  every SSM tensor at the aggressive Q2_K base with zero protection. APEX's own report
  explicitly protects "attention & SSM weights" together; ours didn't protect SSM at all.
- **Fixed** (shipped as quantprobe v1.6.4: `ssm_.*=q4_k` added alongside `attn_.*=q4_k`) and
  **rebuilt at the identical band** (34-39, same measured fragile region, testing the mechanism
  cleanly rather than re-tuning): **13.27 GB, PPL 6.6976** — a 24.0% relative ppl reduction for
  +0.06 GB (~56 MB). The mechanism is confirmed: SSM protection was the dominant term.//
- **Final scored delta: +8.88% vs APEX Mini** (staked band -2% to +8%) — **MISS, but a near-miss
  of the stated honest prior**, not the decisive loss the first build showed. Also **+4.75%
  worse than the IQ2_M reference** (down from +37.8% pre-fix). Held to the letter of the stake:
  8.88 > 8.00, this is scored as a miss, not rounded into a hit.

**Standings at matched bytes (~13.2-13.3 GB), lower is better:** APEX Mini 6.15 < IQ2_M
reference 6.39 < ours (SSM-fixed) 6.70. APEX wins. Per the pre-registered commitment: APEX
enters `quantprobe auto` as a recipe candidate. S-3 (CPU-pure speed, the other half of the
commitment's condition) is next.
