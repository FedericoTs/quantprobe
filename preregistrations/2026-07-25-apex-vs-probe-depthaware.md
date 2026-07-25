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
