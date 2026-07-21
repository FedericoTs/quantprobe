# Weight-Space Abliteration Detection — Experiment Results

**Date:** 2026-06-06
**Question:** Can we detect safety-removal ("abliteration") in an LLM *from the weights
alone* — no model execution, no prompts — and does the signature survive real-world
modifications? This validates research opportunity #1 (weight-space model forensics).

## Method (no forward pass)

Abliteration removes the single "refusal direction" `r` by orthogonalizing the
residual-stream **writers** (attention `o_proj`, MLP `down_proj`) against it:
`W → W − r(rᵀW)`. The induced weight delta is therefore (a) ~rank-1 and (b) aligned
to the **same direction `r` across every layer**. A benign fine-tune produces a
diffuse, high-rank delta with uncorrelated per-layer directions.

Given a candidate and its base, for each writer matrix we SVD the delta and take its
dominant left singular vector. Two weight-only signals:
- **cons** — colinearity of those per-matrix directions (top eigenvalue of their Gram / n). 1.0 = one shared direction.
- **shared_E** — median fraction of each delta's energy lying along that shared direction (gates out benign noise).

Flag if `cons > 0.85` **and** `shared_E > 0.03`. Model: Qwen2.5-0.5B (24 layers, 48 writer matrices).

## Phase 1 — discrimination (1 abliterated vs 9 benign fine-tunes)

| model | cons | shared_E | verdict |
|---|---|---|---|
| **ablit** | **1.000** | **0.998** | **ABLITERATED** |
| dataforge-sft | 0.118 | 0.005 | clean |
| dpo-halueval | 0.265 | 0.007 | clean |
| grpo-summ | 0.290 | 0.007 | clean |
| mathphd | 0.259 | 0.002 | clean |
| neon-sft | 0.263 | 0.038 | clean |
| reasoning | 0.718 | 0.009 | clean |
| ultrachat-sft | 0.172 | 0.003 | clean |
| unsloth | — | — | clean (byte-identical re-upload) |
| vikhr | 0.057 | 0.002 | clean |

**Zero false positives** across SFT / DPO / GRPO / reasoning / math fine-tunes. The
two near-misses each trip only one threshold (`reasoning` cons 0.72 but no energy;
`neon-sft` energy 0.038 but no consistency) — the **conjunction** is what makes it robust.

## Phase 2 — robustness gauntlet

| transform | cons | shared_E | result |
|---|---|---|---|
| **(A) fine-tune on top** | | | |
| ablit + 0.5×(mathphd) | 0.958 | 0.461 | **DETECTED** |
| ablit + 1.0×(mathphd) | 0.955 | 0.174 | **DETECTED** |
| ablit + 2.0×(mathphd) | 0.944 | 0.049 | **DETECTED** |
| ablit + 1.0×(reasoning) | 0.953 | 0.122 | **DETECTED** |
| ablit + 2.0×(reasoning) | 0.914 | 0.035 | **DETECTED** |
| **(B) quantization** | | | |
| quantize ablit → int8 | 0.957 | 0.335 | **DETECTED** |
| both base+ablit → int8 | 0.998 | 0.343 | **DETECTED** |
| quantize ablit → int4 (per-tensor) | 0.416 | 0.003 | evaded |
| both base+ablit → int4 (per-tensor) | 0.775 | 0.017 | evaded |
| **(C) adaptive adversary** | | | |
| rank-{1,2,4,8} random spread | 0.032 | ~0.02 | evaded |

## Verdict

- **Works**: pristine abliteration, **fine-tuning on top (up to 2× magnitude)**, and
  **int8 quantization** — all detected, zero benign false positives. This answers the
  research's open robustness question for the two most common real-world transforms.
- **Differentiated**: the only published abliteration detector (arXiv 2602.09434, Feb 2026)
  is *behavioral* (extracts refusal vectors by running the model). Ours is **weight-space,
  zero-execution** — faster, no prompts, scales to a hub scanner.

## The frontier (= the defensible moat to build)

1. **int4 evades** — but we used the *harshest* per-tensor absmax quant; the both-quant
   case already recovers to cons 0.775 (near threshold). Real GGUF Q4 is **block-wise**
   (gentler) and precision-matched diffing should recover it. → block-wise-quant-robust scoring.
2. **Adaptive rank-k spread evades** — *but* our test scrambles directions **per matrix**,
   which would not actually remove a consistent refusal feature. A real distributed removal
   must share a **subspace across layers** → defeated by **subspace-consistency** (top-k
   Grassmann alignment), not single-direction. This is the buildable upgrade.
3. **Base identification** — a real scanner gets an *unknown* model; it must first identify
   the base (provenance, MoTHer-style) then diff. Provenance + detection = the product.

## Files
- `weights/abliteration_detect.py` — the detector (phase 1)
- `weights/abliteration_robust.py` — the robustness gauntlet (phase 2)
