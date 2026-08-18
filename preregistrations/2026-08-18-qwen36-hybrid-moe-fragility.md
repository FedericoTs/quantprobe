# Pre-registration #103: does depth-localized fragility survive a hybrid linear-attention MoE?

**Author:** Federico Sciuca · **Date staked:** 2026-08-18, **BEFORE the source is quantized
and before any band is probed.** **STAKED.**

## The gap this closes

Law 3 says the fragile band is **measurable, not predictable** - three refuted predictors so
far (architecture family, weight kurtosis, effective rank). Every model in the atlas is
either a full-attention dense, a full-attention MoE, or - since prereg #101 - a hybrid
linear-attention **dense** (Qwen3.8-27B, band 51-64, 2.04x, monotone).

Nothing in the atlas is a **hybrid linear-attention MoE**: linear attention in most layers
AND a routed-expert FFN in every layer. That is the architecture the frontier is converging
on, and it is the one case where the two mechanisms we have probed separately meet. Qwen3.6-
35B-A3B is that model, and it is already on disk.

It also gives the first **same-lineage sibling** comparison in the atlas: Qwen3.5-35B-A3B was
probed 2026-07-25 (band 30-39 of 40, back-fragile) on this exact box, this exact eval, from a
Q6_K base. Probing the 3.6 the same way makes the comparison a controlled one.

## What is being measured

Band probe, identical in method to the qwen3.5-35b entry it will sit beside: quantize one
depth band's FFN tensors to Q2_K at a time, leave the rest at base, measure delta perplexity
against the unmodified reference on held-out WikiText-2.

- **Model:** Qwen3.6-35B-A3B, 40 layers, 34.7B total / 2.9B active, **10 of 40 layers cache
  KV** (hybrid linear attention, read from the GGUF header - `weights/data/qwen36_plan_q8.log`)
- **Source:** `unsloth/Qwen3.6-35B-A3B-GGUF` Q8_0 (36,903,140,320 bytes on disk), **converted
  to Q6_K first** so the base quant matches the qwen3.5-35b entry exactly. Deviating on base
  quant would confound the one sibling comparison this probe exists to make.
- **Bands:** 4, over 40 layers -> (0-9)(10-19)(20-29)(30-39)
- **Eval:** WikiText-2 test, held out, same file as every prior probe
- **Box:** GTX 1060 6GB / 16GB DDR4-3000 / i5-7600K, one machine state (C-14), llama.cpp
  b10098

## Staked predictions

- **P-1 (depth survives the hybrid MoE).** The profile is **back-heavy and monotone-ish**:
  the worst band is one of the last two (20-29 or 30-39) and costs **>= 1.3x** the median
  band's delta. This is the same bar prereg #101 staked for the hybrid dense, and it passed
  there at 2.04x.
- **P-2 (it is depth, not attention type).** A **flat** profile - worst band within **15%** of
  the median - would say fragility tracks the evenly-spread full-attention layers rather than
  depth. Qwen3.6 caches KV on 10 of 40 layers; if those are the fragile ones the four bands
  hold roughly equal numbers of them and the curve reads flat. Flat therefore REFUTES P-1 and
  is the single most informative outcome this probe can produce.
- **P-3 (the sibling lands in the same place).** The Qwen3.6 fragile band is the **same band
  index** as Qwen3.5-35B's (30-39, the last of four). If a half-generation bump moves the
  fragile band, then a recipe cannot be reused across a model's own minor versions, and the
  atlas needs a staleness rule it does not currently have.

## KILL RULES

- **If P-1 fails and the profile is flat**, the "depth-aware" framing does not generalise to
  hybrid MoE, and `quantprobe quantize` must say so for this architecture class the same day -
  it may not silently apply a depth recipe to a family where depth was measured not to matter.
- **If P-3 fails** (fragile band moves between 3.5 and 3.6), every atlas entry gets a
  **model-version scope note** the same day: a recipe is valid for the exact version probed,
  not the family. That is a product-facing change and it fires on one measurement.
- **A probe that cannot complete** (disk, time, OOM) is reported as infeasible-on-this-box
  with the reason, exactly as prereg #101's BF16 arm was. It is not retried at a lower
  standard and then quoted as if it were the planned arm.

## Scoring

By the same rule the atlas already uses: bands ranked by delta perplexity, fragile band =
argmax, ratio = worst/median. The probe's own log is the source of record, and the chart that
reports it must parse that log rather than recompute a rival median (the C-30 lesson).

**Wired into:** pending - `quantprobe/recipes/qwen3.6-35b.json` (atlas entry 7), Law 3's
evidence list, `docs/QUANT_QUALITY.md`.
