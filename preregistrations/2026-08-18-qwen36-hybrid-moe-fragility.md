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


---

## SCORED 2026-08-18: depth survives the hybrid MoE, and the sibling holds. 3 of 3.

Probe complete on the box it was staked for. Reference PPL **5.4669**, four bands, one
machine state, CPU perplexity throughout (the Q6_K intermediate of a 35B is 28.5 GB against
6 GB of VRAM, so the probe's built-in GPU->CPU fallback fired exactly as designed).

| band | delta PPL | |
|---|---|---|
| layers 0-9 | 0.0303 | |
| layers 10-19 | 0.1429 | |
| layers 20-29 | 0.1869 | |
| **layers 30-39** | **0.4179** | **fragile** |

Monotone increasing, back-heavy, worst band **2.53x** the median.

| stake | verdict | evidence |
|---|---|---|
| **P-1** depth survives the hybrid MoE | **PASS** | worst band is 30-39 (one of the last two) at 2.53x the median, against a >= 1.3x bar. The curve rises at every step. |
| **P-2** flat would refute it | **did not fire** | a flat profile within 15% would have said fragility tracks the evenly-spread full-attention layers. The spread is 2.53x, not 1.15x. Depth, not attention type. |
| **P-3** the sibling lands in the same place | **PASS** | Qwen3.5-35B was probed 2026-07-25 on this box, this eval, this base quant: fragile band **30-39**. Qwen3.6-35B: fragile band **30-39**. Same index. |

**The sibling comparison, side by side** - the first controlled same-lineage pair in the
atlas:

| band | Qwen3.5-35B (Jul 25) | Qwen3.6-35B (Aug 18) |
|---|---|---|
| 0-9 | 0.0909 | 0.0303 |
| 10-19 | 0.1531 | 0.1429 |
| 20-29 | 0.1579 | 0.1869 |
| 30-39 | **0.4492** | **0.4179** |

Two independently probed models, half a generation apart, land on the same fragile band with
worst-band deltas within 7% of each other. **No kill rule fires:** the profile is not flat, so
`quantprobe quantize` keeps the depth framing for this class; the band did not move, so no
model-version scope note is owed to the atlas.

**What this adds to Law 3.** Depth-localized fragility has now been measured across four
structurally different families: full-attention dense (Mistral-7B front-fragile, Qwen2.5-7B
back), full-attention MoE (Qwen3-30B), hybrid linear-attention dense (Qwen3.8-27B, prereg
#101), and now hybrid linear-attention **MoE**. The mechanism that would have explained
fragility by attention type predicts a flat curve here and got a 2.53x spread. Law 3's
negative claim - measurable, not predictable - keeps its three refuted predictors, and its
positive claim now spans the architecture the frontier is actually shipping.

Chain of custody: staked 2026-08-18 before the source was quantized (three predictions, two
product-facing kill rules) -> U-52 registered -> probed 6 h on one machine state -> scored
against the staked bars. Atlas entry 7 is `quantprobe/recipes/qwen3.6-35b.json`.

**Protocol note, recorded because the staked text says otherwise.** The stake said the Q8_0
source would be "converted to Q6_K first". It was not converted as a separate step: the
attempt failed, because llama.cpp refuses a Q8_0 -> Q6_K requantize unless the caller passes
`--allow-requantize`, and it turned out to be unnecessary anyway. `quantprobe probe` passes
that flag itself and builds its own Q6_K reference and Q6_K band files from whatever source
it is handed, so the base quant is Q6_K **by construction** and the protocol parity the stake
was protecting is exactly what the run delivered. The outcome matches the stake; the route to
it did not, and that is worth a line so nobody reproducing this wonders why there is no
standalone Q6_K file on disk.
