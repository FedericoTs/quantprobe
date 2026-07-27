# External datapoint: GLM-5.2 (753.3B) on 4× RTX 6000, 2026-07-27

Reported publicly on X, forwarded by Federico. Not our measurement; recorded as an external
observation with its provenance, and scored against our prediction as-is.

| prompt | prefill tok/s | decode tok/s | TTFT |
|---|---|---|---|
| 8K | 1,531.25 | 91.14 | 5.35 s |
| 32K | 1,897.41 | 96.51 | 17.27 s |
| 64K | 2,420.73 | 102.18 | 27.07 s |
| 128K | 2,425.10 | 96.30 | 54.05 s |
| 262K | 2,482.96 | 103.19 | 105.58 s |

Reported configuration: 4× RTX 6000, vision enabled, 1M-token context target. Effective bits
NOT stated — we assume ~3-bit because that is what fits 384 GB of VRAM for a 753.3B model.

## Scored against `quantprobe plan --model glm-744b --bits 3.0`

| context | ours | measured | error |
|---|---|---|---|
| 8K | 166.0 | 91.14 | **+82%** |
| 64K | 116.9 | 102.18 | +14% |
| 262K | **12.6** | 103.19 | **−88%** — we predict it cannot fit and falls to disk-streaming |

Three separate failures, and the third is the serious one.

## What it exposes: our `kvp` for the GLM family is wrong by more than an order of magnitude

`MODELS["glm-744b"]` carries `kvp=188416` bytes/position, marked `[est]`. Two independent lines
say that is far too large:

1. **Capacity.** For 3-bit weights (305 GB) to fit 384 GB VRAM at 262,144 positions, the KV cache
   must be under ~40 GB, i.e. `kvp < 152,600`. Ours exceeds that, which is why we predict a
   configuration that demonstrably runs cannot fit at all.
2. **The flat decode curve, which is stronger.** Their decode is *unchanged* across 32× of
   context (91 → 103). At `kvp=188416`, the KV read at 262K would be **49 GB/token — 3× the
   15.5 GB of weights** — and decode would collapse. For the context term to be invisible, KV
   must be ≤5–10% of weight bytes, i.e. `kvp ≈ 3,000–6,000`: **30–60× smaller than we assume.**

That is the signature of **MLA / compressed-latent KV**, which we already model for
`deepseek-16b` at ~1,152 B/layer. The GLM entry never got the same treatment and was estimated as
dense-GQA.

## Why this is not fixed here

The correction needs GLM-5.2's actual attention architecture, not a back-solved constant. Fitting
`kvp` to make one external datapoint land would be exactly the error this project has already
paid for twice. Recorded as a contradiction against an `[est]` field, to be closed by reading the
architecture.

## Also worth separating before trusting the 8K/64K rows

Their decode *rises* with context (91 → 103), which no bandwidth model predicts. Either KV
bandwidth is genuinely negligible at ~7 TB/s aggregate, or their harness reports a decode metric
that excludes the KV re-read. Until that is known, the +82% at 8K should not be read as a clean
efficiency miss — the two harnesses may not be measuring the same quantity.

Contradicts: `quantprobe/plan.py:MODELS["glm-744b"]["kvp"]` (marked `[est]`).
