# The media portfolio — every chart, its hook, and its committed data

Rule of the house applies to content: **a chart is a claim** — every asset renders from
committed measurements (cite-or-refuse), misses stay on the chart, and nothing ships without
the render-review loop (headless render -> look -> fix -> deliver).

## Data-ready now (committed measurements, CPU-only to build)

1. **The stuck boost state** — *"your GPU silently lost 18% and only a reboot fixes it."*
   Preregs #60/#61: SM clocks locked ~18% low at cool temps after hours of churn; plus the
   ClockSampler sustained-vs-idle data and the gpu-state lines logged around every grid row
   (139 MHz idle vs 1885 MHz sustained in one night's log). Format: before/after bars + a
   clock strip across the night. The most relatable finding we own - every gamer has felt it.
2. **Same gigabytes, different speed** — the format ladder. FORMAT_EBW measured: Q4_0 119.1
   GB/s vs Q2_K 65.4 vs IQ2_XXS 46.0 - a 2.6x speed spread at comparable size - plus the
   IQ4_NL twist (117: "IQ by name, Q4-class by kernel"). Preregs #31/#52/#53/#70/#77.
3. **Prediction vs reality** — the trust chart. Scatter of predicted vs measured: the
   14-model ladder (8.4% median), the externals (E-08 Blackwell +2..7.6%, E-13 AMD +0.1%,
   Kimi retrodiction 0.3%, airllm's 30x spread bracketed), floor band shaded, the misses
   (U-06 -67% disk arm) labeled at the same size as the hits. Nobody else can draw this.
4. **The size-vs-speed map** — every measured model as a bubble: 0.5B to 744B on one machine,
   x = file GB (log), y = tok/s (log), colored by memory tier. The strong-contrast asset:
   "a 744B answered on 16 GB RAM - at 0.13 tok/s; the honest chart shows you both halves."
   Sources: ladder logs + docs/MATRIX.md rows.
5. **The speculation regime map** — the 2x3 grid (dense/MoE x all-VRAM/split x external
   draft/MTP): +33.5%, +11%, 0.74x, +11.4%, and the K-cliff - preregs #67-#71. The "it
   depends, and we mapped every cell" flex.
6. **The KV depth tax** — decode vs context for 3 models (7B loses 66% by 16K; the q8-KV
   +37% rescue at depth; the 3.04x -nkvo penalty). Mostly committed; completed by the queued
   8K-32K sweep. Protocol mixes stated per-point.
7. **9% of a clean corpus collides with the benches** — the Phase B screen ledger as a
   breakdown bar (4,551 exclusions by reason, sum-of-first-n-naturals as the poster child).
   Every "trained on open data" claim in the industry is exposed to this; we have receipts.
8. **The batching inversion** (restyle to brand v2) — U-38's 23 -> 219 tok/s aggregate curve,
   the MMVQ->MMQ kernel cliff at width 9, and U-39's MoE 2.0x cap: "which model to serve
   DEPENDS on how many users you have."
9. **The verification vector** (shipped) — extends with k=32 and collapse-strategy arms (#78).

## Campaign-fed (staked, GPU queue)

10. **Ours vs naive all-layers quantization** — the quality-ladder Pareto (KLD vs size, both
    layouts, tok/s stamped per point). THE centerpiece comparison; prereg
    2026-08-05-quality-ladder-chart.md, gates staked.
11. **Grid v2 rows** — GSM8K/IFEval/GPQA/AIME columns landing on the capability table (#79).

## Build order

Stuck-boost (1) and format ladder (2) first - highest relatability per build-hour; then
prediction-vs-reality (3) and the size-speed map (4) as the trust + contrast pair; the rest
as their data or restyle slots open. One asset per post; every asset carries the receipt
footer and lands here the day it renders clean.

## Status, 2026-08-09 — the data-ready queue is EMPTY

Built, indexed in `media/README.md`, and live on raw GitHub:

| # | asset | what it turned out to say |
|---|---|---|
| 1 | `stuck_boost_state` + `stuck_boost_reddit` | the r/LocalLLaMA cut carries the self-check ON the canvas |
| 2 | `format_ladder_v2` | the spread is the hook; **"IQ is slow" being false** is the value |
| 3 | `prediction_vs_reality` | 8.4% median, the -67% disk miss at full size |
| 4 | `size_vs_speed` | the **inversion**, not the trend: +26% bytes, 4.2x faster |
| — | `pipeline` | the six stages, with `serve` drawn as unshipped |
| — | `fragility_fingerprint` | the fragile band MOVES: Mistral front 27x, every Qwen back |
| — | `depth_vs_uniform` | same bytes, better model - with P3's staked speed MISS |

Three of them (`pipeline`, `fragility_fingerprint`, `depth_vs_uniform`) were built for the
README reshape and are wired into it; the rest are post-ready standalone.

**What is left needs measurement, not build-hours:**

- **5** speculation regime map - BUILT as `speculation_map`. The collation turned out to be
  the asset: twelve cells on one box, 2.41x down to 0.61x, with llama.cpp's own default
  among the losses. Drawn on a log axis about 1.00x because these are ratios.
- **6** KV depth tax - waiting on the queued 8K-32K sweep.
- **7** the 9%-collision ledger - **NOT BUILDABLE AS DESCRIBED, and the plan was wrong to say
  it was.** The headline is solid and citable: 4,551 of 50,661 items in
  `bigcode/self-oss-instruct-sc2-exec-filter-50k` share an 8-gram with HumanEval or MBPP, and
  total - kept reconciles exactly. But the promised breakdown "by reason" does not exist -
  `decon.screen_batch` stored `reasons=excluded[:50]`, so the committed ledger holds why for
  50 of the 4,551, **1.1%**, and says nothing about the truncation. Aggregating those 50 looks
  exactly like a breakdown of all of them; it is not one, and a chart built from it would have
  published a 1.1% sample as if it covered everything. The cap is now removed, so a future
  screen records every reason - but regenerating THIS ledger needs the 50k corpus re-downloaded
  (not in the local HF cache), which is a user-gated decision, not a build-hour.
- **8** batching inversion restyle - **deliberately parked**: it advertises multi-session
  serving we do not ship. It lands with `serve`, not before. The strongest chart in the deck
  must not write a cheque the tool cannot cash.
- **10** quality-ladder Pareto - still gated on the GPU queue.
- **11** EV-1 capability rows - BUILT as `capability_ladder`, rendering live from the
  results on disk and re-graded per C-26. 16 of 21 rows banked; the outstanding 30B/GPQA
  cells draw as explicit gaps, so the asset is honest at every stage and simply gets denser
  as the night lands. RE-RENDER when the night completes - the committed PNG is a snapshot.
  **REFRAMED after Federico pushed back on it.** The first cut was titled "Where size stops
  buying accuracy" over rows labelled 0.6B/4B/7B/30B, which reads as a controlled scaling
  test. It is not one: the rows span Qwen2.5/Qwen3/Qwen3.5, three quantization tiers, and a
  code specialist at sub-4-bit being asked to do competition maths. The lesson is a chart
  rule, not a one-off - **a row label is a claim.** Labelling four files by parameter count
  asserts that parameter count is the variable, and here it was the one thing that did not
  vary alone. Every bar now names its family and quant tier.
- **NEW** the difficulty-band chart from U-43: 73% of the bulk corpus sits in the two easiest
  quartiles, validated against 397 declared-level samples. Data committed
  (`weights/data/phaseb_difficulty.json`); it becomes a chart when Phase C gives it a
  before/after to sit beside.
