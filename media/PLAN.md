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
