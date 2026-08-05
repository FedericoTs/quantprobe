# QUALITY LADDER — the chart campaign (KLD-vs-size Pareto, depth-aware vs stock, tok/s on every point)

**Date staked:** 2026-08-05, before any quant in the campaign was built. Inspired by Atomic
Chat's Ling 3.0 AD-layout chart; ours adds the axis theirs cannot: measured decode speed per
point, on named hardware.

**The claim under test:** our probe-built depth-aware layout dominates the stock llama.cpp
preset on the KLD-vs-size frontier for a model the probe has never tuned — and the margin is
largest in the Q4-band, consistent with A2A (-39.5% median KLD at equal bytes on a different
model).

## Design

- **Model:** Qwen3.5-4B (the Phase-A-promoted lanes engine; small enough for a full ladder in
  one campaign). Reference = BF16, bit-exact, run locally.
- **Arms:** ~8 quant levels (IQ2 band -> Q6/Q8 band) x 2 layouts: stock llama.cpp preset vs
  depth-aware built by `quantprobe probe --apply` from the model's OWN measured fragility
  curve (Law 3: never copied from another model).
- **Metrics, one pinned protocol:** mean KL divergence vs BF16 on held-out text (llama.cpp
  KLD machinery, fixed chunk count, seed recorded) + measured tok/s per point (llama-bench,
  r=3, one machine state, C-14) + file size. Every chart number reproducible from the ledger.
- **The chart** (`weights/make_quality_ladder.py`, house SVG style): KLD-vs-size frontier,
  both curves, matched-size delta callouts, memory-budget verticals for common VRAM classes
  (6/8/12/16/24 GB), tok/s stamped per point, quant table below, repo footer.

## Staked gates

- **P-Q1:** the depth-aware curve sits at-or-below stock at >= 6 of 8 matched sizes, with
  >= -15% KLD at at least one Q4-band point. Below that, A2A's effect failed to transfer to
  this model and THAT is the published result.
- **P-Q2 (the speed axis is honest):** depth-aware costs <= 3% tok/s vs stock at every
  matched size (the free-lunch claim from prior work; a miss is published on the chart
  itself, not hidden).
- **KR-Q1:** one machine state for all speed points; any overlap voids the row (standing).
- **KR-Q2:** the probe's fragility curve for this model is measured fresh - copying another
  model's band is the 25x error Law 3 exists to prevent; the curve ships in the ledger.
- **KR-Q3 (chart honesty):** every number printed on the chart appears in the committed
  ledger; the chart generator refuses to render a point absent from it.

Queue: after Phase B's GPU work (B4) and the Maple check (#77). Raw under
`weights/data/qladder_*`. Verdict appended here.
