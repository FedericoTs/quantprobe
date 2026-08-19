# Pre-registration #109: the excess over Law 4 — a missing term, or just residency?

**Author:** Federico Sciuca · **Date staked:** 2026-08-19, **before any arm on the control model
was run.** **STAKED.**

## Why

Two preregs have now measured the same knob and both overshot Law 4's ceiling:

| | ceiling from the file | measured | excess |
|---|---|---|---|
| decode, k=1 ([#107](2026-08-18-the-k-lever-is-bounded-by-the-always-active-floor.md), re-audited) | 1.242x | 1.430x | **+15.2%** |
| prefill, k=2 ([#108](2026-08-19-is-the-expert-dial-a-prefill-lever.md)) | 1.345x | 3.766x | **+180%** |

#108's scorer routed that excess to its own stake rather than letting it be explained away. This
is that stake. There are two candidate explanations and they are not distinguishable on the
model measured so far, because that model is **larger than free RAM**:

- **(a) A missing per-expert term.** Gather, scatter and per-expert kernel launches cost
  something that scales with k independently of bytes or FLOPs. Law 4 prices neither.
- **(b) Residency (L-29/L-31).** Lowering k shrinks the *set of experts touched*, so more of the
  working set stays page-cache resident. Nothing to do with the knob per se — it is the same
  cache effect that produced a 6.3x span from run order alone.

## The discriminator

**Measure the identical sweep on an MoE that comfortably FITS in free RAM.** Residency cannot
operate there: there is nothing to evict.

- **(a) predicts the excess survives** — per-expert overhead does not care about page cache.
- **(b) predicts the excess vanishes** — the measurement should land on the ceiling.

Control: **`DeepSeek-Coder-V2-Lite-Base-IQ2_XS`**, 5.56 GiB against ~12.2 GB free — **6.6 GiB of
headroom**, versus the Qwen3.6 file's *deficit* of 1.0 GiB. A different architecture
(`deepseek2`, 64 experts, k=6) and a much larger routed share, which makes the test harder on
me rather than easier: its ceilings are 1.84x rather than 1.24x, so there is more room for the
measurement to land inside the band by luck.

Predicted from that file, before measuring:

| k | decode ceiling | prefill ceiling |
|---|---|---|
| 6 (default) | 1.000x | 1.000x |
| 4 | 1.224x | 1.225x |
| 2 | 1.577x | 1.580x |
| 1 | **1.843x** | **1.848x** |

Routed share: **54.9% of active bytes, 55.1% of active params** — nearly equal, unlike Qwen3.6's
22% / 34% split, because this model's experts and always-active path are quantized alike.

## Predictions (staked before any control arm ran)

- **P-1 (decode).** On the fitting model, measured decode gain at **k=1 lands within ±15% of
  1.843x**. *Refuted outside that band.* This is the direct analogue of #107's +15.2% overshoot.
- **P-2 (prefill).** On the fitting model, measured prefill gain at **k=2 lands within ±25% of
  1.580x**. *Refuted outside.* #108's prefill overshot by 180% on the non-fitting model; if that
  was residency, nothing like it should appear here.
- **P-3 (the comparison that decides it).** The fitting model's prefill excess over its ceiling
  is **smaller than 60%** — i.e. less than a third of the +180% measured on the non-fitting
  model. *Refuted at 60% or more.*

## Method

`llama-cli` b10098, `-ngl 12`, same box, one session. **Descending k order only** — no ascending
passes, because L-31 says an arm preceded by a lower k is contaminated and the correct protocol
is now known. Three passes, so every arm has the same predecessor every time. Decode: `-n 128`.
Prefill: `-n 1 -f` the same ~2000-token prompt used in #108. Free RAM recorded beside every arm.
C-14 holds.

## Kill rule (committed before data exists)

Scored by [`weights/prereg109_score.py`](../weights/prereg109_score.py), written and committed
**before** the arms run.

- **P-1, P-2 and P-3 all hold** → the excess is **RESIDENCY**, not a missing term. Law 4 needs no
  new physics; it needs its regime stated, which v1.29–v1.31 already ship. L-29/L-31 absorb the
  finding and the "open edge of Law 4" recorded in #107 and #108 is **closed as explained**.
- **P-1 or P-2 refuted high on a model that fits** → residency cannot be the whole story, and
  there is a real **per-expert term** Law 4 does not contain. That becomes a law-amendment track
  with its own measurement programme, not a footnote.
- **P-3 refuted** (the fitting model overshoots nearly as badly) → the same conclusion, more
  strongly, and the term is architecture-independent enough to matter for every MoE.

Whichever fires, #107's and #108's verdicts stand as scored; this decides only what the excess
*means*.
