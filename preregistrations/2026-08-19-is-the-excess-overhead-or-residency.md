# Pre-registration #109: the excess over Law 4 — a missing term, or just residency?

**Author:** Federico Sciuca · **Date staked:** 2026-08-19, **before any arm on the control model
was run.** **SCORED 2026-08-19 - 1 of 3, two predictions mis-specified, and the control arm later found to be measured in the wrong REGIME. See the verdict and the correction at the foot.**

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

---

## Verdict: SCORED 1/3 — and two of my three predictions were mis-specified (2026-08-19)

Scored by [`weights/prereg109_score.py`](../weights/prereg109_score.py), committed before the
control arms ran. Raw: [`prereg109_run.log`](../weights/data/prereg109_run.log) ·
[`prereg109_control.json`](../weights/data/prereg109_control.json) ·
[`prereg109_verdict.txt`](../weights/data/prereg109_verdict.txt).

**DeepSeek-Coder-V2-Lite-Base-IQ2_XS, 5.56 GiB against 12.8 GB free — it fits, with room.**

| | k=4 | k=2 | k=1 |
|---|---|---|---|
| decode measured | 1.106x | 1.380x | 1.528x |
| decode ceiling | 1.224x | 1.577x | 1.843x |
| | −9.6% | −12.5% | **−17.1%** |
| prefill measured | 1.035x | 1.108x | 1.186x |
| prefill ceiling | 1.225x | 1.580x | 1.848x |
| | −15.5% | **−29.8%** | −35.8% |

- **P-1 decode k=1 within ±15% of 1.843x — MISS (1.528x, −17.1%).**
- **P-2 prefill k=2 within ±25% of 1.580x — MISS (1.108x, −29.8%).**
- **P-3 prefill excess below 60% — HIT (−29.8%, against +180% where the model does not fit).**

**Kill rule as printed: "RESIDENCY IS NOT THE WHOLE STORY."** That branch's text presumes the
misses were HIGH. **They were LOW.** The scorer routed correctly on the booleans it was given and
its prose does not fit the data, exactly as #106's kill rule failed to anticipate its own
(False, True) combination. Reported as printed, and then read properly below.

### The predictions were badly written, and that is the first finding

P-1 and P-2 asked whether the measurement lands **on** the ceiling. A ceiling is an **upper
bound**: it forbids landing above, and says nothing about landing below. Testing it with a
two-sided ±15% band means any model with unmodelled *fixed* costs fails a test it was never in
tension with. That is my error in specification, not evidence of a missing term.

**P-3 was the only one written as a one-sided test, and P-3 is the discriminator.** It hit
decisively: the fitting model shows **no excess at all** (−29.8%) where the non-fitting model
showed **+180%**.

### What the data supports

**The excess appears only where the model does not fit.** On 6.6 GiB of headroom, nothing
exceeds its ceiling on either resource; on a 1.0 GiB deficit, both do. The only structural
difference between the two runs is residency, and that is consistent with L-29/L-31 rather than
with a per-expert term — a per-expert overhead would not care about page cache and would have
followed the knob here too.

**A second, separate finding falls out: the ceiling over-promises on models that fit, by
10–36%.** Real gains fall short of the bound because something *fixed* — the always-active path,
per-token sampling, kernel launch — does not shrink with k. That is an Amdahl floor, not a
per-expert term (a per-expert cost would make low k *faster* than predicted, not slower). It
also means `quantprobe`'s wording is already right: the line says **"buys at most ~1.24x"**, and
"at most" is exactly what a bound of this kind licenses.

### What is NOT claimed

Residency is **not confirmed** by this prereg. P-3 is one correctly-specified test on one control
model, and the two tests meant to corroborate it were unusable by construction. The clean version
is a one-sided stake — *does a fitting model ever EXCEED its ceiling?* — across more than one
architecture, and it is owed before the open edge in #107 and #108 is called closed. Registered
as untried rather than resolved.

#107's and #108's verdicts stand as scored; this decides nothing about them.

---

## Correction, 2026-08-19 (same day): the control arm was measured in the wrong regime

Challenged within hours of the verdict: *is this a real inversion, or GPU/CPU overload?* It is
the second. Two commands settle it, and both were available before the run.

**1. The flag was matched; the condition was not.** `-ngl 12` was copied from #107/#108 so the
two models would "share a placement". They do not:

| | layers | `-ngl 12` is | to VRAM | VRAM idle |
|---|---|---|---|---|
| Qwen3.6-35B | 40 | **30%** of layers | ~3.94 GiB | ~2 GiB |
| DeepSeek-Lite | 27 | **44%** of layers | ~2.47 GiB | **~3.5 GiB** |

**2. The control ran at about half speed, in a different binding regime.** `quantprobe plan` on
the control file recommends 65% of experts in VRAM, predicts **33.0 tok/s** there, and classifies
that row **CAPACITY-BOUND (VRAM)**. Measured here at `-ngl 12`: **16.67**.

The expert ceiling is derived from **Law 4, which prices bandwidth**. Applying it to a row bound
by capacity is a category error, so the 10–36% shortfall is evidence about *the placement I
chose*, not about the ceiling. **L-32 is qualified accordingly and [C-33](../FINDINGS.md) logged.**

**What this does to the verdict.** P-3's numbers stand as measured, but the reading "the excess
appears only where the model does not fit" **cannot be supported**: the two-model comparison
confounds four variables — residency (intended), GPU-layer fraction (30% vs 44%), absolute VRAM
load, and architecture. The prereg's own "What is NOT claimed" section was already right to
refuse the residency conclusion; it was right for a weaker reason than the real one.

**The clean experiment, and it is cheaper than this one was.** Vary residency on **one model, one
placement, one architecture**, changing nothing else: measure the k-curve, then repeat it with
free RAM driven below the model size by holding a memory balloon. Same binding constraint, same
split, same everything — only residency moves. That isolates in one session what two models
cannot isolate at all, and it is staked next rather than folded in here.

**The lesson, which is the generalisable part:** matching a *flag* across models is not matching
a *condition*. `-ngl N` means a different split, a different VRAM load, and possibly a different
binding constraint on every model it touches. The tool prints the regime; I did not read it
before running.
