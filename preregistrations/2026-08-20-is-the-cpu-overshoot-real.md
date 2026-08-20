# Pre-registration #111: is the pure-CPU overshoot real, or was it baseline noise?

**Author:** Federico Sciuca · **Date staked:** 2026-08-20, **before any arm was run.** **VOID 2026-08-20 - baseline unusable at n=5; and the void reopens the residency question I had closed. See the verdict at the foot.**

## Why, and why this is narrow on purpose

The mechanism hunt (U-59) wants to know if the excess over the expert byte-ceiling is a
**per-expert CPU term** — cheaper per routed expert on GPU than on CPU. The clean test would sweep
the same model from experts-on-GPU to experts-on-CPU. **This box cannot run it.** The regime check
that C-33 taught me to do first says so: on a 6 GB card with a 5.56 GiB model, you cannot move
experts between CPU and GPU without also crossing the VRAM capacity boundary. `quantprobe plan`
classes the experts-on-GPU rows **capacity-bound** and only the pure-CPU row **RAM-bandwidth-bound**.
Placement and regime are entangled here; the CPU-vs-GPU contrast is unmeasurable on this hardware.

So this prereg does the one thing the box *can* do cleanly, and it is the load-bearing one. The
whole per-expert-term thread rests on a single observation: prereg #110's probe showed
DeepSeek-Coder-V2-Lite at `-ngl 0` overshooting its byte ceiling ~2× **with the model fitting** —
but on a **47% baseline spread**, past the usability gate, so the multiple was not trustworthy.
Before building anything on that, confirm it. **Does the overshoot survive a usable baseline?**

`-ngl 0` is the cleanest regime available: pure CPU (no VRAM capacity issue), RAM-bandwidth-bound
(the byte ceiling's own regime), model 5.56 GiB against ~13 GiB free (no residency). Capacity and
residency are both excluded by construction, so an overshoot here is the byte ceiling being crossed
in a clean bandwidth regime — the strongest evidence for a per-expert term this box can produce, and
the reason the mechanism is not residency (already shown) or capacity.

## What the file says (no measurement)

DeepSeek-Coder-V2-Lite-Base-IQ2_XS: 64 experts, default k=6, routed byte share 54.9%. Byte
ceilings, `1 / (1 − 0.549·(1 − k/6))`:

| k | byte ceiling |
|---|---|
| 4 | 1.224x |
| 2 | 1.577x |
| 1 | **1.843x** |

## Predictions (staked before any arm ran)

Gain is `median(speed(k)) / median(speed(k=6))`. Excess is `gain(k) / ceiling(k)` — above 1.0 means
the lever beats its bandwidth ceiling.

- **P-1 (the overshoot is real).** `gain(k=1) > 1.90x` — the #110 probe's ~2× survives a usable
  baseline, clearing the 1.843x ceiling with margin. *Refuted at 1.90x or below.*
- **P-2 (the method fix worked).** The k=6 baseline spread is **≤ 15%** (the #108 usability gate).
  *If refuted, the whole result is VOID* — the box cannot measure this cleanly and U-59 needs
  different hardware. This is a method check with teeth: it can void my own headline.
- **P-3 (the excess grows as experts are removed — the per-expert signature).** `excess(k=1) >
  excess(k=2) > excess(k=4)`, i.e. removing more experts overshoots the byte prediction by more.
  *Refuted by any non-increasing step.* A per-expert cost saved on every removed expert is exactly
  what makes the excess grow toward low k; pure bandwidth would keep excess flat at 1.0.

**Where I might be wrong.** IQ2_XS CPU dequant is heavy, so at low k the always-active path (attention,
shared experts) may dominate and *cap* the gain below the ceiling — the opposite of P-1. If P-1 fails,
the #110 probe's 2× was baseline-noise, the per-expert-term thread weakens sharply, and L-32's
"undershoot" reading gains ground. I am staking the overshoot, because that is what the probe showed
and the honest test is whether it holds, not whether I can hedge both ways.

## Method

`llama-cli` b10098, `-ngl 0 -n 128 -t 4 --seed 1234`, one box, one session. **Descending k only**
(L-31), **five passes** so each k has five readings and the median is robust to CPU-scheduler jitter
— the fix for the #110 probe's 47% spread, which came from single readings at ~5 tok/s. Warm-up
discarded. Free RAM logged beside every arm (must stay ≫ model). C-14: nothing else runs. Override on
deepseek2 confirmed honoured in #109.

## Kill rule (committed before data exists)

Scored by [`weights/prereg111_score.py`](../weights/prereg111_score.py), committed **before** the run.

- **P-2 refuted** → **VOID.** Baseline still unusable; this box cannot measure it and the finding is
  a hardware limit, not a result. Recorded as such.
- **P-1 and P-3 hold** → the overshoot is **real and grows with expert removal**, in a clean
  bandwidth-bound fitting regime. Residency and capacity are both excluded, so the excess is a
  **per-expert term** — quantprobe's ceiling line gains a caveat: on CPU-resident-expert placements
  the dial can beat its byte ceiling, and the file-only number is a floor there, not a cap. U-59
  advances from hypothesis to measured-on-one-model.
- **P-1 holds, P-3 refuted** → overshoots but flat: a fixed per-*call* offset, not per-expert.
  Recorded, weaker, and named for its own follow-up.
- **P-1 refuted** → the probe's 2× was noise. The overshoot is not robust, the per-expert-term thread
  weakens, and L-32's undershoot reading stands. Said plainly.

#107–#110 stand as scored. This confirms or dissolves one number; it decides nothing about them.

---

## Verdict: VOID (2026-08-20). And the void reopens something I had closed.

Scored by [`weights/prereg111_score.py`](../weights/prereg111_score.py), committed before the run.
Raw: [`prereg111_ngl0.json`](../weights/data/prereg111_ngl0.json) ·
[`prereg111_run.log`](../weights/data/prereg111_run.log) ·
[`prereg111_verdict.txt`](../weights/data/prereg111_verdict.txt).

| k | median tok/s (n=5) | gain | ceiling | excess |
|---|---|---|---|---|
| 6 | 5.90 | 1.000x | — | 1.000 |
| 4 | 5.90 | 1.000x | 1.224x | 0.817 |
| 2 | 8.30 | 1.407x | 1.577x | 0.892 |
| 1 | 9.30 | **1.576x** | 1.843x | **0.855** |

- **P-2 baseline spread ≤ 15% — MISS (23.7%).** Five passes, median, and the k=6 baseline still
  ranged 4.9–6.0. Pure-CPU decode at ~5 tok/s on four 2016 cores carries irreducible scheduler
  jitter. **Kill rule fires: VOID.** P-1 and P-3 are not evaluated — you cannot build a ratio on a
  divisor this loose. This box cannot measure the overshoot cleanly; the honest label is a hardware
  limit, not a result.

### What the void nonetheless shows — and the claim it forces me to walk back

The gains are VOID and I claim nothing from them as measurements. But one qualitative fact does not
need a tight baseline to read: **the k=1 median gain is 1.58×, nowhere near the probe's 2.0×.**
The [#110](2026-08-19-does-starving-ram-enlarge-the-expert-lever.md) probe's overshoot came from a
single low k=6 reading (3.73) inflating one ratio; five passes put k=6 at 5.9 and the overshoot
disappears.

That matters because **the #110 probe's overshoot was my sole basis for ruling residency out.** I
wrote there that the excess "happens without memory pressure, so residency cannot be the
mechanism." That inference rested on a number this run shows was not robust. So I have to withdraw
the strength of it: **residency is not ruled out.** It is back on the table.

And the larger pattern, read across every attempt, actually points back toward it:

| model | fits free RAM? | ceiling | reading |
|---|---|---|---|
| Qwen3.6 (#107/#108) | **no** (deficit) | crossed | **overshoot** +15% / +180% |
| DeepSeek −ngl 12 (#109) | yes | — | undershoot (but capacity-bound, C-33) |
| DeepSeek −ngl 0 (#110 probe) | yes | — | "overshoot" 2× — **now shown to be noise** |
| DeepSeek −ngl 0 (#111) | yes | — | undershoot-leaning, but VOID |

The overshoot correlates with **not fitting**. The one fitting-model overshoot was an artefact.
That is the residency story #109's kill rule first proposed — reinstated, not by new clean data,
but by the removal of the one datapoint that had displaced it.

### The real conclusion: this box cannot resolve U-59

Three attempts, three non-answers, each for a hardware reason this machine cannot escape:

- **#109** — every GPU placement that fits is capacity-bound, not bandwidth-bound (C-33).
- **#110** — can't hold a fitting MoE's experts on the 6 GB GPU to contrast CPU vs GPU per-expert cost.
- **#111** — pure-CPU decode is too jittery to resolve the ceiling to ±15%, even at n=5.

The 6 GB card and the four old cores structurally entangle or blur every version of the question.
**U-59 needs different hardware** — a box that can hold a fitting MoE's experts fully in VRAM (to
price GPU per-expert cost) or fit a large MoE in RAM with headroom and a CPU fast enough for a tight
baseline. I am marking U-59 blocked-on-hardware and stopping the box thread here rather than
running a fourth confounded variant. Knowing an instrument's floor is a result; pretending past it
is how C-31, C-32 and C-33 happened.

#107 and #108 (the non-fitting model) stand as scored — their overshoot is real and, if anything,
this reinforces its link to non-residency. #109/#110 stand with their existing verdicts; this only
withdraws the *strength* of #110's residency-ruled-out inference.
