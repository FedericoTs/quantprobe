# Pre-registration #111: is the pure-CPU overshoot real, or was it baseline noise?

**Author:** Federico Sciuca · **Date staked:** 2026-08-20, **before any arm was run.** **STAKED.**

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
