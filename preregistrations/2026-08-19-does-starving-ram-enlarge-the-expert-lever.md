# Pre-registration #110: does starving RAM enlarge the expert lever? (the clean residency test)

**Author:** Federico Sciuca · **Date staked:** 2026-08-19, **before any balloon arm was run.**
**VOID / UNSCORED 2026-08-19 - a pre-balloon probe refuted the premise before any treatment arm ran. See the verdict at the foot.**

## Why

Three preregs have circled the same question and none has isolated it:

- #107 (decode) and #108 (prefill) measured the expert knob **overshooting** Law 4's ceiling
  (+15%, +180%) on a model **larger than free RAM**.
- #109 measured it **undershooting** on a model that fits — but [C-33](../FINDINGS.md) then found
  that arm was run in the wrong *regime*: `-ngl 12` on the control left 3.5 GiB of VRAM idle and
  the planner classes that row capacity-bound, not bandwidth-bound. Two models, four confounded
  variables (residency, GPU-layer fraction, VRAM load, architecture). It decided nothing.

The candidate mechanism is **residency**: lowering k shrinks the *set of experts touched per
token*, so on a memory-starved box a larger fraction of that smaller set stays cache-resident,
buying a speedup Law 4's bandwidth model does not contain. If true, it is not a nuisance — it is
**leverage**: the expert dial would be worth *more* precisely on the constrained hardware
quantprobe exists to serve.

## The design that isolates it

**One model, one placement, one architecture. Vary only free RAM.** Everything C-33 confounded
is now held fixed by construction; the only thing that moves between conditions is how much of
the model the page cache can hold.

- **Model:** `DeepSeek-Coder-V2-Lite-Base-IQ2_XS`, 5.56 GiB (deepseek2, 64 experts, default k=6).
- **Placement:** `-ngl 0` — **pure CPU, zero GPU**. This is deliberate and it answers the exact
  objection that sank #109: with no layer on the GPU, the result **cannot** be GPU/CPU overload,
  a capacity cliff, or a layer-fraction artefact. The whole 5.56 GiB is the RAM-resident,
  k-controlled, starve-able working set, and the binding constraint is unambiguously RAM
  bandwidth — the exact regime the byte-share ceiling is derived for.
- **Two conditions**, differing *only* in free RAM, induced by a touched-and-held memory balloon:
  - **FITS:** no balloon, ~11 GiB free ≫ 5.56 GiB model. Verified reachable.
  - **STARVED:** 7.5 GiB balloon → ~3.8 GiB free < 5.56 GiB model, so ~1.8 GiB streams from disk
    every pass. Verified: 8 GiB touched drops free RAM 11.3 → 3.5 GiB.

## Predictions (staked before any arm ran)

Gain within each condition is `gain(k) = speed(k) / speed(k=6)`. The byte-share ceilings are a
property of the file (bandwidth-only): k=4 → 1.224x, k=2 → 1.577x, **k=1 → 1.843x**.

- **P-1 (the leverage claim).** The k-lever is **larger when starved**: `gain(k=1, STARVED) ≥
  1.15 × gain(k=1, FITS)`. *Refuted if starving does not enlarge the lever by at least 15%.*
- **P-2 (fits does not overshoot — #109 re-run, clean).** `gain(k=1, FITS) ≤ 1.90x` — at or below
  the 1.843x ceiling within noise. *Refuted above 1.90x.* This is #109's core claim at a placement
  C-33 cannot touch.
- **P-3 (the excess is residency).** The **starved** condition **overshoots** the ceiling —
  `gain(k=1, STARVED) > 2.03x` (1.843 × 1.10) — while FITS does not. *Refuted if starving fails to
  push the lever above its bandwidth ceiling.* This is the direct analogue of the Qwen3.6 overshoot,
  reproduced on demand by memory pressure alone.

**Where I might be wrong.** Disk streaming may be so slow that the starved k=6 baseline collapses
and *every* k looks fast relative to it, inflating the lever for a trivial reason (a slower
baseline, not a smaller working set). Guarded two ways: gains are within-condition ratios, and the
baseline-usability gate from #108 voids any condition whose k=6 spread exceeds 15%. If the starved
baseline is merely noisy rather than genuinely slower, the gate catches it.

## Method

`llama-cli` b10098, `-ngl 0 -n 96`, one box, one session. **Descending k only** (L-31), three
passes per condition so every arm shares a predecessor and the usability gate has n=3. FITS first,
then STARVED. Balloon allocated and **re-touched on a background thread** so the OS cannot page it
out and hand the model its RAM back. Free RAM sampled beside every arm. Override honoured on
deepseek2 confirmed in #109. C-14: nothing else runs.

## Kill rule (committed before data exists)

Scored by [`weights/prereg110_score.py`](../weights/prereg110_score.py), committed **before** the
arms run.

- **P-1 and P-3 both hold** → **residency is the mechanism, and it is leverage.** The excess in
  #107/#108 is explained, the open edge closes, and quantprobe gains a real recommendation: *on a
  box where your model does not fit RAM, the expert dial buys more than the file predicts* — the
  one regime where trading quality for speed is most likely worth it. L-32 is corrected from "an
  Amdahl floor" to "a residency ceiling that memory pressure lifts."
- **P-1 holds, P-3 refuted** → residency enlarges the lever but not past the ceiling; the effect
  is real but modest, and the recommendation is quantitative, not categorical.
- **P-1 refuted** → residency does **not** enlarge the lever on a fitting model made to starve, so
  the #107/#108 overshoot is *not* residency after all, and the per-expert-term hypothesis returns
  with its own stake. Either way #107/#108/#109 stand as scored; this decides only the mechanism.

---

## Verdict: VOID / UNSCORED (2026-08-19). The premise was refuted by a pre-balloon probe.

No balloon arm ever ran. Before inflating it, I ran a quick k=6 vs k=1 probe at -ngl 0 with the
model FITTING comfortably (~13 GB free, 5.56 GiB model) - and it broke the prereg's own premise.

Raw: [`prereg110_ngl0_probe.txt`](../weights/data/prereg110_ngl0_probe.txt).

| | k=6 | k=1 | gain |
|---|---|---|---|
| -ngl 0, fits, no pressure | 3.73 (3.0-4.4) | 7.53 (7.1-8.0) | **2.018x** |

Byte ceiling at k=1: 1.843x. **The lever is ~2x with the model fitting comfortably in free RAM.**

**Two things this kills.**

1. **P-2 assumed a fitting model does not overshoot the ceiling** (that was #109's core, re-run
   clean). This probe shows a fitting model *at or above* the ceiling, with 13 GB of free RAM and
   nothing to evict. So the overshoot - if it is real - happens **without memory pressure**, and
   **residency cannot be the mechanism.** The whole #110 framing is wrong before the treatment arm.
2. My earlier read of the first three arms as "flat, compute-bound, wrong regime" was a **misread
   of noisy partial data** - I killed the run at k=2 of pass 1, before the k=1 arm where the lever
   lives. The lever is present and large; I was wrong to call it absent.

**What is NOT concluded.** The k=6 baseline spread is 47% - far past the #108 usability gate - so
"2.018x" is not a trustworthy multiple (it flips below 1.843 at the high end of the baseline).
What is robust is only this: **the lever is large and is not smaller without memory pressure.**

**What this points at instead.** #108 named two candidates for the excess: residency, or a
per-expert term (gather + a small matmul launch per routed expert, unpriced by Law 4). Residency
is now the less likely of the two, because the excess shows up with the model fitting. A
per-expert **CPU** overhead fits the pattern - #109 measured this same model *undershooting* at
-ngl 12 (experts partly on GPU) and this probe shows it *overshooting* at -ngl 0 (all on CPU),
same fitting model. That is a **placement** effect, not a memory-pressure effect.

**The clean successor** varies **only -ngl** on **one model, one session, all placements fitting
in RAM**, with tight baselines: does the excess over the byte ceiling grow as the model moves onto
the CPU? That isolates the per-expert-CPU term from residency and does not require a balloon. It is
staked next, carefully, with its regime checked by the planner *before* the run - the lesson of
C-33 and of this void both.

#107, #108, #109 stand as scored. This decides nothing about them; it only redirects the mechanism
hunt away from residency.
