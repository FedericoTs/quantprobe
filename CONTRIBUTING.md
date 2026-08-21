# Contributing

The most valuable contribution is a **measurement**. This project runs on falsifiable numbers.

## Contribute an η data point (10 minutes)
Run `quantprobe calibrate` first: it measures your RAM stream, disk, and GPU sustained clocks
(plus optional CPU/GPU anchor runs on your own GGUF), and `bench` picks the results up
automatically — so your point carries measured bandwidths, not spec-sheet numbers. Hand-typed
spec values are exactly the input class behind the first external replication's 9× miss (E-06).
Then the easiest path: `quantprobe bench ... --contribute` prints a pre-filled, opt-in issue (you review before submitting; nothing auto-sent). Or manually: run `quantprobe bench --gguf <model> --model <preset> [--vram N --vram-bw N --ram N --ram-bw N --disk-bw N]`
and open an issue with the "η data point" template: your hardware, the model/bits, predicted vs measured.
Points that land outside the bands are MORE valuable than ones that confirm them.

## Contribute an atlas entry (~1 hour)
Run `quantprobe probe --gguf <model-f16.gguf> --eval wiki.test.raw` on a model family not yet in
[the atlas](weights/GGUF_DEPTH_RECIPE.md) and open an "atlas entry" issue with the band curve.
A model whose fragile end breaks the current pattern is a finding, not a failure.

## Code
PRs welcome for the `quantprobe` package (plan/calibrate/optimize/auto/target/fetch/quantize/probe/run/bench/dashboard) (keep `python tests/smoke.py` green). For claims/laws,
open an issue with data first — the bar for prose is measurements.

## How this repo is written

Much of the code, the pre-registrations, and the review comments here are drafted with **Claude
(Opus)**, working from the method below. The commit trailers say so — `Co-Authored-By: Claude
Opus 4.8` is on essentially every commit — and this section exists so it is stated plainly in
one place rather than left to be inferred from `git log`.

A human owns every merge, every published number, and every claim. That is the part that does not
delegate.

It is worth saying why this is disclosed without embarrassment: **the method is the thing that
makes a claim here trustworthy, and it does not care who typed it.** A prediction is written and
committed before the data exists; a scorer is committed before the run; the verdict is whatever
that code prints, including when it is a miss or a void. Several entries in
[FINDINGS.md](FINDINGS.md) are corrections of this project's own published numbers, found by that
process and kept at full size. An argument that survives a staked falsification test is worth the
same whoever drafted it — and one that does not, is worth nothing however it was produced.

The practical consequence for you: **argue with the reasoning, not the author.** If a review
comment gets a fact wrong, say so with the number — that has already happened, more than once, and
the review was the thing that changed.

## The method: measure, stake, wire, audit

Every number this tool prints is supposed to be traceable to a measurement. Keeping that true
needs four steps, and the fourth exists because the first three were not enough.

**1. Stake before you measure.** Write a pre-registration in `preregistrations/` naming the
prediction, the falsification condition, and the ship/don't-ship rule — then commit it *before*
running anything. Misses publish with the same prominence as hits; several of the most useful
findings here are misses.

**2. Measure, and log the machine state.** Raw logs go in `weights/data/`, referenced from the
pre-registration. For GPU work, record `nvidia-smi` **memory and clocks** before and after. Both
have burned us: an orphaned process once made a result look 
worse and invented a finding that had to be retracted, and a boosted clock once made a result look
28% *better*, which is more dangerous because a flattering number invites no scrutiny.

**3. Score it, then say where it went.** Add a `## Scored` section, and end the file with a
machine-readable line:

```
**Wired into:** `quantprobe/plan.py:some_symbol` · `tests/smoke.py:t_some_test` — what changed.
```

`**Wired into:** nothing — <reason>` is a perfectly good answer, and a common one: a refuted
hypothesis *should* change no code. What is not acceptable is leaving it implicit.

**4. Audit that the finding actually reached the code.** Run `python audit.py`, or just
`python verify.py`, which runs it as layer 5.

### Why step 4 exists

On 2026-07-25 we measured — and published in `LAWS.md` — that the Pascal low-bit decode collapse
was format-dependent, not bit-width-dependent. The planner went on gating decode efficiency on
bit-width for another full day, telling every user with a sub-4-bit quant that their GPU was
useless for it and recommending a placement **2.4× slower** than the one it rejected.

Nothing was broken in the usual sense. Tests were green. The release gate passed. The finding was
written down. **It simply never reached the code** — and no test can catch that, because the code
was perfectly self-consistent with a belief we had already disproved. Steps 1–3 produce knowledge;
step 4 is the only one that checks the knowledge landed.

The audit also enumerates the planner's whole decision surface and requires every placement it can
recommend to be either measured or listed in `audit.py:UNMEASURED_PLACEMENTS` **with a reason**.
That second half exists for the same reason: every anchor we had was a MoE-hybrid or disk-stream
row, not one covered "all in VRAM" — the most common setup there is — which is exactly where the
9.5× error hid through a public release. A hole in your evidence is invisible until you enumerate
the surface and diff it against your anchors.

### A test that asserts a bug is worse than no test

Two smoke tests had encoded the collapse as expected behaviour, so they actively resisted the fix.
One demanded that a 2-bit model be *slower* than a 4.5-bit one, which is backwards on byte count
alone. When a test fails after a fix, decide which is wrong before changing either — and
mutation-test the replacement: break the code deliberately and confirm the test goes red.
