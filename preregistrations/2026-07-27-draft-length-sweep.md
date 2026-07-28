# Pre-registration #36: is the draft length the binding constraint on speculation?

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The observation that motivates this

Speculation measures 50–59 tok/s on copy-regime output (#28, #35). Its ceiling is NOT bandwidth —
it is the verify forward pass, and this GPU's batched-forward ceiling is measured at **405–445
tok/s** (#26, the prefill convergence). So speculation sits at roughly 1/8 of its own ceiling.

Then the flag list: `--spec-ngram-simple-size-m` — "length of draft m-gram" — **defaults to 48**.
Our measured `mean len` was **46.50**. The accepted run is pressed right against the draft budget,
which is what a binding constraint looks like. If the drafter is allowed to propose more, and the
context genuinely contains the continuation (89–95% acceptance says it does), each verify pass
should return more tokens for the same weight read.

## Arms

`llama-server`, split placement, `-ub 1024`, f16 KV (per #35: q8_0 is a cost at this context),
`--spec-type ngram-simple`, edit task (copy regime), temp 0, r=2.
Sweep `--spec-ngram-simple-size-m` ∈ **{24, 48 (default), 96, 192}**, everything else fixed.

## Stakes

- **P-1 (the default is not optimal).** Some m > 48 beats m=48 by **≥15%**.
- **P-2 (acceptance survives).** At m=96 the draft acceptance rate stays **≥80%** (from 89%).
  If long drafts collapse acceptance, the lever is self-limiting and P-1 cannot hold for the
  right reason.
- **P-3 (an optimum exists, i.e. this is a trade not a free ride).** m=192 is **worse than the
  best** arm — beyond some length the verify batch turns compute-bound and wasted drafted tokens
  cost more than the accepted ones save. A monotone-increasing result would mean we still have
  not found the ceiling and the sweep must extend.
- **P-4 (output identity holds at every m).** All arms byte-identical. Speculation is
  output-preserving by construction; if a draft-length change alters output, the implementation
  has a bug and every speculation number in this project needs re-examination.

## KILL RULE — stated before measuring

**If no m beats the default by more than 5%, this lever is closed**: llama.cpp's default is
already tuned, the 8× gap to the verify ceiling is owned by something else (draft *quality*, not
draft *length*), and the next investigation is the drafter itself rather than its budget.

## What ships

If a better m exists, it goes into the plan output's speculation advice WITH its measured
acceptance rate and the caveat that it is copy-regime-only — the same shape as every other lever
we ship. No law changes either way; this is a flag.

---

## Scored (2026-07-28, log: `weights/data/prereg36_draft_length.log`)

**Verdict: P-1 HIT at +81%, P-2 HIT (exactly at the boundary), P-3 HIT in letter but the
MECHANISM I predicted is wrong, P-4 HIT. The default draft length was leaving 81% on the table.**

| `--spec-ngram-simple-size-m` | tok/s | acceptance | drafted / accepted | output sha |
|---|---|---|---|---|
| 24 | 46.10 | 100.0% | 709 / 709 | `28a5c1e1c014` |
| **48 (default)** | **49.80** | 88.9% | 811 / 721 | `28a5c1e1c014` |
| 96 | 65.00 | 80.0% | 911 / 729 | `28a5c1e1c014` |
| 192 | 78.01 | 66.8% | 1098 / 733 | `28a5c1e1c014` |
| **384** | **90.33** | 66.8% | 1100 / 735 | `28a5c1e1c014` |
| 768 | 89.82 | 66.8% | 1100 / 735 | `28a5c1e1c014` |

Position control (the check that killed the fusion claim): default re-run LAST measures **49.59**
against 49.80 cold, and m=384 re-run last measures **89.85** against 90.33. The effect is not
thermal ordering. **+81% over the shipped default, byte-identical output, one flag.**

- **P-1 (some m>48 beats default by ≥15%): HIT at +81%.**
- **P-2 (acceptance ≥80% at m=96): HIT**, landing exactly on 80.0%.
- **P-3 (m=192 worse than the best): HIT in letter** — 78.01 < 90.33 — **but my stated mechanism
  is refuted.** I predicted the verify batch would turn compute-bound and waste drafted tokens.
  It does not: from m=192 onward the drafted and accepted counts are essentially FROZEN
  (1098/733, 1100/735, 1100/735) while throughput still rises 78 → 90. Identical work, less time.
- **P-4 (identity at every m): HIT.** One sha across the entire sweep, including the 100%- and
  66.8%-acceptance extremes.

### What is actually happening — fewer verify rounds, not more drafting

The frozen counts are the tell. `m` is a per-round draft BUDGET, so raising it delivers the same
total drafted tokens in **fewer, longer rounds**. Each round costs one full weight read of the
model. Same accepted tokens ÷ fewer weight reads = more tokens per byte moved — the axiom-break
of L-11 applied harder, not a new mechanism. And m=768 returns byte-identical counts to m=384
because the DRAFTER has run out: the source file being reproduced contains no longer matching
spans. **The ~90 tok/s plateau is the drafter's ceiling on this task, not a hardware limit.**

Note the acceptance rate falls (100% → 66.8%) while throughput nearly doubles. Acceptance is
therefore the WRONG optimisation target — a metric this project would have chased had it not
measured wall-clock alongside. Rejected drafts are nearly free; skipped weight reads are not.

### Where this leaves the ceiling

| regime | tok/s | vs raw-decode wall (41.1) |
|---|---|---|
| raw decode, measured | 22.25 | 54% |
| **raw-decode wall (physics)** | **41.1** | 100% |
| speculation, shipped default | 49.80 | 121% |
| **speculation, m=384** | **90.33** | **220%** |
| verify-pass ceiling (#26 batched forward) | ~405–445 | ~1000% |

Still ~4.5× below the verify ceiling, and the binding constraint is now identified as **draft
quality/length available from the n-gram store**, not the hardware. That is the next target.

**Wired into:** `quantprobe/plan.py:speculation_advice` (the flag and its measured number) ·
`findings/REGISTER.json:V-04`.
