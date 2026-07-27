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
