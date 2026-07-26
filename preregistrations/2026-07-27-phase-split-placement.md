# Pre-registration #20: prefill and decode want different placements

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The claim

`plan` prints **one** command. If the placement that maximises prompt processing is not the
placement that maximises generation, then that single command is wrong for one of the two phases,
always — and we have never checked which.

Two measured facts say it might be:

- Partial expert offload gives **~2–3× on prefill but only +12% on decode** from the same flags
  (pre-registration #13). The lever is phase-sensitive even if the argmax is not.
- `-ub 2048` gives **+73% prefill and 0% decode** on host-resident experts, and **−39% prefill**
  when the model is fully in VRAM (pre-registration #19). The −39% is a *compute-buffer* cost:
  a large ubatch needs VRAM.

That second fact is what makes an inversion plausible rather than merely possible. The expert
**split** placement deliberately spends VRAM on experts — which is exactly the VRAM the large
ubatch needs. So the placement that wins decode may be starving the lever that wins prefill.

## Configurations, on `Qwen3-30B-A3B-Q2_K`, reference box

| | placement | flags |
|---|---|---|
| **A** | all experts → CPU | `-ngl 99 -ot "exps=CPU" -mmp 0` |
| **B** | split, K=16 experts → VRAM | `-ngl 99 -ot "blk\.(16..47)\.ffn_.*_exps\.=CPU" -mmp 0` |
| **C** | pure CPU | `-ngl 0 -mmp 0` |

Each measured for **both** phases in one session: `pp2048` at `-ub 2048` (the setting v1.13.0 now
ships for host-resident weights) and `tg128`. Warm-up discarded, r=3, GPU memory and temperature
logged.

## Stakes

- **P-1 (the inversion — the claim that matters).** `argmax(prefill) ≠ argmax(decode)` over
  {A, B}. Refuted if the same configuration wins both.
- **P-2 (the mechanism).** **A beats B on `pp2048`**, because B spends VRAM on experts and that
  is the VRAM the `-ub 2048` compute buffer needs. This is the directional claim; P-1 could hold
  for some other reason and P-2 would still be wrong.
- **P-3 (decode ordering, control).** **B beats A on `tg128`**. Already measured twice
  (18.84–18.89 vs 18.46–17.44), so this is a harness check: if it fails, the session is
  contaminated and P-1/P-2 are void.
- **P-4 (magnitude worth acting on).** The phase-optimal pair beats the best single placement by
  **≥15%** on whichever phase the single placement sacrifices. Below that, a two-command
  recommendation is not worth the complexity it adds for users.

## Refuted if

P-1 fails — one placement wins both phases, and a single command is the right answer after all.
That is a perfectly good outcome and would close the dimension rather than open it.

## What ships if it holds

Not a law change. `plan` would report the phase-optimal placement **per phase** and say which one
its single `run it:` command optimises, so a user with long prompts and a user doing chat are not
handed the same flags with the same confidence. llama.cpp can be started with either; nothing
here requires switching placements mid-session (that would need slot save/restore, which is a
separate and much larger question — Law 5 H7).

**Explicitly not claimed:** that we can serve both phases optimally *at once*. This measures
whether the two optima differ, nothing more.
