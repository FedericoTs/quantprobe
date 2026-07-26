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

---

## Scored (2026-07-27, log: `weights/data/prereg20_phase_matrix.log`)

**Verdict: P-1, P-2, P-3, P-4 all HIT. And the measurement caught a regression I had shipped
five hours earlier.**

`Qwen3-30B-A3B-Q2_K`, one session, warm-up discarded, r=3:

| placement | pp2048 @ub512 | pp2048 @ub2048 | tg128 @ub2048 |
|---|---|---|---|
| **A** all experts → CPU | 199.90 ± 1.42 | **349.59 ± 1.78** | 18.54 ± 0.16 |
| **B** split, K=16 → VRAM | **279.07 ± 0.86** | 161.87 ± 0.24 | **20.16 ± 0.18** |

- **P-1 (argmax differs): HIT.** A wins prefill by **2.16×**; B wins decode by 1.09×. One command
  cannot serve both.
- **P-2 (A beats B on prefill): HIT** at the shipped `-ub 2048` — and the mechanism is confirmed
  by the column next to it, which is the more interesting result. See below.
- **P-3 (B beats A on decode): HIT.** 20.16 vs 18.54, consistent with two prior sessions.
- **P-4 (≥15% on the sacrificed phase): HIT, by a wide margin.** Choosing B — the decode
  optimum — costs **116% of the available prefill**. Choosing A costs 8% of decode. The trade is
  wildly asymmetric.

### The deeper finding: placement and batch are not independent

At the default ubatch, **B wins prefill** (279 vs 200) — the original partial-offload result. At
`-ub 2048`, **A wins** (350 vs 162). *Adding a lever inverted which placement is fastest.*

The reason is a resource conflict, not a coincidence: the split placement exists to fill spare
VRAM with experts, and that is precisely the VRAM a larger compute buffer needs. B therefore
**loses 42%** when given the bigger ubatch — the same sign and nearly the same magnitude as the
fully-VRAM-resident control in pre-registration #19 (−39%).

So the search cannot treat placement and batch as separable axes. They compete for one budget.

### The regression this caught in v1.13.0

v1.13.0, shipped five hours before this measurement, gated `-b/-ub` on *"is anything
host-resident"*. The split placement's label is `split experts: N%->VRAM, rest->RAM` — which
satisfies that test. So the tool was recommending, on its own default placement for the flagship
MoE, a flag now measured to cost **42% of prompt processing** there.

The gate now excludes the split explicitly, with the measurement in the comment. Fixed in v1.13.1.

**This is the argument for measuring a lever on every placement rather than the one it was
discovered on.** #19 measured `-ub` on A and on a VRAM-resident control, concluded correctly, and
still shipped a wrong gate — because the split is neither of those two cases, and nobody had
looked. A double dissociation proves a mechanism; it does not enumerate a decision surface.

**Wired into:** `quantprobe/plan.py:ubatch_flags` · `quantprobe/plan.py:phase_advice` · `tests/smoke.py:t_ubatch_only_when_host_resident`
