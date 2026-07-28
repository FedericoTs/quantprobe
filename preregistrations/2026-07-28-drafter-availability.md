# Pre-registration #37: raising draft AVAILABILITY — the constraint #36 left standing

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The constraint, precisely

#36 took speculation from 49.80 to **90.33 tok/s** by raising the draft budget, and identified
where it stops: at `size-m` ≥ 192 the drafted/accepted counts FREEZE (1100/735) — the n-gram
store has no longer matching spans. Meanwhile the batched verify forward sustains ~405–445 tok/s
(#26). So we sit at ~1/4.5 of speculation's own ceiling, bounded by **draft availability**.

Two untried levers attack availability directly:

- **`--spec-ngram-simple-size-n`** (lookup length, default 12): the drafter must match n tokens of
  recent context before it will propose anything. A *shorter* lookup matches more often, so more
  decode steps get a draft at all. It should also produce *worse* drafts — and #36 established
  that acceptance rate is the wrong target, so worse-but-more may still win.
- **Drafter stacking**: `--spec-type` takes a comma-separated list. If one drafter running dry is
  the constraint, a second may fire on the steps the first misses.

## Arms (all at the #36 winner `size-m 384`; edit task, split, f16 KV, temp 0, r=2)

| arm | setting |
|---|---|
| ref | `ngram-simple`, size-n 12 (the #36 winner: 90.33) |
| N6 | size-n 6 |
| N4 | size-n 4 |
| N20 | size-n 20 (the control — availability should FALL) |
| ST | `--spec-type ngram-simple,ngram-cache` |

## Stakes

- **P-1 (availability is the constraint, and n controls it).** N6 drafts **≥30% more tokens**
  than ref (from 1100). Rejected drafts being cheap, more drafting should mean fewer verify
  rounds for the same output.
- **P-2 (the headline).** Some arm beats **104 tok/s** (≥15% over 90.33).
- **P-3 (the control fires the right way).** N20 drafts **fewer** tokens than ref. If a *longer*
  required match somehow increases availability, I do not understand this drafter and P-1/P-2 are
  uninterpretable.
- **P-4 (identity holds).** Every arm byte-identical to ref. Speculation is output-preserving by
  construction; if changing the drafter changes the output, that is a bug in llama.cpp worth
  reporting, and every speculation number in this project needs re-examination.

## KILL RULE — stated before measuring

**If nothing beats 90.33 by more than 5%, drafter tuning is closed.** The n-gram store is then at
its practical ceiling for this workload, the remaining 4.5× to the verify ceiling belongs to a
fundamentally better drafter (a model, an indexed corpus), and #28's finding that a 0.6B draft
model is net-negative here says that road is expensive. We would stop and ship the m=384 result.

## What ships

Any winning combination goes into `speculation_advice` with its measured numbers and its
copy-regime scope, exactly like the m=384 flag. No law changes; these are flags.

---

## Scored (2026-07-28, log: `weights/data/prereg37_availability.log`)

**Verdict: P-1 MISS (availability is NOT the mechanism), P-2 HIT at 108.41, P-3 HIT, P-4 HIT.
The headline lands, and the reason I gave for it was wrong again.**

All arms at `--spec-ngram-simple-size-m 384`, edit task, split placement, r=2:

| arm | tok/s | acceptance | drafted / accepted | sha |
|---|---|---|---|---|
| n=20 (control) | 77.61 | 65.8% | 1068 / 703 | `28a5c1e1c014` |
| n=12 (the #36 winner) | 89.82 / 90.67 | 66.8% | 1100 / 735 | `28a5c1e1c014` |
| n=6 | 105.52 | 67.9% | 1117 / 759 | `28a5c1e1c014` |
| **n=4** | **108.41 / 107.65** | 68.4% | 1121 / 767 | `28a5c1e1c014` |
| n=2 | 81.98 | 52.2% | 1486 / 775 | `28a5c1e1c014` |
| n=4 + `ngram-cache` stacked | 97.16 | 66.8% | 1148 / 767 | `28a5c1e1c014` |

Position-controlled: reference re-run LAST gives 90.67 (vs 89.82 first), winner re-run last gives
107.65 (vs 108.41). Not ordering. **108 tok/s, byte-identical output, two flags.**

- **P-1 (n=6 drafts ≥30% more): MISS badly.** +1.5% (1100 → 1117). Draft *availability* barely
  moved while throughput rose 21%. My stated mechanism — "shorter lookup fires more often" — is
  refuted by its own counter.
- **P-2 (some arm > 104 tok/s): HIT.** 108.41.
- **P-3 (n=20 control drafts fewer): HIT.** 1068 < 1100, and it is the slowest non-degenerate arm.
- **P-4 (identity everywhere): HIT.** One sha across every arm, including the degenerate n=2 and
  the stacked drafter.

### What the counters actually say

Across n = 20 → 12 → 6 → 4, drafted tokens move 1068 → 1121 (+5%) while throughput moves
77.6 → 108.4 (+40%). Tokens drafted is therefore **not** the variable. What must be changing is
the number of VERIFY ROUNDS — the same drafted tokens delivered in fewer, longer runs, each round
costing one full weight read. This is the identical mechanism #36 found (throughput rose while
drafted/accepted froze), now confirmed on an independent axis. **The unit of cost in speculative
decoding on this hardware is the verify round, not the token, and neither `draft_n` nor the
acceptance rate measures it.**

n=2 is the counter-example that proves it: it drafts the MOST (1486, +36%) and accepts the most
(775), yet runs 25% slower than n=4 — because a 2-token lookup matches noise, producing many
short, quickly-rejected runs. Over-drafting costs rounds.

Stacking is a real regression (97.16 vs 108.41): running a second drafter costs more than the 27
extra drafted tokens it contributes.

### Where the ceiling stands now

| | tok/s |
|---|---|
| raw decode, measured | 22.25 |
| raw-decode wall (physics, #27) | 41.1 |
| speculation, llama.cpp defaults | 49.80 |
| speculation, tuned (#36) | 90.33 |
| **speculation, tuned (this) — `size-m 384 size-n 4`** | **108.41** |
| batched-verify ceiling (#26) | ~405–445 |

**2.6× the physical raw-decode wall**, 4.9× the shipped default, on a 2016 GPU, with
byte-identical output. Still ~4× below the verify ceiling — and the next honest question is
whether *rounds* can be instrumented directly rather than inferred, because every tuning result
in #36 and #37 is really about that one hidden number.

**Wired into:** `quantprobe/plan.py:speculation_advice` · `findings/REGISTER.json:V-04`.
