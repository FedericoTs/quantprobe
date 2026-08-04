# X-1 — verify-width surfing: does the batch-9 kernel cliff pay inside single-stream speculation?

**Date staked:** 2026-08-04, before any draft-length sweep on this box.
**Born from:** U-38/U-39 measured a per-STEP wall-clock valley on this card: a fused decode step
over 8 positions costs 148 ms, over 9 positions **84 ms** — nine tokens in less time than
eight, because llama.cpp's mat-vec kernel serves widths ≤8 and the mat-mat path serves ≥9.

**The idea nobody has tested:** speculative decoding's VERIFY step is exactly such a fused step
over draft+1 positions. Every guide tunes draft length by acceptance arithmetic alone. If the
valley applies to the verify pass, then on this card a draft of 5–7 tokens verifies in the SLOW
kernel regime and a draft of 8+ verifies in the FAST one — so the optimal draft length is set by
a kernel boundary, not by acceptance, and drafts of 5–7 are strictly dominated. "Use the batch
to speed up the single stream" — the batch is the verify pass, and we surf it onto the fast
kernel.

**Why it might NOT work (stated before measuring):** the verify pass is one sequence of m+1
positions in one ubatch; batched-bench measured N separate sequences of 1 position each. Same
matmul width, different attention pattern. If the kernel choice keys on something other than
row count, the cliff will not transfer, and this dies.

## Protocol

Qwen2.5-7B-Instruct Q4_K_M all-in-VRAM (`-ngl 99`), the U-38 anchor model. One session, quiet
box. Copy-heavy task so the ngram drafter actually drafts: a ~500-token code file with the
instruction to reproduce it with one identifier renamed, `--temp 0`, `-n 512`.

Sweep `--spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m M` for
M ∈ {4, 6, 7, 8, 9, 12, 16, 24}, plus spec-off baseline. Record decode tok/s and the printed
draft/accept counts per arm.

## Staked predictions

- **P1 (the cliff transfers):** decode tok/s jumps ≥1.25× between M=7 and M=8 or M=9 (verify
  width crossing 8→9), while the acceptance RATE across that same step changes ≤10 points —
  i.e. the jump is kernel, not acceptance.
- **P2 (control):** consecutive ratios below the boundary (M=4→6→7) all ≤1.15 — no jump where
  no boundary is crossed.
- **KR-1 (confound guard):** if acceptance moves >10 points across the claimed jump, the arm is
  UNSCOREABLE for mechanism — no conclusion may be drawn in either direction.
- **KR-2 (refutation):** if no consecutive-M ratio anywhere in {4..24} reaches 1.2, the cliff
  does not transfer to the verify pass and X-1 is dead. That is a real possibility and this
  file exists so it can die in public.

## If P1 holds

The planner's speculation advice gains a hardware-specific rule it can state exactly: *on
pre-Ampere cards, never draft 5–7; draft ≥8 or don't bother* — and `plan` already knows the
card generation. A follow-up would sweep the boundary on Ampere+ via `bench --contribute`.
