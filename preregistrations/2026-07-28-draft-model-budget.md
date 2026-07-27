# Pre-registration #42: the draft-model arm was measured at a CRIPPLED budget — the novel-content path, retested

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The error this suspects in our own record

Novel generation sits at 21 tok/s against a 41.1 wall, and the only mechanism that can pass that
wall is drafting. N-gram drafting is closed for novel content (#41: it has nothing to copy). That
leaves a **draft model**, which predicts without copying — and #28 measured it **net negative
(0.72×)** and closed the line.

**#28 ran it at llama.cpp's default `--spec-draft-n-max 3`.**

That is precisely the trap #36 found for n-gram: the default `size-m 48` captured only 2.4× of an
available 4.7×, because the draft budget — not the drafter — was binding. We then made the same
mistake in the same session on a different flag and closed a line on it. #28 also recorded **81%
draft acceptance**, which is a drafter working well while being allowed only 3 tokens per round.

The arithmetic that motivates this: at 81% per-token acceptance a 3-token budget yields ~2.2
accepted tokens per verify round; a 16-token budget yields ~4.6. Since the cost unit is the
**verify round** (#36/#37), that is roughly 2× the tokens for one target-model weight read, against
a draft cost that is ~50× cheaper per forward (0.6B vs 30B active).

## Arms — target `Qwen3-30B-A3B Q2_K`, draft `Qwen3-0.6B-Q8_0`, split widened to `blk.(12..47)` to
make VRAM room for the draft model (as in #28). Novel-code and novel-prose tasks. Fresh server,
request 1 only (#38). temp 0.

| arm | `--spec-draft-n-max` |
|---|---|
| base | no speculation |
| D3 | 3 (llama.cpp default — reproduces #28's 0.72×) |
| D8 | 8 |
| D16 | 16 |
| D32 | 32 |

## Stakes

- **P-1 (the control reproduces the closed result).** D3 is **below base** on novel tasks,
  reproducing #28's net-negative finding. If D3 is already positive, #28 was wrong for a different
  reason and the whole comparison needs re-basing.
- **P-2 (THE CLAIM — the budget was binding, not the drafter).** Some arm D8/D16/D32 is
  **≥ 1.3× base** on novel code. This is the first mechanism this project would have that
  accelerates FRESH generation.
- **P-3 (an interior optimum).** D32 is worse than the best arm — beyond some length the draft
  model's own forwards and the wasted rejected tokens cost more than the saved rounds. A
  monotone-rising result means the sweep must extend.
- **P-4 (quality).** Output is compared against base at matched request index. Per #41 this is
  NOT expected to be bit-identical — a verify pass perturbs numerics — so the claim under test is
  distributional equivalence, and any divergence is reported, not hidden.

## KILL RULE

**If no budget beats base by more than 10%, the draft-model path is closed for good** on this
hardware: the 0.6B's own forwards plus the VRAM it displaces exceed what verification saves at
every budget, and novel generation is bounded by the 41.1 wall with no software lever left. That
would make the honest answer to "100 tok/s on novel content" a hardware answer, and this project
would say so.

## What ships

If P-2 holds: a second speculation recipe for novel/chat workloads, with its measured numbers, its
VRAM cost, and its quality caveat — and a correction to D-09, which closed this line on
defaults-scoped evidence exactly as D-10 did.
