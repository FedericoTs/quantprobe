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

---

## Scored (2026-07-28, log: `weights/data/prereg42_draft_budget.log`)

**Verdict: P-1 HIT, P-2 MISS — the KILL RULE FIRES, P-3 answered by monotonicity, P-4 divergence
reported. The draft-model path is closed for good, and #28's conclusion was RIGHT for a reason it
did not state.**

| `--spec-draft-n-max` | novel code | acceptance | novel prose | acceptance |
|---|---|---|---|---|
| base (no speculation) | **19.39** | — | **21.61** | — |
| D3 (llama.cpp default) | 15.95 | 75.2% | 9.52 | 35.9% |
| D8 | 14.67 | 56.8% | 6.48 | 17.6% |
| D16 | 10.58 | 34.9% | 3.91 | 8.7% |

- **P-1 (D3 below base): HIT.** #28's net-negative result reproduces exactly.
- **P-2 (some budget ≥1.3× base): MISS, decisively and in the wrong direction.** Every budget
  increase makes it **worse**, monotonically, on both tasks. The kill rule fires.
- **P-3 (interior optimum): answered without running D32.** The decline is monotone across three
  budgets with a large effect and a clear mechanism; D32 can only be worse. Recorded as not-run
  rather than implied.
- **P-4 (quality): divergence present and reported.** D3/D8 novel-code share `f0212e6eca`, D16
  gives `1c2a7a785d`, base gives `7d826928f6` — consistent with #41's finding that a verify pass
  perturbs numerics at temp 0. Not hidden.

### Why the same fix that saved n-gram destroys the draft model

This is the finding, and it explains both results at once:

| drafter | what the draft IS | acceptance vs draft length |
|---|---|---|
| **n-gram** | a literal copy of text that already occurred | **~flat** — 66–68% even at 384 tokens |
| **draft model** | an autoregressive guess by a 50×-smaller model | **collapses geometrically** — 75% → 57% → 35% (code); 36% → 18% → 9% (prose) |

An n-gram draft is *verified-correct by construction*; extending it costs nothing and returns more
tokens per weight read, which is why `size-m 384` won. A model draft compounds its own divergence:
each extra token is exponentially less likely to survive **and costs a full draft-model forward**.
Raising the budget buys wasted work at an accelerating rate.

So #28's "net negative" was correct, and this pre-registration's suspicion — that it was an
artifact of the default budget, as `size-m 48` had been — is **refuted**. The two flags look alike
and behave oppositely, because the drafters differ in kind, not degree. Worth stating plainly: the
lesson from #36 ("the default budget was crippling it") did NOT generalise, and assuming it would
was the reason this test was worth an hour.

### The honest bottom line on novel generation

Every drafting mechanism available on this box is now measured on novel content:

| mechanism | result |
|---|---|
| n-gram (any tuning) | 0 drafts — nothing in context to copy (#41) |
| n-gram chaining (`ngram-mod`) | −21% even on copy work (#38) |
| MTP | unmeasurable, three attempts (#30) |
| **draft model (any budget)** | **−18% to −82%** (this) |

**Novel generation on this hardware has no software lever left.** It sits at ~21 tok/s against a
measured 41.1 wall, and the remaining 1.9× is the kernel/sync residue already priced at ≤1.85×
(#27/#31/#32/#33). 100 tok/s on novel content is a **hardware** statement on this box:

- **~2.4× more effective bandwidth** — the wall is linear in it (DDR5-6000 dual-channel is ~3.5×
  this box's measured 26.1 GB/s), or
- **~2.4× fewer active bytes per token** — a MoE with ~1.4B active parameters instead of
  Qwen3-30B-A3B's 3.3B, at the same total size and quantization.

Both are real, purchasable, and predictable with the tool. Neither is code.

**Wired into:** `findings/REGISTER.json:D-09` (corrected: right conclusion, better reason) ·
`D-13` (the acceptance-decay law) · `quantprobe/plan.py:speculation_advice`.
