# Pre-registration #41: does the TUNED drafter reach novel content? (the untested combination)

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The gap this closes

#28 and #30 measured novel generation at **0% draft acceptance** and closed that line by kill
rule. Both were run at llama.cpp's **defaults: `size-n 12`, `size-m 48`**. #37 then found that
`size-n 4` — a three-times shorter required match — is worth +21% on copy-regime work.

**The combination "tuned drafter × novel content" has never been measured.** A 4-token lookup
matches vastly more often than a 12-token one, and novel prose/code still contains repeated
short spans (`    return `, `self.`, ` the `, indentation runs). D-10 may be an artifact of the
defaults rather than a property of novel generation, and if so this project closed a line early.

## Arms (llama-server, split, Q2_K, temp 0, fresh server, request 1 only per #38)

Two novel tasks — nothing to copy from context:
- **NC** novel code: "write a Python function `schedule(jobs)` … maximum-weight non-overlapping
  jobs by dynamic programming, with a docstring."
- **NP** novel prose: "explain in plain English why reading a file from disk is slower than from
  memory, about 200 words."

| arm | flags |
|---|---|
| base | no speculation |
| def | `ngram-simple` at defaults (m 48, n 12) — reproduces the #28/#30 null |
| tuned | `ngram-simple m 384 n 4` — the untested cell |

## Stakes

- **P-1 (the tuned drafter fires at all on novel content).** `tuned` drafts **> 50 tokens** on at
  least one novel task, against `def`'s ~0. This is the claim that D-10 was defaults-scoped.
- **P-2 (but it does not pay).** `tuned` is within **±10%** of `base` on both novel tasks. I
  expect firing without profit: short matches produce short runs, and #37 showed short runs are
  what makes n=2 lose. **If this is exceeded upward, D-10 must be reopened and novel generation
  is not closed after all.**
- **P-3 (identity).** All arms byte-identical at matched request index.

## Refuted / reopened if

**P-2 exceeded upward (tuned > base by >10% on a novel task).** Then novel-generation speculation
is real at tuned settings, D-10 was closed on defaults-scoped evidence, and the headline extends
from "copy-regime only" to "all content, with a smaller multiplier on novel" — a materially
different claim for users, and one this project would have to publish as a correction of its own
prior conclusion.

## What ships

Either the confirmation that D-10 holds at tuned settings (tightening a claim we already make), or
its reopening with measured numbers. The plan output's "novel generation gains nothing" sentence
is downstream of this either way.

---

## Scored (2026-07-28, log: `weights/data/prereg41_novel_tuned.log`)

**Verdict: P-1 MISS (the tuned drafter fires LESS, not more — and the reason is a trap we were
shipping), P-2 HIT, P-3 VIOLATED reproducibly — which corrects a claim I have repeated all day.**

| arm | task | tok/s | drafted | accepted | output sha |
|---|---|---|---|---|---|
| base | novel code | 21.02 | 0 | 0 | `5bd34c5e2f` |
| base | novel prose | 21.16 | 0 | 0 | `3c0e346e05` |
| default (m48 n12) | novel code | 20.00 | **38** | **0** | **`1c2a7a785d`** |
| default | novel prose | 20.80 | 0 | 0 | `3c0e346e05` |
| **tuned (m384 n4)** | novel code | 21.13 | **0** | 0 | `5bd34c5e2f` |
| **tuned** | novel prose | 20.95 | **0** | 0 | `3c0e346e05` |

- **P-1 (tuned drafts >50 on novel): MISS — it drafts ZERO**, fewer than the default's 38.
- **P-2 (within ±10% of base): HIT.** 21.13 vs 21.02; 20.95 vs 21.16. **D-10 holds.**
- **P-3 (identity): VIOLATED, and it reproduces exactly** (`1c2a7a785d` on both runs).

### Finding 1 — the tuned flags SILENTLY DISABLE speculation on short contexts

`ngram-simple` requires `cur_len > size_n + size_m + 1` before it will draft at all
(`ngram-map.cpp:65`). That threshold is a function of the flags we ship:

| setting | context needed before it can EVER draft |
|---|---|
| `m 48, n 12` (llama.cpp default) | **61 tokens** |
| **`m 384, n 4` (what we ship)** | **389 tokens** |

Our edit task carries a ~950-token prompt, so it clears the bar instantly and we never saw this.
A user with a short prompt gets **nothing at all** from our recommended flags — strictly worse
than the default, which at least tries. This is a shipped-advice defect, found only because the
question "does it work on novel content" forced a short-prompt test.

### Finding 2 — "byte-identical by construction" is FALSE, and I said it repeatedly

The default drafter on novel code drafted 38 tokens, **accepted zero**, and still produced
different output from no-speculation — reproducibly. Zero acceptance cannot change the sampled
tokens through the acceptance path, so the divergence comes from the **verification forward pass
itself**: verifying a draft evaluates several positions in one batched pass, and batched reductions
do not produce bit-identical logits to single-token decode. At temperature 0 a hair's difference
flips an argmax.

So the correct statement is: **speculative decoding is output-preserving in exact arithmetic, not
in floating point.** It was byte-identical in every copy-regime comparison this project ran — many
of them — and that is real evidence, but it is not a guarantee, and I stated it as one. Corrected
in the register and in the shipped advice.

### Answering the three questions this pre-registration was asked to settle

1. **Does it work on novel content?** No. Confirmed at tuned settings, which closes the gap D-10
   left open — and the tuned flags are actively *worse* than defaults there.
2. **Is it the last drop?** For copy-regime at ~3-bit, the drafter axis is closed by #38's kill
   rule and this one. What remains unmeasured is U-10 (speculation × batching).
3. **Parity quality?** Parity in *distribution* and in every copy-regime test, but not
   bit-guaranteed — see Finding 2.

**Wired into:** `quantprobe/plan.py:speculation_advice` (the 389-token context floor + the
corrected identity claim) · `findings/REGISTER.json:V-04`, `D-10`.
