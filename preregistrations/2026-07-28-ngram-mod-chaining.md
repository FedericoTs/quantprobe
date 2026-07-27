# Pre-registration #38: ngram-mod chains across sites — can it beat the single-span ceiling?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The source-level reason #37 plateaued

An adversarially-verified source study of `llama.cpp @ f113e02` found the structural limit:

- **`ngram-simple`** (`ngram-map.cpp:96`) drafts `copy_max = min(size_m, cur_len - match_pos - n)`
  — a **contiguous copy from the single most recent match**. Its draft can never be longer than
  the text that happens to follow that one match. That is the ceiling we hit at 1121 drafted
  tokens, and no `size-m` or `size-n` value can lift it.
- **`ngram-mod`** (`speculative.cpp:1860-1872`) instead **chains autoregressively**: each lookup
  hashes a window sliding over already-drafted tokens, so a draft can walk across *different*
  source sites. It is bounded by `n_max`, not by any one span.

And its defaults explain #30's null: **`n_min = 48` is an all-or-nothing floor** — a chain shorter
than 48 emits ZERO tokens rather than a short draft (`speculative.cpp:1863-1866`).

## The hazard this pre-registration must control

The same study confirms, at source level, the artifact that fooled me in #30: mod's 16 MB table is
a **single member shared by all sequences**, and `begin()` adds the prompt **without clearing**
(`speculative.cpp:1811-1813`), so it accumulates across requests for the process lifetime. A
second identical request therefore replays a table the first one built.

**Protocol consequence, binding:** every mod arm reports **request 1 and request 2 separately**,
and only **request 1 on a freshly started server** counts as the result. Averaging them would
reproduce exactly the error #30 caught.

## Arms (edit task, split, f16 KV, temp 0; fresh server per arm; r1 and r2 reported separately)

| arm | flags |
|---|---|
| ref | `ngram-simple`, m 384, n 4 — the #37 winner (108.41) |
| MOD-def | `ngram-mod` at defaults (n_min 48, n_max 64, n_match 24) |
| MOD-tuned | `ngram-mod --spec-ngram-mod-n-min 1 --spec-ngram-mod-n-max 512 --spec-ngram-mod-n-match 12` |
| MOD-long | as tuned, `--spec-ngram-mod-n-max 1024` (the CLI cap) |

## Stakes

- **P-1 (the control — the source reading is right).** MOD-def drafts **< 200 tokens** on request 1,
  reproducing #30's null, because `n_min=48` suppresses everything shorter.
- **P-2 (the headline).** Some tuned mod arm beats **119 tok/s** (≥10% over 108.41) **on request 1
  of a fresh server**.
- **P-3 (the hazard is real and must be disclosed).** For a tuned mod arm, request 2 is **≥20%
  faster** than request 1 — the cross-request table accumulation, measured rather than inferred.
  If this fails, the persistence exists in source but does not reach throughput, and #30's artifact
  needs a different explanation.
- **P-4 (identity).** Every arm byte-identical to ref. Speculation is output-preserving; a
  divergence here is a llama.cpp bug worth reporting upstream.

## KILL RULE

**If no tuned mod arm beats 108.41 on request 1, chaining is closed** and `ngram-simple` at
`m 384 / n 4` is the shipped optimum. The remaining ~4× to the verify ceiling then belongs to a
fundamentally better drafter, and #28 already measured that road (a 0.6B draft model is net
negative here).

## What ships

Any winner ships in `speculation_advice` with its measured numbers, its copy-regime scope, AND —
non-negotiable — the cross-request accumulation caveat, since a user benchmarking mod twice will
otherwise measure a number they cannot reproduce on fresh input.
