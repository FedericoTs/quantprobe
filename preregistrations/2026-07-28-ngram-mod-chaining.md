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

---

## Scored (2026-07-28, log: `weights/data/prereg38_ngram_mod.log`)

**Verdict: P-1 MISS (the floor suppresses less than I claimed), P-2 MISS — the KILL RULE FIRES,
P-3 REFUTED BY DIRECTION (accumulation makes mod WORSE, not better), P-4 HIT once request index
is matched. `ngram-simple` at `m 384 / n 4` stands as the shipped optimum.**

| arm | req | tok/s | drafted / accepted | acceptance |
|---|---|---|---|---|
| **ref `ngram-simple` m384 n4** | **1** | **109.52** | 559 / 382 | 68.3% |
| ref | 2 | 115.45 | 562 / 385 | 68.5% |
| `ngram-mod` defaults | 1 | 57.41 | 341 / 334 | 97.9% |
| `ngram-mod` defaults | 2 | 54.53 | 408 / 340 | 83.3% |
| mod tuned (n_min 1, n_max 512, n_match 12) | 1 | 86.42 | 556 / 365 | 65.6% |
| mod tuned | 2 | **71.75** | 976 / 384 | 39.3% |
| mod tuned, n_max 1024 | 1 | 86.63 | 556 / 365 | 65.6% |
| mod tuned, n_max 1024 | 2 | **64.65** | 963 / 371 | 38.5% |

- **P-1 (MOD-def drafts <200): MISS.** 341. The `n_min=48` floor suppresses a lot (341 vs ref's
  559) but not nearly everything, and what survives is drafted at **97.9% acceptance** — the floor
  admits only near-certain chains. Correct in direction, wrong in magnitude.
- **P-2 (a tuned mod arm > 119 tok/s): MISS.** Best mod = 86.63 on request 1, **21% BELOW** the
  `ngram-simple` reference. Raising `n_max` from 512 to 1024 changes nothing (556 drafted in
  both) — the chain terminates on a hash miss long before either cap.
- **P-3 (request 2 ≥20% faster): REFUTED BY DIRECTION, and this is the finding.** Request 2 is
  **17–25% SLOWER** (86.42→71.75; 86.63→64.65) while drafting **75% MORE** (556→976) at collapsed
  acceptance (65.6%→39.3%). The accumulated table does not help — it **poisons**. Its 4.2 M slots
  hold one successor each with overwrite-on-write and no key stored, so a second pass adds
  collisions that produce confidently-wrong drafts. More drafting, less accepted, slower.
- **P-4 (identity): HIT, at matched request index.** Every arm's request 1 is `65985276565a` and
  every arm's request 2 is `28a5c1e1c014`. The req1/req2 difference is prompt-cache state
  (`cache_prompt` default true; reused KV is not bit-identical to freshly computed KV), NOT a
  speculation effect — speculation is output-preserving across all four drafter configurations.

### Why chaining loses, when the source says it should have more freedom

The study was right that mod can walk across source sites while simple copies one span — but
freedom is not the binding constraint. Mod's table stores **one successor per hash slot, no key,
overwrite-on-write**: every chaining step is a lookup that can silently be a collision, and the
chain is autoregressive, so a single wrong step corrupts everything after it. Simple's contiguous
copy is *verified-correct by construction* — it reproduces text that actually occurred. On a task
whose output IS a near-copy of the context, the dumber drafter is strictly better, and the
smarter one's error compounds.

### The kill rule fires — and #38's control retires a hazard

Drafter tuning is **closed**. `ngram-simple --spec-ngram-simple-size-m 384 --spec-ngram-simple-size-n 4`
at ~108–115 tok/s is the shipped optimum, and the remaining ~4× to the verify ceiling belongs to a
fundamentally better drafter, which #28 already priced as net-negative on this hardware.

Two things worth keeping from the failure:

1. **The #30 replay artifact is now explained at source level AND measured in the opposite
   direction.** In #30 mod's cross-request table made a repeated identical request *look* fast; on
   a real edit task the same accumulation makes it *slower*. Either way the number is not
   reproducible on fresh input, which is exactly why request 1 of a fresh server is the only
   figure this project will quote for mod.
2. **Our previous identity checks were sound but under-specified.** Earlier harnesses reported
   only the last request's sha, so they compared req2 to req2 — valid, but blind to the fact that
   req1 and req2 differ at all. That difference is prompt-cache reuse, is present with and without
   speculation, and is now recorded as a scope note on every temp-0 identity claim in this project.

**Wired into:** `findings/REGISTER.json:D-12` (chaining drafter refuted) · `V-04` (identity scope
note) · no code change: the shipped flags already carry the winner.
