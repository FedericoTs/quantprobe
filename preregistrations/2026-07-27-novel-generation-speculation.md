# Pre-registration #30: can NOVEL generation be accelerated on this box?

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The user's question, taken at face value

The 50-59 tok/s result (#28) fires only when output reuses context. The common case — fresh
reasoning, fresh prose, fresh code — sits at ~21-22 tok/s against a 41.1 tok/s raw wall. What can
move THAT number? Novel generation cannot copy, so it needs a PREDICTOR: something that guesses
the next tokens from the model's own state rather than from spans.

## Why MTP-on-the-split is the live hypothesis

Law 6's 2×2 (2026-07-26) measured MTP at **0.76×** on all-experts-offloaded MoE and concluded the
expert-union tax catches every speculation mechanism there. But that 2×2 never measured the SPLIT
placement — and #28 just showed the tax is placement-dependent: ngram went from +3% (all-offloaded,
Law 6 arm) to **2.41×** on the split. If the tax softens the same way for MTP, the −24% could flip
positive — and MTP drafts from the model's own trunk, so it works on novel text where ngram gets 0%.

MTP model: `Qwen3.6-35B-A3B-APEX-MTP-I-Nano` (41 layers + 1 nextn head, 10.88 GB), split at
layers 14–40 (~66% of experts to CPU, mirroring the flagship's ratio). Compared against ITS OWN
no-spec baseline — never against the flagship, which is a different model.

## Arms (novel task: write a fresh function; chat template, temp 0, coherent output verified, r=2)

| arm | model | spec |
|---|---|---|
| F-B | flagship 30B, split | none (have: ~21.6) |
| F-V | flagship 30B, split | `ngram-mod`, `ngram-cache`, `ngram-map-k4v` — the statistical variants |
| M-B | APEX-MTP 35B, split | none |
| M-S | APEX-MTP 35B, split | `draft-mtp` |

## Stakes

- **P-1 (statistical ngram variants pay something on novel).** Best F-V arm ≥ **1.10×** F-B.
  Function words and syntax boilerplate are predictable without copying; if no variant clears 10%,
  span-free drafting from statistics is dead here.
- **P-2 (the placement rescue generalises to MTP).** M-S ≥ **1.15×** M-B on the split. This is the
  arm that matters: Law 6's 0.76× was measured one placement away from where we actually ship.
- **P-3 (trying costs nothing).** No arm loses more than 5% vs its own baseline — a lever that
  must be toggled per-workload is advice nobody follows.

## KILL RULE — stated before measuring

If P-1 AND P-2 both miss, **novel-generation speculation on this box is CLOSED as a dead end**,
and the honest answer to the user becomes: the novel path is bounded by the 41.1 wall, reachable
only through kernel/sync engineering (≤1.85× by #27, realistically much less), plus the two
already-shipped levers that help long novel sessions specifically — q8_0 KV at depth (+37% at
16k, #25) and prefix caching for the ingest side (29×, #29). No third measurement will be chased
past this one without new external evidence.

## What ships

If P-2 hits: the speculation advice gains an MTP branch for MTP-capable models on the split, with
the measured number. If the kill rule fires: the advice states the novel-generation bound plainly.

---

## Scored (2026-07-27, log: `weights/data/prereg30_novel_speculation.log`)

**Verdict: P-1 MISS (after diagnosing my own artifact), P-2 MISS (unmeasurable — worse than a
miss), P-3 MISS for one variant. THE KILL RULE FIRES: novel-generation speculation on this box is
CLOSED.**

### P-1, and the artifact that nearly became a headline

The first sweep showed `ngram-mod` at **50.38 tok/s on a novel task, 100% acceptance** — an
apparent 2.38× on fresh generation, which would have overturned the copy-vs-novel scoping a few
hours after it was established. It is not real, and the diagnosis matters more than the number:

**`ngram-mod` keeps its n-gram store across requests.** The r=2 protocol sent the same prompt
twice; run 1 generated at baseline speed and populated the store; run 2, at temp 0, regenerated
identical text — which was now a COPY of run 1. The "novel" speedup was the average of one honest
run and one replay. Re-measured single-shot on a fresh server, seed pinned: **1.03×, zero drafts
fired, output byte-identical to baseline.** The full honest table:

| single-shot, fresh server | novel code | novel prose | edit |
|---|---|---|---|
| baseline | 21.11 | 21.52 | 20.52 |
| `ngram-mod` | 21.67 (1.03×, no drafts) | 21.06 (0.98×, no drafts) | **51.28 (2.50×, 97% acc.)** |
| `ngram-cache` | 19.75 (0.93×, 19% acc.) | — | — |
| `ngram-map-k4v` | 21.10 (1.00×) | — | — |

This is the third time this project measured a spectacular number and found it was the harness
(the `-ub 2048` frontier cell, #28's repetition loop, now this). The common thread: **repetition
is invisible in a throughput number and obvious in the output.** Reading what the model actually
wrote is now, formally, part of the protocol.

- **P-1 (best variant ≥1.10× on novel): MISS.** 1.03× best, no drafts.
- **P-2 (MTP on the split ≥1.15×): MISS, worse — unmeasurable.** Three attempts, `draft-mtp` on
  the APEX model never returned even 96 tokens inside 10 minutes (<0.2 tok/s or hung), on the
  same split where its own no-spec baseline runs 18.19. Law 6's 0.76× was optimistic for this
  build. Also collected in passing: the M-B baseline measured 5.64 then 18.19 on the identical
  command — the #23 VRAM cliff again, fired by 772 MiB of a dying server's memory still draining.
- **P-3 (no arm loses >5%): MISS for `ngram-cache`** (−7% at 19% acceptance). `ngram-mod` and
  `ngram-map-k4v` are harmless.

### What survives, positively

`ngram-mod` matched or beat `ngram-simple` on the edit task (**2.50×** vs 2.36–2.41×) with
byte-identical output, and is measured harmless on novel tasks. One measurement each — not enough
to switch the shipped default, enough to note. Its cross-request replay is also a real (niche)
property: deterministic regeneration of a previous response runs ~2.4×.

### The closed conclusion, per the kill rule

Novel generation on this box is bounded by the raw-decode wall: **41.1 tok/s ceiling, 22 measured,
and no speculation mechanism moves it.** What remains for novel workloads, all already measured:
q8_0 KV at depth (+37% at 16k, #25) for long sessions, prefix caching for the ingest side (29×,
#29), batching for aggregate throughput (~2×, #26), sync-elimination upstream (≤29%, #27), and
hardware — the wall scales linearly with DRAM bandwidth, which is the axis a 2016 box is poorest on.

**Wired into:** `findings/REGISTER.json:D-10` · the protocol rule ("read what the model wrote").
