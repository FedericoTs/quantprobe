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
