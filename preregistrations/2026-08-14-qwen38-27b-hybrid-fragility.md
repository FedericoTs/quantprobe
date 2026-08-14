# Pre-registration #101: does depth-localized fragility survive a hybrid linear-attention model?

**Author:** Federico Sciuca · **Date staked:** 2026-08-14, **the day Qwen3.8-27B's weights landed
on Hugging Face, BEFORE downloading them, BEFORE the fragility probe, before any benchmark.**
**STAKED.**

## Why this model, today

`Qwen/Qwen3.8-27B` dropped this morning (last-modified 2026-08-14, confirmed against the HF repo,
not the countdown blogs). It is the first model this project has met that our laws were **not
built for**, and that is exactly why it is worth staking before we touch it.

Architecture, read from `config.json` (`Qwen3_5ForConditionalGeneration`, `qwen3_5_text`):

- **Dense 27B**, no experts. 64 layers, hidden 5120, head_dim 256, vocab 248,320.
- **HYBRID ATTENTION: 48 linear-attention layers + 16 full-attention layers**, and the full ones
  sit at indices 3, 7, 11, ... 63 - **every 4th layer, strictly periodic** (3 linear : 1 full).
- An MTP head (`mtp_num_hidden_layers = 1`), same inert `nextn` block shape as the 4B carried.

Every prior model we have probed (Mistral-7B, Qwen2.5-7B, Qwen3-30B, Qwen3.5-4B, Qwen3.5-35B) is
full-attention throughout. Qwen3.8-27B breaks that assumption in a way that lets two of our
central claims be tested where they have never been tested.

## The one experiment worth staking: depth vs attention-type fragility

Our fragility probe (Law 3) is a **4-band depth sweep**: split the layers into 4 equal bands,
push each band's FFNs to Q2_K in turn, measure the perplexity cost, and protect the worst band.
This design encodes an assumption we have never had reason to question: **that fragility is
localized by DEPTH.** On the 4B it found the back band (24-31 of 32); on the 35B, the back.

A hybrid model breaks the assumption in a testable way, because the full-attention layers - the
ones that do the long-range mixing linear attention approximates - are spread **evenly across all
four depth bands** (4 per band). So the two hypotheses make OPPOSITE, falsifiable predictions:

- **P-1 DEPTH WINS (the family pattern holds):** the 4-band depth profile is monotone or
  back-heavy, the worst band is **48-63**, and its perplexity delta exceeds the median band by
  **>= 1.3x** (the effect size the 4B and 35B both showed). The depth-aware recipe transfers to
  hybrids unchanged.
- **P-2 THE PROBE IS BLIND TO THIS MODEL (the hybrid surprise):** the depth profile comes back
  **FLAT** - all four bands within **15%** of each other - because the fragile thing is the
  full-attention layers, which every band contains in equal measure. If this lands, the depth
  probe is the wrong instrument for hybrids, and the finding is that fragility here is localized
  by ATTENTION TYPE, not depth. A follow-up attention-type probe (protect the 16 full-attention
  layers, leave the 48 linear at base) would be the corrected tool - NOT built now, named so the
  option survives.
- **P-3 NEITHER (front-fragile or mixed):** any other shape - e.g. front-heavy like Mistral-7B -
  is reported as its own datapoint against the "Qwen family breaks at the back" generalization.

P-1 and P-2 cannot both hold; a flat profile refutes depth-localization, a back-heavy one
confirms it. This is the rare prereg where our own tool's core assumption is the thing on trial.

## Two secondary stakes, carried from what we already measured

- **P-4 C-30 REPLAYS (the always-active path):** naive `llama-quantize Q2_K` will damage
  instruction-following / convergence more than the depth-aware recipe, in the same direction the
  35B showed (#98: +24.0 MATH-500, mostly a convergence collapse per #99). The mechanism there
  was the always-active SSM path left at 2 bits; here the always-active path is the **48 linear-
  attention layers**. Staked direction: ours beats naive on MATH-500 by **>= 5 pts**. Same
  powered-three protocol as #100, scored by the same sealed code.
- **P-5 LAW 4 SURVIVES THE WEIGHT TERM, NOT THE KV TERM:** for single-stream decode the weight-
  bandwidth term is architecture-agnostic (a dense model reads all its weights per token
  regardless of attention type), so quantprobe's decode tok/s prediction, once fed the real GGUF
  header, should land within the band it hits on full-attention models. BUT the KV/context term
  is built for full attention (KV grows O(sequence) per layer); linear attention carries O(1)
  state for 48 of 64 layers, so at long context our model **over-predicts KV memory and
  under-predicts speed**. Prediction: the divergence is negligible at short context (weight-
  bound) and grows with depth. This is a KNOWN tool gap, staked so the miss is a measurement, not
  a surprise - and it is the concrete reason quantprobe needs a linear-attention term.

## Kill rules / discipline

- **KR-1 PROBE BEFORE PEEK:** the 4-band profile is recorded before any benchmark row of any arm,
  same as #100's KR-C6. The band is chosen by the probe, never after a score.
- **KR-2 SAME SOURCE CHAIN:** BF16 original -> depth-aware child + naive child, all from the one
  downloaded BF16 file, same binary, same box (C-14, one measurement at a time - this waits for
  #100's box to free).
- **KR-3 BYTES DISCLOSED:** the ours/naive byte premium is measured at build and printed; > 5% is
  size-confounded (the #98/#100 rule).
- **KR-4 SAME SCORER:** `ev1_report` with re-grade on; a prereg100-style scorer is committed
  before the first benchmark row.
- **KR-5 THE TOOL GAP IS NOT PATCHED TO FIT:** if P-5's KV divergence appears, quantprobe's
  linear-attention support is added as its OWN change with its own before/after, never
  back-fitted to make this prediction look better after the fact.

## What this cannot show

One hybrid model, one size. It cannot say linear-attention fragility generalizes across the
hybrid family, and it cannot separate "linear-attention layers are fragile" from "the full-
attention layers are fragile" unless P-2 lands and the follow-up attention-type probe is run.
The 35B ceiling remains a separate open item; this does not touch it.

## Sequence (queued behind #100, not started)

1. Download BF16 (split, ~54 GB) for autospec + probe; Q4_K_M (15.9 GB) / Q3_K_M (12.9 GB) for
   reference rows.
2. `quantprobe plan --gguf <BF16>` - the real autospec number, replacing today's discarded
   manual-flag output (which applied MoE/default assumptions to a dense model and ignored
   bit-depth; caught, not published).
3. 4-band fragility probe -> score P-1/P-2/P-3.
4. Build ours + naive from BF16 -> ceiling-chain rows -> score P-4.
5. Compare autospec prediction to measured decode -> score P-5.
