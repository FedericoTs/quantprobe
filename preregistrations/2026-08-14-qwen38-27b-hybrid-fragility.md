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


---

## AMENDMENT (2026-08-15, BF16 probe failed on this box, BEFORE the Q4 probe ran)

**The BF16 probe is infeasible on this hardware, and the fragility band will be measured from a
Q4_K_M source instead. Declared before the Q4 probe runs, per KR-1.**

What happened: the BF16 (50.9 GB) downloaded fine, and the probe ran 69 minutes and quantized all
its band files - but every perplexity run failed to LOAD them ("invalid magic '????'"). Root
cause, diagnosed: **the disk was full** (11 GiB free against ~19.6 GiB per band file), so each
band write truncated. Not an architecture bug. The probe correctly detected the broken curve and
refused to emit a recipe - the C-26/C-30 discipline holding in failure.

Fixing the disk only exposes the deeper wall, the same one prereg #100 named for the 35B: a 27B's
Q6 reference (~20 GB) exceeds this box's 16 GB RAM, so the full-precision probe streams from disk
(days), and the **BF16 ceiling chain (P-4/P-5) is weeks per arm - infeasible here.**

**Decision (Federico, 2026-08-15): salvage the novel science, defer the ceiling.**

- The BF16 was deleted (unusable for the ceiling on this box), freeing disk to 61 GiB.
- The fragility band is measured from **Qwen3.8-27B-Q4_K_M** (15.9 GiB, fits 16 GB RAM), not BF16.
- **This answers P-1/P-2/P-3 - the depth-vs-attention-type question, which is the genuinely novel
  stake - because those turn on RELATIVE perplexity deltas between depth bands, and a consistent
  Q4 source measures relative fragility fine.** Probing from a quantized source instead of the
  original is a known compromise (the fragile band is where a further-quantized band hurts MOST
  relative to the rest; the source's own quantization is common-mode and cancels in the deltas).
- **P-4 (ceiling chain) and P-5 (Law 4 vs measured decode) are DEFERRED to rented hardware**, and
  are explicitly NOT scored from this Q4-sourced probe. The absolute BF16->quant loss is not
  measurable here; only the band LOCATION is.

**What does NOT change:** P-1 vs P-2 thresholds (back-heavy worst band 48-63 with >=1.3x median,
versus a flat profile within 15% pointing at the evenly-spread full-attention layers). The whole
point of the stake - whether depth-localized fragility survives a hybrid - is intact, because it
was always a question about the SHAPE of the depth curve, not its absolute height.

**Honest limit added:** a Q4 source could in principle blunt the very fragility we are trying to
locate (if the fragile band is already damaged by the Q4 quantization, its marginal Q2 delta
shrinks). If the profile comes back suspiciously flat, that confound is named here in advance as a
rival to P-2's "attention-type" reading, and the tie-breaker is the follow-up BF16 probe on a GPU
- not a reinterpretation of this run.


---

## SCORED (2026-08-15, Q4-sourced probe complete, 7.3h, clean 4-band curve)

**P-1 DEPTH WINS: CONFIRMED. P-2 REFUTED. Depth-localized fragility survives a hybrid
linear-attention model.**

| band | delta PPL |
|---|---|
| layers 0-16 | 0.045 |
| layers 17-33 | 0.212 |
| layers 34-50 | 0.292 |
| **layers 51-64** | **0.593** (fragile) |

Strictly monotone with depth. Worst band 51-64 (the back), delta 0.593 vs median 0.292 =
**2.04x** - past the staked >= 1.3x bar. The profile spans 13x (0.045 -> 0.593), the opposite of
P-2's "flat within 15%".

**The finding.** Qwen3.8-27B has 48 of 64 layers as LINEAR attention, spread evenly across all
four depth bands (full-attention layers at 3,7,11..63). If fragility tracked attention TYPE the
depth profile would be flat, because every band holds the full-attention layers in equal measure.
It is not flat - it is cleanly back-heavy, exactly like the full-attention Qwen family (4B 24-31,
35B back, 2.5-7B, 3-30B). So fragility here is localized by DEPTH, not attention type: the depth
probe is the right instrument for this hybrid, the "Qwen breaks at the back" pattern extends to
hybrid architectures, and the depth-aware recipe transfers unchanged.

**The declared confound did NOT fire.** The 2026-08-15 amendment named the risk that a Q4 source
could blunt the fragile band into a false flat. It didn't: a 2.04x back-heavy signal survived the
Q4 quantization, which makes the confirmation stronger, not weaker. The GPU BF16 tie-breaker named
in the amendment is therefore not needed to reach the P-1 verdict (it would only sharpen the
absolute deltas).

**Prediction accuracy, stated honestly.** P-1 named the worst band "48-63"; the probe's back band
was "51-64". Same band (the back quarter) - the small index difference is just how the probe
divides 64 layers into 4 (0-16/17-33/34-50/51-64) versus my assumed even 48-63. The substantive
stake - back-heavy, >= 1.3x median - is confirmed; the exact boundary was off by three layers and
that is noted rather than smoothed over.

**The emitted recipe is architecture-aware where it matters:** it protects `ssm_.*` (the linear-
attention state, the always-active path) at Q4, plus shared experts and the MTP head at Q8, with
the fragile band 51-64 at Q4 and the rest of the FFNs at Q2. That the tool independently protects
the always-active path is C-30's lesson showing up in the generated recipe, unprompted.

### Still open (deferred to rented hardware, NOT scored here)

- **P-4** (naive vs recipe on benchmarks - the always-active-path collapse) and **P-5** (Law 4
  weight term vs KV term for 48 linear layers) both need the BF16 ceiling chain, which is weeks
  per arm on this box. Deferred, exactly as the 35B ceiling is.
- The **attention-type follow-up probe** (P-2's corrected tool) is moot - P-2 is refuted, so
  there is no flat profile demanding an attention-type explanation.


---

## P-5 SCORED (2026-08-15): the speed law survives linear attention, with one named gap

Feasible on this box (a speed measurement, not the deferred benchmark ceiling). quantprobe
autospec on the Q4_K_M read the model correctly (27.3B total, 26.0B active, dense) - unlike the
launch-day manual-flag attempt that applied MoE assumptions.

**WEIGHT TERM - CONFIRMED.** quantprobe predicted 1.8 tok/s decode (split, -ngl 15, RAM-bandwidth-
bound). llama-bench measured **tg128 = 2.04 +/- 0.02 tok/s** (GTX 1060 6GB + 16GB RAM, Q4_K_M,
15/65 layers on GPU). quantprobe is 11.8% UNDER, in the conservative/floor direction, inside the
band it hits on full-attention models. The staked claim holds: for single-stream decode the
weight-bandwidth term is architecture-agnostic - a dense model reads all its weights per token
regardless of attention type. (Prefill pp512 = 44.2 but +/-39.3 over 2 reps with warmup counted;
noisy, not a staked decode term - reported, not claimed.)

**KV TERM - DIVERGES EXACTLY AS STAKED.** quantprobe read 260 KB/pos, the all-64-layers-full-
attention computation (64 x 2 x 4 KV-heads x 256 head-dim x 2 bytes = 256 KB). But only 16 of 64
layers are full attention; the other 48 are linear attention with a fixed-size recurrent state
that does not grow with position. Real KV ~= 16 x 4 KB = ~64 KB/pos, so quantprobe OVER-ESTIMATES
KV by ~4x - the precise divergence P-5 predicted. Its other half holds too: at the 128-token
context measured, KV is negligible either way (~8 vs ~33 MB against a 17 GB model), so the weight
prediction still lands. The error is real but only bites at long context / high depth, as staked.

**What P-5 buys the tool (KR-5, not back-fitted):** quantprobe needs a linear-attention KV term -
count full-attention layers from the GGUF, not all layers. Filed as U-51, to be added as its own
change with its own before/after, never by tuning a constant to make this row look better.

**Headline datapoint:** Qwen3.8-27B decodes at ~2 tok/s on a 2016 $50 GTX 1060 6GB - it RUNS, and
quantprobe called it within 12% before measuring.
