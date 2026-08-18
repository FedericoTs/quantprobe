# Pre-registration #104: does the measured recipe beat a naive quant on a hybrid MoE, at equal bytes?

**Author:** Federico Sciuca · **Date staked:** 2026-08-18, **while the depth-aware file is
still building and BEFORE either artifact is scored.** **STAKED.**

## Why this, right now

Prereg #103 (scored today) measured where Qwen3.6-35B-A3B breaks: band 30-39, 2.53x the
median, monotone. That is a **measurement**. It is not yet a **product claim**. The product
claim is that protecting the measured band buys quality a naive quantizer does not, at the
same file size - and that has only ever been tested on full-attention models (prereg #98 on
Qwen3.5-35B, prereg 2026-08-04 on a 7B).

This is the first test of the payoff on a **hybrid linear-attention MoE**, and it is the
artifact people would actually download: 12.4 GB, predicted 22.7 tok/s on a 2016 desktop.

## Arms, built from the same source and matched on bytes

| arm | build |
|---|---|
| **OURS** | `quantprobe quantize --recipe qwen3.6-35b` - band 30-39 held at Q4_K, layers 0-29 FFN to Q2_K, attention/SSM Q4_K, shared experts Q8_0 |
| **NAIVE** | plain `llama-quantize ... Q2_K` from the identical Q8_0 source, no band protection |

Both from `Qwen3.6-35B-A3B-Q8_0.gguf`. Same box, same eval, one machine state (C-14), same
llama.cpp b10098, no imatrix on either arm so the comparison isolates band placement alone.

**The byte gate:** if the two files differ in size by more than **+/-3%**, this is not an
equal-bytes comparison and the quality numbers are not comparable. In that case the result is
reported as a size-mismatched observation, never as "ours beats naive", and the arm is re-run
with the tier adjusted. Bytes are the budget; *where the protection goes* is the treatment.

## Staked predictions

- **P-1 (the recipe pays).** OURS has **lower perplexity** than NAIVE on held-out WikiText-2
  at matched bytes. Direction only - any margin counts, because at 2-bit the naive baseline is
  where quantizers fall apart and the size of the gap is not something I can call in advance.
- **P-2 (the gap is material, not cosmetic).** The perplexity gap is **>= 5%** of NAIVE's
  delta over the Q6_K reference (ref PPL 5.4669, measured in #103). Below 5% the recipe is
  real but not worth telling anyone about, and I would say so.
- **P-3 (speed is unchanged).** Decode tok/s differs by **< 5%** between the two arms. They
  carry the same bytes/token to within the byte gate, so Law 4 says speed cannot move much. A
  bigger difference means something other than band placement changed and the comparison is
  contaminated.

## KILL RULES

- **If P-1 fails** - naive equals or beats the measured recipe on a hybrid MoE - then
  `quantprobe quantize --recipe` must carry a scope note for this architecture class the same
  day, and the atlas entry for qwen3.6-35b gets it too. A measured fragile band that does not
  buy quality is a finding about the *method*, not a footnote.
- **If P-2 fails** (gap under 5%) the recipe is reported as **measurable but not material** on
  this model, and no marketing number is drawn from it.
- **If the byte gate fails**, no quality claim is published from this pair at all.
- **A published headline needs the population, not one number:** perplexity is reported with
  its chunk count, and any speed figure with N and the spread, per C-31.

## What gets published either way

The two GGUFs are the same build anyone can reproduce from the committed recipe JSON and the
raw probe log. If P-1 and P-2 both pass, the depth-aware file is worth putting on Hugging Face
with the fragility curve in the model card. **If either fails, that goes in the model card
instead**, and the file either ships with the caveat or does not ship.

**Wired into:** pending - `quantprobe/recipes/qwen3.6-35b.json` provenance, Law 3's payoff
evidence, and the model card if one is published.


---

## Pre-data amendment - 2026-08-18, after the byte gate fired and BEFORE either arm is scored

**The staked NAIVE arm failed the byte gate and is replaced. Declared before any perplexity
was measured.**

Measured sizes from the identical Q8_0 source:

| arm | bytes | GiB |
|---|---|---|
| OURS (band 30-39 held at Q4_K) | 14,115,658,720 | 13.15 |
| NAIVE (plain `llama-quantize ... Q2_K`) | 12,939,594,720 | 12.05 |
| difference | | **+9.09%** |

That is three times the staked +/-3% window, so under the gate no quality claim may be drawn
from this pair, and the stake says the arm is re-run.

**Why "adjust the tier" is the wrong repair.** Moving the naive arm up a tier (Q3_K_S, or an
IQ3 variant) would land nearer 14.1 GB, but it changes *how many bits the file spends* and, in
the IQ case, the kernel class as well - which would confound P-3's speed prediction with a
codebook penalty we have measured elsewhere (prereg #70). Either way the comparison stops
being about placement.

**The replacement control: SPREAD.** Identical to OURS in every respect - same source, same
attention/SSM at Q4_K, same shared experts at Q8_0, same token embedding, the same **ten**
layers held at Q4_K and every other FFN at Q2_K - except that the ten protected layers are
spread **evenly across depth** (0, 4, 8, 12, 16, 20, 24, 28, 32, 36) instead of concentrated
on the measured fragile band (30-39). Ten protected layers either way, so the files land
within a fraction of a percent of each other by construction, and the *only* difference is
**where the protection sits**.

This is a stronger test than the one originally staked, and it is the test the project already
runs elsewhere - the Gemma 4 12B result quoted in the README is exactly this design
("byte-identical files, 2.25 ppl apart"). It asks the question the recipe actually makes: given
a fixed budget of protected layers, does putting them where the probe says beat spreading them
around?

**The predictions are unchanged and now bind against SPREAD:** P-1 OURS has lower held-out
perplexity than the control; P-2 the gap is >= 5% of the control's delta over the Q6_K
reference (5.4669, from #103); P-3 decode differs < 5%. The byte gate still governs and is now
expected to pass by construction - if it does not, the pair is still unpublishable.

**The plain Q2_K file is kept**, not deleted: it is the receipt for why this amendment exists,
and it remains a legitimate datapoint for the separate question of what a normal user's
download costs in quality - a question this prereg does not answer.
