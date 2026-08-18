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


---

## SCORED 2026-08-18: the measured band pays on a hybrid MoE. 3 of 3, plus one void.

Both arms built from the identical Q8_0 source and **byte-identical in size**:
14,115,658,720 bytes each. Same tiers, same attention/SSM at Q4_K, same shared experts at
Q8_0, the same **ten** layers held at Q4_K. The only difference in either file is *which* ten.

| arm | PPL (32 chunks, held-out WikiText-2) | delta over Q6_K reference 5.4669 |
|---|---|---|
| **OURS** - band 30-39, as the probe measured | **5.7796** | **+0.3127** |
| SPREAD - ten layers at 0,4,8...36 | 5.9088 | +0.4419 |

| stake | verdict | evidence |
|---|---|---|
| **P-1** OURS beats the control at equal bytes | **PASS** | 5.7796 < 5.9088 |
| **P-2** gap >= 5% of the control's delta | **PASS** | gap is **29.2%**, six times the bar |
| **P-3** decode differs < 5% between arms | **PASS** | 1.22% apart - placement moved quality, not speed |
| byte gate +/-3% | **PASS** | 0.000% - the files are the same size to the byte |

**Putting ten layers where the probe says instead of spreading them evenly removes 29% of the
quality loss, for free.** That is the first test of the depth-aware payoff on a hybrid
linear-attention MoE, and the control was strong rather than convenient: SPREAD spends exactly
the same bits on protection, just in the wrong places. A plain Q2_K would have looked far worse
and proved far less - which is precisely why the byte gate rejected it and the arm was replaced
before anything was scored.

### The speed number, and the one that is VOID

The artifact's decode speed, measured on the placement that is actually stable here:

| placement | tok/s | error bar | usable? |
|---|---|---|---|
| CPU only (-ngl 0) | 7.40 +/- 2.09 | 28% | noisy |
| **partial GPU (-ngl 12)** | **14.86 +/- 0.36** | **2.4%** | **yes - this is the number** |
| partial GPU (-ngl 24) | 4.84 +/- 0.60 | 12% | noisy, and 3x SLOWER than -ngl 12 |

**A 35B hybrid MoE at 14.86 tok/s on a GTX 1060 6GB.** N=5 reps, one machine state.

**What is void, and why it is recorded rather than quietly dropped.** Law 4 predicted **22.7
tok/s** for this file on the hybrid row (attention to VRAM, routed experts to RAM). Measured on
that row: **9.94 +/- 5.40** - a **54% error bar**, which our own `report` command classifies as
"quoted for completeness, not usable as a number". SPREAD came back 9.82 +/- 5.31, equally
noisy. The P-3 *ratio* survives that noise because both arms ate it identically; the *absolute*
number does not.

So the 22.7 prediction is **UNSCORED, not missed**. 22.7 against 9.94 looks like a 2.3x miss
and it would have been easy to write down, but a prediction cannot be scored against a
measurement whose error bar is half its value - that is the same error class as the 16-token
headline corrected this morning in C-31. The row is re-run or it gets no verdict.

**The instability was predicted by the tool.** `plan` printed that row with the warning "pins
12GB of 12GB RAM (CUDA host memory) - fails under memory pressure; if it does, drop -ot and let
auto-placement decide". A 13.15 GiB file pinning host memory on a 16 GB box is exactly that
failure, the warning was correct, and it was ignored on the first attempt. The fallback the
tool recommended is what produced the usable number.

### Unstaked observation: the -ngl cliff

Going from `-ngl 12` to `-ngl 24` costs **3x decode** (14.86 -> 4.84) on this file and box. More
GPU layers made it dramatically slower - a VRAM-overcommit cliff, not a gentle curve. This was
not staked and is recorded as an observation only; it is a candidate for its own prereg, since
the planner currently reasons about placements rather than about where a placement's own
-ngl optimum sits.

Chain of custody: staked while the OURS file was still being written -> byte gate fired at
+9.09% on the originally staked NAIVE arm -> control replaced with SPREAD and the change
declared before any scoring -> both arms measured on one machine state -> scored against bars
fixed in advance.
