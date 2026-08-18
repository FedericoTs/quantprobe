# Pre-registration #107: how much speed can you buy by using fewer experts?

**Author:** Federico Sciuca · **Date staked:** 2026-08-18, **before any speed or quality arm was
run.** **SCORED 2026-08-19 - 4 of 4. See the verdict at the foot of this file.**

## Why

`--override-kv <arch>.expert_used_count=int:K` is a **runtime** knob: same file, same bytes on
disk, fewer expert FFNs read per token. It is widely treated as a free speed dial for MoE models
on constrained hardware. Feasibility confirmed before staking — `print_info: n_expert_used` reports
2 and 1 when overridden, so llama.cpp honours it on `qwen35moe`.

Law 4 says `tok/s = eta * BW / bytes_per_token`, so the lever's ceiling is decided by what share
of the active bytes the routed experts actually own. That is readable from the file, before any
measurement.

## What the file says (metadata only, no measurement)

`Qwen3.6-35B-A3B-OURS-depthaware.gguf`: **256 experts, k=8 by default**, expert FFN length 512.

| | params | bytes | bits |
|---|---|---|---|
| routed experts (all 256) | 32.212 B | 11.602 GiB | 3.09 |
| always-active path | 2.448 B | 1.268 GiB | 4.45 |

`token_embd` (0.266 GiB) is excluded as a gather, not a stream (U-26 / prereg #76).

Bytes, not params, are what Law 4 divides — and the split is worse for the lever in bytes than in
params, because the always-active path is carried at 4.45 bits against the experts' 3.09:

| k | active bytes | vs k=8 | Law 4 predicted speedup |
|---|---|---|---|
| 8 (default) | 1669.7 MiB | 1.000 | 1.000x |
| 6 | 1576.9 MiB | 0.944 | 1.059x |
| 4 | 1484.1 MiB | 0.889 | **1.125x** |
| 2 | 1391.3 MiB | 0.833 | **1.200x** |
| 1 | 1344.9 MiB | 0.805 | **1.242x** |

**At k=1 — one expert of 256, which should wreck the model — 96.5% of the active bytes are still
being read.** The routed experts own only 22% of the active byte budget at the default k. If Law 4
holds, this knob cannot buy more than about a quarter, no matter how much quality is burned for it.

## Predictions (staked before any arm ran)

- **P-1 (the headline).** The k lever is **weak**: measured speedup at **k=1 is below 1.50x**.
  *Refuted at 1.50x or above.*
- **P-2.** Speed is **monotone non-decreasing as k falls** across k ∈ {8, 4, 2, 1}, allowing
  overlap within the error bars. *Refuted by a reversal larger than the combined error bars.*
- **P-3 (the quantitative Law 4 test).** Measured speedup at **k=2 lands within ±15% of the
  predicted 1.200x**, i.e. in [1.02, 1.38]. *Refuted outside that band.*
- **P-4 (the trade).** The lever is **not cheap**: perplexity at k=4 is **at least +0.50 worse**
  than at k=8 (5.7796 → ≥ 6.28 on the same 32 chunks of held-out WikiText-2).
  *Refuted if halving the experts costs less than 0.50 PPL* — which would make k a genuinely
  cheap lever and a far more interesting result than the one I expect.

**Where I might be wrong, stated up front.** This model does **not fit free RAM** (13.15 GiB file,
~12.2 GB free — L-29, C-32). Lowering k shrinks not just the per-token byte count but the *set of
experts touched across a whole run*, which could raise the page-cache hit rate and produce a
speedup **larger** than the bandwidth-only ceiling. If that happens, P-1 or P-3 fails and the
reason is residency, not a broken law. I am staking the bandwidth-only numbers anyway, because a
prediction hedged against both outcomes is not a prediction.

## Amendment, pre-data (2026-08-18): the speed harness changes, the predictions do not

`llama-bench` in b10098 **does not accept `--override-kv`**. Its help lists only
`-ot / --override-tensor`; passing the KV override makes it print usage and exit, which is why
the first attempt returned a rate for k=8 and nothing for k=4, 2 and 1. The feasibility check
before staking was run with `llama-cli`, which does accept it - so the flag works on this
build, just not in the benchmark binary. That is a harness limitation I should have checked in
the binary I intended to measure with, not in a different one.

**No k != 8 speed number exists**, so this is a method change before data, not after it.

Speed arms move to **`llama-cli`**, which accepts the override, using its own reported
generation rate: `-ngl 12 -n 128 -st --simple-io --seed 1234`, three reps per k, **every arm
including the k=8 baseline on the same binary**. Absolute values will sit below llama-bench's
(llama-cli's figure includes sampling and detokenization - measured ~11.4 against llama-bench's
~14.4 on this file), but **every prediction in this document is a RATIO against the k=8 arm**,
and the ratio is preserved when the baseline moves with the treatment. P-1, P-2 and P-3 stand
exactly as staked; none of their thresholds are touched.

Retained from the killed attempt because it is evidence and not a treatment: the warm-up
discards ran 9.91 -> 12.70 -> 14.49 tok/s, and the first measured k=8 arm was 14.42. That is
L-29's cold-start ramp reproducing on demand - a 46% climb across three runs of one unchanged
command - and it is why the warm-up is in the method at all.

The perplexity arms are unaffected: `llama-perplexity` shares `llama-cli`'s argument parser.
Their built-in check is that k=8 must reproduce prereg #104's committed 5.7796 on the same
corpus and chunk count; if all four k values come back identical, the override was silently
ignored and the quality arms are VOID rather than flat.

## Method

Speed: `llama-bench -ngl 12 -p 0 -n 128 -r 3`, one binary (b10098), one file, one session, arms
differing only in `--override-kv`. **Warmed first** — L-29 says the first runs of a file this size
are cold and climb, so a cold k=8 baseline against a warm k=1 arm would manufacture the result.
Free RAM recorded beside every arm. C-14: nothing else runs.

Quality: `llama-perplexity -f wikitext-2 --chunks 32 -ngl 0`, identical chunks to prereg #104, so
k=8 is directly comparable to the committed 5.7796.

## Kill rule (committed before data exists)

Scored by [`weights/prereg107_score.py`](../weights/prereg107_score.py), written and committed
**before** the arms run.

- **P-1 and P-4 both hold** → the k knob is a **bounded, expensive lever** on this architecture.
  It goes in the register as such, and `quantprobe` states the ceiling *computed from the file*
  rather than offering the knob as a win. The useful output is the ceiling formula, not the dial.
- **P-4 refuted** (halving experts costs under 0.50 PPL) → k is a **cheap** lever and the tool
  should expose it, with the measured quality curve attached.
- **P-1 refuted** (k=1 beats 1.50x) → the bandwidth-only model is incomplete for a model that
  exceeds free RAM, and the residency term needs to enter the prediction, not just the disclosure.
  That is a Law 4 amendment and it gets its own stake.
- **P-3 refuted with P-1 holding** → the lever is weak as claimed but Law 4 mis-sizes it; publish
  the miss and the measured curve at the same size as the ceiling claim.

---

## Verdict: SCORED 4/4 (2026-08-19)

Scored by [`weights/prereg107_score.py`](../weights/prereg107_score.py), committed before the
arms ran. Raw: [`prereg107_kcurve.json`](../weights/data/prereg107_kcurve.json) ·
[`prereg107_speed.log`](../weights/data/prereg107_speed.log) ·
[`prereg107_run.log`](../weights/data/prereg107_run.log) ·
[`prereg107_verdict.txt`](../weights/data/prereg107_verdict.txt).

| k | tok/s (llama-cli, N=3) | measured | Law 4 predicted | error | PPL | cost vs k=8 |
|---|---|---|---|---|---|---|
| 8 | 13.90 +/- 0.55 | 1.000x | 1.000x | - | 5.9618 | - |
| 4 | 15.93 +/- 0.45 | 1.146x | 1.125x | **+1.9%** | 7.4708 | +1.51 |
| 2 | 16.33 +/- 1.05 | 1.175x | 1.200x | **-2.1%** | 21.4196 | +15.46 |
| 1 | 20.17 +/- 0.35 | **1.451x** | 1.242x | **+16.8%** | 2277.15 | destroyed |

- **P-1 k=1 below 1.50x — HIT (1.451x).** By 0.049. The lever is weak, and it nearly wasn't.
- **P-2 monotone as k falls — HIT.**
- **P-3 k=2 within ±15% of 1.200x — HIT (1.175x, -2.1%).** Law 4 sized the lever from the file's
  byte split, before any measurement, to within 2%.
- **P-4 halving the experts costs ≥ 0.50 PPL — HIT (+1.51).**

**Kill rule: BOUNDED, EXPENSIVE LEVER.** The knob is real and Law 4 prices it, but the ceiling
is set by the always-active floor and the quality bill arrives immediately. The useful product
output is the **ceiling computed from the file**, not the dial.

### The headline for a user

Turning experts down on this model buys **15% at k=4 and costs +1.51 perplexity** — the model is
measurably worse for a gain most people would not notice. At k=2 it buys 18% and costs 15.5 PPL.
At k=1 it buys 45% and the model is gone (2277). There is no good point on this curve. That is
not a defect of the implementation; it is arithmetic that was readable from the file before
anything ran: the routed experts own **22% of the active bytes**, so 78% of the work is untouched
no matter what k does.

### Where the prediction broke, exactly where the stake said it might

k=4 and k=2 land within 2% of Law 4. **k=1 overshoots by 16.8%** — 1.451x measured against a
1.242x bandwidth-only ceiling.

The stake named this in advance: *"Lowering k shrinks not just the per-token byte count but the
set of experts touched across a whole run, which could raise the page-cache hit rate and produce a
speedup larger than the bandwidth-only ceiling."* The signature fits — the excess appears **only**
at the extreme, where the touched-expert set shrinks 8x and this file is larger than free RAM
(13.15 GiB against ~12.6 GB, L-29/C-32). At k=4 and k=2 the working set is still too large to
change residency and Law 4 is exact.

So P-1 survives, but the honest reading is that it survived **narrowly and for a reason the
bandwidth model does not contain**. Registered as the open edge of Law 4, not as a win.

### Two errors made and corrected in this prereg

1. **Wrong binary.** The feasibility check used `llama-cli`; the method specified `llama-bench`,
   which has no `--override-kv` in b10098. Caught on the first pass because k=8 returned a number
   and every override arm returned nothing. Amended pre-data — all arms moved to `llama-cli`, and
   since every prediction is a ratio against k=8, no threshold moved.
2. **Wrong corpus.** These arms ran on `weights/data/wikitext2_test.raw` (1,307,975 bytes);
   prereg #104's 5.7796 was measured on `D:/evo-compress-data/eval/wiki.test.raw` (1,290,590
   bytes, different hash). I verified `quant_probe.py`'s default instead of what #104's own script
   ran. **Absolute PPL here is not comparable to #103/#104**; every k value here is mutually
   comparable, which is what all four predictions need. The scorer computed P-4 from the session's
   own k=8 and was correct throughout — but its *printed* delta line used the #104 constant and
   showed +1.69 where the truth is +1.51. Fixed, and the line now says which baseline it used.

Both errors are the same shape: **verifying a plausible proxy instead of the thing itself.** Same
family as C-31 (grepping a log's first match instead of its population).

### Observed, not staked, and confounded

Perplexity wall time fell 341s → 142s → 112s → 96s as k dropped. That looks like a large prefill
effect, and prefill is compute-bound rather than bandwidth-bound, so k should bite harder there.
But **k=8 ran first and was therefore cold**, and L-29 prices that at up to 46% on this file. Among
the warm arms alone it is 142s → 96s. Suggestive, confounded, unstaked — a candidate for its own
pre-registration, not a finding.
