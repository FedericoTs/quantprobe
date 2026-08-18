# Pre-registration #107: how much speed can you buy by using fewer experts?

**Author:** Federico Sciuca · **Date staked:** 2026-08-18, **before any speed or quality arm was
run.** **STAKED.**

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
