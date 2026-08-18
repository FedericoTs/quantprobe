# Pre-registration #108: the expert dial is a bad decode lever — is it a good prefill one?

**Author:** Federico Sciuca · **Date staked:** 2026-08-19, **before any prefill arm was run.**
**SCORED 2026-08-19 - 2 of 3, P-1 refuted high. See the verdict at the foot.**

## Why

[Prereg #107](2026-08-18-the-k-lever-is-bounded-by-the-always-active-floor.md) measured the
expert-count knob on **decode** and it is a bounded, expensive lever: 1.146x at k=4 for +1.51
perplexity, ceiling 1.24x. Decode is **bandwidth**-bound, and the routed experts own only **22% of
the active bytes**, so most of the work is untouched whatever k does.

**Prefill is a different resource.** It is compute-bound, and compute scales with *parameters*
rather than *bytes* — quantization shrinks bytes, not FLOPs. The routed experts own **34.2% of the
active params** against 22% of the active bytes, so the same knob has a **larger ceiling on
prefill than on decode** purely by changing which unit the ceiling is computed in.

This matters because prefill is where agentic and coding workloads actually live: long files,
retrieved context, repeated re-reads of a large prompt. A lever worth nothing at decode can still
be worth having if it moves time-to-first-token.

## What the file says (metadata only, no measurement)

Active params `= 1.9398 + 32.2123 * k/256` B. Routed share at the default k=8: **34.16%**.

| k | active params | vs k=8 | predicted prefill speedup | measured DECODE (#107) |
|---|---|---|---|---|
| 8 | 2.9464 B | 1.000 | 1.000x | 1.000x |
| 4 | 2.4431 B | 0.829 | **1.206x** | 1.146x |
| 2 | 2.1915 B | 0.744 | **1.345x** | 1.175x |
| 1 | 2.0656 B | 0.701 | **1.426x** | 1.451x |

## Predictions (staked before any prefill arm ran)

- **P-1 (the law test).** Prefill speedup at **k=4 lands within ±15% of 1.206x**, i.e. in
  [1.03, 1.39]. *Refuted outside that band.*
- **P-2 (the point of the whole thing).** Prefill gains **exceed** #107's decode gains at k=4 and
  k=2 — the knob belongs to prefill if it belongs anywhere. *Refuted if prefill fails to beat
  decode at either.*
- **P-3 (is it worth having).** Prefill speedup at **k=2 is at least 1.25x**. Below that the lever
  is not worth the quality it costs even on the workload it suits best. *Refuted below 1.25x.*

**Quality is not re-measured.** k is k: prereg #107's curve already prices it (k=4 costs +1.51
PPL, k=2 costs +15.5). This prereg only asks whether the *speed* side of that trade looks
different on prefill. Any recommendation must carry #107's quality bill unchanged.

**Where I might be wrong, stated up front.** #107's perplexity wall times fell 341s → 142s → 112s
→ 96s as k dropped, which is a far steeper fall than 1.426x. That observation is **confounded** —
k=8 ran first and was cold, and L-29 prices a cold start at up to 46% on this file — but if the
true prefill effect really is that large, P-1 fails high and the FLOP-share model is missing
something (most likely per-expert gather overhead that scales with k independently of FLOPs).
I am staking the FLOP-share numbers rather than the wall-clock hint, because the hint is
contaminated and staking around contaminated data is how #105 died.

## Method

`llama-cli` (b10098) — `llama-bench` has no `--override-kv` (#107's amendment). Prefill rate is
llama-cli's own reported `Prompt: N t/s`, measured on a **~2000-token prompt read from a file** so
the number is a real prefill and not the 20-token noise of a chat turn. `-ngl 12 -n 1`, three reps
per k, arms differing only in `--override-kv`, **after a discarded warm-up** (L-29). Free RAM
recorded beside every arm. C-14: nothing else runs. Same placement as #107's decode arms so P-2
compares like with like.

## Kill rule (committed before data exists)

Scored by [`weights/prereg108_score.py`](../weights/prereg108_score.py), written and committed
**before** the arms run.

- **P-2 and P-3 both hold** → the dial is a **prefill** lever, and `quantprobe` says so: the
  ceiling line shipped in v1.30 gains a second number, computed from the param share, and states
  that the gain lands on time-to-first-token and never comes free of #107's quality bill.
- **P-2 refuted** → the knob is bad everywhere. V-22 is upgraded from "not recommended" to
  "no known workload", and the tool keeps exactly one ceiling line.
- **P-1 refuted high with P-2 holding** → the FLOP-share model under-prices prefill, and the extra
  is per-expert overhead rather than arithmetic. That is a new term, and it gets its own stake
  before anything is claimed about it.

---

## Verdict: SCORED 2/3, P-1 refuted high (2026-08-19)

Scored by [`weights/prereg108_score.py`](../weights/prereg108_score.py), committed before the
arms ran. Raw: [`prereg108_run.log`](../weights/data/prereg108_run.log) (pass 1) ·
[`prereg108_run2.log`](../weights/data/prereg108_run2.log) (pass 2) ·
[`prereg108_order_effect.txt`](../weights/data/prereg108_order_effect.txt) ·
[`prereg108_prefill_controlled.json`](../weights/data/prereg108_prefill_controlled.json) ·
[`prereg108_verdict.txt`](../weights/data/prereg108_verdict.txt).

| k | prefill tok/s | gain | predicted | error | decode (#107) |
|---|---|---|---|---|---|
| 8 | 51.53 ± 0.95 | 1.000x | 1.000x | — | 1.000x |
| 4 | 83.10 ± 0.65 | **1.613x** | 1.206x | **+33.7%** | 1.146x |
| 2 | 194.07 ± 1.95 | **3.766x** | 1.345x | +180% | 1.175x |
| 1 | 202.80 ± 5.65 | **3.935x** | 1.426x | +176% | 1.451x |

- **P-1 k=4 within ±15% of 1.206x — MISS (1.613x, +33.7%).** The FLOP-share model
  **under-prices prefill**, and by more than the band allows.
- **P-2 prefill beats decode at k=4 and k=2 — HIT.** 1.613x against 1.146x, and 3.766x against
  1.175x. The dial belongs to prefill.
- **P-3 k=2 reaches 1.25x — HIT (3.766x).** By a factor of three.

**Kill rule: IT IS A PREFILL LEVER.** With the note the scorer attaches automatically — P-1
refuted while P-2 holds means the excess is a term the FLOP model does not contain, and it gets
its own stake before anything is claimed about it.

### The measurement nearly reported the wrong number, twice

**Pass 1 had an unusable baseline.** k=8 came in at 20.2, 48.5, 68.2 — a **53% spread** against
the project's 15% bar. Divide by the mean and k=4 reads 2.026x (a spectacular MISS); divide by
the highest sample and it reads 1.356x (a HIT). A verdict that flips depending on which sample
you divide by is not a verdict, so P-1 was recorded **VOID**, and the scorer gained a
baseline-usability gate — the same bar `runtime.py` already applies and the same call prereg
#104 made on a 9.94 ± 5.40 arm.

**Pass 2 revealed the spread was not noise.** Five interleaved passes, and the scatter sorted
itself perfectly by *which k ran immediately before*:

| arm | preceded by | n | spread |
|---|---|---|---|
| k=8 | k=8 | 3 | **1.8%** |
| k=8 | k=4 | 2 | **72.4%** |
| k=4 | k=8 | 3 | **0.8%** |
| k=4 | k=2 | 2 | **20.9%** |
| k=2 | k=4 | 3 | 1.0% |

**An arm preceded by a lower-k arm is contaminated; an arm preceded by an equal-or-higher k is
clean to within 2%.** The mechanism is the one L-29 already names: this file is larger than free
RAM, so the page cache carries the previous run's expert working set into the next one, and a
low-k predecessor leaves the wrong pages resident for a high-k successor.

The interleaving was in the method for exactly this reason — *"so residual drift shows up as
disagreement between passes instead of hiding inside the trend"* — and it earned its place. The
scored table above is the **three descending passes only** (p1/p3/p5), where every arm has the
same predecessor in every pass, and every arm is tight to ≤2.8%.

**This is post-hoc subset selection and it is declared as such.** Two things keep it honest.
First, the split is *mechanistic and pre-named*, not fitted: it falls out of L-29, which was
measured yesterday on a different question. Second, **the conclusion does not depend on it** —
under the pooled pass-1 analysis P-1 is VOID, under the controlled analysis P-1 is a MISS at
+33.7%, and under the worst-case divisor P-2 and P-3 still hold. P-1 is never a HIT on any
reading, so *"the FLOP-share model under-prices prefill"* survives every way of cutting the data.
A first cut of the filter (`predecessor >= k`) was wrong and swept each ascending pass's first
arm back in; caught by checking the sample lists, not by the number looking plausible.

### What this means

The expert dial is a **bad decode lever and a real prefill lever**: ~1.6x time-to-first-token at
k=4, ~3.8x at k=2 — against decode's 1.15x and 1.18x. But it carries prereg #107's quality bill
unchanged: **k=4 costs +1.51 perplexity, k=2 costs +15.5.** Nothing here makes the trade good on
general work; it makes it *arguable* for a prompt-dominated pipeline that can tolerate the loss,
which is a narrower claim and the only one the data supports.
