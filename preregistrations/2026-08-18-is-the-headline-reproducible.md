# Pre-registration #106: is our published headline reproducible on the box that produced it?

**Author:** Federico Sciuca · **Date staked:** 2026-08-18, **before any arm below was run.**
**SCORED 2026-08-18 - 2 of 4, with P-1 and P-3 refuted against me. See the verdict at the foot of this file.**

Successor to [#105](2026-08-18-published-speed-vs-experienced-speed.md), which went VOID when its
reference arm refuted its premise. Nothing here reuses #105's predictions.

## What is already known (the data that killed #105)

`llama-bench -m Qwen3.6-35B-A3B-OURS-depthaware.gguf -ngl 12 -p 0 -n 128 -r 5 -o json` produced
**14.86 ± 0.36 tok/s** ([`prereg104_speed_stable.log`](../weights/data/prereg104_speed_stable.log)),
and that number is published on the HuggingFace model card, in the recipe atlas, and in C-32.

Today, the same command on the same file and the same binary returns **11.0**. `llama-cli` returns
**11.40, 11.40, 11.40** — no spread at all. So today's box is not jittery; it is *stably slower*.
Something changed persistently, and the tool that is supposed to predict speed did not notice.

Diagnostics already taken with the box idle: desktop VRAM 409 MiB of 6144, GPU at P0 with memory
clock 4004 of 4006 MHz, **free RAM 12.24 GB, model file 13.15 GiB**.

## Hypothesis

**The file is larger than free RAM, so it cannot stay resident, and 14.86 was measured while it
was still hot in page cache from having just been written by the quantizer.** Under that story the
original figure is not wrong — it is a *first-run-after-build* number that no later run, ours or a
downloader's, can reach. Everything since streams part of the weights off disk every pass, which
is a different regime from the one Law 4 prices.

## Predictions (staked before any arm ran)

- **P-1.** The published figure does not reproduce: the mean of **six fresh invocations of the
  exact published command** is **below 13.0 tok/s**. *Refuted if the mean is 13.0 or above.*
- **P-2.** No single one of those six reaches **14.50** (the bottom of the published error bar).
  *Refuted by any invocation at 14.50 or above.*
- **P-3.** Residency is the mechanism: one invocation run **immediately after sequentially reading
  the whole file** (priming the page cache) beats the six-run mean by **at least 1.0 tok/s**.
  *Refuted if priming buys less than 1.0.*
- **P-4.** The instability is size-dependent: a model that comfortably fits free RAM
  (`Qwen2.5-7B-Instruct-Q4_K_M`, 4.68 GB, same command, same `-ngl 12`) has a **relative spread
  `(max-min)/mean` no larger than the 13.15 GiB file's**. *Refuted if the small model is the
  more variable of the two.*

P-3 is the one I expect to be hardest on me. Priming cannot work fully — the file exceeds free RAM,
so filling the cache with its tail evicts its head. If P-1 and P-2 hold but P-3 fails, the residency
story is incomplete and I will have a reproducibility failure with no mechanism, which is a worse
place to stand than the one I am in now. It stays staked at 1.0 anyway.

## Method

One binary (`llama-bench`, b10098), one box, arms run back-to-back in a single session. C-14 holds:
nothing else runs. Free RAM is sampled immediately before every invocation and recorded next to its
result — the omission that made this whole mess possible.

## Kill rule (committed before data exists)

Scored by [`weights/prereg106_score.py`](../weights/prereg106_score.py), written and committed
**before** the arms run.

- **P-1 and P-2 both hold** → the published 14.86 is **not reproducible** and comes down. The model
  card, the recipe atlas entry and README all move to the measured distribution, quoted with N, the
  spread, and the free-RAM condition it was measured under. Same day.
- **P-3 also holds** → the mechanism is page-cache residency. `quantprobe` ships a check that
  records free RAM beside any tok/s figure and **refuses to present a number as stable when
  `model_bytes > free_RAM`**, because that is exactly the regime where our own headline broke.
- **P-2 refuted** (something does reach 14.50) → the figure is reachable but unstable, and the card
  publishes a **range with its conditions**, never again a point estimate.
- **P-1 refuted** → today's 11.0 was the anomaly, not the 14.86. Then the defect is that we cannot
  tell the two apart without re-measuring, and the tooling change in P-3's branch ships regardless.

Whatever the outcome, the correction is published at the same size as the original claim.

---

## Verdict: SCORED 2/4 (2026-08-18)

Scored by [`weights/prereg106_score.py`](../weights/prereg106_score.py), committed before the
arms ran. Raw: [`prereg106_reproduce.json`](../weights/data/prereg106_reproduce.json) ·
[`prereg106_run.log`](../weights/data/prereg106_run.log) ·
[`prereg106_verdict.txt`](../weights/data/prereg106_verdict.txt).

| | tok/s |
|---|---|
| published on the card | 14.86 +/- 0.36 |
| six fresh runs, identical command | 13.04 · 13.14 · 13.89 · 14.33 · **14.43** · 14.23 |
| after priming the page cache | **11.89** |
| control, 4.68 GB model that fits free RAM | 9.14 mean, **2.1%** spread |

- **P-1 six-run mean below 13.0 — MISS (13.84).** I predicted a persistently slow box. Wrong.
  It is not slow; it *warms*, monotonically, from 13.04 to a plateau near 14.4.
- **P-2 nothing reaches 14.50 — HIT (best 14.43).** Six attempts at the exact published command
  and not one landed inside the published error bar. The point estimate stands unreproduced,
  the best try 2.9% short.
- **P-3 priming buys at least +1.0 — MISS (-1.95), and the sign is wrong.** See below.
- **P-4 the control is no more variable — HIT (2.1% vs 10.0%).** A model that fits free RAM is
  stable and shows no ramp at all. The instability is a property of the size relationship, not
  of this box.

**Kill rule, as printed by the scorer: TODAY WAS THE ANOMALY (P-1 refuted).** *"The defect is
then that we cannot tell the two states apart without re-measuring. The free-RAM disclosure ships
regardless."* Both of those follow. The 11.0 and 11.40 readings that started #105 were the cold
end of a real range, not a broken box — and the range is 11.0 to 14.43 with nothing about the
file changing. 14.86 sits above all of it.

So the card loses its point estimate and gains a range with its condition, and `bench` ships the
residency line. Note what the kill rule did NOT let me do: P-1 was mine and it failed, and the
comfortable move would have been to read the ramp as vindication. It is not. I predicted the
wrong shape.

### P-3 is the result worth keeping

I staked that sequentially reading the whole file would warm the cache and buy at least a
tok/s. It **cost 1.95** — 11.89 against a 13.84 mean, and against the 14.43 the box had just
reached one run earlier. Priming did not fail to help; it actively destroyed a good state.

The mechanism, in hindsight: a file larger than free RAM cannot be held whole, so a sequential
read ends with the cache holding the **last** ~12 GB of the file. Six consecutive real runs
instead leave it holding the pages the workload actually re-reads — for a sparse MoE, the hot
experts and the layers that stream every token. Priming swaps a **frequency**-adapted cache for a
**position**-adapted one, and position is the wrong key.

That refutes the common `cat model.gguf > /dev/null` folklore in precisely the regime where
people reach for it: a model too big for their RAM. Registered as D-29. The warming behaviour is
L-29.

### What shipped from this

- `quantprobe bench` prints free RAM against model size on every measurement, **before** the
  noisy-run guard — 14.86 +/- 0.36 was a 2.4% spread and sailed through that guard while being
  unreproducible. Variance and reproducibility are different questions.
- The planner's own `rc - 4` OS reserve predicted 12.0 GB available against 12.22-12.29 measured
  across six runs — a 2% error, and the first direct check of that heuristic. The tool *knew* a
  13.15 GiB model would not fit. It simply never said so at bench time.
