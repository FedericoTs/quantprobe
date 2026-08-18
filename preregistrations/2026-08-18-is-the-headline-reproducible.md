# Pre-registration #106: is our published headline reproducible on the box that produced it?

**Author:** Federico Sciuca · **Date staked:** 2026-08-18, **before any arm below was run.**
**STAKED.**

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
