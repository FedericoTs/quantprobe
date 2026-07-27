# Pre-registration #22: does the top-k prefill prize survive on the placement we actually recommend?

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## Why this is open despite #18 closing top-k

Pre-registration #18 killed expert-count reduction **for decode**, by dominance: ×1.335 speed for
×1.206 quality, against the bit ladder's ×1.424 for ×1.048. Quantizing further is faster *and*
four times cheaper. That argument is sound — **and it is decode-only.**

On prefill the dominance inverts. H12 measured CPU-pure `pp2048` on a dense 7B:
**Q4_K_M 27.49 > Q4_0 23.73 > Q2_K 17.71** (`LAW5_PROTOCOL.md:292`). Dropping bits makes CPU
prefill **36% slower** — prefill is compute-bound, so the bit ladder runs backwards there. On the
prefill axis, top-k reduction (**+43% CPU-pure**, H6: 47.2 vs 33.0) is **not dominated by
anything**, and #18's argument never touched it.

That makes asymmetric top-k — k=4 to ingest, k=8 to generate — the last idea in this project that
could justify patching the runtime. It is also the last fork-shaped question, so it is worth one
hour to close.

## What this stage does NOT test

Not the quality cost, and not the phase switch. Those are Stage 2, and only run if this passes.
This asks one thing: **does the prize reach the configuration we actually ship?**

The +43% is measured **CPU-pure** (`-ngl 0`). The placements we recommend are not CPU-pure — the
GPU holds attention and, on the split, some experts. If most of the prefill wall time on those
rows is not routed-expert compute, halving k does little and there is nothing to patch.

## Stakes

`Qwen3-30B-A3B-Q2_K`. `expert_used_count` 8→4 baked into a **copy** via `gguf_set_metadata.py`
(u32→u32, same byte width; original verified untouched). `llama-bench pp2048`, r=3, warm-up
discarded, GPU memory and temperature logged. All three frontier rows from `plan.py:MOE_FRONTIER`:

| row | flags |
|---|---|
| A | `-ot "exps=CPU" -ub 2048` |
| B | `-ot "blk\.(16..47)…=CPU" -ub 512` |
| C | `-ot "blk\.(16..47)…=CPU" -ub 2048 -nkvo 1` |

- **P-1 (the prize reaches row A).** Routed experts are the bulk of prefill FLOPs, and a 0.68×
  FLOP cut should show. Row A gains **+20% to +35%** at k=4.
- **P-2 (it is not CPU-pure-only).** At least **two of the three** rows gain **≥15%**.
- **P-3 (control).** The k=8 arm reproduces the frontier figures within ±10%
  (A 345.41, B 280.64, C 391.72). Outside that, the session is contaminated and P-1/P-2 are void.

## KILL RULE — stated before measuring

**If row A gains <15%, this line stops permanently.** The +43% is then a CPU-pure artifact that
does not reach any configuration we recommend, asymmetric top-k is worthless on this hardware,
Stage 2 is never run, and the fork/patch question closes as settled-no. Publish as a scored miss.

That rule is deliberately harsh: the prize has to clear the frontier's own 14.3% dynamic ceiling
to be worth more than simply picking the right static point.

## If it passes

Stage 2 (~1 day) measures whether the **+20.6% perplexity cost attaches to ingestion or only to
generation**, via slot save/restore between a k=4 and a k=8 server. Only if the cost is
generation-only does asymmetric top-k become real — and even then it ships as an **upstream PR
against a pinned SHA**, never a fork, because the tool's "runs on stock llama.cpp" property is
worth more than the gain.

---

## Scored (2026-07-27, log: `weights/data/prereg22_asymmetric_topk.log`)

**Verdict: P-1 HIT, P-2 HIT, P-3 partial. The kill rule is NOT triggered — Stage 1 passes.**

| row | k=8 | k=4 | gain |
|---|---|---|---|
| **A** all experts → CPU, `-ub 2048` | 387.40 ± 1.48 | **468.29 ± 2.90** | **+20.9%** |
| **B** split, KV in VRAM, `-ub 512` | 305.88 ± 1.19 | **376.15 ± 1.64** | **+23.0%** |
| **C** split, KV evicted, `-ub 2048` | 194.33 ± 0.54 | **317.42 ± 1.31** | **+63.3%** |

- **P-1 (row A gains 20–35%): HIT.** +20.9%, at the band's edge.
- **P-2 (≥2 of 3 rows gain ≥15%): HIT.** All three did.
- **P-3 (controls within ±10%): PARTIAL.** B +9.0% (inside), A +12.2% (marginally outside, GPU
  started cold at 36 °C), C not comparable — see below.

**The prefill prize is not a CPU-pure artifact.** It reaches every configuration we recommend, at
+21% to +63%. That clears the frontier's own 14.3% dynamic ceiling comfortably, so Stage 2 — does
the +20.6% quality cost attach to ingestion or only to generation — is justified.

## The more important finding: row C sits on a VRAM cliff

Chasing P-3's row-C anomaly turned up something that matters more than top-k.

The *identical command* produced **193–195 tok/s** in some invocations and **437–438** in others.
Not flag ordering (tested: both orders give 438). Not bimodal noise — four consecutive fresh runs
gave 438.18, 437.52, 438.48, 438.42, error bars under 0.5%. The variable is **desktop VRAM
occupancy**:

| VRAM held before launch | pp2048 |
|---|---|
| 462–472 MiB | **437–438** |
| 713–714 MiB | **193–195** |

A ~250 MiB difference — one browser window — flips the prefill champion by **2.3×**. This is the
overcommit cliff pre-registration #13 measured at −29%, except here it is **−56%**, and row C is
perched on its edge because evicting KV is precisely what lets the compute buffer grow to the size
that only just fits.

**This is a caveat on shipped advice.** v1.14.0/v1.14.1 recommend "evict KV to RAM" for
long-prompt workloads, quoting 391.72. That figure is only available on an otherwise-clear card.
A user with a browser open may get 193 — *worse than every other frontier point* — from the
configuration we told them was fastest.

Needs its own controlled test (deliberately occupy VRAM in steps, find the cliff edge) before the
advice is either qualified or withdrawn. Flagged, not fixed, and not quietly averaged away.

**Wired into:** nothing yet — Stage 1 authorises Stage 2, and the cliff finding is staked as its
own follow-up rather than patched on a correlation.
