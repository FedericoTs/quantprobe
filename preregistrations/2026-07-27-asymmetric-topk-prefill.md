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
