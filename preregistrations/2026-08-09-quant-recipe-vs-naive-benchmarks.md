# Pre-registration #98: does a depth-aware quantization recipe change BENCHMARK scores, or only perplexity?

**Author:** Federico Sciuca · **Date staked:** 2026-08-09, **while the naive arm is still
building and BEFORE either arm has been evaluated on a single item.** **STAKED.**

## The one comparison the capability ladder cannot make

EV-1 measured four models on one card and produced a useful table, but it is **not a
controlled experiment** and the chart now says so: three model generations, three quantization
tiers and one code specialist sit on those rows. Nothing in it isolates a single variable.

This does. Same weights, same source file, same quantizer binary, same machine, same harness,
same protocol, same day. **The only thing that differs is which layers got which bits.**

| arm | build | bytes |
|---|---|---|
| **NAIVE** | `llama-quantize --allow-requantize SRC OUT Q2_K 8` — no `--tensor-type` at all | measured at build |
| **OURS** | the `qwen3.5-35b` recipe: FFN `q2_k` outside the fragile band, `q4_k` inside it, `attn_.*=q4_k`, `--token-embedding-type q4_k` | 13,272,701,568 |

Source for both: `D:/evo-compress-data/gguf/Qwen3.5-35B-A3B-Q8_0.gguf` (36,903,139,968 bytes).
The OURS arm already exists and its build command is recorded verbatim in
`weights/data/apex_ab_stageB2_resize.log:942`. NAIVE is being built now with the same binary
(`tools/llamacpp-b10098/llama-quantize.exe`), same thread count, from the same source.

"Naive" means what a normal person gets by typing the obvious command. It is not a strawman:
llama.cpp's own Q2_K already applies internal per-tensor rules. The question is whether OUR
per-layer plan, chosen from a measured fragility probe, beats the sensible default.

## Why this is the honest version of the "we quantize better" claim

Every prior comparison we own is perplexity or KLD on held-out text. That is a proxy. A reader
is entitled to ask whether it survives contact with the benchmarks people actually quote, and
the answer is not obvious - a recipe can reduce divergence from the fp16 model while leaving
task accuracy untouched, because accuracy is a coarse threshold on top of the distribution.

**P3 below is that outcome, and it is stated as a prediction rather than kept as an excuse.**

## Arms and protocol

Both arms run the EV-1 protocol exactly: lm-eval 0.4.12, `local-chat-completions`, temp 0,
full sets, `--reasoning off`, the answer-format instruction on boxed tasks only, budgets
derived from the slot plan, and scoring through `weights/ev1_report.py` (so both arms get the
C-26 re-grade identically).

Benchmarks: **MATH-500 (500), GSM8K (1,319), IFEval (541), AIME 2024 (30), AIME 2025 (30)**.

**POWER IS DECLARED UP FRONT, because it decides what this experiment can and cannot say.**
At n=30 the AIME rows have a standard error near 7pp and can only detect a landslide. The
powered benchmarks are MATH-500 (stderr ~1.9pp), GSM8K (~1.1pp) and IFEval (~1.7pp). **P1/P2
are scored on the powered three only.** AIME is reported for completeness and explicitly
excluded from the verdict - deciding that afterwards would be choosing the benchmark that
gave the answer we liked.

## Staked predictions

- **P1 — THE RECIPE PAYS (confirmed):** OURS beats NAIVE on MATH-500 by **>= 2.0 pts**, and
  is **not worse by more than 1.0 pt** on either GSM8K or IFEval.
- **P2 — THE RECIPE HURTS (refuted):** NAIVE beats OURS on MATH-500 by >= 2.0 pts, or OURS
  loses by more than 1.0 pt on two or more of the powered three. This is the outcome that
  would retire the depth-aware recipe as a quality claim, and it ships if it happens.
- **P3 — PERPLEXITY IS NOT ACCURACY (null):** every powered benchmark differs by **< 2.0 pts
  in absolute value**. The recipe demonstrably moves KLD and perplexity; if it does not move
  benchmark accuracy at this size, that is a real and publishable limit on what the recipe
  buys, and it does NOT retroactively become "we were only ever claiming perplexity".

## Kill rules

- **KR-1 BYTES:** the two files must land within **1.5%** of each other. Outside that, size is
  a confound and the comparison is reported as size-confounded, not as a recipe result - the
  exact mistake the capability ladder made and had to be rebuilt for. If NAIVE comes out far
  from 13,272,701,568 bytes, the fix is a differently-sized naive base chosen BEFORE seeing any
  score, or an honest disclosure. Never a re-pick afterwards.
- **KR-2 ONE MACHINE STATE:** C-14. One row per server session, the shared lock held, GPU
  state logged, no other measurement on the box. Both arms run back to back on the same day.
- **KR-3 SAME SCORER:** both arms scored by `ev1_report` with re-grading on. Any extractor
  change after the first arm runs invalidates both.
- **KR-4 DEGRADED ROWS:** if either arm trips KR-E3 (>10% empty or truncated) on a benchmark,
  that benchmark is dropped from the verdict **for both arms**, not just for the one that
  tripped it.
- **KR-5 NO PEEKING:** the arms are run to completion before any comparison is drawn. Partial
  results are not consulted to decide whether to continue.

## What this cannot show

One model, one family, one size, one bit-depth (Q2_K-class), one card. It cannot say the
recipe generalises to dense models, to 4-bit, or to other architectures - the fragile band is
model-specific by construction (Law 3), and this is a single instance of applying it. It also
cannot separate "protecting the right layers" from "spending more bits on attention and the
token embedding", because the recipe does both. Isolating those needs a third arm and is out
of scope here; it is named so nobody mistakes this for the decomposition.
