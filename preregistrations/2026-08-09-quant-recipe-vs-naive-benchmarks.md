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


---

## AMENDMENT (2026-08-09, naive arm built, BEFORE any benchmark item was evaluated)

**KR-1 FAILED. The arms are NOT byte-matched, and this is a size-confounded comparison.**

| arm | bytes | vs ours |
|---|---|---|
| OURS depth-aware | 13,272,701,568 | - |
| NAIVE plain Q2_K | 12,939,594,368 | **-2.51%** |

Band was +/-1.50%. Build took 1,046 s.

**No re-pick is possible without a worse confound, and that is why this is a disclosure
rather than a second attempt.** A base inside the window would have to land between 12.18 and
12.55 GB. llama.cpp's tiers are coarser than the tolerance: Q2_K is the largest K-quant below
Q3_K_S, and Q3_K_S is roughly 15% larger. The only thing in the window is IQ3_XXS at 3.06 bpw
- a **codebook** format. Swapping format family to fix 2.5% of bytes would introduce exactly
the confound L-15 says dominates, so it is not a fix, it is a different experiment.

**How this comparison must now be read.** OURS carries a 2.51% byte premium. The direction is
conservative for the sceptic and unfavourable to us: any win OURS posts is partly bought with
more space, so a confirmed P1 means *"the recipe plus 2.5% more bytes"*, not *"the recipe"*.
A refuted P2 would be correspondingly stronger, since OURS would be losing while spending more.

**Thresholds are UNCHANGED.** Moving P1's 2.0-point bar after a kill rule failed is moving the
goalposts, and the whole point of writing KR-1 down was to make this moment cost something.

**Row order fixed now:** MATH-500, GSM8K, IFEval, then AIME 2024, AIME 2025. The three
verdict-bearing benchmarks run first, so an interrupted run still decides the question. This
is sequencing, not scope - all five still run, and AIME is still excluded from the verdict
under the power argument above.

**Cost, stated so it is a decision and not a surprise:** roughly 12 hours per arm on this card
at 30B-class speeds, about a day for the pair.


---

## COST CORRECTION (2026-08-09 17:50, one row in flight, no scores seen)

**My cost estimate above was wrong by 1.6x and is corrected here rather than quietly left.**

I wrote "roughly 12 hours per arm, about a day for the pair". The real figure, derived from
the EV-1 rows this box has already measured on a comparable 30B-A3B at Q2_K-class, is:

| benchmark | 30B measured | 35B estimate (x1.05) |
|---|---|---|
| MATH-500 | 484 min | 508 min (8.5 h) |
| GSM8K | 222 min | 233 min (3.9 h) |
| IFEval | 208 min | 218 min (3.6 h) |
| AIME 2024 | 96 min | 101 min (1.7 h) |
| AIME 2025 | 99 min | 104 min (1.7 h) |

**~19.4 h per arm. ~38.8 h - about 1.6 days - for the pair.**

Where the estimate went wrong: I priced the pair off GSM8K and IFEval, which are the rows I
had watched most recently, and MATH-500 is by far the longest row in the suite at 484 minutes.
Its answers are long and its budget is 3,072 tokens. Anchoring on the rows most available to
memory rather than the one that dominates the total is an ordinary estimation error, and the
fix is that the numbers were on disk the whole time and I did not look until the run was live.

**SCOPE IS UNCHANGED. All five benchmarks still run.** Dropping AIME would save 6.8 h and
cost nothing for the verdict, since AIME is already excluded from P1/P2/P3 on power grounds -
but trimming scope to save my own wall-clock is Federico's call, not mine, and it is offered
rather than taken. If he says trim, the amendment lands before any result is read.


---

## CORRECTION (2026-08-09 19:30, still no scores read)

**The KR-1 amendment above contains a false statement and I am not leaving it standing.**

It says: *"No re-pick is possible without a worse confound."* That is wrong. A byte-IDENTICAL
arm is possible, and this project published the method three weeks ago.

`docs/DEEP-DIVE.md` and the arXiv paper both carry it: **Gemma-4-12B, two byte-identical 5.22
GB files, first-12 FFN blocks protected (12.27 ppl) versus last-12 protected (10.02 ppl) -
a 2.25 ppl swing from position alone.** The bytes match exactly because the arms protect the
SAME NUMBER of blocks with the SAME bits, and differ only in WHICH blocks.

Applied here that is a POSITION-SWAPPED arm: protect layers 0-9 with `q4_k` instead of the
recipe's 30-39, everything else identical. Byte-identical by construction, no format change,
no tier hunting. The reason I reported KR-1 as unfixable is that I only considered changing
the naive arm's BASE, and never considered holding the recipe's shape and moving its position.

**The two designs answer different questions and both are worth having:**

| arm pair | isolates | byte-matched |
|---|---|---|
| OURS vs NAIVE (running) | does the recipe beat the default command? | no, -2.51% |
| OURS vs POSITION-SWAPPED | does WHERE the protected bits sit matter? | exactly |

The first is the product question. The second is the science question, and it is the one Law
3 and the placement thesis actually rest on - which makes its absence here the bigger gap.

**NOT DONE, and deliberately:** the swapped arm is not built. Quantizing is ~17 minutes of
8-thread CPU work and night 3 is measuring right now; contaminating a running comparison to
improve a future one is a bad trade and a C-14 violation. Evaluating it would also add ~19
hours, which is a scope increase and Federico's call. Recorded here so the option survives
this session, and so the false sentence above is not the last word on it.


---

## SCORED (2026-08-11 05:30, all six verdict-bearing rows complete, scored by pre-written code)

**P1 CONFIRMED. P2 refuted. P3 refuted.** Scored by `weights/prereg98_score.py`, which was written
and committed (a856a47) while five of the six rows did not yet exist and none had been read.

| benchmark | NAIVE | OURS | delta | staked bar |
|---|---|---|---|---|
| MATH-500 (primary) | 57.0% | 81.0% | **+24.0** | >= +2.0 |
| GSM8K | 75.8% | 84.9% | **+9.1** | not worse than -1.0 |
| IFEval | 72.5% | 84.7% | **+12.2** | not worse than -1.0 |

Row wall-clock: MATH-500 616/593 min, GSM8K 272/287, IFEval 202/191 (NAIVE/OURS).

### The margin is 12x the staked bar, so the artifact was checked before the number was believed

Per C-28 - a check that reads the code tests what you meant; only a check that reads the artifact
tests what the machine did. All five passed:

- **Response health:** 0.0% empty and 0.0% truncated on all six rows. KR-4 drops nothing.
- **Prompt parity:** `ev1_report.verify_prompts()` returns no mismatches. Both arms were SENT the
  same thing, including the boxed-answer system instruction.
- **Scorer parity:** the C-26 re-grade moved 0 items on both arms (0 rescued, 0 lost), so the
  extractor fix cannot be an asymmetry between them.
- **File identity:** the SESSION START banners record `Qwen3.5-35B-A3B-naive-q2k.gguf` and
  `Qwen3.5-35B-A3B-ours-depthaware.gguf` on their own arms. The arms are not swapped.
- **KR-1 stands:** ours carries +2.57% bytes. Every win reads as "the recipe PLUS 2.5% more bytes".

### WHAT THE WIN ACTUALLY IS, AND IT IS NOT BETTER ARITHMETIC

The same scorer computes format compliance for both arms, and it reframes the headline:

| | emitted a `\boxed{}` | exact_match | correct GIVEN a box |
|---|---|---|---|
| NAIVE | 64.4% | 57.0% | 88.5% |
| OURS | 86.4% | 81.0% | 93.8% |

**+22.0 of the +24.0 gap is whether the model emitted an answer in the requested format at all.**
Conditional on emitting one, the arms differ by 5.2 points, not 24. Naive Q2_K did not mostly
forget how to do maths; it largely lost the ability to follow the answer-format instruction.

The three benchmarks corroborate this and are ordered exactly as the format hypothesis predicts -
strictest format shows the largest gap, most format-tolerant the smallest:

    MATH-500  strict \boxed{} required, our extractor       +24.0
    IFEval    measures instruction-following directly       +12.2
    GSM8K     flexible-extract, format-tolerant by design    +9.1

IFEval is the independent confirmation: it needs no extractor at all and still shows a 12-point
instruction-following collapse in the naive arm.

**The defensible claim is therefore narrower and more interesting than "our quantization scores
24 points higher on maths":** naive Q2_K on this MoE substantially damages instruction-following,
and a depth-aware recipe that spends q4_k on attention and the token embedding preserves it, for
2.5% more bytes. That is a claim about what low-bit quantization breaks FIRST.

### A re-scoring we are NOT doing

The obvious next move is to re-score MATH-500 with a format-agnostic extractor to see how much of
the gap survives. **KR-3 forbids it:** any extractor change after the arms have run invalidates
both. The `emitted_boxed` column above is admissible because the SAME scorer already computed it
for both arms as a diagnostic; swapping the extractor would be a different experiment and must be
staked separately, before it is run, or it is post-hoc re-scoring wearing an analysis.

### Limits, unchanged from the top of this document

One model, one family, one size, one bit-depth, one card. This still cannot separate "protecting
the right layers" from "spending more bits on attention and the token embedding" - the recipe does
both, and the position-swapped byte-identical arm named in the 19:30 correction is what would
decompose it. AIME 2024/2025 are running now and remain excluded from the verdict on power.
