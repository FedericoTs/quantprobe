# Pre-registration #100: the ceiling chain — what does the recipe cost against the ORIGINAL model?

**Author:** Federico Sciuca · **Date staked:** 2026-08-11, **before the BF16 file finished
downloading, before the fragility probe has run, before any arm exists.** **STAKED.**

## Why this exists, and why on the 4B

Prereg #98 compared two children of the same Q8_0 parent and confirmed the recipe beats the naive
default (+24.0 MATH-500). What it cannot say is **what either child lost against the original
model** — the ceiling was never run. Federico's framing, verbatim in intent: *we would expect to
score less than the original (some quality loss is the honest price) but better than the naive
quantizations people find around, because we protect the fragile layers.*

The 35B ceiling is out of reach on this box: 15.9 GB RAM against a 34.4 GB Q8_0 parent means
~1 GB/token streamed off NVMe at a measured 0.476 GB/s — 20–60 days per arm. Decision recorded
2026-08-11: run the chain on **Qwen3.5-4B**, where the TRUE original — **BF16, not quantized at
all** — fits in VRAM whole (7.85 GiB). The 35B ceiling remains open and needs rented hardware
(≥40 GB); it is not answered here and nobody should read this as if it were.

**What the 4B adds beyond feasibility:** it is a SECOND model for the recipe, so it tests
generalization (#98's own limits section: "a single instance"). And it is hybrid like the 35B —
**168 `ssm_` tensors across 34 layers** — so the always-active-path mechanism from the #98
tensor audit replays: a naive Q2_K will crush the SSM path, the recipe will protect it. That
bundle IS the product; the decomposition of bundle-vs-band stays a separate question.

## Arms

| arm | file | provenance |
|---|---|---|
| **BF16** | `Qwen3.5-4B-BF16.gguf` (8,429,494,272 B expected ~7.85 GiB) | unsloth/Qwen3.5-4B-GGUF, straight conversion — THE ORIGINAL |
| **Q4_K_M** | existing EV-1 rows | downloaded quant, **different provenance** — reported as a reference point, not part of the built chain |
| **OURS-Q2K** | to build | recipe from a fragility probe run on THIS model, built from BF16 with `tools/llamacpp-b10098/llama-quantize.exe` |
| **NAIVE-Q2K** | to build | `llama-quantize --allow-requantize BF16 OUT Q2_K 8` — the command people type, same binary, same source |

Both Q2 children are built **from the BF16 arm's exact file**, so the chain parent→children is
fully local and provenance-clean. The Q4_K_M is the one impure arm and is labeled so.

## Protocol

EV-1 v3.1 exactly, unchanged from nights 2 and 3: lm-eval 0.4.12, `local-chat-completions`,
temp 0, `--reasoning off`, scoped system instruction on boxed tasks, budgets from the slot plan,
scored by `ev1_report` with the C-26 re-grade on. **Benchmarks: the powered three only** —
MATH-500, GSM8K, IFEval — as quoted in the decision (~24h box). AIME stays out: n=30 decides
nothing and its cost is real. If AIME is ever added it is an amendment BEFORE those rows run.

Existing 4B Q4_K_M rows (77.6 / 81.7 / 80.6) are reused, not re-run — same protocol, same box,
same scorer. Re-running them would be better (same-week machine state); cost says reuse, and
this line is the disclosure.

## Staked predictions (Federico's expectation, made falsifiable)

Ordered claim: **BF16 ≥ OURS-Q2K > NAIVE-Q2K**, with a real but modest loss at the top.

- **P-C1 — THE EXPECTED PICTURE:** BF16 beats OURS-Q2K on MATH-500 by **(0, 12] pts**, AND
  OURS-Q2K beats NAIVE-Q2K on MATH-500 by **≥ 5 pts**. Quality loss is real, the recipe pays
  anyway. (The 12 allows for a 4B being genuinely harder to hold at 2 bits than a 35B-A3B —
  fewer parameters, less redundancy.)
- **P-C2 — NEAR-LOSSLESS RECIPE:** OURS-Q2K within **2 pts** of BF16 on every powered benchmark.
  Would be the strongest product headline available ("3× smaller than BF16's already-quantized
  Q4, indistinguishable scores") and is NOT expected at 2 bits on a 4B.
- **P-C3 — THE RECIPE CANNOT SAVE A 4B AT 2 BITS:** OURS-Q2K loses to BF16 by **> 25 pts** on
  MATH-500. If simultaneously OURS still beats NAIVE by ≥ 5, the recipe helps but the size class
  is the binding constraint — a publishable limit on the recipe, and it ships.
- **P-C4 — ORDERING VIOLATION:** NAIVE-Q2K ≥ OURS-Q2K on MATH-500, or OURS-Q2K > BF16 by more
  than noise (> 2 pts) anywhere. Either inverts the thesis and gets front-page treatment, not a
  footnote.

Gaps between bands are possible (e.g. BF16−OURS in (12, 25]); an outcome landing there is
reported as BETWEEN STAKED BANDS, not rounded — the #99 lesson, now structural.

## Kill rules

- **KR-C1 BYTES, STATED UP FRONT THIS TIME:** OURS-Q2K vs NAIVE-Q2K will NOT be byte-identical —
  the recipe protects attention/SSM/embedding and llama.cpp's tiers are coarser than the gap.
  The premium is measured at build time and disclosed in the header of every table. If it
  exceeds **5%**, the pair is reported as size-confounded. (#98's KR-1 failed at 2.51%; the
  expected premium here is of that order.)
- **KR-C2 ONE MACHINE STATE:** C-14. One row per server session, lock held, GPU state logged.
- **KR-C3 SAME SCORER:** all arms through `ev1_report`, re-grade on. Any extractor change after
  the first new row invalidates the comparison.
- **KR-C4 DEGRADED ROWS:** KR-E3 (>10% empty/truncated) drops that benchmark for ALL arms.
- **KR-C5 NO PARTIAL VERDICTS:** scored only by `prereg100_score.py`, which must be committed
  before the first new-arm row lands. Rows are read row-at-a-time as they bank; the ordered
  claim is evaluated only when all three new arms hold the powered three.
- **KR-C6 PROBE BEFORE PEEK:** the fragility probe that picks the protected band runs BEFORE any
  benchmark row of any new arm, and its output (band, bytes) is committed when the recipe file
  is built. Choosing the band after seeing a benchmark number would be tuning.

## Cost, stated with the humility two prior misses earned

BF16 fits in VRAM; expect roughly Q4-row speeds × (bytes ratio ~3×) worst case, likely better
since it stays on-GPU: estimate **4–8 h** for its powered three. Q2 children are ~1.5 GiB,
fastest arms yet: **~3–5 h each**. Probe ~0.5 h, builds ~0.5 h, download in progress. Total
**~12–20 h of box time**. The first completed row recalibrates this line, as it did twice in #98.

## What this cannot show

The 35B ceiling (needs rented hardware). Whether the recipe's win is placement or the
always-active bundle (needs the decomposition arm, still unbuilt). And BF16-vs-BF16-served
equivalence with the official API — this BF16 runs through llama.cpp like everything else here,
which is the point: one harness, one box, one scorer, and the only variable is the file.


---

## AMENDMENT (2026-08-11, BF16 downloaded, BEFORE the probe and before any arm ran)

**A provenance investigation, run because the two 4B files disagreed on layer count - and its
resolution, which changes labels but no arm and no staked outcome.**

The downloaded BF16 has 32 blocks; our existing Q4_K_M has 33, and its size matches nothing in
today's unsloth or bartowski repos. Ground truth from `Qwen/Qwen3.5-4B` `config.json`:
`text_config.num_hidden_layers = 32` **plus `mtp_num_hidden_layers = 1`**. Reading the extra
block directly settled it: `blk.32` carries `nextn.eh_proj / enorm / hnorm / shared_head_norm` -
**it is the MTP speculation head**, GLM-style layout, present in our older unsloth conversion
and stripped from the current one (unsloth re-uploaded the whole repo on 2026-03-02).

Consequences, in order of what they touch:

1. **Same model everywhere.** Both files are the same 32-layer checkpoint; the MTP head is inert
   during standard decode and EV-1 never enabled speculation. The Q4_K_M's EV-1 rows remain
   valid references. Label sharpened from "different provenance" to: *older unsloth conversion,
   MTP head present but inert, possibly imatrix-assisted - reference point, outside the chain.*
2. **The chain parent lacks the MTP head.** Children built from this BF16 cannot carry it.
   Irrelevant here (speculation is never enabled in this protocol) and recorded so nobody is
   surprised when the children's tensor count differs from the Q4_K_M's.
3. **Nothing staked moves.** P-C1..P-C4 reference BF16 / OURS-Q2K / NAIVE-Q2K only.

Probe command fixed now, before running: `quantprobe probe --gguf <BF16> --bands 4 --chunks 32
--eval <wikitext-2 test> --llama-dir tools/llamacpp-b10098` - the same tool and band count that
produced the 35B recipe, per KR-C6.


---

## SCORED (2026-08-14, all nine cells banked, scored by pre-written code)

**P-C3 CONFIRMED: the recipe cannot save a 4B at 2 bits. And P-C1 - Federico's expected "modest
loss" - is REFUTED: the loss is large, not modest.**

| benchmark | BF16 (original) | OURS Q2_K | NAIVE Q2_K | Q4_K_M ref* |
|---|---|---|---|---|
| MATH-500 | 81.0 | 50.2 | 2.6 | 77.6 |
| GSM8K | 82.9 | 71.0 | 0.4 | 81.7 |
| IFEval | 83.7 | 61.7 | 17.0 | 80.6 |

`* Q4_K_M is the older conversion (inert MTP head) - context, not a chain arm.`

MATH-500: **BF16 - OURS = +30.8** (staked "expected" ceiling was 12), **OURS - NAIVE = +47.6**.
So P-C3's two clauses both hold: deficit > 25 AND ours still beats naive by >= 5.

### Verified before believed (the numbers are dramatic, so they were checked)

- **Health:** 0.0% empty/truncated on 8 of 9 rows; NAIVE/ifeval 0.2% empty. The naive collapse
  is REAL, not degraded rows. KR-C4 drops nothing.
- **Prompt parity:** clean - all three arms were sent identical prompts.
- **Format decomposition, MATH-500:**

  | | emitted a box | exact | correct GIVEN a box |
  |---|---|---|---|
  | BF16 | 87.8% | 81.0% | 92.3% |
  | OURS | 62.0% | 50.2% | 81.0% |
  | NAIVE | 22.4% | 2.6% | 11.6% |

### What it means, honestly

1. **Naive Q2_K on a 4B is catastrophic** - 2.6% on MATH-500, 0.4% on GSM8K. The model is
   effectively destroyed: only 22% of items even produce a boxed answer, and of those only 11.6%
   are right (near chance). This is C-30's mechanism on a hybrid model - naive 2-bit wrecks the
   always-active path (here the SSM layers) and the model stops converging.
2. **The depth-aware recipe rescues it enormously** - +47.6 on MATH-500 (2.6 -> 50.2), 0.4 -> 71.0
   on GSM8K. Placement is doing real, large work: the difference between destroyed and usable.
3. **But 2 bits on a 4B costs 30.8 points versus the original, and the recipe cannot hide it.**
   Unlike the 35B (#98/#99), where the loss was almost all answer-FORMAT, here OURS loses on BOTH
   axes: format (62.0% vs 87.8% emit a box) AND conditional reasoning (81.0% vs 92.3% correct when
   it does box). A 4B simply has too few parameters to survive 2 bits, even with perfect placement.
4. **The sensible quant for a 4B is Q4, and it is near-lossless** - Q4_K_M 77.6 vs BF16 81.0, a
   3.4-point loss. 2-bit is the wrong tier for this size class regardless of recipe.

### The finding, stated against #98

**2-bit viability is size-dependent.** On the 35B-A3B (#98) the depth-aware recipe made 2-bit
genuinely strong. On the 4B it makes 2-bit *survivable* but not good. The recipe's job -
protecting the always-active path - it does at both sizes; what changes is whether the base
model has the redundancy to absorb 2-bit at all. Big models do; a 4B does not. The honest product
line is therefore: **use the recipe to reach aggressive quant on LARGE models; use Q4 (near-
lossless) on small ones, where 2-bit is a false economy the recipe can rescue from catastrophe but
not from mediocrity.**

Federico's stake was that ours would lose to the original by a modest margin. It lost by a large
one. The prereg predicted this outcome as P-C3 and it ships as staked.

### KR-C2 disclosure

BF16/gsm8k and BF16/ifeval FAILED on first attempt (a timeout floor calibrated for quantized
models; BF16 is the first unquantized arm and ran below it - fixed in commit 38d14f8, floor 3 t/s
-> 1 t/s). They were re-run days after their OURS/NAIVE siblings. Temp-0 scoring is
clock-independent so the verdict is unaffected, but the machine-state drift is real and named
rather than hidden. MATH-500 already carried this property (BF16 led its children by ~21h).
