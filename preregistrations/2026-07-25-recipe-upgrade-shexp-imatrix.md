# Pre-registration #12: closing the two gaps APEX exposed — shared-expert protection + imatrix

**Author:** Federico Sciuca · **Date staked:** 2026-07-25, committed BEFORE any build or
measurement in this program was run. **Scoring: same day.**

## Context

Pre-registration #11 (APEX Mini vs our probe-built depth-aware, matched ~13.3 GB on
Qwen3.5-35B-A3B) ended: APEX 6.1511, community IQ2_M 6.3939, ours 6.6976 — we lost quality by
+8.88%, won CPU prefill by only x1.16. Two concrete gaps in OUR recipe were identified there and
are tested here, isolated one at a time:

1. **Shared-expert tensors unprotected.** `ffn_*_shexp` (the ALWAYS-active expert, distinct from
   the 256 routed ones) falls into our generic `ffn_.*` band pattern and lands at the aggressive
   Q2_K base. APEX pins these at Q8_0 everywhere, citing heavy-tailed weight distributions
   (kurtosis 13.10 vs 3.41 for routed experts) — a structural argument, not a depth one. Every
   token passes through this tensor; only ~8/256 routed experts fire per token.
2. **No importance-matrix calibration anywhere in this project.** `llama-quantize --imatrix` is
   stock and we have never used it, on any model, ever. The field (APEX's i-variants, Unsloth's
   UD quants — note the IQ2_M "reference" that beat us is itself a smart dynamic recipe) has
   broadly converged on it.

**Verified before staking** (not assumed): llama.cpp resolves overlapping `--tensor-type`
patterns **first-match-wins** (tested on Qwen3-0.6B: `ffn_.*=q2_k` beat a later
`ffn_down.*=q4_k`). The shared-expert rule is therefore placed FIRST in the command; placed
last it would silently do nothing and this experiment would have measured a false null.

## Method

Source: `Qwen3.5-35B-A3B-Q8_0.gguf`. Band held FIXED at layers 34-39 (the S-1 measured fragile
band) across every build — only the tested variable changes. Quality: WikiText-2 test,
ctx 2048, 32 chunks, hybrid (`-ngl 99 -ot exps=CPU`), identical to #11. Speed: CPU-pure
(`-dev none`) pp2048, r2. GPU state logged per Law-5 convention.

**Calibration corpus: `wiki.train.raw`** — the TRAIN split, distinct from the `wiki.test.raw`
eval split. No test-set contamination.

**Disclosed limitation, stated before measuring:** this is *in-domain* calibration (Wikipedia
train → Wikipedia test), the most favourable possible setup for a wikitext-ppl metric. APEX's
i-variants deliberately use a DIVERSE corpus (chat/code/reasoning/tool-calling, explicitly no
Wikipedia) because they optimise for real-world accuracy over wikitext ppl. Any gain measured
here is therefore an **upper bound** on what a general-purpose build would show, and is not
directly comparable to APEX's i-variant methodology. Treat P-2/P-3 as "does imatrix work at
all, and how big is the lever" — not as a general-purpose quality claim.

## Builds (each isolates one variable)

| build | change vs #11's build | expected size |
|---|---|---|
| **A** | + `ffn_.*_shexp.*=q8_0` (placed first) | ~13.36 GB (+87 MiB) |
| **B** | + `--imatrix` (wiki.train, 100 chunks) | ~13.27 GB (unchanged) |
| **C** | both | ~13.36 GB |

Baseline for all deltas: #11's SSM-fixed build, **6.6976 @ 13.27 GB**.

## Stakes

- **P-1 (shared-expert protection alone).** Ppl reduction of **2-8%** → lands in
  **[6.16, 6.56]**. Mechanism: always-active + heavy-tailed is the worst possible combination
  for 2-bit. Below 2% means the kurtosis argument doesn't transfer to this model and APEX's
  Q8_0 pinning is over-engineering; above 8% means always-active tensors dominate far more than
  their 0.35% byte share suggests.
- **P-2 (imatrix alone).** Ppl reduction of **5-15%** → lands in **[5.69, 6.36]**. Below 5%
  would mean the field's convergence on imatrix doesn't hold at this bit-level on MoE, which
  would itself be a publishable negative.
- **P-3 (combined — the headline).** **The combined build beats APEX Mini's 6.1511** at matched
  bytes, landing in **[5.60, 6.15]**. Not fully additive: staked at 8-16% total reduction.
  This is the falsifiable claim — if C lands above 6.1511 we still lose the head-to-head after
  closing both known gaps, and that publishes as a loss.
- **P-4 (speed is unaffected).** CPU-pure pp2048 of build C stays within **±5% of 43.67**
  (i.e. [41.5, 45.9]). imatrix changes weights, not format; the shexp promotion is ~0.65% of
  bytes. A miss here means quantization content affects CPU GEMM throughput in a way our
  format-level η_pp model doesn't capture.

## Refuted if

Any of the above bands is missed. Misses publish with the same prominence as hits. If P-1 or
P-2 individually miss but P-3 hits (or vice versa), both results publish — the isolation is the
point, and a surprising interaction is a finding, not an inconvenience.

## Commitment

Whatever wins is baked into `quantprobe` for ALL models, not just this one — with the
architectural lesson from #11 applied: protection should be **structural** (always-active
tensors, attention, SSM, embeddings protected by default; the measured depth-gradient applied
only to genuinely redundant routed-expert weights), not a growing list of named regex
exceptions. A win here is a product change, not a paper result.
