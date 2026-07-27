# Pre-registration #24: is η a function of bytes per token, or of format?

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## Why this is open

`verify.py` layer 3 is RED. The law under-predicts the configuration most users actually land on —
a model that simply fits in VRAM — by **+12% to +85%** across eight points in two sessions. This is
the opposite of the usual failure direction: we are telling people a model is *less* usable than it
is, which pushes them toward complicated offload placements that are slower for them than `-ngl 99`.

Two explanations have already been eliminated, and recording that is the point of doing this
properly:

- **The fixed-overhead model is refuted** (pre-registration #15). It predicted a small model would
  be *slow*; the 0.6B is fast, and the existing law is right to within 1.5% there. Not a near-miss —
  the mechanism is wrong.
- **GPU clock state is refuted.** Stated as a hypothesis before testing: a warm card should measure
  higher. Cold gave 144.21, warm 143.37. Refuted by its own prediction.

What survives is #15's observation that **measured η rises with bytes per token** — 0.354 → 0.461 →
0.560 across 0.73 → 5.38 GB/token. #15 refused to fit a curve to three points. This adds the
intermediate points it specified, and asks the question three points could not: **is the axis bytes
per token, or is it format?**

That distinction matters because I got it wrong earlier today. Seeing Qwen2.5-7B at Q2_K miss by
+20% and the same model at Q4_K_M miss by +85%, I read it as a format effect. It is not obviously
that: the two also differ by 1.55 GB of bytes per token, and once ordered on that axis a format term
is unnecessary. Both readings fit four points. They make **different predictions** about I-quants,
which is what this measures.

## Design

`Qwen2.5-7B-Instruct` at four quantizations — **one architecture, two format families, bytes per
token spanning 1.55×**. Nothing else varies. This is the comparison the register did not have.

| arm | file | GB | family |
|---|---|---|---|
| A1 | `Qwen2.5-7B-Instruct-Q2_K` | 2.809 | K-quant |
| A2 | `Qwen2.5-7B-Instruct-IQ3_XS` | 3.116 | **I-quant** |
| A3 | `Qwen2.5-7B-Instruct-IQ3_M` | 3.329 | **I-quant** |
| A4 | `Qwen2.5-7B-Instruct-Q4_K_M` | 4.361 | K-quant |

Plus **B1** `Qwen3.5-4B-Q4_K_M` as a within-session reproducibility control.

All `-ngl 99` (all-in-VRAM), `llama-bench`, `r=3`, ONE session, GPU state logged before and after.

**Method for η.** Not recomputed by hand — derived from the tool's own arithmetic, so it cannot
disagree with the shipped law. The law is `predicted = η_assumed × BW / bytes`, so
`η_actual = η_assumed × measured / predicted` with `η_assumed = 0.35`. Bytes per token likewise
follows from `predicted`. This keeps the analysis inside the model being tested.

## Stakes

- **P-1 (bytes per token is the axis).** Within arm A — fixed architecture — measured η rises
  **monotonically** with bytes per token across all four points.
- **P-2 (format is a SEPARATE axis).** The two I-quant points fall **≥5% BELOW** the line drawn
  through the two K-quant points (A1, A4). I-quants are known to cost more compute per byte to
  dequantize, so if η is really about how well the memory system is used, the extra compute should
  pull them off a purely byte-driven curve. **If they land on the line, format is not a separate
  axis and bytes per token alone explains η** — which is the cleaner result and the one that would
  let a one-parameter correction ship.
- **P-3 (the between-session drift is real and bounds everything).** B1 reproduces today's 30.89
  within **±3%** in this session, confirming that the 27.30 measured on 2026-07-26 differs by
  session and not by change. Sub-1% error bars within a session and 13% between them is the reason
  no constant has moved yet.
- **P-4 (it is not a small-model artifact).** All four arm-A points under-predict by **≥15%**.

## KILL RULE — stated before measuring

**If P-1 fails — η is not monotone in bytes per token at fixed architecture — then bytes per token
is not the axis, #15's lead is dead, and nothing is fitted to it.** The register entry C-02 gets
its leading hypothesis struck out and the search starts again. I am not going to keep an
explanation alive by adding terms to it; that is how the overhead model survived a day longer than
it deserved.

## What ships

**Nothing automatic, and possibly nothing at all.** A curve through five points on one GPU is a
lead, not a calibration — that was #15's conclusion and it is not weakened by having more points on
the same single card. What ships is at most:

1. The register entry C-02 updated with which axis survived.
2. If P-1 and P-3 both hold, a *staked* proposal for a bytes-per-token-dependent η, to be measured
   on a second GPU before any published number moves.

Layer 4's anchors must be audited for all-in-VRAM membership before any constant changes. If an
anchor is all-in-VRAM and currently retrodicted correctly, then it and this finding are in direct
conflict and that conflict is the result, not an obstacle to route around.

**Explicitly NOT claimed:** that this generalises off a GTX 1060. Every η here is measured on one
memory system, and "larger tensors use this GPU's memory system better" is a hypothesis about *this
GPU*.
