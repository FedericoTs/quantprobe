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

---

## Scored (2026-07-27, log: `weights/data/prereg24_eta_bytes_per_token.log`)

**Verdict: P-1 HIT, P-2 HIT, P-3 HIT, P-4 HIT. Four for four — and the control point carries the
result, which is not what the stakes were pointing at.**

η derived from the tool's own arithmetic (`η = 0.35 × measured/predicted`), so the analysis cannot
disagree with the law it is testing. GB/token = `66.9 / predicted`, the constant cross-checked
against #15's published table (0.73 × 91.7 = 66.9, 3.24 × 20.6 = 66.7, 5.38 × 12.5 = 67.3).

| arm | GB/token | measured | predicted | gap | **η** |
|---|---|---|---|---|---|
| B1 4B Q4_K_M *(control)* | 3.25 | 30.03 ± 0.79 | 20.6 | +46% | **0.510** |
| A1 7B Q2_K | 3.76 | 21.63 ± 0.12 | 17.8 | +22% | **0.425** |
| A2 7B IQ3_XS | 4.05 | 20.57 ± 0.04 | 16.5 | +25% | **0.436** |
| A3 7B IQ3_M | 4.26 | 19.86 ± 0.06 | 15.7 | +27% | **0.443** |
| A4 7B Q4_K_M | 5.35 | 22.55 ± 0.07 | 12.5 | +81% | **0.631** |

- **P-1 (η monotone in bytes/token at fixed architecture): HIT.** 0.425 → 0.436 → 0.443 → 0.631
  across arm A. The kill rule is not triggered.
- **P-2 (format is a SEPARATE axis): HIT.** The K-quant line through A1 and A4 predicts η = 0.463
  at A2's bytes and 0.490 at A3's. Measured 0.436 and 0.443 — **5.9% and 9.6% below**, both clearing
  the staked 5%.
- **P-3 (within-session reproducibility): HIT.** B1 gave 30.03 against 30.89 earlier the same day,
  **−2.8%**, inside the staked ±3% — while sitting **+10%** above 2026-07-26's 27.30. The drift is
  between sessions, not within them, exactly as feared.
- **P-4 (not a small-model artifact): HIT.** All four arm-A points under-predict by ≥15%
  (+22%, +25%, +27%, +81%).

### The control refutes the hypothesis the stakes were built around

P-1 passing looks like support for bytes per token. **It is not, and the control is why.**

**B1 has FEWER bytes per token than every sub-4-bit 7B point (3.25 vs 3.76–4.26) and a HIGHER η
(0.510 vs 0.425–0.443).** If bytes per token drove η, that is impossible. Arm A's monotonicity is
confounded: within one architecture, bytes and bit-width rise together, so a purely format-driven η
would produce the same monotone sequence.

Read by format class instead, all five points fall into place, and so do the eight from before:

| format class | η |
|---|---|
| sub-4-bit (Q2_K, IQ3_XS, IQ3_M) | 0.425, 0.436, 0.443 — **flat within 4% across a 13% spread in bytes** |
| 4-bit (Q4_K_M) | 0.510 (4B), 0.631 (7B) — and #15 measured 0.461 (4B), 0.560 (7B) |

Two factors, both real, neither sufficient alone:

1. **Format class sets the level.** At matched bytes, 4-bit runs far closer to peak bandwidth than
   sub-4-bit. The three sub-4-bit points are nearly flat while bytes move 13%; the 4-bit points sit
   0.07–0.19 higher.
2. **Within a format, η still rises with size.** Q4_K_M: 0.510 → 0.631 here, 0.461 → 0.560 in #15.
   The same direction in both sessions.

This is the same shape as **D-06**, where a bit-width gate had to be replaced by a format property —
and it is the second time in this project that a quantity assumed constant across formats turned out
to be a property *of* the format. That is now a pattern worth naming rather than rediscovering.

### What ships

**Nothing, yet — and the reason is P-3.** Between-session drift is 10–13% with sub-1% error bars
inside a session. A two-factor η fitted across sessions would be fitting that drift, and a
one-parameter correction is now known to be wrong anyway. What changes is the register:
bytes-per-token is **demoted from leading hypothesis to a second-order term within a format**, and
format class is promoted to the primary axis.

The remaining blocker is unchanged and is stated again because it gates everything: **layer 4's
anchors must be audited for all-in-VRAM membership before any constant moves.** If an anchor is
all-in-VRAM and currently retrodicted correctly, it is in direct conflict with a +81% miss on the
same tier, and that conflict is the result rather than an obstacle.

**Explicitly NOT claimed:** that any of this generalises off a GTX 1060. Every η here is one memory
system, and "4-bit kernels use it better" is a hypothesis about *this GPU*.

**Wired into:** `findings/REGISTER.json:C-02` (leading hypothesis replaced) and the new
`findings/REGISTER.json:C-05`. Deliberately NOT wired into the planner — no constant moves until the
anchor audit runs and a second GPU confirms it.
