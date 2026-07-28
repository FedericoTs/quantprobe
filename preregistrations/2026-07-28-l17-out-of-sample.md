# Pre-registration #59: does L-17 predict models it has never seen?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurements. **STAKED.**

## Why this exists

L-16 + L-17 were built on exactly two configurations (the MoE flagship split, the 7B Q2_K
all-in-VRAM) and reconciled both within 5% — but the floor constant was FIT to the same data it
explains. A law earns the word only out-of-sample. This stakes numeric predictions for **two
models the law has never seen**, at opposite extremes of the call-granularity regime, computed
from GGUF structure + the session's constants BEFORE running tg128.

Structural inputs (measured with a 9-token count probe — counts only, not times):

| | Qwen2.5-0.5B Q8_0, all-in-VRAM | DeepSeek-Coder-V2-Lite Q4_K_M, split |
|---|---|---|
| architecture | 24 layers, hidden 896, dense | 27 layers, hidden 2048, 64 experts (6 used), **MLA attention** |
| GPU nodes/token | **293** | **747** (17 expert layers on CPU via `-ot`) |
| GPU bytes/token | 0.525 GB (tied lm_head reads embd) | ~0.94 GB (0.63 always-active + 9 resident MoE layers × 0.0347) |
| CPU bytes/token | — | ~0.59 GB (17 layers × 0.0347) |

## The predictions, with every assumption stated

Constants carried in: floor 16 µs/call (L-17, fitted on the two prior arms); Q8_0 format rate
**123 ± 8 GB/s** (assumed from Q4_0's 119 e2e + kernelprobe int8 124 — Q8_0's metadata is coarser
per byte than Q4_0's, L-16 predicts ≥); DS GPU mix (Q4_K/Q8_0/Q5) **100 ± 12 GB/s**; CPU 28 GB/s
(L-11); glue: +1.0 ± 0.5 ms (all-in-VRAM), +5 ± 3 ms (split, 17 boundaries).

**Model A (0.5B):** device = 0.525/0.123 + 293×16µs = 4.27 + 4.69 = 8.96 ms; token ≈ 9.96 ms →
**~100 tok/s**. The byte-only model (no floor) predicts **189 tok/s** — the floor term nearly
doubles the token, so this arm DISCRIMINATES hard.

**Model B (DS-Lite split):** GPU = 0.94/0.100 + 747×16µs = 9.4 + 12.0 = 21.4 ms; CPU = 21.0 ms;
token ≈ 47.4 ms → **~21.1 tok/s**. Byte-only predicts 28.2.

## Stakes

- **P-1 (0.5B).** Measured tg128 lands in **[85, 130] tok/s**. Explicit alternative outcomes,
  decided in advance: **> 160** → the byte-only model wins and L-17's floor is refuted;
  **[130, 160]** → the floor is real but the 16 µs constant is size-dependent (small kernels pay
  less) — the law's FORM survives, its single-constant version does not, and I say so.
- **P-2 (DS-Lite).** Measured tg128 lands in **[17.5, 25.5] tok/s**. **> 27** → floor
  overestimated; **< 16** → the model is missing a term (MLA graph cost, scheduler, PCIe).
- **P-3 (the generalization claim).** BOTH P-1 and P-2 inside their primary bands → L-17
  transfers across architectures (dense/MoE, GQA/MLA, Q8_0/K-quant mix) ON THIS BOX with fixed
  constants. Machine generalization remains untested (one box) and is claimed by NO outcome here;
  the form-with-recalibration path is `quantprobe bench --contribute`.

## KILL RULE

If EITHER arm lands outside both its primary and named-alternative interpretation (e.g. 0.5B
< 85, DS < 16), L-17 is not a law but a two-point fit, it does NOT go into plan.py, and the
register entry is downgraded from established to open.

**Wired into:** pending; a P-3 hit wires L-17 into `plan.py` as the mechanistic split model.
