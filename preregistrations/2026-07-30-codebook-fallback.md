# Pre-registration #77: the withheld-format fallback must be conservative, not optimistic

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE implementation. **STAKED.**

When known formats cover <60% of bytes we withhold `fmt_bw` on the principle that no number beats
a wrong number — but the fallback assumes K-quant-class decode. Isolated on two files of the same
architecture and active size: UD-Q2_K_XL (priced, fmt_bw 65.1) errs **+8.5%**; APEX-MTP-Nano
(37% IQ2_XXS + 22% IQ2_S, both unpriced) errs **+49.5%**. #70 measured codebook formats 36–52%
below K-quants per byte, so the withheld path assumes exactly what #70 refuted.

**Change:** (a) measure IQ2_XXS on a 7B against the #70 Q4_K_M control and add the entry; (b) when
unknown formats are codebook-named (`IQ*`) and cover >25% of bytes, price them at the **worst
known codebook** rather than falling back to a generic eta.

## Stakes
- **P-1 (the measurement).** IQ2_XXS's derived ebw lands **below IQ2_XS's 51.1** — a lower-bit
  codebook cannot decode faster per byte. (If it does, #70's mechanism story is wrong.)
- **P-2 (the fix, out-of-sample on the arm that failed).** APEX-MTP-Nano's error moves from
  +49.5% to **inside ±15%**. Not ±5%: a conservative fallback is a floor, not a fit, and saying
  so in advance is the point.
- **P-3 (no collateral).** Every arm whose formats ARE priced moves by <1% — including the
  sibling UD-Q2_K_XL at +8.5%. A fallback must not touch the non-fallback path.

## KILL RULE
If P-1 fails, the codebook mechanism does not generalise below IQ2_XS and the entry does not ship.
If P-3 fails, the change leaked outside its regime and is reverted.

**Wired into:** pending; `spec.FORMAT_EBW` + the coverage rule in `spec.from_gguf`.

---

## SCORED — 2026-07-30

Raw log: `weights/data/prereg77_codebook.log` (IQ2_XXS 25.93 tok/s @ 2.11 GiB vs the same-session
Q4_K_M control 22.87 @ 4.36 GiB).

- **P-1 HIT.** IQ2_XXS solves to **46.0** effective GB/s — below IQ2_XS's 51.1. The measured
  codebook ladder now reads **IQ2_XXS 46.0 < IQ2_XS 51.1 < IQ3_S 61.1 < IQ3_XXS 68.3 <<
  IQ4_NL 117.0**: monotone in bit-width, exactly as #70's mechanism requires.
- **P-3 HIT, exactly.** Every arm whose formats were already priced moved **0.0 points** — all
  thirteen. A fallback that touches the non-fallback path is a bug; this one does not.
- **P-2 MISS, by 0.9 points.** APEX-MTP-Nano went **+70.3% → +15.9%** (a 4.4× error reduction)
  against a staked ±15%. Published as a miss. The kill rule covered P-1 and P-3 only, both of
  which held, so the change ships — but the stake was the stake and it was not met.

**Why the residual is still +15.9%, stated rather than tuned away:** that file's remaining
unpriced format is IQ2_S (22% of bytes), now priced at IQ2_XXS's 46.0. IQ2_S is a *higher*-bit
codebook, so 46.0 under-prices it — the conservative direction, which cannot explain an
over-prediction. Something else in the APEX mixed-precision build is unaccounted for. Left open
rather than fitted.

**Ladder after #76 + #77: median |error| 15.6% → 7.4%** (I first wrote 5.6% here from a hand-sorted list and corrected it against the data before publishing; the improvement is real, the number was mine). Remaining over-predictors are all
fully-priced IQ files (DS-Lite IQ2_XS +28.4%, Qwen3.6-Q2_K_XL +25.3%) — a third mechanism, not
this one.
