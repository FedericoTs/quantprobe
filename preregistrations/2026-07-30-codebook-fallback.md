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
