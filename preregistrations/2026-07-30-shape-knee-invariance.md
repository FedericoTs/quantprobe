# Pre-registration #81: is the shape knee a ROW count or a BYTE count? (testing L-20 before wiring it)

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE running either binary. **STAKED.**

#80 established that decode bandwidth depends on tensor shape (3.2x across 128→16384 rows/tensor,
monotone). Before that becomes a term in the tool it has to survive a test that could break it,
and it has to be pinned down to a FORM — the two candidate mechanisms make opposite predictions:

- **(A) occupancy / grid-size floor** — what matters is how many BLOCKS a launch has to fill the
  SMs. Then the knee sits at a fixed **row count**, independent of how many bytes each row holds.
- **(B) launch-overhead amortisation** — what matters is how much WORK each launch does. Then the
  knee sits at a fixed **bytes-per-tensor**, and doubling bytes/row halves the knee's row count.

The probe now takes `K` (input dim) at compile time. `K=2048` gives 1152 B/row; `K=4096` gives
2304 B/row — **the same row counts carry double the bytes**. Everything else is identical.

## Stakes

- **P-1 (the discriminator).** Compare the row count at which each sweep first reaches 90% of its
  own maximum. If (A): the two knees land at the **same row count** (within one sweep step). If
  (B): the K=4096 knee lands at **half** the rows of K=2048. One of these is true; the prereg
  commits to reporting whichever, and to naming the mechanism accordingly.
- **P-2 (L-20 survives at all).** The K=4096 sweep is also monotone non-decreasing and also spans
  ≥2x from its smallest to its largest shape. If L-20 were an artifact of one layout, this is
  where it breaks.
- **P-3 (the ceiling is the format's, not the shape's).** Both sweeps saturate at approximately
  the SAME peak GB/s (within 10%) — the shape penalty should vanish at large tensors, not shift
  the ceiling. If K=4096 saturates materially higher or lower, bytes/row affects the ceiling too
  and L-20's "upper bound" framing needs revision.

## KILL RULE

**If P-2 fails, L-20 is refuted as a general law** and is rewritten as a single-layout
observation, with #80's numbers kept as a scoped curiosity. If P-3 fails, L-20's claim that
FORMAT_EBW entries are shape-upper-bounds is wrong in form and must be restated before anything
is wired. P-1 cannot "fail" — it selects between two mechanisms — but the tool cannot be given a
shape term until it is answered, because (A) and (B) imply different lookup keys (rows vs bytes).

**Wired into:** pending; decides the KEY of any future `FORMAT_EBW[fmt][shape]` table.
