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

---

## SCORED — 2026-07-30. **L-20 SURVIVES, BUT ITS FORM WAS WRONG AND IS RESTATED.**

Raw log: `weights/data/prereg81_knee.log`. Same probe, same budget, only `K` differs
(1152 vs 2304 bytes per row).

| rows/tensor | K=2048 penalty | K=4096 penalty |
|---|---|---|
| 128 | −69.4% | −64.5% |
| 512 | −38.1% | −33.3% |
| 2048 | −10.4% | −12.4% |
| 4096 | −4.6% | −7.1% |
| 16384 | 0.0% (98.7 GB/s) | 0.0% (**138.5 GB/s**) |

- **P-1 ANSWERED: mechanism (A), the key is ROWS.** Both layouts reach 90% of their own maximum
  at the **same 4096 rows/tensor**, though that is 4.6 MB/tensor in one and 9.2 MB in the other.
  The knee is a **grid-size/occupancy floor** — how many blocks a launch has to fill the SMs —
  not launch-overhead amortisation. Any shape term must be keyed on **rows, not bytes**.
- **P-2 HIT.** The second layout is also strictly monotone and spans 2.8x. L-20 is not an
  artifact of one layout.
- **P-3 MISS — and this is the useful part.** The two sweeps do NOT share a ceiling: 98.7 vs
  **138.5 GB/s, +40%**. Doubling bytes-per-row raises the ceiling substantially. So the geometry
  has **two independent axes**, and #80 measured only one of them.

**Restated L-20.** Effective decode bandwidth is `f(format, rows-per-tensor, bytes-per-row)`:
*rows/tensor* sets how close you get to the ceiling (knee ~4096 rows, occupancy-bound,
invariant to row width); *bytes/row* sets where the ceiling **is** (+40% for 2x width here).
`FORMAT_EBW`'s entries therefore carry an implicit row-width from whichever model measured them,
on top of the implicit FFN shape already identified — two hidden variables, not one.

**Consequence, and why nothing is wired yet:** a `FORMAT_EBW[fmt][shape]` table keyed only on
rows would have baked our 7B's hidden dimension into every prediction as an invisible constant —
precisely the class of error this whole day has been about. The correct key is
(rows, bytes-per-row), which needs the per-architecture read-set work (U-29) to supply both.
