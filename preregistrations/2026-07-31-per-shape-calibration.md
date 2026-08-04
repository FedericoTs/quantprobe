# Pre-registration #92 — per-SHAPE calibration: measure the L-20 curve through the product path, then ask whether it moves ANY emitted decision

**Author:** Federico Sciuca · **Date staked:** 2026-07-31, BEFORE compiling the probe or running
any arm. **STAKED.**

**Script:** `weights/exp_per_shape_calibration.py` · **Raw output:** `weights/data/exp_per_shape_*`
**Exit codes:** 0 = all evaluable gates PASS · 1 = a kill rule fired (including the P-4 true
null, which is a MISS of the staked speed prediction, published at equal prominence) · 2 =
precondition missing, no number produced · 3 = UNABLE TO BIND (a gate could not be evaluated;
must never be cited as a pass OR as a null).

## Why

`quantprobe calibrate` measures the user's box and stores scalars: one RAM stream number, one
disk number, per-format bandwidths measured once on the reference card. L-20 (preregs #80/#81)
measured that effective decode bandwidth is not a format constant — it is
`f(format, rows-per-tensor, bytes-per-row)`: 30.2 GB/s at 128 rows rising to 98.7 at 16384
(knee ~4096 rows, occupancy-bound), with the ceiling itself set by bytes/row (+40% at 2× width).
Every FORMAT_EBW entry was measured at the FFN end of that curve and is therefore an upper
bound. We run at 55–62% of spec peak and L-20 says shape is where much of the loss is.

The idea under test: `calibrate` should run the shape sweep on the USER's card and emit a
per-shape bandwidth CURVE, and `plan` should price each model's tensors against their actual
geometry read from the GGUF. No placement tool does this. But two prior results demand
suspicion before any wiring:

- **Prereg #82 (refuted):** the synthetic L-20 curve plus one machine constant did NOT transfer
  to real files (worst arm +25.5% against a ±20% stake; median got WORSE, 6.9%→8.1%).
- **U-32 (open):** the zero-parameter shape correction gets the DIRECTION right on every row
  (r = +0.769, n = 6) but systematically under-corrects. **The PREDICTION half is already
  staked there** — LOO median |error| < 8.7% and LOO max < 18.6% against a fresh ladder — and
  this prereg does NOT restake or pre-empt it.

This prereg stakes the two halves U-32 does not cover:

1. **The CALIBRATION half:** can the shape sweep be productized — compiled and run from the
   tool's own code path on the user's card, in minutes, stable enough to store as a
   calibration artifact — and does it reproduce the research measurement it claims lineage
   from?
2. **The DECISION half (the only part that can move real tok/s):** does per-shape pricing
   change the SPLIT POINT the tool emits — the winning placement row and its flags — anywhere
   on an unselected grid? **The honest risk, staked as its own prediction (P-4): it may
   improve prediction accuracy while changing zero placement decisions, in which case this is
   accuracy work with zero speed, and that null must be published as a MISS, not buried as a
   partial pass.**

## Method

### Phase A — the calibration half (GPU, single session)

**Disclosure up front: the #80/#81 probe binary (`shape.cu`) was never committed — only its
logged output survives** (`weights/data/prereg80_shape.log`, `weights/data/prereg81_knee.log`).
The script therefore carries a REWRITE of the probe, embedded as CUDA source and compiled at
run time with the user's `nvcc` — exactly what a shipped `calibrate --shapes` would have to do.
That makes P-1 a genuine replication gate, not a formality: a rewrite that does not reproduce
the logged curve is a rewrite of something else.

Probe geometry (fixed by #80/#81, restated here so someone else can run it):

- 4.5-bit layout, byte-identical in size to Q4_K: per 256-weight superblock, 128 B packed
  nibbles + 8 fp16 sub-block scales; row bytes = 9K/16.
- **One output row per block, one kernel launch per tensor** — the llama.cpp decode geometry
  (#55). Grid = rows-per-tensor blocks of 128 threads.
- ~384 MiB touched per sweep point; GB/s = bytes touched / event-timed wall time; best of 3
  reps after 1 warmup pass.
- Rows sweep: 128, 256, 512, 1024, 2048, 4096, 8192, 16384. Two widths: K = 2048 (1152 B/row)
  and K = 4096 (2304 B/row) — both L-20 axes (#81: rows set distance-to-ceiling; bytes/row set
  the ceiling).
- The full two-width sweep is run TWICE back-to-back (for P-3). nvidia-smi clocks are logged
  before and after (stuck-boost context, #60/#61 — logged, not gated).

Output artifact: `weights/data/exp_per_shape_curve.json` — the prototype of what
`calibrate --shapes` would store in `~/.quantprobe/calibration.json`: device name, date, and
`{bytes_per_row: {rows: GB/s}}`, i.e. a per-machine curve instead of a Pascal-only constant.

### Phase B — the decision half (NO GPU; deterministic arithmetic)

- **Eval set (unselected):** every readable `*.gguf` in `D:\evo-compress-data\gguf`. No
  hand-picking; a curated set that happens to flip would prove only that a set can be curated.
- **Machine grid (unselected):** every shipped `plan.MACHINES` preset × ctx ∈ {0, 16384}.
- Per model, from the GGUF tensor table: each 2D/3D weight tensor contributes
  (rows = ne[1], bytes/row = n_bytes / (rows × experts)); classed expert vs non-expert by the
  same name rules `spec.from_gguf` uses; `token_embd` excluded when untied (U-26 gather).
- Per-tensor shape factor from the Phase A curve: penalty = GB/s(rows, width) normalized to
  that width's 16384-row ceiling; log2-interpolated in rows (clamped 128–16384) and in
  bytes/row (clamped to the two measured widths — a disclosed extrapolation limit, e.g. a 7B
  down-projection at ~10 kB/row clamps to the 2304 B/row column).
- Group factors F_attn, F_exp: **bytes-weighted HARMONIC means** (time = bytes/BW; averaging
  bandwidths arithmetically would be the wrong mean), then **normalized so the whole-model mix
  factor = 1**. This convention is pinned NOW: the fitted eta already absorbs the average
  shape penalty of a full mix (that is how it was calibrated, U-32), so per-shape pricing must
  redistribute time between tiers/subsets, not re-fit the level. Consequence, stated before
  running: the all-in-VRAM row barely moves; rows whose GPU tier reads a shape-BIASED subset
  (hybrid = attention-only, expert-split = attention + resident experts) are the ones that can
  re-rank.
- The script calls the SHIPPED `plan.evaluate()` for every cell, then re-prices each
  recognized row's GPU weight-read seconds by 1/F. It does NOT copy plan's formulas on faith:
  it recomputes act_ne/act_ex/f from plan's own imported constants and **refuses (exit 2)
  unless the recomputation reproduces every recognized row's own `terms` decomposition to
  1e-9** — the exp54 lesson (a reimplementation must be verified against shipped output
  before it is allowed to score anything).
- KV-read seconds are NOT re-priced (the L-20 sweep never measured the KV access pattern);
  ctx 16384 cells are included because the KV term changes which rows exist and how close the
  top-2 sit, not because KV itself gets a shape price.
- Winner = argmax tok/s row per cell. Compared: shipped winner vs shape-priced winner,
  identity = (placement name, emitted flags).

## Staked predictions and kill rules

- **P-1 (replication through the product path).** The rewritten probe, on this card,
  reproduces the reference curve: at every (width, rows) point the normalized penalty is
  within **±6 percentage points** of `prereg81_knee.log`, and each width's 16384-row ceiling
  is within **±10%** of the logged ceiling (98.7 / 138.5 GB/s).
  **KILL:** any point outside → the productized sweep does not measure what #80 measured; a
  shipped `calibrate --shapes` claiming L-20 lineage would be mislabeled. Per-shape
  calibration does not ship from this rewrite.
- **P-2 (the form survives the product path).** Both widths monotone non-decreasing in rows
  (±3% noise), span ≥2× from 128→16384, and the 90%-of-own-max knee lands at **4096 rows ±1
  sweep step on BOTH widths** (rows-keyed, #81's mechanism (A)).
  **KILL:** knee differs by >1 step between widths → the curve is not rows-keyed on this path
  and a rows-indexed calibration table is the wrong data structure; nothing ships.
- **P-3 (stable enough to be a calibration artifact).** Between the two back-to-back sweeps,
  every point agrees within **5%**.
  **KILL:** any point >5% → `calibrate --shapes` would store noise; ship refused regardless of
  P-1/P-2. (Context: #80 vs #81, same card different sessions, drifted up to 5.6% at 512
  rows — this gate is expected to be tight, and that is deliberate: a calibration artifact
  that needs a lucky session is not an artifact.)
- **P-4 (THE SPEED QUESTION, staked separately so the null is visible).** Per-shape pricing
  changes the emitted winner (name or flags) in **≥1 scoreable grid cell**, and every flipped
  cell's shipped top-2 margin is smaller than that model's attn/exp factor differential
  (flips happen only where they are arithmetically possible — a flip in a wide-margin cell
  means the re-pricing is broken, and also fires this kill).
  **KILL (the pre-written null):** zero cells flip while the bindability guard (below) shows
  ≥1 cell where a flip WAS arithmetically possible → **TRUE NULL, scored as a MISS of this
  prediction and published at equal prominence: per-shape calibration is accuracy work with
  zero effect on any decision the tool currently emits.** Its only path to shipping is then
  U-32's accuracy gate, and the "changes your split point" pitch is dead on the shipped
  planner.
- **P-5 (structural: what per-shape pricing CANNOT move, verified mechanically).** Reading
  plan.py says the MoE expert-split fraction (`f = v_free/experts_gb`) and the dense layer
  split are CAPACITY-determined — no bandwidth term enters them. Staked: calling the shipped
  `evaluate()` with vb and geta scaled ×0.5 and ×2 changes tok/s in every GPU cell but changes
  **no split fraction in any row name and no emitted `-ot`/`-ngl` flag** anywhere on the grid.
  **If this HOLDS** (predicted), the claim surface of P-4 is exactly "the winner row", and
  wiring a shape-aware fraction chooser is a separate future experiment. **If it FAILS**, my
  structural reading is wrong, P-4's scope statement is invalid, and the whole run is scored
  INCOMPLETE (exit 3) pending a corrected design — not silently rescoped.

## The failing input, constructed BEFORE running (mandatory)

The signature to hunt is the **measurement that cannot vary**. This experiment has three ways
to fake a result, and each has a concrete constructed input plus a mechanical guard:

1. **The guaranteed-null grid (the KV-depth analogue).** Concrete input: an eval set of only
   dense-7B-class files (Qwen2.5-7B: q/o-proj 3584 rows, FFN 18944 rows, all at similar
   bytes/row) on machines where each cell offers one placement (every `vc=0` preset offers no
   GPU row at all; a 6 GB card offers a 30B MoE exactly one hybrid/split row), or where the
   top-2 rows sit 2.4× apart (pure-CPU 3.9 vs all-in-VRAM 9.56, the #16 pair). In every such
   cell the winner CANNOT change no matter what the curve says — the run returns "0 flips"
   and looks like a clean null while measuring nothing.
   **Guard G-1 (bindability):** a cell is scoreable for P-4 only if it has ≥2 rows including
   a GPU-priced one; the P-4 null may only be declared if ≥1 scoreable cell has shipped top-2
   margin < the model's factor differential. If no cell can flip, verdict is **UNABLE TO BIND,
   exit 3** — explicitly NOT a null, NOT a pass.
2. **The flat-factor model.** Concrete input: a model whose GPU-read tensors all sit above the
   4096-row knee at one width — every factor ≈ 1.00 after normalization, differential ≈ 0, the
   correction is a global scalar and ranking is invariant BY CONSTRUCTION.
   **Guard G-2:** models with attn/exp differential < 1% are excluded from the P-4 scoreable
   population (they can still be reported, but cannot support a null).
3. **The self-checking reimplementation (exp54 failure class 3).** If the script's
   recomputation of act_ne/act_ex/f drifts from shipped plan.py (a constant renamed, a formula
   changed), re-pricing would silently score a model of the planner instead of the planner.
   **Guard G-3:** every recognized row's `terms` must be reproduced to 1e-9 from imported
   constants, else exit 2. No fallback.

**Verified before any GPU run:** `--self-test` constructs exactly these inputs — a flat
synthetic curve + uniform-geometry model (must yield differential 0 and G-2 exclusion), a grid
of single-row and wide-margin cells (must yield UNABLE TO BIND, the exit-3 path), a
non-monotone synthetic curve (must fire P-2's kill), and a tampered decomposition (must fire
G-3's refusal). The self-test exits non-zero unless every guard demonstrably fires on its
constructed failing input. A version of this script whose self-test cannot fail is itself the
failure signature.

## Scope and honest limitations

- **n = 1 machine.** The per-MACHINE claim ("the curve differs across cards and must be
  measured, not copied") is NOT tested here and cannot be with one card; this prereg tests
  only that the measurement is productizable and reproducible on the card we have. The
  cross-machine claim stays an open conjecture and is labeled as such in any output.
- Bytes/row is clamped to the two measured widths; real tensors sit outside that range
  (disclosed per-model in the output as the share of bytes priced at a clamped width).
- KV reads keep their L-24/ETA_KV pricing; the shape curve does not apply to them.
- The normalization convention (whole-model mix = 1) is a choice, pinned before running; a
  different convention (e.g. anchoring FFN-shaped to 1) would shift levels but the winner
  comparison is invariant to a global rescale of the GPU tier only for rows with identical
  GPU byte mixes — which is exactly why the convention is pinned in advance rather than
  chosen after seeing results.
- C-14: Phase A is one session (both repeats back-to-back, one device fingerprint recorded in
  the curve artifact). Phase B is deterministic arithmetic; C-14 does not bind there. Phase B
  refuses to run against a curve whose recorded device differs from the current one.
- Prereg #82's ghost: that failure was a synthetic curve asked to predict LEVELS of real
  files. Here the curve is asked only to re-rank rows (levels are frozen by the
  normalization), which is a strictly weaker ask — that is the design response to #82, stated
  before the result is known.

**Wired into (only if P-1..P-3 pass AND P-5 holds):** `calibrate --shapes` writing the curve
into `calibration.json`, and — only if P-4 also hits — a per-shape term in plan's GPU-tier
pricing. On the P-4 null: calibration ships as measurement-only (curve stored, reported,
nothing re-priced) pending U-32's separately-staked accuracy gate.

---

## VERDICT (scored 2026-08-04): FAIL - kill rules fired, per-shape calibration does NOT ship

Run as staked (`--phase a`, embedded probe compiled at runtime with the box's own nvcc + MSVC
Build Tools host compiler; self-test passed first; one session, GPU clean, lockfile held).

- **P-2 KILLED, 3 violations:** at K=4096 the measured span is 1.41x against a staked >=2.0x;
  the 90% knee sits at 512 rows, outside the staked [2048..8192] window; and the knee moved
  between widths (2048 vs 512 rows), so it is not rows-keyed as the research characterization
  claimed.
- **P-3 KILLED, 8 violations:** the productized rewrite reads systematically ~7% below the
  logged #80/#81 curve (e.g. 39.9 vs 43.0 GB/s at K=4096/16384 rows) against a staked +/-5%.
  This prereg's own disclosure anticipated exactly this failure mode: the original probe binary
  was never committed, and "a rewrite that does not reproduce the logged curve is a rewrite of
  something else."
- Phase B (the decision half) is gated on Phase A and therefore does not run. **No per-shape
  term ships.** U-32's separately staked prediction half is untouched by this verdict.
- First attempt exited 2 (precondition: nvcc could not find cl.exe); resolved by putting the
  MSVC Build Tools host compiler on PATH - recorded because exit codes 2 and 1 mean different
  things and only the second is a result.

Evidence: `weights/data/exp_per_shape_phaseA.log`, `weights/data/exp_per_shape_calibration.json`.
