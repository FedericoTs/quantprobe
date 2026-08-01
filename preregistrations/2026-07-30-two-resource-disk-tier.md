# Pre-registration #89: the streaming/disk tier needs a second resource — does the two-resource form earn its way in?

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE any line of `quantprobe` was
changed and BEFORE the model was wired anywhere. **STAKED.** · **Experiment #53** ·
**Task:** #53 ("two-resource model for the streaming/disk tier: compute bytes vs I/O bytes") ·
**Register touched:** U-33, U-34 (BigMoeOnEdge), C-06. **No register edit is made by this
document**; scoring may propose one. Sibling: **#88** (naming *which* resource binds) — #88
classifies inside our existing single-resource model, #89 asks whether a *second* resource term
belongs there at all.

---

## The observation this comes from

Law 4 (`tok/s = eta x bandwidth / active-bytes`) has one resource. On the streaming/disk tier
that is visibly not enough, and BigMoeOnEdge hands us the cleanest possible demonstration:

> Their `k=8 -> k=6` result on Qwen3-30B-A3B at cache 4000 MiB is **x1.250 measured**.
> Our Law 4 compute-byte ratio predicts **x1.175**. Their flash-byte ratio is **x1.364**.

The truth sits *strictly between two single-resource bounds*. Neither resource alone can produce
it; a model with both can. That is the entire motivation, and it is also the sharpest available
test, because the bracketing is a **structural** property that does not depend on any fitted
constant.

They also ship an `--overlap` flag which, in principle, turns the sum of the two resources into
their maximum. Whether it actually does is the second thing this experiment decides.

---

## 0. What kind of claim this is, stated before anything else

**This is a RETRODICTION against tables published before this experiment existed, and the author
ran the arithmetic before writing this document.** Every number in §4 was computed by
`weights/exp53_two_resource_disk_tier.py --stake` and pasted here. Pretending otherwise would be
the exact dishonesty this project exists to avoid, so it is stated first, in the same terms
#86 and #88 used.

**What staking therefore buys, and the only things it buys:**

1. **Every selection rule is fixed in this document before it can be re-chosen** — which rows,
   which models, which files, which statistic, which exclusions, which tolerance. The scoring
   script hard-codes them and re-reads this file.
2. **Every threshold comes from somewhere else.** K-1's 15% is the project's standing
   external-comparison tolerance, staked in #86 and recorded in `U-33.predicted_effect`. K-2's
   factor 2 and K-3's "no same-cost rival beats it" are parameter-count fairness rules, not
   numbers chosen to make an answer look good. K-4's tolerance is 10% and its gate is 4 of 5.
   **No threshold in this document was moved after a number was seen.** The evidence for that
   claim is that the staked verdict below is a **FAIL** — a document written to flatter its own
   hypothesis would not stake its own refutation.
3. **The staked verdict itself is falsifiable by re-running.** §4 stakes
   `verdict_m2_pass = 0`. Anyone who runs the script and gets a PASS has caught us.
4. **The script refuses to run rather than produce a number** when any input drifts, and in
   scoring mode re-reads §4 out of *this file* and aborts if the fresh computation no longer
   reproduces all 23 values.

**QUALIFICATION added 2026-07-30 by adversarial review, and it limits points 2 and 3 above.**
"The staked verdict is a FAIL, so no threshold was moved" is evidence about **M2 and nothing
else**. The form that actually leaves this experiment is the pre-declared fallback **N3**, and
§4 stakes `verdict_n3_pass = 1`. For the arm that ships, the argument in point 2 is simply not
available: §0 states the arithmetic was run before this document existed, so N3 was named as the
fallback *knowing it was the best same-cost form*. **N3's pass is a selection, not a prediction.**
§5 now enumerates which of its four gates can fail and which cannot, and reports the tolerance
window over which its K-4 verdict is unchanged. Point 4's "reproduces all 23 values" was also
overstated: the check was an absolute ±0.005, i.e. ±12% on the smallest staked value, against a
documented 0.5%. It is genuinely relative now.

**What staking does NOT buy:** blinding. Their tables are public, the GGUF headers are public,
the fit is deterministic. Weigh this as "does a fixed, shipped, refusable procedure reproduce a
dataset it did not generate", not as "we predicted their results". §8 says what a real
prediction would look like.

---

## 1. The model, fixed here

For one decoded token on a device that streams routed experts from storage:

```
t_token = max(C, I)        if the row was run with --overlap        <- THE STAKED FORM (M2)
t_token = C + I            otherwise

  C = active_bytes(model, k) / S_c(device)      compute-byte term  (Law 4, unchanged)
  I = flash_bytes(row)      / S_i(device)       streamed-read term (new)
```

- **`S_c` and `S_i` are one effective rate per DEVICE.** Not per model, not per row, not per
  configuration. Two constants explain four models across 18 rows. This is the whole reason the
  test has teeth: a per-model rate would fit anything.
- **`active_bytes(model, k)`** is `quantprobe/plan.py:640-669` evaluated at routing width `k`,
  including finding **#76**'s embedding-gather correction, read from the model's own GGUF
  header. `ACT_MULT = 1.15` and the `max(bits, 4.5)` always-active floor are the shipped values.
  This is the **primary** convention because it is what the tool ships. The alternative
  ("exact per-tensor bytes", the U-29 convention) is computed in the same run and reported as a
  disclosed sensitivity — it must not change the verdict, and §4 stakes both.
- **`flash_bytes(row)`** is **their** instrumented `Flash/token` counter, taken verbatim. We do
  not model it in Arm A. Arm C models it separately.
- **`S_i` absorbs cache management on purpose.** Their telemetry splits a token into flash I/O,
  cache management and compute; the bytes memcpy'd into the expert cache are the same bytes read
  from flash, so a single effective rate proportional to flash bytes is the right shape. Stated
  now so that a good fit is not later read as evidence that cache management is free.

**Declared rivals, all fixed before scoring** — the model is scored *against these*, not against
nothing:

| id | form | free constants |
|---|---|---:|
| **M2** | **max under `--overlap`, sum without — THE STAKED MODEL** | 2 |
| N1 | compute only (`t = C`; Law 4 alone, storage invisible) | 1 |
| N2 | I/O only (`t = I`; compute invisible) | 1 |
| N3 | additive always (`t = C + I`; `--overlap` earns no byte-level credit) | 2 |
| N4 | max always (`t = max(C, I)`; perfect overlap everywhere) | 2 |
| M2b | partial overlap (`t = C + I - phi*min(C,I)` under `--overlap`) | 3 |

---

## 2. The dataset, and every exclusion, fixed here

Held-out set: the four README benchmark tables plus the desktop table of
`github.com/Helldez/BigMoeOnEdge`, read 2026-07-30. Transcribed verbatim, with provenance,
column meanings, their own caveats and six known defects, into
**`weights/data/external_bigmoe_tables.md`**. The scoring script asserts that every row in the
scorer is present verbatim in that file *and* still present verbatim in the live README, and
aborts otherwise — the transcription is not decoration.

**Scored (Arm A): the 18 streamed phone rows** of gpt-oss-120b, Qwen3.6-35B-A3B, Qwen3-30B-A3B
and Gemma-4-26B-A4B on their 12 GB / UFS 4.x phone.

**Excluded, and why, before any fit:**

- **All five `mmap baseline` rows.** Different code path (page-cache faulting, not the streaming
  engine), no published bytes/token, and three of five are labelled "unstable" by the authors.
- **Desktop rows 2 and 3 (`--drop-cold-experts 0.75`).** Dropping skips routed experts, so the
  active-byte count — the quantity `C` predicts — changes by an amount they do not publish, and
  they state the output is not reproducible.
- **Desktop row 1 is not fitted either.** One usable row cannot support two device constants.
  The desktop is §6, a disclosure arm with **no kill power**.

**Model file identification.** One rule, fixed: the `Q4_K_M` (or, for gpt-oss-120b, the native
MXFP4 their README says "streams unchanged") whose published size is closest to the size they
state; for Qwen3.6-35B-A3B the file is **inherited unchanged from experiment #51** so the two
experiments cannot disagree about it. That rule is *not* what does the work — **P-0 is**. Each
file's own routed-byte arithmetic has to reproduce their independent flash counter, and a file
that cannot is disqualified by an abort, not by a judgement call.

| model | file | their stated size | actual |
|---|---|---:|---:|
| gpt-oss-120b | `ggml-org/gpt-oss-120b-GGUF / gpt-oss-120b-MXFP4.gguf` | ~60 GB | 63.387 GB |
| Qwen3.6-35B-A3B | `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF / ...-Q4_K_M.gguf` | 22.3 GB | 22.285 GB |
| Qwen3-30B-A3B | `ggml-org/Qwen3-30B-A3B-GGUF / ...-Q4_K_M.gguf` | 18.5 GB | 18.557 GB |
| Gemma-4-26B-A4B | `bartowski/google_gemma-4-26B-A4B-it-GGUF / ...-Q4_K_M.gguf` | 17.0 GB | 17.035 GB |

The gpt-oss size mismatch is real and disclosed: their prose says "Q4_K_M, ~60 GB" but the only
file whose experts stream unchanged is MXFP4 at 63.4 GB. **P-0 arbitrates**, not the label.

---

## 3. Method — enough detail to run it without us

```
python weights/exp53_two_resource_disk_tier.py --stake     # freeze the numbers in §4
python weights/exp53_two_resource_disk_tier.py             # score them against the kill rule
```

0. **Derived tables are derived, not re-typed.** *[Added 2026-07-30 by the second adversarial
   review.]* P-0's anchor list and Arm B's five k-pairs repeat numbers that already live in the
   18-row table — and only that table is checked against the transcription and the live README.
   The script now resolves every anchor and every pair back to a unique verified row and asserts
   the premises this document states about them: a P-0 anchor must be a genuine no-cache row, and
   a "clean" k-pair must be two rows of the same model at the same cache and the same overlap
   whose published tok/s are the stated speedup. Any mismatch aborts.
1. **Preconditions.** `requests`, `gguf`, `quantprobe` and `weights/exp51_external_retrodiction.py`
   must import. #53 deliberately does **not** re-implement the GGUF header parser or the
   `quantprobe.spec.from_gguf` mirror: it imports #51's, and runs #51's self-test, which proves
   the mirror reproduces `quantprobe.spec.from_gguf` exactly on local GGUF files. Any
   disagreement aborts.
2. **Headers, not files.** One HTTP Range request per model pulls only the GGUF header (a few MB
   of a 17–63 GB file) and caches it under `weights/data/exp53_headers/`, so re-runs are offline
   and byte-identical. Each parse must satisfy
   `header + alignment pad + sum(per-tensor bytes) == remote Content-Length`, and each file must
   report the architecture, expert count, routing width and block count staked in the script.
3. **P-0, before anything is fitted.** For every row that ran with **no cache at all**, their
   `Flash/token` is the model's *full* routed-expert read at that width. Compare it with
   `exp_bytes x k / E` straight from the header. Four comparisons, zero parameters, 3% gate.
4. **Arm A.** For each of the two active-byte conventions and each of the six model forms, fit
   `S_c`, `S_i` (and `phi` for M2b) by minimising the sum of squared log-ratios of predicted to
   measured tok/s, using a **deterministic** log grid (46 x 46, ratios 1.08 / 1.09) followed by a
   fixed 50-round coordinate descent. No RNG, no scipy, no restarts.
   Then score three ways:
   - **in-sample** (reported, believed for nothing),
   - **LOO** — leave one *row* out, refit, predict it,
   - **LOMO** — leave one *model* out, refit on the other three, predict all of the held-out
     model's rows. **LOMO is the headline**, because wiring this into `quantprobe` means
     predicting a model the tool has never measured.
5. **Arm B.** The five clean `k -> k'` pairs (same model, same cache, same overlap, only `k`
   changes). For each, check the prediction is **strictly between** the compute-only bound
   `A(k)/A(k')` and the io-only bound `F(k)/F(k')`, **and** within 10% of measured.
6. **Arm C.** Cache hit as a *modelled* quantity: `h = (M/N)^beta`, where `M` = cache bytes /
   one expert slot's bytes and `N = n_layer x E`. **One `beta` for every model, on the phone
   only**, chosen on a fixed 1/2000 grid to minimise median absolute error. Structural null:
   `h = M/N` (uniform routing). 12 de-duplicated `(model, k, cache)` rows.
   *[Corrected 2026-07-30 by adversarial review: this line originally read "one beta for every
   model **and both devices**", which the script has never done and must not do — the desktop
   row publishes `cache auto` with no realised size, and pooling two machines into one constant
   is the cross-machine-state comparison C-14 forbids. The code was right and the prose was
   wrong; the prose is now the code.]*
7. **Scoring.** Re-read §4's ```stake``` block out of this file; abort if the fresh computation
   does not reproduce all 23 values to 0.5%; then evaluate the kill rule.

Raw output: `weights/data/exp53_two_resource_disk_tier.json` and `.log`,
`weights/data/exp53_stake.json`, cached headers in `weights/data/exp53_headers/`.
Runtime ~55 s on the author's box, dominated by 276 deterministic fits.

---

## 4. The staked numbers

```stake
p0_max_abs_rel_error     = 0.0045
p0_gate                  = 0.0300
m2_lomo_median           = 0.3539
m2_lomo_max              = 1.1567
m2_loo_median            = 0.1963
n1_lomo_median           = 0.4301
n2_lomo_median           = 0.3192
n3_lomo_median           = 0.0404
n4_lomo_median           = 0.1372
m2b_lomo_median          = 0.0861
m2b_phi                  = 0.1800
k1_lomo_median_max       = 0.1500
k1_lomo_max_max          = 0.3500
k2_min_beat_factor       = 2.0000
kpair_m2_pass_count      = 2.0000
kpair_n3_pass_count      = 4.0000
kpair_gate_count         = 4.0000
armc_beta                = 0.1965
armc_within_count        = 9.0000
armc_null_median         = 0.4962
armc_model_median        = 0.0478
verdict_m2_pass          = 0.0000
verdict_n3_pass          = 1.0000
```

Supporting detail, all reproduced by the script in the same run:

**P-0 — the byte model against their own counter, zero parameters:**

| model | width | ours (header) | theirs (no-cache Flash/token) | error |
|---|---:|---:|---:|---:|
| gpt-oss-120b | k=4 | 1820.1 MiB | 1817 MiB | **+0.17%** |
| gpt-oss-120b | k=2 | 910.1 MiB | 909 MiB | **+0.12%** |
| Qwen3-30B-A3B | k=8 | 1046.2 MiB | 1051 MiB | **−0.45%** |
| Gemma-4-26B-A4B | k=8 | 901.8 MiB | 904 MiB | **−0.24%** |

**Arm A — held-out error, primary (shipped) convention:**

| form | free | in-sample | LOO median | **LOMO median** | LOMO max |
|---|---:|---:|---:|---:|---:|
| **M2 (staked)** | 2 | 18.50% | 19.63% | **35.39%** | 115.67% |
| N1 compute-only | 1 | 38.48% | 40.19% | 43.01% | 269.28% |
| N2 io-only | 1 | 28.35% | 29.75% | 31.92% | 115.48% |
| N3 additive-always | 2 | 4.45% | 4.89% | **4.04%** | 28.66% |
| N4 max-always | 2 | 13.67% | 15.33% | 13.72% | 54.53% |
| M2b partial (phi=0.180) | 3 | 7.01% | 8.20% | 8.61% | 32.15% |

Fitted device constants for the phone: N3 gives `S_c = 17.24 GB/s` effective compute rate and
`S_i = 1896 MiB/s` effective streamed-read rate. M2 gives `11.93 GB/s` and `1747 MiB/s`.

**Arm B — the five clean k-pairs (M2 / N3, primary convention):**

| model | k → k' | measured | compute-only bound | io-only bound | M2 | N3 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 8→6 | 1.250 | 1.175 | 1.364 | 1.247 | 1.261 |
| Gemma-4-26B-A4B | 8→6 | 1.220 | 1.103 | 1.469 | 1.176 | 1.194 |
| Qwen3.6-35B-A3B (c2000) | 8→6 | 1.256 | 1.094 | 1.504 | **1.094** | 1.252 |
| Qwen3.6-35B-A3B (c3000) | 8→6 | 1.160 | 1.094 | 1.582 | **1.094** | 1.237 |
| gpt-oss-120b | 4→2 | 1.692 | 1.527 | 2.190 | **2.190** | 2.002 |

The three bold M2 entries are *degenerate*: `max()` collapses onto one resource, so the
prediction lands exactly **on** a single-resource bound instead of between them — the precise
failure the motivating observation was about.

---

## 5. Predictions and the KILL RULE

- **P-0 (parameter-free).** Our GGUF routed-byte arithmetic reproduces their independent
  no-cache `Flash/token` counter on all four comparisons, worst |error| **< 3%**.
- **P-1.** The two-resource form beats **both** one-resource nulls on LOMO median by at least
  **2x**. (This is the claim "we lack a term", stated so it can fail.)
- **P-2.** The staked form M2 has LOMO median **< 15%** and LOMO max **< 35%**.
- **P-3.** No same-cost rival (N3, N4 — also two constants) predicts held-out *models* better
  than M2. A model that is beaten by an equally cheap rival has not earned its form.
- **P-4.** On **at least 4 of the 5** clean k-pairs, M2's prediction is strictly between the two
  single-resource bounds **and** within 10% of measured.

**KILL RULE.** The staked two-resource form **M2 is wired into `quantprobe` only if P-0 AND P-1
AND P-2 AND P-3 AND P-4 all hold.** If any fails:

- **M2 does not ship.** Not in `plan.py`, not in the tiered waterfall, not in the docs.
- The miss is published in `FINDINGS.md` at equal prominence to a hit, per protocol.
- **No constant is retuned to recover it.** `ACT_MULT`, the 4.5-bit floor and `eta_r` are
  calibrated on other evidence; back-fitting them to one external dataset would destroy the only
  property that makes this test worth running.

**Pre-declared fallback, fixed here so it cannot be invented afterwards.** If M2's kill rule
fires, exactly one rescue is permitted, and only if it passes **the same four gates evaluated
identically**: the additive form **N3** (`t = C + I`, `--overlap` earning no byte-level credit).
If N3 passes all four, the published conclusion is:

> the two-resource **accounting** survives and the two-resource **combine rule** dies. `--overlap`
> does not turn a sum into a maximum.

and what may be wired in is N3, in a later, separate commit, with the overlap flag carrying no
credit. If **neither** M2 nor N3 passes, nothing is wired in at all and the disk tier stays
single-resource with the failure on the record. M2b (fitted partial overlap) is **not** an
allowed rescue: it costs a third parameter and §4 already shows it does not beat `phi = 0`
held-out, so choosing it would be paying for a worse answer.

**What the fallback's gates can and cannot detect — added 2026-07-30 by adversarial review, before
N3 is wired anywhere.** The four gates above were written for M2, a form named before the fit.
Applied unchanged to the *fallback*, two of them are empty, and this must be on the record next to
the pass, not discovered later:

- **P-0 did not gate the fallback at all.** The scorer computed `verdict = M2_passes AND P-0` but
  evaluated the fallback as `N3_passes` alone. A P-0 failure would therefore have killed only the
  model that was already dead and left the model that actually ships untouched — the exact
  opposite of what §8 promises. **Fixed in the script: P-0 is now a conjunct of every gate set.**
  This changes no staked value (P-0 holds at 0.45% against a 3% gate), which is why it could be
  fixed without re-staking; it changes what would have happened had P-0 failed.
- **K-3 cannot fail for the fallback, by construction.** The fallback was *named* as the best
  same-cost form. K-3 asks whether it is the best same-cost form. For M2 the gate has teeth and
  M2 fails it; for N3 it is a tautology and carries no evidence. The script now prints
  `K3_informative=False` beside it.
- **K-4's `bracketed` conjunct is an algebraic identity for any additive form.** For `t = C + I`
  the predicted ratio `(C_hi+I_hi)/(C_lo+I_lo)` is the *mediant* of the compute-only bound
  `A_hi/A_lo` and the io-only bound `F_hi/F_lo`, and a mediant always lies between its two
  ratios — 200,000 random draws produce zero violations, while the `max()` form lands on or
  outside a bound about half the time. So the "structural property that does not depend on any
  fitted constant" motivating this whole experiment is a genuine discriminator **against M2** and
  a free pass **for N3**. On the five pairs, K-4 for N3 reduces to the 10% tolerance alone. The
  script now labels every forced pair and reports how many informative ones remain.

**No threshold is changed in response to any of this.** Retuning a gate after seeing the numbers
is the failure this protocol exists to prevent; the correct repair is disclosure plus a
sensitivity that the reader can check. That sensitivity, computed by the script and reported in
the log and the JSON: **N3's K-4 pass count stays 4 for any tolerance in (6.67%, 18.31%]** — so
the staked 10% is not knife-edge on tolerance, but it *is* exactly at the gate on count, 4 of 5,
one row from failing. **M2's K-4 is unreachable at every tolerance**: only 2 of its 5 pairs are
bracketed at all, so no choice of tolerance could have saved it. What survives as real evidence
for N3 is K-1, K-2, and K-4's tolerance half.

*[Corrected 2026-07-30 by the second adversarial review: this sentence read "(6.64%, 18.32%]",
which is not what the script prints — the window is **(6.67%, 18.31%]**. The window is also a
statement about the pass **count** staying exactly 4, not about the verdict, which survives any
tolerance above 6.67%. Neither number is in the ```stake``` block, so nothing checked them; a
prereg literal that the script contradicts is the same defect §6 already caught once.]*

**Their tok/s are printed to one decimal, so K-4's measured ratios are bands, not points** — added
in the same review, because K-4 clears its gate at exactly 4 of 5 and is declared the softest gate.
`2.2 / 1.3` is anything from `2.15 / 1.35` to `2.25 / 1.25`, a −5.9%/+6.4% band before any physics.
The script now computes each pair's band and reports how many pairs' 10% verdict is decided by
their printing precision rather than by the data. **It is 0 of 5, for M2, N3 and M2b alike** — the
one pair N3 misses (gpt-oss, +18.3%) misses across its whole band, and the marginal pair it clears
(6.67%) clears across its whole band. This is a disclosure that came out in the model's favour and
is reported for the same reason it would have been reported had it not.

**Arm C gate (independent, does not gate the wiring of Arms A/B).** The modelled cache hit must
(a) beat the structural null `h = M/N` on median absolute error and (b) land within **0.10
absolute** on a **majority** of the 12 rows. Failing it means the model can only be used where
the caller supplies a measured hit rate, and the register must say so.

*[Disclosure added 2026-07-30 by the second adversarial review, threshold unchanged: **conjunct
(a) is very nearly unfalsifiable.** The "structural null" `h = M/N` is not an outside comparison —
it is exactly the `beta = 1` member of the fitted family `h = (M/N)^beta`, and `beta = 1.0` is in
the 1/2000 grid the fit minimises over. The fitted median error is therefore `<=` the null's by
construction, and (a) can fail only on an exact tie. It is the same defect class as K-3, found in
the same place for the same reason: a gate scored against a rival the procedure was free to
become. **Only conjunct (b) carries evidence** — landing within 0.10 on a majority is out-of-family
and can fail; it currently holds 9 of 12. The script prints this beside the Arm C verdict and
records `armC.null_disclosure.informative = false`.]*

---

## 6. The desktop — disclosure arm, NO kill power

One usable row cannot support two device constants, so nothing here can pass or fail anything.
It is published because it is the only second machine in the dataset and because it is where the
`max()` rule is most directly visible.

| quantity | value | sourced from |
|---|---:|---|
| #51's staked compute term for this model | 112.1 ms/token (8.92 tok/s ceiling) | our arithmetic, our 51 GB/s DDR4 preset — **not their number** |
| their own stated compute | ~110 ms/token | their prose, "~0.11 s/token in every cell" |
| measured | 208.3 ms/token (4.8 tok/s) at 74 MiB/token | their table 5, row 1 |
| implied effective read rate (additive model) | **0.806 GB/s** | derived; refuses to print if the residual is ≤5% of the token |
| their prose calls the NVMe | ~3 GB/s | their prose |
| additive prediction at 3 GB/s | 7.25 tok/s vs 4.8 measured (**+51%**) | derived |
| their `--overlap` A/B on the drop rows | 6.8 → 7.3 tok/s = **x1.074** | their table 5, rows 2–3 — **rows §2 excludes from every fit** |

*[Corrected 2026-07-30 by adversarial review. Two defects in the table above. (1) "their own
instrumented compute | ~115 ms/token" was **unsourced** — their prose says ~0.11 s/token, i.e.
110 ms; 115 appears nowhere in the transcription or upstream. (2) Every literal in this table
reached the write-up through an f-string and was checked against **nothing**: `verify_source`
only ever validated the 18 fitted rows plus the desktop row 1. The 6.8/7.3 A/B and the prose
quotes are now named constants verified against the transcription and the live README on every
run, exactly like the fitted rows. A number with no kill power is still a number that gets read.]*

Two things to read from this and nothing more. First, **the sequential NVMe spec is not the
right rate**: effective streamed-expert read is ~4x below it, which is why `S_i` is fitted per
device rather than taken from a datasheet — and any use of this model on an unmeasured machine
inherits that uncertainty. Second, **x1.074 is not a max()**: a perfect overlap at these terms
would buy far more, and this is the same verdict Arm A reaches on the phone by a completely
different route.

---

## 7. Known defects in the inputs and the method, listed before the result

1. **Their rows are best-of, not means**, "can come from different benchmark sessions", and
   phone throughput "moves a lot with device state". Best-of is biased upward and unequally so.
2. **The Qwen3.6 phone table is a single 96-token run**, by their own warning, not their
   256-token protocol. It is four of the eighteen scored rows.
3. **I/O lane count varies inside the gpt-oss table** (8 lanes cached, 4 uncached). One `S_i`
   per device charges that difference to model error. Not corrected — disclosed.
4. **qwen35moe's flash counter and its GGUF disagree by ~24-38%.** It is the one model with no
   no-cache row, so P-0 cannot test it; back-deriving `F/(1-hit)` and scaling to k=8 gives
   379-468 MiB against our 607 MiB, on the phone **and** on the desktop, so it is structural and
   not run noise. It has **no kill power here** because Arm A takes `I` from their counter, but
   it is a real open question about that architecture and belongs in the register either way.
5. **Cache hit is published to whole percent**, so back-derived byte totals carry ~3% error at
   h = 0.68 and worse above.
6. **Arm C's `h` is independent of `k`.** Measured hit does move with `k` (gpt-oss 0.32 at k=2
   vs 0.27 at k=4), and a k-free model cannot capture that. Disclosed, not fixed.
7. **`S_c` and `S_i` are fitted, not derived.** Two constants for a device we have never touched.
   That is why LOMO, not in-sample, is the headline, and why §8's held-out test matters more
   than anything in §4.
8. **Every model's active-byte count comes from a file we *chose*.** P-0 is what makes that
   choice defensible for three of the four models, and there is no equivalent check for the
   fourth (defect 4).

*Defects 9–11 added 2026-07-30 by adversarial review.*

9. **Arm B — the only row-level arm with kill power — compares across machine states, and C-14
   forbids that.** Every k-pair is a ratio of two rows the authors say "can come from different
   benchmark sessions", on a phone whose throughput "moves a lot with device state (heat, free
   memory)", each cell a best-of rather than a mean. C-14 exists because two calibrations hours
   apart on our own idle box moved every arm by 5–12 points. There is no `cal_id` available for
   somebody else's phone and no way to construct one from published tables, so **this defect
   cannot be fixed within this dataset — only declared.** Its consequence is asymmetric and
   should be read that way: a k-pair *failure* is informative (the model is wrong by more than
   session noise), a k-pair *pass* is weak (session noise is of the same order as the 10%
   tolerance). K-4 is therefore the softest of the four gates, not the hardest, despite being the
   one the motivating observation came from.
10. **P-0's coverage is three models, not four, and the missing one is the one that would fail
    it.** qwen35moe carries 4 of the 18 fitted rows, 2 of the 5 k-pairs and the entire desktop
    arm, and its byte model disagrees with the back-derived counter by 24–38% (defect 4). P-0's
    "worst |error| 0.45% against a 3% gate" is a statement about gpt-oss-120b, Qwen3-30B-A3B and
    Gemma-4-26B-A4B. This is now a first-class field in the output JSON rather than a log line —
    a defect that is harder to machine-read than a success is published at lower prominence than
    a hit, which the protocol forbids.
11. **The desktop arm divided by an unguarded residual.** `implied_read_gbs = flash_bytes / (1/measured − C)`
    was printed with no check that the residual is positive. If our compute term ever exceeds the
    measured token time — precisely what happens when the model, the machine or `eta` is wrong —
    that prints a negative or near-infinite "measured" read rate in the same confident format as
    a valid one. This is the shape of the artifact this project nearly published once before (a
    profiler run on a model too large for the card). **Fixed: the arm aborts unless the residual
    is positive and at least 5% of the token.** It currently sits at 46%.

*Defects 12–14 added 2026-07-30 by a second adversarial review. Each was found by constructing the
input that would exploit it and running it, not by reading the code.*

12. **P-0's anchors and Arm B's k-pairs were re-typed copies of the 18-row table, and only the
    18-row table was checked.** Constructed input: change one k-pair's flash figure from `165` to
    `156` — a digit transposition of a number that *is* verified where it lives in the fitted
    table. Result: the published io-only bound moved 1.364 → 1.442 and N3's prediction 1.261 →
    1.292, the transcription check passed, the live-README check passed, "all 23 staked values
    reproduced to 0.5% relative" printed, and the script exited 1 with an ordinary verdict. A
    published table was silently wrong and nothing objected. **Fixed:** both derived tables now
    resolve to unique verified rows, and §3 step 0's premises are asserted rather than written.
13. **An abort left the previous run's JSON on disk as the apparent current result.** Constructed
    input: make our routed-byte model wrong by 10%, the wrong-GGUF failure P-0 exists to catch.
    Result: P-0 failed at 10.19% against its 3% gate, `verdict_n3_pass` correctly fell to 0, the
    run aborted with exit 2 — and `weights/data/exp53_two_resource_disk_tier.json` was left
    **byte-identical to the last good run**, still asserting `P0.holds = true`,
    `gates.N3.PASS = true`, verdict `FAIL`, `publishable = true`. The log was honest; the
    machine-readable artifact, which is what anything downstream reads, said the shipping form had
    cleared a gate it had just lost. **Fixed:** every exit path writes that file. An abort writes
    an abort record carrying no `verdict` key, `publishable = false`, and whatever the run computed
    before refusing — including the fresh gate evaluation, so a lost gate is recorded rather than
    discarded with the run.
14. **In scoring mode the kill rule replays the staked verdict; it does not test it.** Every
    operand of P-0 and K-1..K-4 is a staked value, both verdicts are staked, and the 0.5%-relative
    reproduction gate is checked *before* §5's thresholds. So no input can make §5 print a
    different answer: anything that would flip a gate aborts at the reproduction step instead.
    That is prereg **#86**'s shape — a tight drift gate upstream of a loose kill threshold — and it
    means `exit 0` is unreachable in scoring mode against this document. The refusal is correct and
    is **not loosened**; the repair is that it stops being disguised. A drift in a verdict-valued
    key is now diagnosed as "a staked verdict changed — a scientific event, not a transcription
    slip", names the conjunct that flipped, and is carried into the abort record. The honest
    reading of a passing scoring run is therefore *"the staked answer still reproduces"*, not
    *"the gates fired again"*: they fired once, at stake time, and §4 is where they are on record.

---

## 8. What would REFUTE this, and what a PASS does not prove

**Refuted by**, mechanically, in the script:

- **P-0 failing** (any no-cache comparison off by ≥3%). That would mean our byte model does not
  describe their engine, and *nothing downstream of it may be wired in whatever Arm A says*. The
  script says so explicitly and the verdict is FAIL regardless.
- **P-1 failing** — the two-resource form failing to beat the better one-resource null by 2x.
  That is the direct refutation of "the disk tier lacks a term": if one resource explains the
  held-out models as well as two, the second term is decoration.
- **P-2 failing** — LOMO median ≥ 15% or LOMO max ≥ 35%.
- **P-3 failing** — a same-cost rival predicting held-out models better.
- **P-4 failing** — fewer than 4 of 5 k-pairs bracketed and inside 10%.

- **P-0 failing now also refutes the fallback.** Until the 2026-07-30 review it did not: P-0 was
  conjoined to M2's verdict only, so the model that ships could have shipped on a byte model
  known not to describe their engine.

**Refuted independently of any number** — each aborts with exit code 2 and produces nothing:
exp51's mirror disagreeing with `quantprobe.spec.from_gguf`; byte accounting not closing on a
header; a file's metadata not matching the staked architecture/E/k/L; the transcription
disagreeing with the scorer or with the live README — **including the desktop disclosure
literals, which nothing checked before the 2026-07-30 review**; the fresh computation not
reproducing §4; and the desktop arm being asked to divide by a residual that is negative or
smaller than 5% of the token; **a P-0 anchor or a k-pair that does not resolve to a unique verified
row, or a k-pair whose two rows disagree on cache or overlap or whose stated speedup is not the two
rows' own tok/s** (added in the second review, §7 defect 12). **A wrong number is worse than no
number** — and, added in the same review, **an abort now replaces the output JSON with an abort
record carrying no `verdict` key**, because the previous run's number surviving as the apparent
current one is the same failure wearing a different coat (§7 defect 13).

**Not publishable, as distinct from refuted:** a `--offline` scoring run. It skips the live source
check, so the verdict is stamped `-UNVERIFIED-SOURCE` and the exit code is non-zero even if every
gate is cleared. Previously such a run produced an ordinary PASS/FAIL with a warning printed
*after* the verdict banner.

**A PASS — of M2 or, under §5's fallback, of N3 — does NOT prove:**

- **that the model predicts an unmeasured machine.** `S_c` and `S_i` are fitted per device on
  that device's own rows. LOMO holds out a *model*, never a *device*. The honest next test is a
  device held out entirely, and this dataset has only one usable row on the second device (§6).
  Until that test exists, any `quantprobe` output using this must present the two rates as
  **calibration inputs**, not as predictions.
- **that Law 4's compute term is right.** The fitted `S_c` is a free constant absorbing eta,
  DRAM bandwidth and every unmodelled per-token cost. #86 tests the compute term against a
  device where the bandwidth is known; this experiment does not.
- **that `--overlap` is useless.** Their measured x1.074-x1.30 gains are real. What the result
  would show is that those gains do **not** appear as hidden I/O bytes at token granularity —
  which is a statement about where the benefit comes from, not whether it exists. That distinction
  goes in the register as an open question, not as a dead end.
- **anything about prefill, about context depth, or about the resident/all-in-VRAM regimes.**
  This is the streaming/disk tier only.
- **that we predicted anything.** We retrodicted published tables. §0.

---

## 9. Adversarial review, 2026-07-30 — what an attacker found and what changed

Reviewed against one question: *in what way can this experiment not fail?* The record, so the
repair is auditable and so no threshold is quietly credited with more than it did.

**The staked arm is not theatre.** M2's kill rule fires, and it fires on **all four** gates, not
one: LOMO median 35.39% against a 15% gate; 2x35.39% = 70.78% against a 31.92% best one-resource
null; beaten at its own cost by N3 at 4.04%; and 2 of 5 k-pairs against a gate of 4. The concrete
outcome that triggers the kill rule is the outcome that was staked. Better: M2's K-4 is
**unreachable at any tolerance**, because only 2 of its 5 pairs are bracketed at all — there is no
choice of `KPAIR_TOL` that could have rescued it, which is the strongest form of "this gate could
fire".

**The soft spot is the fallback, not the staked model.** Everything below concerns N3, the form
that actually leaves this experiment.

| # | finding | severity | disposition |
|---|---|---|---|
| 1 | P-0 gated M2 but **not** the fallback that ships | high | fixed in scorer; P-0 is a conjunct of every gate set |
| 2 | K-4's `bracketed` conjunct is an **algebraic tautology** for additive forms (mediant identity; 0 violations in 200k draws) | high | disclosed, counted, and labelled per pair; threshold **not** changed |
| 3 | K-3 **cannot fail** for the fallback, which was selected as the argmin of the same-cost set | high | disclosed as `K3_informative=False`; threshold **not** changed |
| 4 | §0's good-faith argument ("the staked verdict is a FAIL") covers only M2, not the arm that passes | high | qualified in §0 |
| 5 | Arm B forms ratios across sessions and device states — **C-14** | medium, unfixable in this dataset | §7 defect 9; K-4 declared the softest gate, failures informative, passes weak |
| 6 | desktop arm divided by an **unguarded** residual that goes negative when the compute term is wrong | medium | scorer aborts below a 5%-of-token residual |
| 7 | desktop literals (6.8→7.3 A/B, "~3 GB/s", compute ms) reached the write-up through f-strings, checked against nothing; one of them (**115 ms**) was unsourced | medium | named constants, verified against transcription and live README; corrected to 110 ms from their "~0.11 s/token" |
| 8 | stake-reproduction tolerance was absolute ±0.005 (±12% on the smallest value) while documented as 0.5% | medium | genuinely relative now |
| 9 | `--offline` bought a full publishable verdict for a run that never checked the source | medium | verdict stamped `-UNVERIFIED-SOURCE`, non-zero exit |
| 10 | P-0's coverage is 3 of 4 models and the untested one is the one that would fail it; that disclosure existed only as a log line while every success went to JSON | medium | first-class JSON field; §7 defect 10 |
| 11 | §3 claimed Arm C fits one beta across **both devices**; the code fits the phone only (and pooling would violate C-14) | low, prose-only | prose corrected to the code |

**Second adversarial review, 2026-07-30 — same question, asked again of the repaired script.** The
first review reasoned about the code; this one built the input that would exploit each hole and ran
it. Three of the five findings below were only visible that way.

| # | finding | how it was demonstrated | severity | disposition |
|---|---|---|---|---|
| 12 | P-0's anchors and Arm B's k-pairs **re-typed** numbers from the only table that is verified upstream | flash `165`→`156`: bound 1.364→1.442, prediction 1.261→1.292, **all checks passed, exit 1, ordinary verdict** | high | derived tables now resolve to unique verified rows; §3 step 0's premises asserted; threshold **not** changed |
| 13 | an **abort left the previous run's JSON** on disk as the apparent current result | byte model wrong by 10% → P-0 fails 10.19% vs 3%, N3 loses its gate, exit 2, **JSON byte-identical**, still `P0.holds=true`, `gates.N3.PASS=true` | high | every exit path writes the file; abort record carries no `verdict` key and preserves the fresh gate evaluation |
| 14 | the stake-reproduction gate **pre-empts the kill rule** — prereg #86's shape | a verdict-valued drift was reported in the same words as a transcription slip; `exit 0` is unreachable in scoring mode | medium | refusal **kept and not loosened**; verdict drift now diagnosed as a scientific event and names the flipped conjunct; `score_mode_kill_rule_is_replay` published |
| 15 | Arm C's conjunct (a) is **near-unfalsifiable**: the null is the `beta=1` member of the fitted family and `beta=1.0` is in the grid | algebraic, plus the grid endpoint | medium | disclosed as `armC.null_disclosure.informative=false`; only conjunct (b) counted as evidence; threshold **not** changed |
| 16 | K-4's measured ratios are **bands**, not points — their tok/s print to one decimal, and K-4 clears at exactly 4 of 5 | each pair's band computed against the fixed 10% | low | disclosed; **0 of 5** pairs' verdicts are decided by their rounding, for M2, N3 and M2b alike |

**Again, no threshold was moved and no gate was loosened.** All 23 staked values in §4 reproduce
unchanged after every repair above, and the staked verdicts are unchanged: M2 fails, N3 passes.

**No threshold was moved, and no gate was loosened or tightened.** Every staked value in §4 is
unchanged by this review, which is the test that the repairs are disclosure and refusal rather
than retuning: P-0 holds either way, so conjoining it changes no verdict today — it changes what
would happen on the day it fails.

**What is still weak after the repair, stated plainly.** N3 passes on K-1, K-2 and the tolerance
half of K-4, on 18 best-of rows from someone else's phone, with two free constants, with K-3 and
half of K-4 carrying no evidence, with one of four models unable to be checked by P-0, and with
every k-pair a cross-session ratio C-14 would reject if it were our own measurement. That is
enough to publish as a retrodiction and not enough to call a law. §8's requirement stands
unchanged and is now the binding constraint: the honest next test holds out a **device**, and
this dataset has one usable row on the second device.

---

**Wired into:** nothing, deliberately, in this commit. This experiment exists precisely to decide
whether the term is allowed into `quantprobe` at all. A PASS authorises a *separate, later*
commit that wires in the form this document names — and that commit inherits every caveat in §7
and §8, including the requirement that the two device rates be presented as calibration and not
as prophecy.
