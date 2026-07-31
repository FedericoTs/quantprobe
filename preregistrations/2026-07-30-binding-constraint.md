# Pre-registration #88: name WHICH resource binds, or admit the label carries no information

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE the sweep was run and BEFORE
`binding_constraint()` existed in `quantprobe/plan.py`. **STAKED.** · **Experiment #54** ·
**Task:** #54 ("print WHICH resource binds, not just the tok/s number") ·
**Register touched:** U-33, U-34 (BigMoeOnEdge), E-01/E-02 (expert residency). No register edit
is made by this document; scoring may propose one.

---

## The observation this comes from

BigMoeOnEdge (github.com/Helldez/BigMoeOnEdge, Apache-2.0) publishes the sharpest operational
statement in the edge-MoE literature we have read: **the same model and the same config is
I/O-bound on their phone and DRAM-bandwidth-bound on their laptop**, and *which knob pays depends
entirely on which one binds*. On the laptop their compute time per token sits at ~0.11 s in every
cell of three rounds — so more I/O lanes, more threads and a faster NVMe were **all measured
neutral**, and ~9 tok/s is a ceiling even at zero I/O.

`quantprobe plan` today prints a tok/s number and then a page of advice, and **most of that
advice silently assumes a regime**. Telling a DRAM-bandwidth-bound user to buy faster storage is
not a weak recommendation, it is a wrong one. This experiment adds the classification, and — more
importantly — tests whether the classification carries any information at all, because the
failure mode that would embarrass us is a label that reads "bandwidth-bound" on every row a user
will ever see.

---

## 0. What kind of claim each part of this is, stated before anything else

Three different epistemic grades live in this document and they must not be blurred:

1. **P-1, P-2, P-4 are properties of deterministic code.** Anyone can run the sweep; staking
   them buys *no* blinding. What staking buys, and the only thing it buys, is that **the
   thresholds and the grid are fixed before the answer is seen**. Threshold-shopping is the real
   risk in a classifier (pick 1.15 because 1.20 makes the output look worse), and §1 removes it
   by construction: **this change introduces no new numeric threshold at all** — every constant
   in the classification rule is one already shipped in `plan.py`.
2. **P-3 and K-3 are regression facts** about the diff, checked mechanically against
   `git show HEAD:quantprobe/plan.py`. Nothing to blind; everything to verify.
3. **P-5 is a RETRODICTION against an external result published 2026-07-24.** Their numbers were
   already quoted in the register before this experiment existed. Worse for us than #86: **the
   author computed the §3 arithmetic by hand while drafting this document.** It is written down
   here, in full, so that the script's job is to prove the *shipped code* reproduces it — not to
   discover it. Read P-5 as "does the classifier, as implemented, agree with the only external
   ground truth we have", not as "we predicted their result".

**What this experiment cannot do, stated up front:** it cannot establish that the class *predicts
which lever pays*. That requires an A/B where a lever the classifier calls dead measures neutral.
We have exactly one such datapoint and we did not collect it (§6, their laptop). Everything else
here is information content and soundness. §7 says what a real validation would look like.

---

## 1. The classification rule, fixed here

`plan.evaluate` already builds each placement's per-token time as a sum of physical terms and
then throws the decomposition away, returning only `1/T`. The change keeps it.

**R1 — term attribution.** Every term in the sum is labelled with exactly one resource:

| term in `evaluate` | resource |
|---|---|
| `act*/(geta*vb)`, `kv_gb/(ETA_KV*vb)` | `vram_bw` |
| `act*/(eta_r*rb)`, `kv_gb/(ETA_KV*rb)` | `ram_bw` |
| `streamable*miss/db` | `io` |
| `n_layer*ctx*CPU_ATTN_S_PER_POS_LAYER` (L-19) | `cpu_compute` |

`T_r` = sum of that resource's terms; `T` = Σ`T_r`; `share_r` = `T_r/T`. Rows are emitted with
their terms attached; a row whose terms do not reconstruct its own tok/s to 1e-9 is a bug
(smoke-tested, and K-2). **The parenthetical was false when written** — see §8.5 D-5/D-6: the
smoke grid missed a whole row family and priced a zero-byte KV cache, and the scorer did not check
reconstruction at all. It is true now: 1527 rows per scoring run, all seven families in the smoke
grid, non-zero KV.

**R2 — the time-binding resource.** `bind = argmax_r T_r`. Ties break in the fixed order
`io, ram_bw, vram_bw, cpu_compute` so the answer is deterministic.

**R3 — margin and ceiling.**
`margin_x = T_bind / T_second` — *how much faster the binding resource must get before the next
one takes over*. `ceiling_x = 1/(1 - share_bind)` — *the most any lever that touches only the
binding resource can ever buy* (Amdahl). `ceiling_x` is the quantity that makes BigMoeOnEdge's
"9 tok/s even at zero I/O" statement printable.

**R4 — capacity.** Capacity never appears in `T`; it is what *selected* the placement. Two
probes, both leaving every bandwidth untouched:
* **lift** — re-evaluate with `vc' = (size+kv)/0.90` (VRAM), or `rc' = size+kv+4` (RAM), one at a
  time;
* **shave** — the tier-boundary probe already shipped in `run()` (`act_scale = fit_scale`),
  unchanged.

Capacity is the binding constraint **iff** some probe yields a best row ≥ **1.15×** the winner
**and** the shortfall is ≤ **0.30 ×** model size. Both numbers are lifted verbatim from the
shipped tier-boundary advisor (`plan.py`: `> best[1] * 1.15`, `gap > size_now * 0.30`) and the
advisor is refactored to call the same probe, so **the advisor firing and the classifier saying
`capacity-bound` are the same event by construction** — not two rules that can disagree.

**R5 — the class.** `capacity-bound (<tier>)` if R4 fires; otherwise
`bandwidth-bound (VRAM)` / `bandwidth-bound (system RAM)` / `IO-bound` / `compute-bound (CPU
attention)` from R2. Scoring collapses the two bandwidth labels into the single class
**bandwidth-bound**, which is the harder test.

**R6 — the mandatory caveat.** When the winner is `all in VRAM` **and** bits < 4.5, the printed
class MUST carry the disclosure that #16/#52 *measured* this cell as unpack-limited, not
byte-limited (Q2_K is 36% smaller and 4% slower than Q4_K_M; Q4_0 is +19% over Q4_K_M at equal
bytes). Our `geta` folds bandwidth and unpack ALU into one number and **cannot separate them**;
printing a bare "bandwidth-bound" there would contradict our own measurements. This is K-4.

**Scope printed with the label:** the classification is of the **decode token only**. Prefill has
a different binding constraint (compute), which is exactly why `-ub 2048` pays +73% on prefill and
0% on decode. Any printed class that does not say so is a defect.

---

## 2. Method — enough to run without us

```
python weights/exp54_binding_constraint.py            # score against the kill rule
python weights/exp54_binding_constraint.py --json     # machine-readable only
```

1. **Preconditions, all of which ABORT with exit 2 rather than produce a number:**
   `quantprobe.plan.binding_constraint` must exist; `git show HEAD:quantprobe/plan.py` must
   succeed (K-3 needs the pre-change baseline); this file must exist and its ```` ```stake ````
   block must parse; and the preset grid must still be **10 models × 17 machines**, or the run
   aborts (a preset added after staking silently changes the denominator of P-1).
2. **Grid — unselected, no cell chosen by hand.** The FULL cross product of every shipped model
   preset and every shipped machine preset, at `bits = 2.5` (the tool's own default) and
   `n_layer` from each preset:
   * **G-A**: 170 cells at `ctx = 0`
   * **G-B**: 170 cells at `ctx = 16384`
   340 cells. Choosing the grid as "everything we ship" is the point: a hand-picked grid spanning
   four regimes would prove only that we can pick four cells.
3. For each cell: run `evaluate`, take the winning row, classify it (§1), record class, shares,
   `margin_x`, `ceiling_x`, and the winner's name/tok/s/flags.
4. **K-3 baseline:** load `git show <pre-change ref>:quantprobe/plan.py` as a separate module and
   run the identical grid through it. Every cell's winning row name, tok/s (rel. diff ≤ 1e-9) and
   flags must match. The classifier is a **reporting layer**: if any predicted number moves, the
   implementation is wrong and is reverted. **Two arms** (§8.5 D-9): the staked 340 cells, and the
   same 340 preset pairs replayed with a pinned `true_size_gb` and a non-zero `codebook_share` —
   the `--gguf` code paths every locked ladder row uses, which the staked arm never touches. A
   cell whose row list goes empty on either side is a regression, not a cell to skip.
4b. **K-2 reconstruction (§8.5 D-5):** every row of every cell — 1527 of them, winners and losers
   alike — must satisfy `tok_s == eff / Σ terms` to 1e-9. Staked in §1 R1 and §5 K-2 from the
   start; implemented only after the pre-run audit found it was scored nowhere.
5. **P-3 probes** (the advice-changes test) run on the same grid plus two synthetic arms
   described in §4.
6. **P-5** runs the external arm of §3 through the shipped `evaluate`.

Raw output: `weights/data/exp54_binding_constraint.json` and `.log`; the extracted baseline is
cached at `weights/data/exp54_plan_head.py.txt` so re-runs are offline and byte-identical.
Idempotent: re-running overwrites the same three files and produces the same verdict.

---

## 3. The staked numbers

```stake
grid_models               = 10
grid_machines             = 17
grid_cells                = 340
p1_min_distinct_classes   = 3
p1_max_modal_share        = 0.85
p4_min_frac_ceiling_lt_2  = 0.10
p5_hot_gb                 = 1.428168
p5_streamable_gb          = 0.744573
p5_miss_uniform           = 0.515369
p5_io_s_uniform           = 0.127910
p5_ram_s_uniform          = 0.092312
p5_class_uniform          = IO-bound
p5_miss_measured          = 0.200000
p5_io_s_measured          = 0.049638
p5_ram_s_measured         = 0.104429
p5_class_measured         = bandwidth-bound
p5_their_compute_floor_s  = 0.115000
p5_kill_rel_error         = 0.150000
cap_promotion_min         = 1.15
cap_shave_max_share       = 0.30
regression_rel_tol        = 1e-9
```

### Derivation of the P-5 arm (external, BigMoeOnEdge laptop)

Every input is already public and already in the register or in prereg #86 §2. **Nothing is
fitted.** Their box: 8 cores, 16 GB dual-channel DDR4, ~3 GB/s NVMe, `Qwen3.6-35B-A3B Q4_K_M`
(22.285 GB, ~1.5× RAM), their own flash-streaming engine.

| quantity | value | source |
|---|---:|---|
| always-active params `ne` | 1.9791 B | prereg #86 §2 (their file's header, #76 applied) |
| active params/token | 3.0109 B | prereg #86 §2 |
| effective bits | 5.02 | prereg #86 §2 |
| `hot` = always-active bytes | **1.428168 GB** | `1.9791 × 5.02 / 8 × 1.15` |
| `streamable` = routed-active bytes | **0.744573 GB** | `1.0318 × 5.02 / 8 × 1.15` |
| `eta_r × rb` (MoE RAM tier) | 19.38 GB/s | `0.38` (`plan.py`, shipped since `d590749`) × 51.0 |
| `db` | 3.0 GB/s | their README |
| usable RAM `ra` | 12 GB | `rc − 4`, shipped |

**Arm C1 — our uniform-residency assumption (what `plan.evaluate` computes today).**
`miss = 1 − (ra × 0.9)/size = 1 − 10.8/22.285 = 0.515369`

    io  = 0.744573 × 0.515369 / 3.0            = 0.127910 s
    ram = (0.744573 × 0.484631 + 1.428168)/19.38 = 0.092312 s
    -> io share 58.1%  ->  IO-bound       margin 1.386x   ceiling 2.386x

**Arm C2 — their MEASURED hot-expert cache hit rate (76–84%; midpoint 0.80 fixed here).**

    io  = 0.744573 × 0.20 / 3.0                = 0.049638 s
    ram = (0.744573 × 0.80 + 1.428168)/19.38   = 0.104429 s
    -> RAM share 67.8%  ->  bandwidth-bound     margin 2.104x   ceiling 3.104x

and `0.104429 / 0.115 = 0.908`, i.e. **−9.2% against the compute floor they published**
(median of their seven full-width cells, the identical extraction rule fixed in prereg #86 §1.6).

C1 is computed by the **shipped** `evaluate` (custom model spec, `vc=0, rc=16, rb=51, db=3.0`,
`true_size_gb=22.285`, `ctx=0`). C2 substitutes `miss = 0.20` into the same formula **in the
script, not in shipped code** — disclosed, because it is not something the tool can currently do.

---

## 4. Predictions

- **P-1 — the label carries information.** Over the 340 unselected cells, **at least 3 of the 4
  classes appear**, and **no single class covers more than 85%**. Scored on the four collapsed
  classes (bandwidth / IO / capacity / compute), which is stricter than scoring
  VRAM-vs-RAM-bandwidth separately.

- **P-2 — soundness (no phantom sensitivity, correct wiring).** Both halves must hold on every
  cell:
  * **(a)** For every cell whose winning row has **no** `io` term, changing *only* `db` between
    0.45 and 5.0 GB/s leaves the class, the binding resource, `margin_x` and `ceiling_x`
    bit-identical. A diagnosis that moves when an irrelevant resource moves is not a diagnosis.
  * **(b)** For a **fixed placement**, doubling `rb` strictly decreases that row's `ram_bw` share
    and doubling `vb` strictly decreases its `vram_bw` share. This is forced by the model, so it
    tests the implementation's wiring, not nature — and it is exactly the wiring a copy-paste
    error breaks silently.

- **P-3 — the advice actually changes.** At least one of the following, each mechanically
  checked, and **each reported individually so a partial result cannot be read as a full one**:
  * **(a)** ≥1 cell where the upgrade-advisor line differs once its counterfactual is given the
    **same `n_layer` and `true_size_gb` as the baseline**. *This is a defect found by reading:*
    the three `evaluate` calls in the upgrade advisor pass neither, so the counterfactual is
    drawn from a **smaller row menu** (no MoE split, no dense split — both require `n_layer`) and,
    with `--gguf`, from a **different model size** (the `4.5/bits` dense inflation the code's own
    comment says "wrongly evicted the all-in-VRAM row"). Its signature is one-directional: it can
    only ever **suppress** an upgrade that would genuinely help.
    > **⚠ The staked part of P-3a held (31 cells) but the sentence immediately above is REFUTED —
    > see §8.6.** It is left standing here because staked text is never rewritten. Measured
    > direction: **30 upgrades INVENTED vs 3 suppressed**; the missing row menu suppresses, but
    > `(n_layer or 32)` prices the counterfactual as a different (shallower) model than the
    > baseline and manufactures gain, which is the larger and the more expensive error.
  * **(b)** ≥1 synthetic arm where the codebook/IQ-on-CPU warning's honest bounded gain is
    **< 5%** — i.e. the shipped text tells the user to re-download a whole model for a placement
    where the RAM weight term is a small share of the token. Bound (exact, from the shipped
    constants): removing the penalty scales the RAM term by `1/(1+cs·k)`, so
    `speedup ≤ T / (T − T_ram · cs·k/(1+cs·k))` with `k = IQ_CPU_TG_PENALTY = 0.456`.
  * **(c)** ≥1 cell where the speculation ceiling
    `T / (T_bandwidth/(K+1) + T_compute)` at `K = 2` is **< 1.5×**, against printed advice whose
    headline is 4.7×. Speculation amortizes weight *reads* across the verify batch; it does
    **not** amortize CPU attention, which is per-position compute. So on a compute-dominated row
    the 4.7× is unreachable *arithmetically*, and the tool currently does not say so.

- **P-4 — the ceiling print is not decorative.** On **≥10%** of the 340 cells, `ceiling_x < 2.0`.
  Note the arithmetic: with only two active resources the larger share is ≥ 0.5 and the ceiling is
  therefore ≥ 2.0 **always**, so this predicts that ≥10% of cells split their token across
  **three or more** resources. If it comes in near zero, "even an infinite upgrade buys at most
  Nx" is a line that never says anything surprising and should not be given headline space.

- **P-5 — the external arm (retrodiction, §0.3).** All four must hold:
  `class(C1) = IO-bound`; `class(C2) = bandwidth-bound`; each of the seven staked P-5 quantities
  reproduced by the shipped code to within **2e-3** relative; and
  `|ram_s_measured / 0.115 − 1| < 15%`.

  > **Corrected 2026-07-30 by adversarial review (§8, D-3).** As first written the scoring script
  > recomputed `hot`, `streamable` and `miss` from formulas copy-pasted out of `plan.py` and
  > hard-coded `eta_r = 0.38`, so **four of the seven quantities were produced by the test itself**
  > and the phrase "reproduced by the shipped code" was false for them: a sign error inside
  > `evaluate`'s disk row would have left every one of them matching its staked value. They are now
  > **inverted out of the shipped row's own `terms`** (`streamable = io_s·db/miss`,
  > `hot = ram_s·eta_r·rb − streamable(1−miss)`), cross-checked against `evaluate`'s own `act` to
  > 1e-9 with an abort on mismatch, and `eta_r` is imported as `plan.ETA_R_MOE`. Verified by fault
  > injection: moving the shipped constant to 0.40 now breaks P-5 by −5.0%, where before it changed
  > nothing. **No staked number moved** — all seven still reproduce exactly.
  >
  > **And the honest grade of the ±15% clause:** §0.3 already discloses that the author computed
  > §3 by hand before staking. That includes the **−9.2%**. The ±15% band was therefore chosen
  > *knowing the answer*, which makes K-5's world-facing clause a consistency check with a known
  > result, not a blind prediction. K-5 can fire on an implementation bug — that is now genuinely
  > true and demonstrated — but it cannot fire on a surprise from BigMoeOnEdge, because no surprise
  > remains to have. Read K-5's PASS as "the shipped code reproduces arithmetic we already did",
  > and nothing stronger.

---

## 5. KILL RULES

Mechanically checkable, and each names the consequence, not just the verdict.

* **K-1 — P-1 fails ⇒ the headline claim is dead.** If one class covers > 85% of the grid, we do
  **not** ship "quantprobe names which resource binds" as a feature. The classifier is demoted to
  printing `margin_x` / `ceiling_x` only, the class is printed without emphasis, and FINDINGS
  records at equal prominence that on our own shipped preset grid the binding constraint is
  almost always the same one — which would mean BigMoeOnEdge's phone-vs-laptop contrast does not
  reproduce across the machines we model, and that is the interesting result, not a footnote.
* **K-2 — P-2 fails ⇒ nothing ships.** Not a demotion, a revert. A classifier that responds to a
  resource the winning row does not use, or whose shares move the wrong way when a bandwidth
  moves, is worse than no classifier: it is confident and wrong. Also fires if any row's terms
  fail to reconstruct its own tok/s to 1e-9.

  > **Corrected 2026-07-30 by the second adversarial review (§8.5, D-5/D-6).** That last sentence
  > was staked here and in §1 R1 and then **scored nowhere**: the scoring script did not check it
  > at all, and the smoke test §1 R1 points at ran a hand-picked six-config grid that never emitted
  > the dense-split row — a winner on 8 of the 340 cells — and whose `ctx` entries carried no
  > `kvp`, so every KV attribution term in every family was multiplied by zero. Reconstruction is
  > now checked on **every row of every cell** (1527 rows) and counted, and the smoke grid covers
  > all seven families with a non-zero KV cache and asserts that it does. Verified by fault
  > injection: dropping the dead-simple `(1-g)·kv_gb/(ETA_KV·rb)` term from the dense split leaves
  > every tok/s bit-identical and every other arm clean — it used to print PASS at exit 0, and now
  > fires K-2 on 10 rows.
* **K-3 — any predicted number moves ⇒ revert.** Across all 340 cells the winning row name,
  tok/s (rel. tol 1e-9) and emitted flags must be **identical** to the **pre-change** `plan.py`.
  The single permitted exception is the upgrade-advisor line under P-3(a), which is a disclosed
  behaviour fix and must be enumerated cell-by-cell in the output. Any *other* difference means
  the "reporting layer" framing is false.

  > **Corrected 2026-07-30 by adversarial review (§8, D-1).** This originally said "identical to
  > `HEAD`", and the script defaulted `--baseline-ref` to `HEAD` and folded the result into the
  > verdict as `k3 or k3_vacuous`. **The instant this change was committed, `HEAD` would contain
  > the classifier**, every cell would compare the module to itself, zero regressions would be
  > found *by construction*, and the script would print `VERDICT: PASS`. A kill rule that
  > switches itself off the moment the work lands is not a kill rule. The baseline is now the
  > **pre-change commit**, named explicitly; a baseline that already contains
  > `binding_constraint` is a **refusal** (exit 2), or with `--allow-vacuous-regression` an
  > **INCOMPLETE** verdict (exit 3) that records `K3: null`. It can never read as PASS again.
  >
  > **Extended 2026-07-30 by the second adversarial review (§8.5, D-9).** K-3's grid pins
  > `true_size_gb=None` and `codebook_share=0.0`, and **every one of the 14 locked ladder rows is a
  > `--gguf` run** — i.e. exactly the two inputs the grid never supplies. A second deterministic
  > arm now replays the same 340 preset pairs with a pinned size and a codebook share. Verified by
  > fault injection (`size = true_size_gb * 1.01`): the staked arm reports **0** cells moved while
  > the new arm catches **124**. P-1's staked denominator is untouched. Separately (D-8), a run in
  > which a *scored* kill rule fired now reads **FAIL (exit 1)**, not INCOMPLETE — the vacuous-K-3
  > branch used to be tested first and could mask two dead rules behind exit 3.
* **K-4 — the honesty gate.** A run in which any `all in VRAM` winner at bits < 4.5 is labelled
  bandwidth-bound **without** the R6 unpack caveat fails, regardless of every other result.
  Scored on **every all-in-VRAM winner in the 340-cell grid** (137 of them at 2.5 bits), each put
  through the same `binding_report(bc, bits, placement=best[0])` call `run()` makes. A run that
  checked **zero** such winners **fails** — an honesty gate with an empty population is not a gate.

  > **Corrected 2026-07-30 by adversarial review (§8, D-2).** As first implemented `check_k4`
  > tested **one hand-built row** constructed with `placement="all in VRAM"` — the exact literal
  > `binding_report`'s gate compares against. It asserted that a function handed the string it
  > tests for finds that string, and **it could not fire from the sweep however the sweep came
  > out**. It was in particular blind to the realistic failure: the caveat gate is
  > `placement == "all in VRAM"`, an *exact-equality* test, so the day that row's name gains a
  > suffix the caveat silently stops printing on every cell. Fault injection confirms the repaired
  > gate: renaming the row to `all in VRAM (KV in VRAM)` now fires K-4 on **137/137** cells, where
  > the old check passed. The >4.5-bit negative control is retained so the gate cannot pass by
  > printing the caveat unconditionally.
* **K-5 — P-5 fails ⇒ the class does not ship as a diagnosis.** If the classifier disagrees with
  the only external ground truth we have, we print shares and margins and stop calling the label
  a diagnosis. **`db`, `eta_r`, `rb` and the 1.15/0.30 thresholds are NOT retuned to recover it** —
  back-fitting the one external check destroys the only thing that makes it worth running.

**Predictions with NO kill rule, stated rather than left to omission** (added by §8; both were
already true of this document, and neither was written down, which is how a reader would have
assumed the coverage was complete):

* **P-3 has no kill rule.** §7.4 names "it changes nothing" as a sincere refutation, but nothing
  in this section fires on it. All three probes could have come back empty and the verdict would
  still have read PASS. That asymmetry is a defect of *this pre-registration*, not of the result.
  It is staked forward as **K-6** in §8 and is **not** applied retroactively to this run.
* **P-4 has no kill rule.** Deliberately: P-4 is a claim about whether a *printed number deserves
  headline space*, and its stated consequence (demote it) is editorial, not a revert. P-4 **did
  miss** — 0.6% against a staked 10% — and the consequence was applied in `binding_report`, which
  now prints the miss on the page next to the number it demotes.

**Partial-result rule.** P-1 failing while P-2/P-3/P-5 pass is the most likely mixed outcome. It
is scored as **K-1 fires, feature demoted, experiment successful** — a refutation of the headline
with a working instrument is a result, and it gets the same prominence as a pass.

---

## 6. The advice audit — every line in `run()`, and the regime it silently assumes

Written before the code change, because "check every advice line" is the part of this task that
is easy to skip and expensive to skip.

| advice line | regime it assumes | verdict | change |
|---|---|---|---|
| upgrade advisor: XMP / +16 GB RAM / NVMe | already regime-gated by re-evaluating (an upgrade that does not help is not printed) | **defective inputs, not defective logic** | counterfactual gets the baseline's `n_layer` + `true_size_gb`; each line labelled with the resource it attacks (P-3a) |
| IQ / codebook on CPU tier ("re-download as Q_K") | assumes the RAM weight term dominates the token | **assumes RAM-bandwidth-bound** | state the exact bounded gain; below 5% it is a note, not a WARNING (P-3b) |
| speculation (4.7× ngram, 2.10× dense, +33% split) | every measurement is on a bandwidth-bound row; the verify batch amortizes *reads* | **assumes bandwidth-bound** | print the arithmetic ceiling at K=2 for THIS row; CPU attention does not amortize (P-3c) |
| `-b/-ub 2048` prefill lever (+73%) | host-resident weights + VRAM headroom — already gated | **sound, but scope-blind** | say it attacks the PREFILL/PCIe term, which the decode classification does not price |
| concurrency ("~2× with 8 slots") | batching amortizes weight reads across slots | **assumes bandwidth-bound** | on a compute-bound winner, say the ~2× will not appear |
| `fits_in_vram_advice` (the ≥0.90× floor) | all-in-VRAM only; already scoped | sound | carries the R6 caveat |
| `format_advice` (Q4_0 +19%, avoid codebook IQ) | pre-Ampere, ALU-scarce — i.e. explicitly an **unpack-limited** claim | sound and already scoped in prose | it is the evidence for R6: this line and a bare "bandwidth-bound" label contradict each other |
| `depth_scope_warning` (L-19) | dense split at depth = compute regime | sound | invariant: it fires ⟺ the classifier says compute-bound on that row |
| `phase_advice` / `workload_frontier` | prefill-vs-decode trade, not a resource trade | sound | untouched |
| `mmap_decision` / pinning warnings | capacity/paging domain | sound | untouched |
| tier-boundary advisor | capacity | sound | refactored to share the R4 probe, so it cannot disagree with the class |

**Known limitations, listed before the result:**

* **`geta` fuses VRAM bandwidth and unpack ALU.** We cannot separate them, so `bandwidth-bound
  (VRAM)` is a *model* statement and R6/K-4 exist because our own #16/#52 measurements
  contradict the naive reading of it.
* **Uniform expert residency.** The disk row assumes a cache hit rate set purely by capacity.
  BigMoeOnEdge measure 76–84% by caching *hot* experts. That gap is the whole distance between
  P-5's C1 and C2, and it is already an open item (tasks #52/#55, U-02/E-01/E-02). If P-5 splits
  the way §3 says, the classifier's I/O verdict on streaming rows is only as good as that
  assumption, and the honest print must say so.
* **Decode only.** No prefill term is classified. Prefill is compute-bound in a way this model
  does not represent at all.
* **No PCIe/transfer resource.** Host-resident weights crossing PCIe per token are priced inside
  `eta_r`, not as their own resource, so a genuinely PCIe-bound configuration would be reported
  as RAM-bandwidth-bound. Named here so the gap cannot be discovered later and called a surprise.
* **`margin_x` is a ratio inside one model.** It is not a measurement of the second constraint;
  it is where our arithmetic says the second constraint would take over.

---

## 7. What would REFUTE this idea

Stated as concretely as the confirmations, because the failure modes are the reason to run it:

1. **The label is a constant.** One class > 85% of 340 cells (K-1). Then "which resource binds"
   is a distinction our model cannot draw on the hardware we model, and the honest response is to
   stop advertising it. **This is the single most likely refutation** — every machine preset we
   ship is bandwidth-heavy by construction, and the two bandwidth pools collapse into one class
   under the §1 R5 scoring rule *deliberately*, to make this failure reachable.
2. **The label is unstable** (K-2): it moves when a resource the row does not use is changed, or
   the shares move the wrong way. That is an implementation refutation and it kills the change.
3. **It disagrees with the one external ground truth** (K-5): their laptop is DRAM-bound and we
   call it something else even with their measured hit rate substituted.
4. **It changes nothing.** All three P-3 probes coming back empty means the classification is
   decoration: correct, printable, and with no consequence for a single line of advice. That is a
   *sincere* possible outcome — the upgrade advisor is already gain-gated, and if the IQ and
   speculation bounds never bite on any real cell, then the honest report is "we found one latent
   input bug and otherwise the shipped advice was already regime-safe." That is a good result for
   the users and a bad result for this idea, and it must be published as the latter.
5. **The deep refutation, which this experiment CANNOT perform.** A class that does not predict
   which lever pays. The test is an A/B: take a cell the classifier calls IO-bound, apply a
   storage upgrade, and measure. Take a cell it calls RAM-bandwidth-bound, apply the same
   upgrade, and measure neutral. We have one such pair and it is **theirs**, not ours (their
   laptop: I/O lanes, threads and NVMe all neutral, which is a *positive* datapoint for the
   bandwidth-bound verdict at C2). Until we run our own, the correct status for the classifier is
   **"consistent with one external observation"**, not "validated", and the shipped text must say
   the weaker thing.

**A PASS does not prove:** that our four resources are the right basis (§6 lists two missing
ones), that `eta_r`/`geta` are correct (U-29/U-31 already established they are entangled with the
byte convention), or anything about prefill, quality or multi-user serving.

---

---

## 8. ADVERSARIAL REVIEW — the ways this experiment could not fail

**Added 2026-07-30, AFTER the first scoring run**, by a review whose only brief was to find ways
this pre-registration cannot lose. Published here at equal prominence to §4 because *"a design
that cannot fail"* is a worse defect than any miss it could have reported. Everything below is
disclosed as post-hoc. **No threshold was moved and no verdict changed**: the four defects were
all in the *scoring*, and every staked number still reproduces exactly after the repairs.

### 8.1 Defects found and fixed

| # | defect | why it is fatal | fix | verified by |
|---|---|---|---|---|
| **D-1** | **K-3 auto-passed on any re-run after commit.** The verdict was `all([..., k3 or k3_vacuous, ...])` and `--baseline-ref` defaulted to `HEAD`. | The moment the change was committed, `git show HEAD:` returned a baseline that *already contained the classifier*; all 340 cells compared the module to itself, found 0 regressions by construction, and printed **PASS**. The regression gate silently deleted itself at exactly the moment it started to matter. | Vacuous baseline ⇒ **refuse** (exit 2). With `--allow-vacuous-regression` ⇒ verdict **INCOMPLETE** (exit 3), JSON records `K3: null`. `PASS` is now unreachable without a real baseline. | both paths executed: exit 2 and exit 3 |
| **D-2** | **K-4 could not fire from the grid.** `check_k4` tested one hand-built `Row` whose `placement` was the exact literal the caveat gate compares against. | A tautology: it asserted that a function given the string it tests for finds that string. The 340 cells were never checked, and the realistic failure — the gate is `placement == "all in VRAM"`, exact equality, so a row-name suffix silently kills the caveat everywhere — was invisible to it. | Scored on all **137** all-in-VRAM winners in the grid via the same `binding_report` call `run()` makes; **empty population ⇒ FAIL**; >4.5-bit negative control retained. | fault injection: row renamed ⇒ **137/137 fire**; empty population ⇒ fail |
| **D-3** | **P-5 partly tested itself.** `hot`, `streamable`, `miss` were recomputed from formulas copy-pasted out of `plan.py`, and `eta_r = 0.38` was hard-coded — 4 of 7 scored quantities never touched shipped code. | The prereg claimed they were "reproduced by the shipped code". A sign error inside `evaluate`'s disk row, or a move in the shipped `eta_r`, would have left all four still matching their staked values. | Inverted out of the shipped row's own `terms`; reconstruction cross-checked against `evaluate`'s `act` to 1e-9 with an **abort** on mismatch; `eta_r` imported as `plan.ETA_R_MOE`. | fault injection: `ETA_R_MOE → 0.40` now breaks P-5 by **−5.0%**; perturbing `act` **aborts** instead of scoring |
| **D-4** | **P-3c counted cells the advice never reaches.** 35 cells were reported as over-sold the speculation headline; `run()` prints that block only where `speculation_advice()` fires, which is **17** of them. | Counting a cell the shipped block never reaches as "over-sold advice" is the same error the experiment exists to catch, one level up. | Population gated on the identical `speculation_advice()` call `run()` makes. | 17/127 after gating; **P-3c still passes** |

Two further hardening changes, neither of which found a live defect: a **machine-preset audit**
that refuses the sweep if any preset would be priced by one of `run()`'s `or` fallbacks (`rc or
16`, `rb or 40`, `db … or 0.5`) or carries a calibration key `cell_kwargs` does not reproduce;
and `make_ev` no longer *silently* drops `true_size_gb_scale` but applies it exactly as `run()`
does. Both exist because "a profiler run on a model too big for the card produced a streaming
artifact nearly published as an MLA measurement" is this project's own history, and both of these
were the same shape.

### 8.2 The defect that could NOT be fixed: K-1's bar was never live

Fixing this would require moving a staked threshold after seeing the answer, which is the
back-fit the protocol forbids. So it is measured and published instead. The scoring script now
sweeps the identical 340-cell grid across bit-widths and prints the result:

| bits | 2.0 | 2.5 | 3.5 | 4.5 | 6.0 | 8.0 |
|---|---|---|---|---|---|---|
| modal class share | 55.9% | **52.4%** | 50.9% | 48.2% | 43.2% | 46.5% |
| distinct classes | 4 | **4** | 4 | 4 | 4 | 4 |

The staked bar is **>85% ⇒ K-1 fires**. The worst modal share **anywhere** in a 4× bit sweep is
55.9%, and the "≥3 of 4 classes" clause is met with 4 classes at every single point — indeed the
`ctx = 0` half-grid alone already produces 3. **K-1 could not have fired on any configuration we
ship.** §7.1 called it "the single most likely refutation"; that was wrong, and it is corrected
here rather than left standing. **K-1's PASS is weak evidence and is not cited as a confirmed
prediction.** What P-1 does establish — that the label is not a constant — survives; what it does
not establish is that the bar was a real risk.

The same criticism, in a milder form, applies to **P-4's** direction: it *missed*, and a
prediction that misses is a prediction that could fail, so P-4 is the one number in this document
that demonstrably had a live bar. That it carried no kill rule is recorded in §5.

### 8.3 Staked FORWARD for the successor experiment (not applied to this run)

These are live bars, chosen against the distribution now known, and they are staked here so the
next run cannot choose them afterwards:

```stake-forward
k1_max_modal_share_v2       = 0.60      # vs 55.9% worst observed - a bar 4.1 points away
k1_min_distinct_classes_v2  = 4         # 3 was met by half the grid alone
k6_p3_min_probes_firing     = 1         # K-6: all three P-3 probes empty => the classification is
                                        # decoration; the advice gates do not ship and FINDINGS
                                        # records that the shipped advice was already regime-safe
p4_min_frac_ceiling_lt_2_v2 = 0.00      # WITHDRAWN, not re-staked: 0.6% measured; the quantity
                                        # that carries information is the NON-binding lever cap
```

**K-6 is stated but does not fire on this run.** P-3(a) found a real latent input bug (31 cells),
so it would have passed anyway; applying a kill rule retroactively to a result already scored
would be indistinguishable from writing the rule to fit the outcome.

### 8.4 What the first review did NOT find

C-14 is not violated: nothing here touches the GPU or any machine state, every number is
deterministic arithmetic over shipped preset tables, and the only cross-machine comparison is the
disclosed external retrodiction against BigMoeOnEdge's published figure. Null results are reported
at least as prominently as positives — P-4's miss gets eight lines of transcript and a permanent
demotion note inside `binding_report` itself.

It also asserted that "K-2 is a genuinely live gate (508 real checks, and it counts its own checks
so an unexecuted soundness pass cannot masquerade as a clean one)". **That was wrong, and §8.5 is
where it is corrected**: K-2 has two staked clauses and the 508 checks covered one and a half of
them. The counting discipline the sentence is proud of was applied to the clauses that were
implemented and told us nothing about the clause that was not.

### 8.5 SECOND ADVERSARIAL REVIEW — the pre-run audit

**Added 2026-07-30, BEFORE this run was published**, by a second pass whose only brief was to
*construct the concrete input that makes this experiment fail* and verify it does. Disclosed as
post-hoc relative to §4 and §8.1, exactly as §8 is. **No threshold was moved, no staked number
changed, and the verdict is unchanged (PASS).** Every defect below is in the *scoring or the
regression tests*, never in `evaluate`'s arithmetic — K-3's own numbers were bit-identical before
and after all five repairs.

| # | defect | why it is fatal | fix | verified by |
|---|---|---|---|---|
| **D-5** | **K-2's SECOND staked clause was scored NOWHERE.** §5 K-2 reads "Also fires if any row's terms fail to reconstruct its own tok/s to 1e-9", and §1 R1 calls it "smoke-tested, and K-2". The scoring script never checked it; the smoke test that did ran a **hand-picked six-config grid** that emits six of the seven row families and **never the dense-split row** — three resources, KV split across two tiers, and a WINNER on **8 of the 340 cells**. | The exploit: drop `(1-g)·kv_gb/(ETA_KV·rb)` from that row's `ram_bw` attribution. tok/s is **bit-identical** so K-3 passes; no phantom or monotonicity check bites so K-2 passes; the script printed **VERDICT: PASS at exit 0** while the row reconstructed **4.94% off its own speed** and the **printed** binding share and ceiling moved 64.5%→67.7% and 2.82x→3.09x. A wrong number on the page, through a green gate. | Reconstruction now scored on **every row of every cell** (1527 rows), counted, and folded into P-2/K-2. The smoke grid is widened to all seven families and **asserts** the families are present. | injection ⇒ **10 failures, K-2 FIRED, exit 1**; clean tree ⇒ 0/1527 |
| **D-6** | **The smoke grid's `ctx` entries carried no `kvp`**, so `kv_gb = ctx·kvp/1e9` was **identically zero** on every row that test has ever built. | Every KV attribution term was multiplied by zero, in *every* row family. The grid knob reads as "depth is covered" while the quantity it controls cannot vary — the #85 arms C/D shape. Dropping the KV term from the **hybrid** row was equally invisible. | `kvp` is explicit on every ctx entry, plus an assertion that some entry carries a non-zero `kv_gb`. | both KV drops (dense split, hybrid) now caught; both passed before |
| **D-7** | **An abort left the previous run's result on disk as the apparent current one.** `die()` exits before anything is written and both outputs are written only at the end of `main()`. | After a refusal, `exp54_binding_constraint.json` still said `"verdict": "PASS"` from the last completed run, with nothing distinguishing "scored and passed" from "never scored". Every consumer reads that file. The docstring's "idempotent: re-running overwrites the same three files" was false on exactly the path where it mattered. Same shape as D-1, one file downstream. | `die()` overwrites both outputs with a `REFUSED` record and `K1..K5 = null` first. | bad `--baseline-ref` ⇒ JSON flips `FAIL` → `REFUSED`, exit 2 |
| **D-8** | **A fired kill rule could be reported as INCOMPLETE rather than FAIL.** The verdict branch tested `k3_vacuous` first. | With a vacuous baseline, a run in which **K-1 and K-5 both fired** exited **3** — the code reserved for "a kill rule could not be evaluated" — so a wrapper treating 3 as "re-run with a proper baseline" would never surface the two rules that died. | FAIL takes precedence; INCOMPLETE is reserved for a run where every rule that *could* be scored held. | injection ⇒ exit **1** (was 3); clean tree + vacuous baseline still ⇒ exit 3 |
| **D-9** | **K-3's grid never exercised the pinned-file-size or codebook paths.** `cell_kwargs` fixes `true_size_gb=None` and `codebook_share=0.0`. | **Every one of the 14 locked ladder rows is a `--gguf` run** — a real file size pinned (the input whose absence `evaluate`'s own comment says "wrongly evicted the all-in-VRAM row") and, for the IQ files, a non-zero codebook share. A regression that only bites when a size is pinned was invisible to K-3. | A second **deterministic** regression arm replays the identical 340 preset pairs with a pinned size (×0.8) and `codebook_share=0.5` — no GGUF, no machine state, no change to P-1's staked denominator. | injection `size = true_size_gb*1.01` ⇒ **arm 1 reports 0 moved, arm 2 catches 124** |

Two smaller items, neither of which found a live defect: K-3 skipped the baseline comparison
entirely for any cell whose row list was empty (`if res is None: continue` ran *before* the
comparison), so the largest possible regression — every row for a cell deleted — was filed as
`klass: null` and never compared; and `--json`, documented in §2, did nothing at all. Both fixed.

**The 14 locked ladder rows were also checked directly, out of band**: all 14 predictions and all
14 placements reproduce unchanged against `weights/data/ladder_state_locked.json` under the current
tree. That check is *not* wired into the scoring script — it needs 14 local GGUFs and a calibration
state, which would make the verdict machine-dependent and put a C-14 comparison inside a
deterministic experiment. The deterministic arm-2 replay above covers the same **code paths**.

**All five kill rules were individually fault-injected and each produced exit 1**: K-1 (constant
`RESOURCE_CLASS`), K-2 (swapped `vram_bw`/`ram_bw` keys on the dense split — fires with K-3 clean,
which is the true reporting-layer bug; and a `db`-dependent phantom term ⇒ 52 phantom failures),
K-3 (`CPU_ATTN_S_PER_POS_LAYER` 1.55e-6 → 1.60e-6 ⇒ 53 cells), K-4 (row renamed to
`all in VRAM (KV in VRAM)` ⇒ 137/137), K-5 (`ETA_R_MOE` 0.38 → 0.40).

### 8.6 THIRD PASS — §4 P-3a's direction rationale is REFUTED (post-run, published here)

**Added 2026-07-31**, by a pass whose brief was to re-derive P-3a's before/after independently
instead of trusting it. **The staked part of P-3a held and its number is unchanged: 31 of 340
cells.** What is refuted is the *rationale sentence* attached to it in §4 — an unstaked
characterisation that nevertheless shipped, in this document, in `upgrade_advisor`'s docstring and
in the L-23 register entry:

> "Its signature is one-directional: it can only ever **suppress** an upgrade that would
> genuinely help."

**That is wrong, and the dominant direction is the opposite one.**

**Why the original evidence could not see it.** `weights/exp54_binding_constraint.py` scores P-3a
with `old_upgrade_lines()`, a *re-implementation* of the three pre-fix call sites. A reproduction
can only ever agree with the reading that produced it, and it reports the *set of fired lines*,
not their direction. The replay below instead injects the defect **into `run()` itself** — the two
kwargs are dropped from `evaluate` for the duration of `upgrade_advisor` and for nothing else —
and reads `upgrade_advisor`'s own return, so every disagreement is classifiable by sign. It
reproduces **31 changed cells**, and independently confirms **0 of 340** winning rows and **0 of
340** emitted commands move. Reproducible from the repo:
`python weights/exp54_p3a_direction.py` → `weights/data/exp54_p3a_direction.json`. It is
deterministic (no GPU, no machine state, nothing timed) and **refuses at exit 1 if 0 cells
change**, so an injection that silently fails to reach the advisor cannot be read as "the claim
held".

| direction | count | what it means for the user |
|---|---|---|
| **INVENTED** (fires only with the bug) | **30** | the tool recommended hardware the fixed advisor retracts |
| SUPPRESSED (fires only once fixed) | 3 | the tool hid an upgrade that genuinely helps |
| overstated tok/s (same upgrade) | 14 | gain printed too high, worst **×2.16** |
| understated tok/s (same upgrade) | 7 | gain printed too low |

**Two mechanisms with opposite signs**, which is why "one-directional" was never available:

* **the row menu suppresses.** A counterfactual with no `n_layer` has no split placements to win
  with, so it can only be *slower*. All 3 suppressions are `enable XMP (free)` on the slow-RAM
  `2016` box — `deepseek-16b` at ctx 0 **and** at 16384, `qwen3-30b` at ctx 0 — so suppression is
  not confined to ctx = 0, it is confined to the machine where XMP is a live lever at all.
* **the depth term invents.** `(n_layer or 32)` prices the counterfactual's CPU attention as a
  32-layer model while the baseline pays for 80, so the "gain" is a comparison between two
  **different models**, not two machines. All 30 inventions are at **ctx > 0**. The clearest is
  `llama-70b` on `colibri` (128 GB RAM, no disk tier in play) at ctx 16384, where the bug printed
  **`+16 GB RAM` and `NVMe SSD` at an identical ×1.70** — a RAM-capacity lever and an I/O lever
  cannot honestly buy the same number, and the fixed advisor prints neither.

**Why this correction is not cosmetic.** Suppressed advice costs a user speed they could have had.
Invented advice costs them money — €40 of RAM, €100 of NVMe — on hardware that will not move their
token. The second is the worse failure and it is precisely the one the shipped rationale asserted
could not occur, on 30 cells against 3.

**Consequence applied in the same commit, not argued away.** The sentence is withdrawn from
`upgrade_advisor`'s docstring and from the L-23 register entry, replaced by the measured
breakdown; and a smoke test now pins the **printed line** rather than the plumbing —
`tests/smoke.py::t_p88_upgrade_advice_is_not_invented_by_a_depth_mismatch` asserts that the
`colibri` cell fires **no** upgrade line while still *naming* both dead levers, and that the
`deepseek-16b`/`2016` cell fires XMP on a split-experts row. It fails on the pre-fix tree in both
directions. The pre-existing `t_p88_upgrade_counterfactual_shares_the_baseline_inputs` asserts only
that the kwargs travel, which would still pass if `evaluate` ignored `n_layer` entirely.

**No staked number moved and the verdict is unchanged (PASS).** §4 P-3a required ≥1 cell; 31 were
found before this pass and 31 after. The 14 locked ladder rows (cal_id `a19aeee4`) were replayed
through the defect-injected tree **and** the fixed tree: 14/14 reproduce prediction, placement and
full emitted command in **both**, which is the expected result — the ladder's winners are chosen by
`evaluate`, and P-3a lives entirely in the advice layer downstream of it.

---

**Wired into:** `quantprobe/plan.py` — `binding_constraint()`, `capacity_probe()`,
`spec_ceiling()`, the conditional advice gates, the upgrade-advisor input fix, and (from §8, D-3)
the named `ETA_R_MOE` / `ETA_R_DENSE` constants that replace two inline literals so the external
arm imports the number instead of copying it. No law changes, no constant *values* move, and K-3
requires every predicted tok/s to be bit-identical to the version shipped before this document.

**§8.5 touched no shipped code.** All five second-pass repairs live in
`weights/exp54_binding_constraint.py` and `tests/smoke.py`; `quantprobe/plan.py` is byte-identical
before and after them, and both K-3 arms report 0 cells moved. Independently of the gate, the 14
rows of `weights/data/ladder_state_locked.json` were replayed through the current tree and every
prediction and every placement reproduces unchanged.
