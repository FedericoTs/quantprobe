# ROADMAP R25 v2 — Corrected State After the First-5, and the Ranked Next Wave

Lead-researcher synthesis. Supersedes the optimistic R25 mid-execution notes. Every claim below
was re-derived against the actual code (`codec_zoo.py:263`, `ecvq_cmcore.py:40`, `lattice_rescore.py:39`,
`scale_plane_codec.py:42-62`) so the plan builds only on what is genuinely validated.

---

## 0. CORRECTED STATE AFTER THE FIRST-5 (the honest frontier)

### Champion (Qwen2.5-0.5B)
Per-group(g=128) signed-Hadamard incoherence rotation + AWQ scaling + scalar entropy-constrained
quantization (ECVQ) + entropy-coded index stream + 0.5% largest-|w| kept fp16 (outliers).

**Honest operating point: 4.483 ppl @ ~3.00 b/w** (re-baselined from 3.130 by A1).
Second measured point: 4.169 ppl @ ~3.79 b/w (ECVQ.003). The two lowest measured points imply a
near-vertical local R-D slope of about **-5.5 b/w per ppl** — this steepness is why every
"near-lossless-end" comparison must be done CO-LOCATED in ppl, never by extrapolation.

### A1 (scale-plane side-info recompression) — SOUND, with caveats
- Real byte-identical container recompresses the champion's amax + outlier side info from
  0.2876 b/w (raw 16b/group amax + 32b/outlier) to **0.1608 b/w**, a verified **0.127 b/w bank**
  (zstd-only fallback 0.117) at IDENTICAL perplexity. Verified round-trip: amax-hi 2.796->0.746MB,
  amax-lo incompressible, out-pos 1.919MB, out-hi 0.602MB, out-lo 1.128MB.
- **Not a double-count.** `codec_zoo.py:263` confirms the index rate `ent*W.size` codes
  amax-NORMALIZED weights (`Rn = R/amax`); A1 banks only the disjoint `16*N.shape[0]` (amax) +
  `32*mask.sum()` (outlier) terms.
- **CAVEAT 1 (cross-round fairness):** of the 0.127, ~0.07 is the **amax bank** that EVERY
  per-group-amax codec is equally entitled to — including the E8/D4 lattices (`codec_zoo` lines
  474/534 pay the identical `16*N.shape[0]`). Only ~0.077 (the **outlier bank**) is
  champion-exclusive. Re-baselining ONLY the champion by the full 0.127 and comparing to
  un-rebaselined competitors inflates the champion's apparent edge by ~0.07 b/w.
- **CAVEAT 2 (not bit-identical to deployed):** `scale_plane_codec.planes()` selects outliers/amax
  on RAW |W| (`scale_plane_codec.py:50-52`), but the deployed champion does AWQ-scaling FIRST and
  selects on the scaled `Ws` (`_had_ecvq`). Outlier-set Jaccard is only 0.35-0.55, so A1's container
  is statistically-equivalent, not bit-identical. Bank magnitude is unaffected (amax-hi order-0 rate
  0.0188 b/w either way) but the round-trip proves A1's OWN container, not the deployed side info.
- **CAVEAT 3 (lower bound):** the champion pays a THIRD uncoded stream A1 ignores — AWQ per-column
  scale at 16b/col (`codec_zoo.py:270`, `+16*W.shape[1]`) = 0.011 b/w + codebook levels 0.0005 b/w.
  So 0.127 is a LOWER bound on recoverable side info.
- **amax-snapping** (drop the incompressible lo byte, snap up to log-bin ceiling, ~+0.0625 b/w) is
  genuinely ADDITIVE to the 0.127 but is LOSSY and MUST pass the ppl gate. Report it as a SEPARATE
  b/w line, never summed with the lossless bank.

### A2 (noise floor) — CORRECTED READING
fp16 ppl on 8 enwik8 offsets swings 2.2-17.2 because offsets hit raw-Wikipedia MARKUP/non-English,
not prose. That measured slice-CONTENT heterogeneity, NOT codec noise. Correct read: the arena is
**deterministic + paired** (fixed seeds, fixed 120k PROSE slice), so codec-vs-codec margins on that
slice are decidable to ~0. Generalization ACROSS content is content-dependent (gap-to-fp16 ranges
0.18-8.3). A1's lossless win is ppl-invariant and therefore immune to this. **Operational lesson:
every comparison is co-located on the fixed prose slice; paired margins below the slice's
ppl-noise-equivalent are not real.**

### C1 (divergence battery) — FLAWED → scope downgraded
- **Defensible claim:** single-slice enwik8-ppl is a FAITHFUL RATE-DISTORTION proxy among PRINCIPLED
  codecs (hand-computed Spearman=1.000, n=8, vs worst-of-6-domain mean teacher-forced KL on ~360
  tokens), with the 2.4b point degrading roughly uniformly (KL 1.2-2.0).
- **NOT established:** "verifier not overfit at >=2.4b." n_overfit=0 (all 8 codecs are closed-form
  single-knob, none loop-selected); Spearman=1.000 on a monotone R-D family is near-tautological;
  probes are single sentences (25-85 tokens); the spec'd 24-champion retro-audit, exact-match
  capability canaries, and sealed escrow were all SKIPPED. Honest status: **overfitting-robustness
  untested.** This is what E-experiment "C1-PC" (positive control) closes.

### A3 (lattice rescore) — FLAWED → NOT reopened, UNDECIDED
- A3 correctly showed the old per-coordinate-entropy metric OVERCHARGED lattices by ~0.7-1.0 b/w
  (real, worth banking) and that a real decodable E8 q0.08 container costs ~4.64 b/w honest at
  ppl 4.020.
- The claimed "0.08-0.16 b/w edge" is an ARTIFACT of three errors, all confirmed in code:
  - **ASYMMETRY:** `lattice_rescore.py:39` compares against a hardcoded `CHAMP_NOTE` string; the
    champion index stream is priced at ORDER-0 while the lattice coords are cmcore/zstd context-coded.
    `ecvq_cmcore.py` exists but its result is never folded in.
  - **EXTRAPOLATION:** E8 q0.08 ppl 4.020 is 0.089 ppl BELOW the champion's lowest MEASURED point,
    so the "4.72-4.80 envelope" is convex extrapolation on a -5.5 b/w/ppl tail (linear gives ~4.60,
    edge -0.04; convex gives -0.12 to -0.16 — the sign is set by the model, not data).
  - **UN-VALIDATED HEAD:** the cmcore 5.248->4.561 (13%) drop extrapolates a 16MB head
    (`lattice_rescore.py:42`); the convergence sweep was never run.
- Even if real, the edge lives only at the ~4.6 b/w near-lossless end; at the ~3 b/w operating point
  the archived E8 q.22 (4.535@3.92) and q.28 (4.944@3.58) LOSE to the champion.

### A4 (trellis preflight) — PRIZE CONFIRMED PRESENT
gapSLB measured 0.298 @ lam.003, 0.312 @ lam.008, 0.375 @ lam.040 — all well above the 0.08 b/w
kill line, and GROWING as rate drops toward the headline. The space-filling ceiling
(0.5*log2(2πe/12) = 0.2546 b/w) is largely unrealized. ECTCQ build proceeds.

---

## 1. THE SYMMETRY DECISION RULE (the gate everything keys on)

Let **D_idx** = champion ECVQ-index-stream bank = order-0 b/w minus best(zstd-19, cmcore) b/w,
measured on FULL / convergence-checked streams at lambda matched to each comparison ppl. The
A5-X1 oracle computes a tighter, un-extrapolated lower bound on D_idx via explicit conditional
entropy on decoder-shipped contexts.

- **CASE A — D_idx <= 0.01:** indices ~iid after rotation; champion order-0 accounting was right.
  Decide lattices by a lever-matched, co-located comparison only; lattices most likely STAY CLOSED.
- **CASE B — D_idx >= 0.05:** indices NOT iid; champion shifts DOWN by D_idx at EVERY operating
  point, which is >= the entire claimed lattice edge. Lattices STAY CLOSED (edge inverts) AND the
  champion banks another free, ppl-invariant win of D_idx.
- **CASE C — 0.01 < D_idx < 0.05:** borderline; proceed to the full co-located, lever-matched,
  convergence-validated comparison with the champion total reduced by D_idx. Default = STAY CLOSED.

In ALL cases the comparison must be (i) SYMMETRIC — both streams through the SAME coder on FULL or
convergence-checked data (cmcore at 4/8/16/24/full-MB heads, require <0.01 b/w ratio drift before
trusting any extrapolation); (ii) LEVER-MATCHED — add the champion's 0.5% fp16 outliers to the E8
path; (iii) CO-LOCATED in ppl — sweep ECVQ lambda until the champion lands at E8's EXACT ppl 4.020.
Absent all three, lattices remain CLOSED by default.

---

## 2. RANKED NEXT WAVE (by expected information gain per GPU-hour, diversity preserved)

Four lenses, deliberately spread: **(I) index-stream symmetry / accounting**, **(II) the open
R-D prize (trellis + allocation + embedding)**, **(III) exact-symmetry gauge canonicalization**,
**(IV) new asset classes (KV-cache, optimizer, model-family).** Ranking weights three things:
how much it MOVES or SETTLES the frontier, how DECISIVE the verifier is, and GPU-hours (most index
work is CPU-only, so it dominates the per-GPU-hour metric).

### TIER 0 — RUN FIRST (cheap, CPU-mostly, gates everything else)

**R1. A5-X1 — Cross-stream conditional-entropy oracle for the ECVQ index stream** [Lens I]
- *Why #1:* it is the single number (D_idx) that the lattice rule (Case A/B/C), the A5 free-win
  claim, and the fair lattice baseline ALL key on, and it computes it WITHOUT cmcore's 24MB-head
  extrapolation tax — replacing the weakest link in both gate scripts. ~10 min, no GPU, no ppl gate
  (lossless). Highest information-per-GPU-hour in the wave (GPU-hours ≈ 0).
- *Mechanism:* reuse the exact rotation+ECVQ pipeline from `ecvq_cmcore.py` to materialize the
  per-tensor index array plus each coord's decoder-shipped metadata (group-id, position-in-group
  0..127, amax-octile, row-id, layer-depth, role). For each candidate context C compute order-0
  H(idx) and conditional H(idx|C) via vectorized 2-D bincount; sweep single contexts + best
  2/3-way conjunctions. Conditional entropy is a coder-achievable lower bound and every context is
  decoder-reconstructable, so the number is honest.
- *What is evolved/built:* `weights/ecvq_idx_oracle.py` — fully-vectorized numpy conditional-entropy
  probe (no coder, no training). The winning context is then realized as a decodable per-context
  static-table arithmetic coder. Contexts are enumerated, not LLM-mutated, to fix the ceiling first.
- *Verifier:* conditional entropy is the honest lower bound; any winner >=0.03 b/w is realized as an
  actual arithmetic-coded container (per-context frequency tables, table bytes INCLUDED) over the
  full index stream and byte-exact round-tripped. **Train/code split: fit context tables on a
  hash-fixed HALF of layers, CODE the other half, report bank ONLY on coded layers** (kills
  bucket-overfit flattery). ppl unchanged (lossless re-coding).
- *Runtime on 1060:* CPU-only. Extract ~2 min, sweep ~1-3 min, realize+round-trip ~1-2 min. < 10 min.
- *Expected:* either D_idx >= 0.03 (most likely from position-in-group + amax-bin) → CASE B/C fires,
  champion ~3.00 → ~2.93-2.97 b/w AND the lattice "edge" shrinks/inverts; or D_idx < 0.01 → CASE A,
  order-0 confirmed.
- *Kill (pre-registered):* if the best single/conjunction context gives H(idx|C)-H0 < 0.01 b/w on
  the CODED (held-out) layer half, declare indices iid-after-rotation, set D_idx=0, STOP — the
  A5/X2/permutation index-coding ideas die together (clean "rotation whitens to near-iid" result).

**R2. C1-PC — Loop-overfit positive control + 24-champion retro-audit** [Lens I/verifier]
- *Why #2:* closes the one CLAIM the project cannot currently defend ("verifier not overfit").
  CPU-only manufacturing + cheap re-quant of archived champions. Decisive either way: a slice-tuned
  control that DEMOTES under worst-domain-KL proves the battery catches what ppl hides; if rho stays
  1.000 even WITH the control, THAT is the real not-overfit result. Near-zero GPU.
- *Mechanism:* (1) manufacture a genuinely loop-overfit codec by directly MINIMIZING held-out enwik8
  ppl over 50-200 candidate ECVQ level-sets / per-layer lambda schedules on the SAME fixed 120k prose
  slice the loop selects on (keep the argmin on that slice). (2) Score it in `divergence_battery.py`
  alongside the 8 principled codecs. (3) Discriminating prediction: an enwik8-tuned codec sits BETTER
  on enwik8-ppl than its worst-domain-KL warrants → it should DEMOTE under worst-domain-KL ranking,
  breaking Spearman=1.000. Also run the spec'd 24-champion retro-audit (re-quantize each round's
  champion from `codec_zoo`, ~2-3 min each) to plot worked-slice ppl vs escrow ppl vs KL.
- *What is evolved/built:* a slice-tuned ECVQ level/lambda searcher + the retro-audit harness over
  the 24 archived champions; both reuse `codec_zoo` + the validated `load_quant/ppl` harness.
- *Verifier:* re-run the EXACT rank check with the control in the set; report whether rho stays 1.000
  or the control re-ranks. The retro-audit yields the actual selection-overfitting curve.
- *Runtime on 1060:* CPU-mostly for the search; ~2-3 min GPU ppl per archived champion × 24 ≈ 1
  GPU-hour for the retro-audit, the search itself negligible.
- *Expected:* either the battery catches the slice-tuned control (battery validated as a real
  overfitting probe) or rho holds (verifier robustness upgraded from "untested" to "demonstrated").
- *Kill:* if the manufactured control cannot be made to beat the principled frontier on the worked
  slice at all (i.e. ECVQ has no slack to overfit), report that the codec family is too rigid to
  overfit and the question is moot.

**R3. E1 — RMSNorm-Gain Absorption Canonicalization** [Lens III, exact symmetry]
- *Why #3:* an EXACT symmetry (function-preserving through RMSNorm) that is currently NEVER applied
  (`quant_keys` touches only the 7 linears, never the layernorm gammas). Measured layer-0 input
  gamma has 34.8x abs dynamic range. Folding gamma into the next linear's input columns and setting
  norm=1 is ppl-invariant yet smooths the amax field (better cmcore) and de-double-counts AWQ.
  ~9 min, one ppl pass purely as a function-preservation check. High value-per-GPU-hour because the
  ppl gate is a sanity check, not a search.
- *Mechanism:* per block, multiply q/k/v_proj input columns by `input_layernorm.gamma` and
  gate/up_proj input columns by `post_attention_layernorm.gamma`, rewrite both norm tensors to ones
  (a 21k-param side stream, counted), then run the UNMODIFIED champion pipeline (ecvq lam=0.008 +
  A1 scale_plane_codec). Recompute the AWQ scale on the folded model so it no longer redundantly
  encodes gamma.
- *What is evolved/built:* `codec_zoo.absorb_rmsnorm(Wdict)`; evolved knob = WHERE the diagonal lands
  (fold-into-next-linear vs split-half sqrt(gamma) vs fold-then-re-AWQ).
- *Verifier:* (a) ppl must match the un-absorbed champion to <0.003 (function-preserving check —
  gamma=1 norms round-trip byte-exact); (b) honest bits = ecvq index entropy + cmcore-coded amax
  (SAME scale_plane_codec as baseline → pure canonicalization delta) + outliers + 21k norm cost;
  (c) reconstruct W, re-fold gamma OUT, assert within fp16 ULP.
- *Runtime on 1060:* ~6-9 min (vectorized column scale + one pipeline pass + one ppl + one cmcore
  amax pass).
- *Expected:* amax-plane cmcore ratio improves (smoother field, 3.74x → ~4.5-5x) for ~0.02-0.04 b/w
  + a tighter index entropy ~0.01-0.03 b/w. Total ~0.03-0.06 b/w at IDENTICAL ppl, ppl-invariant
  like A1.
- *Kill:* if cmcore amax bank improves <0.01 b/w AND index entropy drops <0.005 b/w, gamma's scale
  was already captured by the per-group amax; abandon in one ~9-min run. (If ppl moves >0.005 it is
  a bug, since the fold is exact — debug, do not kill.)

### TIER 1 — THE OPEN R-D PRIZE (decisive on the north-star metric, GPU-heavier)

**R4. ECTCQ-RM — entropy-constrained trellis with rate IN the Viterbi metric** [Lens II]
- *Why #4 (top of GPU-heavy tier):* this is the ONE experiment that can move the HEADLINE operating
  point (unlike the lattice edge, which only exists at near-lossless). A4 already CONFIRMED the prize
  (gapSLB 0.30-0.38 >> 0.15 build threshold) and that it GROWS toward the ~3 b/w point. Expected
  +0.10-0.15 b/w at iso-ppl → champion ~3.00 → ~2.88-2.90 @ 4.483. Higher GPU cost (~1 h/candidate
  encode) demotes it below the Tier-0 oracles on per-GPU-hour, but its frontier impact is the largest
  in the wave.
- *Mechanism:* an 8-256 state ECTCQ whose branch metric is `(x-c_k)^2 + lambda*(-log2 p(branch|state))`,
  with p(branch|state) re-estimated ECVQ-style each iteration (subsample 5% of g=128 groups). A
  g=128 group is coded as a Viterbi PATH; the decoder replays the same state machine. Recovers the
  unrealized space-filling loss. Batched Viterbi vectorized over thousands of groups (states×branches
  as gather+argmin), 50k-group chunking, states capped at 256. Seed = Fischer-Wang 1992 8-state.
- *What is evolved/built:* `weights/tcq_zoo.py:make_ectcq(n_states, branch_lambda, tstats)`; evolve
  only trellis topology, per-state subset-codebook generator, branch_lambda schedule.
- *Verifier:* honest rate = EMITTED bytes of a real arithmetic round-trip of the decision stream
  (gate `|emitted_bits - claimed_entropy| < 0.5%` blocks the trellis joint-entropy-undercount). ALL
  decodable artifacts counted (per-state prob tables, per-state codebooks 16b each, amax via
  scale_plane_codec ~0.05, 0.5% fp16 outliers). Byte-exact decode replays the state machine and must
  reconstruct W bit-identically (enforces causality). Quality = held-out enwik8 ppl. **Co-located:
  sweep ECTCQ lambda until ppl matches champion ECVQ.008/.003 EXACTLY — no extrapolation.**
- *Runtime on 1060:* CPU/numpy batched Viterbi ~30-60 min/candidate at 256 states; arithmetic
  round-trip ~2-5 min; ppl ~30-60 s GPU. ~1 h/candidate; 4 rounds × ~6 candidates ≈ 1 GPU-day +
  CPU encode.
- *Expected:* +0.10-0.15 b/w at iso-ppl at 64 or 256 states. Clean negative
  ("entropy-coded scalar matches 256-state ECTCQ on rotated LLM weights") fills the arXiv 2510.11234
  empty cell.
- *Kill:* pre-flight already PASSED (A4). Post-build: both 64- and 256-state gain <0.05 b/w at
  iso-ppl after 4 rounds → ship the negative. Hard fail: emitted bits exceed claimed entropy by
  >0.5% → reject the candidate.

**R5. PPL-ALLOC — rate allocation across layers by MEASURED ppl sensitivity** [Lens II]
- *Why #5:* now that C1 validated ppl as a faithful R-D proxy, allocation can finally be driven by
  DIRECTLY MEASURED `d(ppl)/d(bits)` instead of the activation-magnitude proxy that failed twice
  (`lloyd_mixed`, `ecvq_adaptive`). KKT-optimal water-filling equalizes the slope across tensors.
  The sensitivity table is a one-time ~2-3 GPU-hour cost, then each candidate is ~3-5 min — so the
  per-GPU-hour AFTER the table is excellent, but the upfront table cost ranks it below ECTCQ.
- *Mechanism:* (1) measure per-tensor ppl-sensitivity by quantizing ONE tensor at a reference lambda
  while the rest stay fp16 (~168 evals = the sensitivity table). (2) water-fill per-tensor lambda on
  the measured slope under a global honest-bit budget (inner lambda bisection to hit a target b/w).
  (3) optionally evolve the slope→lambda map and a per-ROLE grouping. Reuses `ecvq()`.
- *What is evolved/built:* `weights/ppl_alloc.py`; evolved knob = slope→lambda mapping + role
  grouping.
- *Verifier:* honest b/w via existing per-tensor accounting (allocator only sets lambda, cannot touch
  the denominator). **Anti-gaming: measure sensitivity on one slice, report the FINAL comparison on a
  SECOND disjoint enwik8 slice; measure the ppl noise floor first (champion × 3 calib shuffles) so the
  0.03 b/w kill line is decidable.**
- *Runtime on 1060:* sensitivity table ~2-3 GPU-hours ONE-TIME (cached); each candidate ~3-5 min;
  3 rounds × ~15 candidates ≈ 1 GPU-hour after the table.
- *Expected:* 0.03-0.08 b/w at iso-ppl (champion ~3.00 → ~2.93-2.97 @ 4.483), confirmed on the
  second slice above the noise floor. Modest prior (two adjacent levers failed in-loop; ECVQ already
  learns levels per layer).
- *Kill:* oracle headroom <0.05 b/w from the sensitivity table (slopes already equal post-rotation)
  → kill before the loop, bank "role/layer rate-uniformity holds post-rotation." Else: best <0.03
  over uniform after 3 rounds, or no replication on the second slice → ship the null.

### TIER 2 — GAUGE / ACCOUNTING FOLLOW-ONS (cheap, conditional on Tier 0)

**R6. E8-SYM — lattice settled by the symmetric coset-conditional coder + D_idx gate** [Lens I]
- *Why:* the experiment that DECIDES the lattice branch, but only worth its CPU-hours AFTER R1
  delivers D_idx (if R1 fires CASE B, E8-SYM is a one-pass confirmation that lattices stay closed and
  is largely pre-decided). Near-zero GPU (ppl values archived).
- *Mechanism:* build `weights/lattice_sym.py` — the spec'd coset-conditional sequential coder for
  E8/D4 (1 coset bit/8-vec, coset-conditioned coords, parity-derived 8th not stored). Fold in D_idx
  from R1/`ecvq_cmcore` on the FULL champion index stream. Add the champion's 0.5% fp16 outliers to
  the E8 path (lever-match). Give E8 the symmetric amax discount. Co-locate by sweeping ECVQ lambda
  to E8 q.08's EXACT ppl 4.020.
- *What is evolved/built:* `lattice_sym.py` (no evolution — a decisive accounting experiment reusing
  `lattice_rescore._RN_CACHE`).
- *Verifier:* both streams through the SAME real decodable coder on FULL/convergence-checked data
  (cmcore 4/8/16/24/full-MB heads, <0.01 b/w drift required); coset-coord stream round-trips
  byte-identical (`array_equal`). Decision: reopen ONLY if TOTAL_E8 < TOTAL_champ by > ppl-noise-
  equivalent bits at a co-located ppl AND at >=1 point <= 3.5 b/w.
- *Runtime on 1060:* ~2-3 CPU-hours, near-zero GPU. cmcore full-stream head-sweep is the long pole.
- *Expected:* most likely D_idx ∈ [0.05,0.15] (A1 amax precedent) → edge inverts, lattices STAY
  CLOSED, champion banks D_idx. Even at D_idx~0, coset-conditional E8 + symmetric side + outliers
  most likely loses at ~3 b/w.
- *Kill:* D_idx >= 0.05 → lattices closed immediately. cmcore head-ratio fails convergence → use
  zstd-19 only, report the conservative bound. If E8 never beats by > ppl-noise at any point <= 3.5
  b/w → close the lattice branch permanently, bank the per-coord-overcount fix as the only result.

**R7. A1-X4 — Lossy amax-snapping with a context-coded escape stream, co-located ppl gate** [Lens I/III]
- *Why:* the one deferred A1 upside; small Pareto add (~+0.03-0.05 b/w on the amax side) but cheap
  and the escape stream protects the few high-error groups. Reports lossless and lossy banks as
  SEPARATE lines per the A1 audit mandate.
- *Mechanism:* replace the raw fp16 amax plane with a snapped log-quant amax index at b bits/group
  (sweep b ∈ {5,6,7,8} bins/octave), reconstruct at the bin CEILING (never under-scale). Add an
  error-targeted per-group ESCAPE storing true fp16 amax for the worst groups (varint-flagged). Both
  streams cmcore/zstd-coded and byte-exact round-tripped via `scale_plane_codec`.
- *What is evolved/built:* `weights/amax_snap_codec.py` extending `scale_plane_codec.py`; evolved knob
  = bin grid + escape budget.
- *Verifier:* REAL decodable container, byte-exact round trip. Snapping is LOSSY → decode and run the
  ppl gate at the SAME operating point; require delta-ppl <= ppl-noise-equivalent (co-located). Report
  TWO separate numbers (lossless 0.127 vs lossy snap), never summed.
- *Runtime on 1060:* ~2 min container + ~2-5 min cmcore (FULL stream, no extrapolation) + ~6 min ppl
  per grid × 4 grids ≈ 25 min GPU. < 35 min total.
- *Expected:* finest grid → delta-ppl within noise, amax side ~0.03-0.05 b/w (vs 0.079 lossless) →
  champion ~2.95 b/w. If even the finest grid blows ppl past noise → "amax precision below fp16-lo is
  load-bearing" (clean bound).
- *Kill:* finest grid (8 bins/octave + full escape) still moves ppl past noise → ship only the
  lossless recode. Or cmcore-coded snapped-hi fails to beat its own order-0 by >0.005 b/w → snapping
  buys nothing.

**R8. E2 — Within-group channel-sort canonicalization (SwiGLU + GQA, requant-free)** [Lens III]
- *Why:* exploits 4864 = 38×128 exactness for a requant-free within-group permutation symmetry with
  a key reconstructible from the shipped amax plane (zero side cost). Cheap, exact, gated by a 10-min
  oracle. Ranks below E1 because the available structure is likely already whitened by per-group
  Hadamard.
- *Mechanism:* per block, argsort intermediate channels WITHIN each 128-block by L2 energy, co-permute
  gate_proj rows + up_proj rows + down_proj columns (exact SwiGLU coupling); independently sort q-heads
  and co-permute o_proj. Optional exact up/down sign flips. Run champion ECVQ + A1 amax coder.
- *What is evolved/built:* `weights/sym_permute.py`; evolved knobs = sort key, scope, sign flips.
- *Verifier:* ppl-invariance <0.003; honest bits include the permutation side stream (FREE if the key
  is reconstructible from the shipped amax plane). Un-permute, assert byte-identical to champion.
- *Runtime on 1060:* ~7-10 min.
- *Expected:* 0.01-0.04 b/w at iso-ppl IF the key is shipped-derivable.
- *Kill:* 10-min oracle (H(idx | sorted order) vs H0 on 3 largest tensors): gain <0.01 b/w AND amax
  cmcore ratio <0.005 b/w improvement → per-group Hadamard already whitened channel structure; kill
  in 10 min.

**R9. A5-X2 — Amortized micro entropy model over indices (train-layers/code-layers split)** [Lens I]
- *Why:* the natural follow-on to R1 IF R1 finds the exploitable conditioning is cross-stream
  (amax-bin × position × depth) rather than neighbor-index. A single <=50k-param shared model
  amortizes over 357M weights at ~0.001 b/w overhead. Gated entirely by R1; do not run standalone.
- *Mechanism:* a fixed micro-coder (logistic mixture / 2-layer MLP, int8, counted) consumes
  decoder-reconstructable causal features → probability over the 64 levels for arithmetic coding.
  Feature extractor LLM-evolved (FunSearch-style), seeded by R1's winning contexts. Train on a
  hash-fixed HALF of layers, CODE the other half.
- *What is evolved/built:* `weights/ctx_entropy_zoo.py`; model weights int8 and counted.
- *Verifier:* real arithmetic encode + byte-exact decode over the full index stream, model bits
  INCLUDED; report bank ONLY on coded layers. Finalists require the real round trip, not teacher-
  forced cross-entropy.
- *Runtime on 1060:* GPU-assisted training seconds-to-minutes/candidate; feature extraction ~2 min
  CPU; round-trip ~2 min; ~5-10 min/candidate; 2 rounds.
- *Expected:* if R1 found D_idx ~0.05-0.10, recover ~0.04-0.08 b/w on coded layers → champion
  ~2.92-2.96 b/w. A null upgrades the iid claim from order-0 to mixer-bound.
- *Kill:* if R1's oracle gate <0.03 b/w AND the model <0.04 b/w after 2 rounds → ship only the static
  recode. Also kill if coded-half bank is positive but train-half exceeds it by >2x (overfit).

### TIER 3 — NEW ASSET CLASSES (diversity / new tracks; orthogonal to the weight frontier)

**R10. KVQ-Zoo — evolved KV-cache codec, scored by long-context next-token agreement** [Lens IV]
- *Why:* a VIRGIN asset class (no prior round touched runtime activations) with a different paper
  (longer context in fixed VRAM) and a new p95-KL attention-fidelity verifier. Cheap (~2-4
  min/candidate). Ranked here because it does not move the weight north-star, but it is the highest-
  value diversity bet.
- *Mechanism:* capture a real KV-cache over a 4k-token prose slice; rotate each head_dim vector with
  the signed Hadamard, scale by per-head/per-group amax, scalar ECVQ; ASYMMETRIC key-vs-value bit
  split (keys feed softmax non-linearly → sensitive; values averaged → tolerant); RoPE-aware key
  rotation in the pre-RoPE frame.
- *What is evolved/built:* codec functions over [layers, heads, seq, head_dim]; evolved levers as
  above.
- *Verifier:* decodable container (byte-split amax + cmcore best-of) round-trips K,V byte-identical;
  behavioral gate = mean + p95 KL(fp16-cache-logits || quant-cache-logits) over the final 512 query
  positions (teacher-forced, deterministic, paired); admissible only if p95 KL < tau.
- *Runtime on 1060:* ~2-4 min/candidate (one 4k forward pass + container + re-decode + last-layer
  logits).
- *Expected:* rotated ECVQ ~3-4 b/elem at p95 KL < 0.05 vs naive int8's 8 b/elem (~2x), enabling ~2x
  context in 6GB.
- *Kill:* rotated ECVQ fails to beat per-token int8 by >1.5x at p95 KL < 0.05, OR capture/replay
  cannot be made deterministic+paired (A2 lesson).

**R11. OptiCodec — lossy Adam optimizer-state codec, gated by bounded-trajectory-divergence** [Lens IV]
- *Why:* new track (lossy optimizer-state compression for checkpoint/resume) with a NEW un-gameable
  behavioral verifier class (bounded-trajectory-divergence). Cheap (~3-5 min/candidate), pure numpy
  dynamics. Ranked below KVQ because the deployment story (checkpoint storage) is narrower than the
  context-length win.
- *Mechanism:* v is strictly-positive ~log-normal → code log2(v) with per-group amax + A1 byte-split
  container; m is signed near-zero → champion Hadamard+ECVQ. Resume K=20 steps under a fixed-seed
  synthetic-gradient stream from fp32 vs decoded state; 1/sqrt(v) is provably insensitive to v's low
  bits.
- *What is evolved/built:* codec functions for m/v planes + an evolvable m-vs-v bit-budget split.
- *Verifier:* container round-trips m,v byte-identical (SHA-256 on QUANTIZED tensors); divergence gate
  = max-over-steps relative ||W_quant_resume - W_fp32_resume||_F < tau (1e-3) AND final-resumed-weights
  enwik8 ppl within +0.01 of fp32-resumed.
- *Runtime on 1060:* ~3-5 min/candidate.
- *Expected:* optimizer (m+v) ~4-6 b/w honest vs ~8-12 lossless, ~2x smaller checkpoints, divergence
  well under tau.
- *Kill:* any budget beating wcodec by >1.3x forces divergence >= tau or ppl drift > 0.01, OR wcodec
  lossless m+v already < 5 b/w on a REAL checkpoint (no headroom).

### TIER 4 — LOWER-PRIORITY / CONDITIONAL (de-duped, kept for completeness)

**R12. EMBED-COND — embedded ECVQ with per-cell conditional refinement vs nested-int** [Lens II].
Capability/comparative claim (one artifact, three rates) rather than a frontier move; ~1-2 GPU-hours.
Run only after R4/R5 settle the headline. *Kill:* penalty > 0.1 b/w at any truncation, or nested-int
pays < 0.3 b/w (claim collapses).

**R13. E3 — Cross-tensor shared ECVQ codebook via sign/scale canonicalization** [Lens III].
Direct codebook saving is tiny (audit-confirmed 0.0005 b/w); the only real upside is JOINT cmcore on
the concatenated index stream — which R1 (and R9) measure more directly per-tensor. **De-dupe note:
the joint-context question overlaps R1/R9; run E3 only if R1 shows meaningful cross-stream structure
AND the shared-codebook ppl gate (<0.03 regression) holds.** *Kill:* shared-codebook ppl regresses
>0.03, or joint-cmcore banks <0.015 over per-tensor best.

**R14. E4 — Fitted pseudo-init residual (cross-layer shared base in a canonicalized gauge)** [Lens III].
Closed-form (no training, unlike B3). Modest, possibly null if layers are near-independent post-gauge.
*Kill (fast):* per-role shared base captures <2% of same-role variance (R^2 < 0.02, measured in
seconds) → layers independent given the gauge; kill before quantizing.

**R15. B7-X3 — Encoder-only permutation gauge that CREATES cross-row index correlation** [Lens III].
The only experiment that CREATES rather than measures index correlation, but the long cmcore
head-convergence pole (~30 min/pass) and the likelihood that shipped-amax conditioning already
dominates make it a lower per-GPU-hour bet than R1/E1/E2. *Kill:* 10-min oracle <0.02 b/w on the 3
largest tensors → shipped-scale conditioning dominates.

**R16. FamilyJoint / PatchQuant — joint family coding / quantized-model patches** [Lens IV].
New systems/storage track (quantized base + ppl-safe lossless patches; OTA quantized-model updates).
Orthogonal to the single-model frontier and the abliteration assets are on disk; valuable but
lowest-urgency. *De-dupe note:* PatchQuant's index-delta also probes D_idx symmetry on a SECOND model
— fold its finding into the R1/R6 symmetry verdict rather than running it as an independent decider.
*Kill:* quantizing the base inflates derivative deltas more than it shrinks the base, or the
quantized→quantized index delta is not smaller than shipping the derivative's quantized indices
standalone.

---

## 3. RUN-NOW SEQUENCE (the 3-5 to execute immediately) + RATIONALE

These build ONLY on validated state, are CPU-mostly (high info-per-GPU-hour), and each is decisive
with a pre-registered kill line. They also span all four lenses on day one.

1. **R1 — A5-X1 index-stream conditional-entropy oracle.** *Run first.* It is THE immediate decider:
   it produces D_idx without cmcore's 24MB-head extrapolation, which simultaneously (a) tests the A5
   free-win, (b) sets the lattice rule Case A/B/C, and (c) supplies the symmetric baseline both gate
   scripts currently lack (`ecvq_cmcore.py` extrapolates; `lattice_rescore.py:39` hardcodes
   `CHAMP_NOTE`). ~10 min, ~0 GPU-hours. Its kill line cleanly terminates the entire index-coding
   sub-tree if indices are iid-after-rotation.

2. **R3 — E1 RMSNorm-gain absorption.** Run in parallel with R1 (independent, both CPU-mostly). An
   EXACT symmetry never applied before; the ppl pass is only a function-preservation check, so the
   ~0.03-0.06 b/w upside is essentially ppl-invariant like A1. If E1 banks, every downstream codec
   inherits the smoother gauge.

3. **R2 — C1-PC positive control + 24-champion retro-audit.** Run on the GPU while R1/R3 occupy CPU.
   This closes the project's one undefended claim ("verifier not overfit") with a decisive, cheap
   test, and the retro-audit (~1 GPU-hour) yields the selection-overfitting curve the original C1
   skipped. Independent of R1/R3.

4. **R6 — E8-SYM**, but ONLY as the immediate follow-up GATED on R1's D_idx. If R1 fires CASE B
   (D_idx >= 0.05), E8-SYM collapses to a one-pass confirmation that the lattice edge inverts and
   lattices stay closed (near-zero additional GPU). If R1 is CASE A/C, run the full co-located,
   lever-matched, convergence-validated comparison. Either way it permanently settles the lattice
   branch that has been "undecided" since A3.

5. **R4 — ECTCQ-RM**, kicked off as the first GPU-heavy build once R1-R3 report (its A4 pre-flight
   already PASSED). It is the only run-now-tier experiment that can move the HEADLINE ~3 b/w operating
   point (+0.10-0.15 b/w at iso-ppl), so it should start its ~1-GPU-day encode in the background
   immediately after the cheap deciders, rather than waiting.

**Why these five and not others:** R5 (PPL-ALLOC) and R10/R11 (new assets) are high-value but carry a
larger upfront GPU cost (sensitivity table) or are orthogonal to the weight frontier, so they queue
behind the deciders. R9/E3 are explicitly GATED on R1's outcome (no point running an index entropy
model before the oracle says the structure exists). R15/B7-X3 has the worst per-GPU-hour profile of
the gauge experiments. The run-now five are the minimal set that (i) settles the symmetry question
that blocks the lattice verdict and re-prices the champion, (ii) banks two exact-symmetry / lossless
wins (E1, plus whatever A5 confirms), (iii) closes the verifier-overfitting claim, and (iv) launches
the single experiment most likely to move the north-star metric — covering Lenses I, II, III, and IV
in the first batch.

---

## 4. DEDUPE LEDGER (collapsed overlaps)

- **D_idx is computed ONCE** (R1), then consumed by R6 (E8-SYM), R9 (A5-X2), and R16 (PatchQuant's
  second-model probe). `ecvq_cmcore.py`'s 24MB-head extrapolation is RETIRED in favor of R1's
  un-extrapolated conditional-entropy lower bound + a full-stream realized container.
- **Joint cross-tensor index context** appears in R1, R9, and E3 (R13); R1/R9 measure it directly and
  per-tensor, so E3 runs only if R1 shows cross-stream structure AND its shared-codebook ppl gate
  holds — it is not an independent decider.
- **The gauge family** (E1/E2/E3/E4, R15) is ordered by exactness-and-cheapness: E1 (exact RMSNorm,
  ppl-invariant) > E2 (exact within-group permutation, oracle-gated) > E4 (closed-form residual,
  fast R^2 pre-gate) > E3 (lossy shared codebook, ppl-gated) > B7-X3 (long cmcore pole). E1 runs now;
  the rest are gated by their oracles/pre-gates.
- **Lattice settlement** is a single thread: R1 → R6, with PatchQuant's index-delta finding folded in
  as corroboration, not a parallel decider.
