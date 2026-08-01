# Pre-registration #90: cache-aware expert dropping on the GPU/CPU split — does the quality cost fit inside the speed win?

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE any perplexity or KL run.
**STAKED.** · **Task:** #55 · **Gates:** U-34 (BigMoeOnEdge's two paying levers), E-10 (our own
KV-quant advice was half-evidenced — speed measured, quality not) · **Sibling:** #87 / task #52.

BigMoeOnEdge's `--drop-cold-experts` skips a routed expert when **two** predicates hold at once:
fetching it would cost an I/O read, **and** the router barely weighted it. They report **+55% at
F=0.75** and **+84% at F=1.0** on their phone, bootstrap intervals disjoint. Our GPU/CPU split has
the same shape — CPU-side experts *are* our expensive fetch — so the lever is worth porting.

This document decides whether a single line of runtime code should ever be written for it.

**It is staked with a bias against my own hypothesis.** §3 derives a bounding lemma whose
consequence, *if the quality cost turns out affordable*, is that **the patch is unnecessary**, and
KR-3 stakes that refutation in advance. I would rather publish "the feature we wanted to build is
redundant" than build it. On my own point predictions (§7, P-1) the likelier outcome is harsher
still: the lever refuted outright.

**This document was adversarially reviewed on 2026-07-30, before any Arm A number existed.** Seven
ways it could not fail were found and fixed — one of them fatal, in that the script would have
refused on its first invocation and no kill rule could ever have fired. Everything the review
changed, and everything it deliberately did not change, is in **§0.5**.

---

## 0. DISCLOSURE — what the author had already seen when staking this

Protocol requires predictions before measurement. Three things were established while building
this document, and staking without saying so would make the prereg theatre.

**(1) Every number in Arm B (§5) was computed before this document was written.** Arm B is
*arithmetic over GGUF headers*, not a measurement, and pretending it was blind would be the exact
dishonesty this project exists to avoid. It is labelled a **prediction, never a measurement**, and
it is not the headline.

**(2) Model metadata and tensor layout were read from the GGUF headers.** Architecture, expert
counts, routing width, block counts and per-tensor byte sizes — all in §2. This is what makes the
structural argument in §3 checkable rather than asserted.

**(3) The instrument's output format was read out of the shipped binary's string table**
(`llama-perplexity-impl.dll`, `llama.dll`) so the parser targets real strings. No model was
loaded and no forward pass was run to do this.

**(4) The instrument was executed against a 0.6B model during adversarial review** (2026-07-30,
after staking, before any Arm A run), to check that what §0.3 read out of the string table is
what the tool actually prints. It is not, in one decisive respect, and the correction is
recorded in §0.5. The probe used `Qwen3-0.6B-Q8_0` — a **dense** model, not an arm of this
experiment — plus one 1-chunk load of the primary model to confirm that `--override-kv` takes
effect. No perplexity, KL or top-token number was produced for either staked arm at any width.

**Arm A — the headline, and the only thing with kill power over quality — is fully blind.** Not
one perplexity, KL divergence or top-token number has been computed at any routing width, on
either model, in either domain. Zero.

---

## 0.5 AMENDMENT, 2026-07-30 — adversarial review, BEFORE any Arm A run

This document was reviewed against one question: *in what ways can this experiment not fail?*
Seven defects were found and fixed. **No threshold, no k grid, no context length, no chunk count
and no corpus hash was touched** — 0.99, 0.05, 1.20x, `{7,6,5,4}`, `-c 512`, `--chunks 12` and
both SHA-256 pins stand exactly as staked. What changed is the machinery that decides whether
those numbers can ever be reached, and the honesty of how the outcome is labelled.

**(A) The experiment could not run at all — every kill rule was vacuous.** KR-5 demands the
loader print `n_expert_used` as proof the override took effect. At this build's **default
verbosity that line is never emitted**: a stock base run prints twelve lines and the model-info
banner is not among them. The script would have refused on its first invocation and produced
`REFUSED`, forever. Verified by running it; verified fixed by running it again with `-v`, which
prints `print_info: n_expert_used = 6` under `--override-kv qwen3moe.expert_used_count=int:6` on
the primary model. **The regex is now anchored on `print_info:`** — the loader also dumps the raw
metadata table, which prints the file's *original* count under the banner "KV overrides do not
apply in this output", and matching that line would have turned KR-5 into a check that proves
nothing. `--no-log-prefix --no-log-timestamps` was added after observing the logger inject a
timestamp into the middle of the `validate_override:` line.

**(B) KR-1 had a hole in the direction it exists to catch.** It was implemented as: pick the cell
with the lowest top-1 agreement, then apply *both* gates to that one cell. A cell that passed
top-1 comfortably but blew the KLD gate escaped KR-1 entirely — and the KLD gate's whole stated
purpose is "catching distribution shift that survives an unchanged argmax". **Each gate is now
applied to its own worst cell.** A regression test constructs exactly that outcome and confirms
it now fires.

**(C) KR-3's verdict word was `PASS`.** KR-3 is the branch where every gate clears, and what it
concludes is *the runtime patch is refuted as unnecessary*. Headlining that with the word for
success is precisely the reporting asymmetry this project exists to avoid. **The status
vocabulary no longer contains `PASS`**: the outcomes are `REFUTED-AS-UNNECESSARY`,
`REFUTED-LEVER`, `MARGINAL`, `MARGINAL-UNSOUND`, `UNSOUND`, `VOID`, and each carries a plain
sentence in the JSON and the log.

**(D) KR-2's speed half has no kill power, and §7 implied it did.** Arm B is deterministic
arithmetic over GGUF headers that was computed *before this document was written* (§0.1), so
every number the 1.20x bar is compared against was already known at staking time, and k'=6/5/4
clear it by construction. Under KR-4 monotonicity **KR-2 reduces to KR-1**: only a quality
failure can make it fire. The claim in §7 that "a bar that only the intended answer clears is not
a bar" is still true of the *k'=7 rejection*, but it does not make KR-2 a second independent
hurdle. Recorded in the JSON as `kr2_speed_gate_kill_power: false`.

**(E) KR-4 could pass vacuously, and could be swallowed.** With fewer than three measured widths
in a cell the monotonicity loop makes at most one comparison — an unfireable rule dressed as a
check; that now VOIDs. And a `MARGINAL` result returned early, discarding a KR-4 failure that
§7 promises is "reported even when KR-1 and KR-2 pass"; that outcome is now `MARGINAL-UNSOUND`.

**(F) P-3 and P-4 were staked but never scored.** A directional prediction that no code evaluates
is a prediction that can be quietly forgotten when it misses. Both are now computed at k'=6 and
written to the JSON whatever the headline verdict is. **P-4 carries a caveat with its number**:
it compares mean KLD *across two models* with different vocabularies, quantisations and base
distributions. That is directional evidence, not a matched test, and it was staked without
saying so.

**(G) Four silent-substitution paths, all closed.** The script fell back to any
`llama-perplexity` on `PATH` when the staked binary was missing — swapping the instrument
mid-experiment while still stamping the run `headline: true`. The base-logits filename keyed on
`ngl`/`ctx`/`chunks` but not on **thread count or build**, so a base could be KL-compared against
a test from a different CPU reduction order or a different binary — the exact hazard §6's
"placement hygiene" paragraph exists to prevent. `--ngl`, `-t` and the binary now also decide
`headline`. A `--dry-run` overwrote the scored result JSON with `DRY-RUN`; dry runs now write to
`*.dryrun.json`. And the logits cleanup swept the whole directory by glob, so a run restricted
with `--model` deleted the *other* arm's base logits.

**Cost, which this document never stated.** One 1-chunk pass on the primary model took ~34 s of
CPU compute after an ~11 s load. At `--chunks 12`, 4 base runs + 16 KL runs is on the order of
**3–5 hours**. Base logits are ~1 byte per vocab entry per token, not the 4 the script reserves —
~2.5 GB peak, not 19.7 GB.

---

## 1. Why this is not simply "run it and see"

Two independent obstacles, both real, both established before staking.

**Obstacle 1 — the faithful port needs an instrument we do not have.** Their second predicate is a
*value* threshold on the router weight. Per-token router weights are not obtainable on this box
today:

| route | status |
|---|---|
| `llama-imatrix` (the #87 instrument) | emits per-expert **counts**, never weights |
| llama.cpp generally | exposes no router weight through any flag or output |
| PyTorch (`weights/router_confidence.py`, the ancestor of this idea) | **dead**: `torch`, `transformers` and `safetensors` are all uninstalled, and the DeepSeek-V2-Lite safetensors under `D:\evo-compress-data\` have been deleted — only an empty HF-hub `refs` stub survives |

So the value-threshold variant is blocked on an instrument that must be rebuilt or repatched first.
That is a Stage 2 precondition, recorded in §8, not something this experiment can wave away.

**Obstacle 2 — #52 does not unblock it either.** The task queue lists #55 as blocked on #52. That
dependency is real but it is **weaker and differently-shaped than it looks**, and the honest thing
is to say so before #52 is scored rather than after:

> **#52 measures routing *counts*. The drop rule needs routing *weights*. #52 cannot supply the
> missing predicate no matter which way it lands.**

What #52 *does* decide is whether a **frequency-ranked hot-expert cache** transfers to our rows.
§4 gives the full branch table. The short version, staked now so it cannot be read as a post-hoc
rescue: **a refutation in #52 does not kill #55**, because our expensive-fetch predicate is static
*placement*, which uses no frequency information at all.

This experiment is therefore deliberately designed to be **independent of #52**. That is not
scope-dodging, and §3 is the reason it is legitimate: the quality gate is a **necessary condition
for every variant of the lever**, so measuring it first can kill all of them at once, cheaply, with
a stock binary — before anyone waits on an instrument that does not exist.

---

## 2. The facts this rests on, read from the files

Read from the GGUF headers on 2026-07-30. The script re-reads and re-asserts every one of them at
runtime and **aborts** rather than proceed on a mismatch.

| arm | file | arch | layers | experts | top-k | vocab |
|---|---|---|---:|---:|---:|---:|
| `qwen3-30b-a3b-q2k` (**primary**) | `Qwen3-30B-A3B-Q2_K.gguf` | `qwen3moe` | 48 | 128 | 8 | 151,936 |
| `q35-A-shexp` (**replication**) | `q35-A-shexp.gguf` | `qwen35moe` | 40 | 256 | 8 (+shared) | 248,320 |

**Experts are fused into one tensor per layer per projection.** Verified:

```
blk.0.ffn_gate_exps.weight   [2048, 768, 128]   Q2_K     <- all 128 experts, one tensor
blk.0.ffn_up_exps.weight     [2048, 768, 128]   Q2_K
blk.0.ffn_down_exps.weight   [ 768, 2048, 128]  Q3_K
```

144 `_exps` tensors on the 48-layer model, 120 on the 40-layer model — exactly `3 x n_layer`. The
last dimension *is* the expert index. `q35-A-shexp` additionally carries always-on
`ffn_{gate,up,down}_shexp` tensors, which no drop rule may ever touch.

The second arm is included **because its shared experts are a plausible cushion**: a model that
always runs a shared FFN may tolerate routed-expert dropping better than one that does not. That is
a directional stake (P-4), not decoration.

---

## 3. The structural port — and the bounding lemma that makes Stage 1 possible

This section contains the whole idea. It is derivation, not measurement, and it is checkable.

### 3.1 Our miss predicate is static and layer-granular

`-ot` / `--override-tensor` matches **tensor names**. Because a layer's experts are one fused
tensor (§2), `-ot` can move a whole layer's experts and nothing finer. Therefore, on our split:

> **"fetching this expert costs an I/O read" ≡ "this layer's fused expert tensor is CPU-resident".**

That predicate is fixed at load time. It is not a cache and it has no state.

**Consequence A — their reproducibility caveat does not transfer, and must be corrected rather
than parroted.** They warn that output is not reproducible because what gets skipped depends on
cache state. Our residency is a *load-time placement*, so **our port is fully reproducible**: the
same prompt yields the same tokens every run. This is a genuine improvement of the ported design,
and it is the one caveat of theirs I am explicitly **not** carrying forward. It would return
immediately if a *dynamic* hot cache were ever substituted for static placement — noted so the
exemption cannot be silently inherited by a later variant.

**Consequence B — cache-aware dropping on our split *is* layer-selective top-k reduction.** Drop on
CPU-resident MoE layers; leave GPU-resident layers at full width.

### 3.2 The zero-code member of the same rule family

`--override-kv <arch>.expert_used_count=int:k'` reduces the routing width globally. That is not a
crude approximation of the drop rule — it is a **genuine member of it**:

> Reducing k from 8 to k' drops exactly the experts ranked k'+1…8 **by router weight**, per token.
> It is the *rank*-threshold parameterization; their `F` is the *value*-threshold parameterization.

The only thing the value threshold buys over the rank threshold is sensitivity to per-token router
confidence — it drops more where the router is decisive, fewer where it is torn. That extra is
exactly what needs the missing weight instrument (§1) and a patch.

**Disclosed semantic difference.** llama.cpp **renormalizes** the surviving weights over the
retained top-k'. Their implementation may simply skip the contribution, which shrinks the layer
output. These are different rules and ours is likely the better-behaved one. Stage 2 must match
whichever is actually being ported; not checked here.

### 3.3 THE BOUNDING LEMMA

Dropping an expert on a **GPU-resident** layer costs quality and buys **nothing** — that expert is
already in VRAM and was never an I/O read. So, at any fixed k':

> **Placement-aware dropping and uniform dropping deliver the *same speed*, but placement-aware
> perturbs only the CPU-resident layers while uniform perturbs all of them.**
>
> ⇒ **Uniform dropping's quality cost is an UPPER BOUND on the placement-aware port's quality cost
> at matched speed.**

This is what lets Stage 1 use a stock binary to bound a feature nobody has written. It is also the
step most likely to be wrong, so it is not assumed — **KR-4 tests it** (§7). The lemma relies on
"perturbing more layers does not damage less", which is an empirical monotonicity claim, not a
theorem, and our own data must support it or the bound is void.

---

## 4. The #52 branch table, staked before #52 is scored

| #52 outcome | frequency-ranked hot-cache variant | **placement-aware variant (what #55 actually ports)** |
|---|---|---|
| **PASS** (skew ≥ 0.55, retention ≥ 0.90) | becomes available — but still needs the per-expert residency runtime change #87 priced | **unaffected** |
| **KR-1 fires** (skew < 0.55) | **dead** for our rows | **unaffected** — uses no frequency information |
| **KR-2 fires** (retention < 0.90) | **dead** as a *static* pin | **unaffected** |
| **VOID** | no information | **unaffected** |

In every branch, Arm A of this experiment is untouched, because quality cost is a function of
*which routed experts are dropped*, not of *why they were expensive*. The register wording that
#55 "loses the gate it was waiting on" if #52 refutes skew is, on this analysis, **too strong**:
what dies is the faithful hot-cache port, not the lever. Scoring may propose that correction; this
document does not edit the register.

---

## 5. Arm B — the speed envelope (COMPUTED, NOT MEASURED)

Two-tier byte model, all-experts bytes from the GGUF tensor table, routed bytes per token
`= expert_bytes x k / n_expert`, non-expert tensors on GPU, expert layers split with GPU share `g`:

```
t_token = (routed_bytes(k') * g + other_bytes) / BW_gpu_eff  +  routed_bytes(k') * (1-g) / BW_cpu_eff
```

At `g = 0.32` (the share the shipped `-ot` regex keeps resident), `BW_gpu_eff = 130 GB/s`,
`BW_cpu_eff = 15.3 GB/s`:

| k' | q3-30b speedup | q35-A speedup |
|---:|---:|---:|
| 8 | 1.000x | 1.000x |
| 7 | 1.128x | 1.096x |
| 6 | **1.294x** | **1.213x** |
| 5 | 1.517x | 1.358x |
| 4 | 1.833x | 1.541x |

Reproduced by `python weights\exp55_cache_aware_dropping.py --dry-run`, which recomputes this
table from the headers and executes nothing.

**Sensitivity, disclosed before the bar is set.** Sweeping `BW_gpu_eff` 100–160 GB/s,
`BW_cpu_eff` 10–25 GB/s and `g` 0.20–0.45, the k'=6 speedup stays in **1.255–1.314x** for q3-30b
and **1.145–1.264x** for q35-A; k'=7 stays in **1.113–1.136x** and **1.068–1.116x**. The ratio is
robust because dropping scales the dominant term
proportionally and the fitted constants largely cancel — this is a *ratio* claim, and **the
absolute tok/s levels above are not claimed at all** (they are optimistic against our own measured
rows, which is exactly why only the ratio is used).

**q35-A straddles the KR-2 bar (1.145–1.264 across the sweep).** Stated now: **KR-2 is evaluated on
the primary model only**, and q35-A's envelope is reported without kill power.

**Deflation of the imported number, stated before measuring.** Their +55% came from a tier reading
at roughly 1.9 GB/s (UFS flash). Ours is DRAM at ~15 GB/s effective — about 8x faster relative to
compute. **The same drop rate must therefore buy substantially less on our split**, and ~1.29x is
the honest expectation, not +55%. Any later result near +55% should be treated as suspicious rather
than as vindication.

---

## 6. Arm A — the measurement (the headline)

**Instrument.** Stock `tools/llamacpp-b10098/llama-perplexity.exe`. No patch, no custom build.
That exact path is required: there is no fall back to a `llama-perplexity` found on `PATH`, and
passing `--ppl-bin` explicitly forfeits headline status (KR-6). A different build is a different
instrument, and base logits written by one build must never be KL-compared against another.

Every invocation carries `-v --no-log-prefix --no-log-timestamps`. This is not cosmetic — see
§0.4.

- base: `-m M -f SHARD -c 512 --chunks 12 -ngl 0 --seed 1 -v --no-log-prefix --no-log-timestamps
  --save-all-logits BASE.bin`
- test: same, plus `--override-kv <arch>.expert_used_count=int:k'` and
  `--kl-divergence --kl-divergence-base BASE.bin`

**Metrics** (parsed from format strings read out of the binary, §0.3): `Mean KLD`, `Median KLD`,
`99.0% KLD`, `Maximum KLD`, `Same top p`, `Mean PPL(Q)`, `Mean PPL(base)`, `Final estimate: PPL`.

**Why KL and top-token agreement, and not perplexity.** BigMoeOnEdge's quality evidence was **15
GSM8K questions** — enough to rule out a collapse, nowhere near enough to see a subtle cost, and
they say so. E-10 caught this project shipping KV-quant advice with speed numbers and no quality
numbers. Aggregate perplexity is the weakest of the three metrics available here: it can sit flat
while the output distribution moves substantially. **Mean KLD** sees the distribution shift
directly, and **top-1 agreement** is the number a coding user actually feels — one wrong identifier
breaks a compile. Perplexity is recorded and reported, but it is not a gate.

**Corpus.** The four SHA-256-pinned shards from #87, re-hashed at runtime; the script refuses on
mismatch. Scoring uses the **B** shards, so #52's ranking surface (A) and this experiment's
scoring surface stay disjoint:

| shard | bytes | SHA-256 |
|---|---:|---|
| `code_B` | 25,550 | `dccbb1bd4af5b47b363d6cce7090bbebd73c9cf112249583f92a98618f5a1593` |
| `prose_B` | 119,800 | `5fb4087ab6b5c048366614e0525127f3adb70055eb1efeead2e9eb9ac785d304` |

Two domains, because E-10's whole point was that wikitext does not see Elixir. C# is not Elixir, but
it is not prose either, and a code/prose split is the sharpest domain contrast available on this box
without new data.

**Budget.** `-c 512`, `--chunks 12` = **6144 tokens per cell**, equal across domains so the
comparison is not confounded by unequal token counts. 2 models x 2 domains x {8,7,6,5,4} = 4 base
runs + 16 KL runs.

**Placement hygiene — the C-14 analogue for a non-timing measurement.** Counts and log-probs are not
timings, so C-14 does not bind. But base and test logits **must** come from the same backend, or the
KL picks up reduction-order noise instead of the drop. `-ngl 0` (pure CPU) is the default, matching
#87; the value is stamped into the base-logits filename and **a mismatch refuses to run**.

---

## 7. Predictions and KILL RULES

Point predictions with bands. Arm A is fully blind (§0).

- **P-1 (primary model, k'=6, blind).** Top-1 agreement on the worst domain: **0.975**, band
  0.95–0.99. Direction genuinely uncertain — the 7th and 8th of 8 experts carry little
  renormalized weight, but these are already Q2_K models with little quality headroom left.

  > **Read P-1 against the bar before reading §8.** The point prediction 0.975 is **below** the
  > 0.99 gate, and the whole band 0.95–0.99 lies at or under it. By my own staked numbers the
  > single most likely outcome of this experiment is that **KR-1 fires and KR-2 with it — the
  > lever refuted**, not KR-3. §8 originally called KR-3 "the most likely outcome"; that was
  > narrative, and it contradicted this line. Corrected 2026-07-30 (§0.5), before any run. The
  > predictions are left exactly as staked; only the prose that misdescribed them is fixed.
- **P-2 (primary model, k'=6, blind).** Mean KLD on the worst domain: **0.045 nats**, band
  0.02–0.12.
- **P-3 (domain, blind).** Code degrades **more** than prose at equal k', because code is the
  lower-entropy domain and a single wrong token is unrecoverable. Directional; no band.
- **P-4 (shared-expert cushion, blind).** `q35-A-shexp` degrades **less** than `q3-30b` at equal
  k', because its always-on shared FFN carries part of the layer output that no drop can remove.
  Directional; no band. A miss here is interesting on its own.
- **P-5 (monotonicity, near-certain).** Mean KLD rises monotonically as k' falls 8→7→6→5→4 in
  every cell. Its job is to fail loudly if the bounding lemma's premise does not hold.

**The bar, and where it comes from.** Set before any number was seen.

- **Top-1 agreement ≥ 0.99.** At 1% argmax disagreement a 500-token completion changes ~5 tokens.
  For code that is a broken build. Anything worse cannot ship as a **default**.
- **Mean KLD ≤ 0.05 nats.** The secondary gate, catching distribution shift that survives an
  unchanged argmax.
- **Speed ≥ 1.20x computed.** #87 required 1.5x on paper for a change of comparable cost, and this
  project's byte arithmetic has over-promised before (U-06: predicted 2.0 tok/s, measured 0.66).
  1.20x is lower than #87's bar because §5 shows this ratio is far more robust than a level
  prediction — but it is still discriminating: **k'=7 fails it on both models (1.128x, 1.096x) and
  k'=6 passes on the primary (1.294x)**. k'=7 fails across the *entire* sensitivity sweep
  (1.113–1.136x), so the rejection is not an artifact of the two bandwidth constants. A bar that
  only the intended answer clears is not a bar; this one rejects the adjacent option robustly.

  **But it carries no kill power of its own, and saying otherwise would be the defect this
  document is supposed to be immune to.** Arm B is deterministic arithmetic over headers that was
  computed before this document existed (§0.1). Every value the bar is compared against was
  already known when the bar was chosen, and k'=6/5/4 clear it by construction. It rejects k'=7;
  it can never *fail*. **Under KR-4 monotonicity, KR-2 therefore reduces to KR-1** — only a
  quality number can make it fire. The script records this as
  `kr2_speed_gate_kill_power: false` so KR-2 can never be quoted as a second hurdle the lever
  cleared. (Added §0.5, before any run.)

### KILL RULES

Mechanically checked by `weights/exp55_cache_aware_dropping.py`; the verdict is printed and written
to `weights/data/exp55_cache_aware_dropping.json` as `overall`.

- **KR-1 (quality).** At k'=6: top-1 agreement **< 0.99** on the cell with the **lowest top-1**,
  OR mean KLD **> 0.05** on the cell with the **highest mean KLD** — each gate against its own
  worst cell, which may be a different cell (corrected §0.5(B); the original wording let a
  KLD-only failure escape) ⇒ dropping at k'=6 is **refuted as a shippable default**. By the bounding
  lemma this does **not** license the conclusion that the placement-aware port fails — only that our
  *upper bound* is too loose to authorize a patch on this evidence. The documented remedy is the
  largest k' that passes, re-scored against KR-2.
- **KR-2 (worth-building).** If no k' both passes KR-1 and reaches **≥ 1.20x** computed on the
  **primary** model, **the entire lever is refuted for our split** — uniform and placement-aware
  alike — because the speed we need requires a drop depth whose quality cost we cannot bound below
  the bar. Published as a miss at equal prominence, per protocol. q35-A carries no kill power here
  (§5).
- **KR-3 (redundancy — a CONCLUSION, not a hurdle).** It fires exactly when everything else
  clears, so it cannot refute me and it is not evidence of rigour; what it *is* good for is
  binding my hands on the success branch. Its verdict word is **`REFUTED-AS-UNNECESSARY`**, never
  `PASS` (§0.5(C)). It was originally billed as "the rule most likely to fire"; P-1 says
  otherwise (see §7) and that billing is withdrawn. If some k' passes **both**
  KR-1 and KR-2, then **the runtime patch is REFUTED AS UNNECESSARY**: the win is already available
  through `--override-kv <arch>.expert_used_count=int:k'`, a flag that ships today and costs zero
  lines of code. What ships is **documentation of the flag with its measured quality cost attached**
  — never the flag alone, per E-10. **No cache-aware runtime change may be written on a KR-3 pass.**
- **KR-4 (bounding-lemma validity).** If mean KLD is **not** monotone non-decreasing as k' falls, in
  any cell, the monotonicity premise of §3.3 is unsupported by our own data. The bound is then
  declared **UNSOUND**, no result may be carried into Stage 2 without a direct placement-aware
  measurement, and this is reported even when KR-1 and KR-2 pass — including when the quality
  number is marginal, which is reported as `MARGINAL-UNSOUND` rather than swallowed (§0.5(E)).
  A cell carrying **fewer than three measured widths** makes at most one comparison and cannot
  test monotonicity at all; that is **VOID**, not a KR-4 pass.
  *Honesty note:* KLD rising as more experts are dropped is close to certain (P-5 says so), so
  this rule is unlikely to fire. It is a premise check, and it should not be counted as one of
  the rules that gives this design its kill power. The rules that can realistically fire are
  KR-1 (and KR-2 through it) and KR-5.
- **KR-5 (instrument VOID).** Any of — the loader's printed `n_expert_used` not equal to the
  requested k' (the override silently did not take); fewer chunks processed than staked; a corpus
  SHA-256 mismatch; a missing or truncated base-logits file; a required metric absent from the
  parsed output; base and test runs disagreeing on model, ngl, ctx or chunks — ⇒ **VOID**. No
  verdict, no register entry, fix the instrument and rerun. **A void is not a failure and must not
  be reported as one.**
- **KR-6 (anti-fudge).** Corpus shards are SHA-256-pinned above and re-checked at runtime. The k
  grid, chunk count, context length and all three thresholds are fixed in this document. `--ctx`,
  `--chunks` and the thresholds may be moved only for **clearly-labelled sensitivity runs**; any run
  with non-default knobs is stamped `headline: false` in the JSON and may never be quoted as the
  headline. **`headline` also requires `-ngl 0`, the staked thread setting, and the staked
  binary** — each of those changes the backend the logits come from, and leaving them out of the
  test let a run with a swapped backend still be stamped `headline: true` (§0.5(G)).
- **KR-7 (marginality).** If top-1 agreement lands within **±0.002** of 0.99, or mean KLD within
  **±0.005** of 0.05, the script reports **MARGINAL**, not PASS/FAIL. The remedy is more chunks and
  a re-stake; a marginal number must not be rounded into a verdict in either direction.

---

## 8. What would REFUTE this — stated plainly

This project publishes misses at equal prominence, and the interesting outcomes here are negative.

**The lever dies entirely (KR-2).** If every k' that reaches 1.20x costs more than 1% of argmax
agreement, then expert dropping does not fit our split at all. BigMoeOnEdge's +55% would stand as a
property of a flash-streaming phone — a tier 8x slower relative to compute than ours (§5) — and the
correct engineering decision is **to build nothing and keep full routing width**. Their result would
not be wrong; it would be theirs.

**The patch dies as redundant (KR-3).** *(Originally "and this is the most likely outcome" — that
claim is withdrawn: P-1's own point prediction of 0.975 top-1 sits below the 0.99 gate, so by my
staked numbers the likeliest outcome is the paragraph above this one, not this one. §0.5.)* If the quality cost is
affordable, then the bounding lemma says the placement-aware version is *at most* that expensive —
but the zero-code flag already delivers the same speed. There is then **no evidence-based reason to
write the runtime change**, and writing it anyway would be building machinery to reach a number a
flag already reaches. Recognising this *before* the code is written is the entire value of staking.

**The bounding lemma dies (KR-4).** If damage is not monotone in drop depth, §3.3's dominance
argument collapses and every downstream inference with it, including a passing KR-1.

**The structural premise dies.** If any future model stores experts as separate per-expert tensors,
`-ot` could address individual experts, the miss predicate stops being layer-granular, and this
entire design is the wrong one for that model. The script asserts fusion per model and aborts
otherwise.

**Note the shape of the outcome space: there is no branch in which this experiment authorizes
building the feature.** Either the flag suffices (KR-3), or the lever is refuted (KR-2), or the
result is unsound (KR-4), or quality fails and only a *measurement* — never a ship — is authorized.
That asymmetry is deliberate.

### What a `REFUTED-AS-UNNECESSARY` (all gates cleared) does NOT prove

- **No tok/s was measured.** Arm B is arithmetic (§0.1, §5). Every speed figure is a prediction, and
  a KR-3 pass authorizes documenting a flag, not claiming a speedup. `llama-bench` does not accept
  `--override-kv` (checked), so measuring even the flag needs a different harness under one
  `cal_id` — a separate experiment.
- **Nothing about context depth.** All of this is `-c 512`. E-10's complaint was about 100k+
  context, where dropping may behave completely differently. Out of scope; #47 owns depth.
- **Nothing about the value-threshold variant.** Their `F` remains unmeasured, blocked on the
  instrument in §1. Its only claimed advantage over the rank threshold is exploiting per-token
  router confidence, and that advantage is untested here.
- **Two domains are not all domains.** C#/WikiText-2 is a sharper contrast than wikitext alone and a
  far weaker one than the real distribution of user tasks. KL over 6144 tokens is a distributional
  metric, not a task score — it is strictly better evidence than 15 GSM8K questions, and it is still
  not a benchmark suite.
- **Nothing about `q35-A-shexp`'s speed envelope**, which straddles the bar (§5).

---

**Wired into:** nothing, deliberately. This experiment exists to decide whether the lever earns any
code at all — and §8 explains why its most likely verdict is that it does not.
`findings/REGISTER.json` is updated only after this is scored, and this document edits nothing.

**Reproduce:**

```
cd <repo>
python weights\exp55_cache_aware_dropping.py
```

Idempotent (completed runs are reused; `--force` redoes them). A cached run is reused only if its
ctx, chunks, ngl, thread count **and build id** match, and a cached base whose logits file has
been deleted is re-run rather than trusted. Raw output under
`weights/data/exp55_cache_aware_dropping.{json,log}` and `weights/data/exp55_runs/`;
`--dry-run` writes to `*.dryrun.{json,log}` and never touches a scored result.

The verdict is the `overall` field. Its possible values are **`REFUTED-AS-UNNECESSARY`**,
**`REFUTED-LEVER`**, `MARGINAL`, `MARGINAL-UNSOUND`, `UNSOUND`, `VOID`, `REFUSED`, `DRY-RUN` —
there is deliberately no `PASS`, because the branch where every gate clears is the branch where
the thing we wanted to build is refuted. `verdict_sentence` spells that out in prose.
