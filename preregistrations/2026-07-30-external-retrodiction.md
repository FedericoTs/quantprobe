# Pre-registration #86: score the decode law on a machine we have never touched

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE the comparison cell was
evaluated. **STAKED.** · **Experiment #51** · **Register:** U-33 (open), C-06 (open)
· **Adversarially reviewed 2026-07-30** — the review found that the score run *could not fail*.
Read **§0-bis** before §2, and **§8** for everything the review changed. The staked block in §2
is unmodified.

BigMoeOnEdge (github.com/Helldez/BigMoeOnEdge, Apache-2.0) published a desktop campaign on a
Windows laptop nobody here has touched: 8 cores, 16 GB dual-channel DDR4, ~3 GB/s NVMe, running
`Qwen3.6-35B-A3B Q4_K_M` (22.3 GB, ~1.5x RAM) through their own flash-streaming engine. Their
finding is that streamed decode there is **DRAM-bandwidth-bound in compute**: the compute time
per token sits at ~0.11 s in *every* cell of three rounds, so **~9 tok/s is a ceiling even at
zero I/O**, and io lanes, compute threads and the NVMe are all dead levers.

That is a direct, independent statement of Law 4 (`tok/s = eta x bandwidth / active-bytes`) on
hardware, a runtime, a model and an OS we do not control. U-33 already noted the coincidence and
refused to score it, for two stated reasons: the active-byte count came from our own
`q35-A-shexp` sibling GGUF instead of **their** file, and the arithmetic was done by hand in a
session log rather than by shipped code. This experiment removes both objections.

---

## 0. What kind of claim this is, stated before anything else

**This is a RETRODICTION, not a prediction.** Their ~9 tok/s was published on 2026-07-24 and was
already quoted in `findings/REGISTER.json` (commit `93b53f0`) before this experiment existed. The
author of this document computed the prediction and the target before writing it. Staking
therefore buys **less** here than in a measurement pre-registration, and pretending otherwise
would be the exact dishonesty this project exists to avoid.

What staking *does* buy, and what a reader should hold us to:

1. **Every constant is a shipped value with a citation and a commit**, not a knob turned until it
   fit. `eta_r = 0.38` has been on `quantprobe/plan.py:674` since commit `d590749`
   (2026-07-21, "quantprobe v1.0") — nine days before U-33 was logged and before this task
   existed. `51 GB/s` is the DDR4-3200 preset already in `MACHINES`, and U-33 already named it.
   The `1.15` activation multiplier and the `max(bits, 4.5)` floor are `plan.py:640-668`.
2. **Every selection rule is fixed in this document** — which file, which cells, which statistic
   — so none of them can be chosen after seeing the answer.
3. **The script refuses to run rather than produce a number** when any input drifts, and in
   `--score` mode it re-reads the staked numbers out of *this file* and aborts if the fresh
   computation no longer reproduces them.
4. **Zero parameters are fitted to the target.** Not one. The whole chain is
   GGUF header -> active bytes -> divide.

What staking does **not** buy: blinding. Both sides of the comparison are public and
deterministic; anyone can do the division. Weigh the result accordingly — and see §7.

### 0-bis. The score run cannot fail. Said plainly, before the verdict.

*Added 2026-07-30 after an adversarial review of this document, before the score run was
committed and before any register text was written from it.*

The `--stake` run emitted the prediction **and** the target in the same output. Both operands
were therefore fixed before `--score` ever executed, and `--score` re-derives them from the same
frozen inputs. On top of that, the drift gate that guards the staked numbers used to fire at
0.5% while the kill rule fires at 15% — a gate 30x tighter than the threshold it sits in front
of. Any deviation large enough to fail P-1 (>12% move), P-2 (>14%) or P-3 (>18%) was caught by
the gate first and converted into `ABORT`, not `FAIL`. **The verdict of the score run was
decided the moment the stake block was pasted into §2. It could only ever print PASS or ABORT.**

That is the worst defect a pre-registration can have, so it is stated here rather than left for
a reader to derive. Three things changed in response, and none of them retroactively make this
run a test:

1. The drift gate now guards **only our side** (prediction, active bytes, kill threshold).
   Their published cells are re-extracted from the live document at score time under the rule in
   §1.6 and are **scored, never aborted on**. If they revise their findings, the target moves and
   the kill rule can fire. Previously a revised source aborted — closing the one door a
   falsification could have come through.
2. Every `--score` run prints a **KILL-RULE REACHABILITY** block that computes the interval of
   predictions in which all three predicates hold, and states in its own output whether any input
   that run could have produced would have flipped the verdict. On the run committed with this
   document it prints `KILL RULE UNREACHABLE THIS RUN`.
3. The one genuinely unknown input — their DRAM grade — is promoted from a disclosed sensitivity
   to a **staked falsifier with a pre-committed consequence** (§5-A). It is the only part of this
   experiment whose outcome is not yet determined.

**How to weigh a PASS here:** as a reproduction of a retrodiction, computed by shipped code from
their actual file rather than by hand from a sibling. That is strictly more than U-33 had, and
strictly less than a test.

---

## 1. Method (enough detail to run it without us)

    python weights/exp51_external_retrodiction.py --stake     # freeze the prediction
    python weights/exp51_external_retrodiction.py             # score it

1. **Identify their file.** Their README says "Qwen3.6-35B-A3B (Q4_K_M): 22.3 GB" and nothing
   more, so the file must be identified by size against a frozen candidate list of public repos.
   **Rule (fixed here):** keep every candidate whose published size rounds to **22.3 GB at 0.1 GB
   resolution**; exactly one must survive or the run aborts. Published sizes, read live:

   | repo / file | GB | rounds to |
   |---|---:|---:|
   | `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` / `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf` | 22.285 | **22.3 <- selected** |
   | `unsloth/Qwen3.6-35B-A3B-GGUF` / `...-UD-Q4_K_M.gguf` | 22.135 | 22.1 |
   | `unsloth/Qwen3.6-35B-A3B-GGUF` / `...-UD-Q4_K_XL.gguf` | 22.360 | 22.4 |
   | `unsloth/Qwen3.6-35B-A3B-GGUF` / `...-UD-Q4_K_S.gguf` | 20.893 | 20.9 |
   | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` / `...-UD-Q4_K_M.gguf` | 22.663 | 22.7 |
   | `ggml-org/Qwen3.6-35B-A3B-GGUF` / `...-Q4_K_M.gguf` | 20.420 | 20.4 |
   | `lmstudio-community/Qwen3.6-35B-A3B-GGUF` / `...-Q4_K_M.gguf` | 21.167 | 21.2 |

   Every candidate must be sized successfully. A candidate that cannot be reached **aborts the
   run**: "exactly one of seven matches" is an identification only if all seven were measured,
   and an unmeasured candidate cannot be excluded. (The script previously logged `UNAVAILABLE`
   and carried on with a partial list — a silent-substitution channel, fixed 2026-07-30.)

2. **Fetch only the header.** An HTTP Range request pulls the first 4 MiB and doubles until the
   tensor table is complete (16 MiB for this file; the header is 10,990,692 B), and a
   self-contained parser reads the 753-tensor table (`gguf.GGUFReader` cannot: it maps the
   whole file). The parse is proved correct by an accounting identity that must close exactly:
   `header bytes + alignment pad + sum(per-tensor bytes) == remote Content-Length`.
3. **Check it is the model we staked on:** `general.architecture == qwen35moe`,
   `expert_count == 256`, `expert_used_count == 8`, `block_count == 41`. Otherwise abort.
4. **Recompute active bytes with `quantprobe.spec` arithmetic**, including finding **#76**'s
   embedding-gather correction (`token_embd` leaves the always-active set iff a separate
   `output`/`lm_head` tensor exists — it does here, so the correction applies).
   The re-implementation is **proved faithful** by a self-test against the real
   `quantprobe.spec.from_gguf` on local GGUFs; any disagreement on
   `t/a/ne/moe/bits/kvp/n_layer/arch/codebook_share` aborts the run. The self-test must
   **exercise the branches the staked number depends on** — the MoE branch, and *both* sides of
   #76's tied/untied test — or it aborts and asks for more files. (It previously took the two
   smallest files plus one name-hinted MoE and never checked what was covered.)

4-bis. **Prove the second half of the chain too.** The self-test above covers bytes. The step
   from bytes to tok/s used to be an equally unverified re-implementation, one layer downstream.
   The script now calls `quantprobe.plan.evaluate` directly and requires that its
   `pure CPU (GPU idle)` row at `ctx = 0` reproduce both the active bytes and the predicted
   tok/s **to 1e-9**, and it re-derives `eta_r`, the `1.15` multiplier and the `max(bits, 4.5)`
   floor out of the shipped source rather than trusting the literals copied into this script.
   That check found the citations in §2 were stale: `eta_r` is at `plan.py:904` today, not 674.
   A copied constant with a line number in a comment is a claim, not provenance.
5. **Predict the zero-I/O ceiling** as the weight-byte term of `plan.evaluate`'s pure-CPU row at
   `ctx -> 0`: `tok/s = eta_r * ram_bw / active_GB`. No KV term, no disk term, no context term —
   a ceiling is by definition the zero-I/O, zero-context limit, which is exactly the quantity
   they report.
6. **Target extraction rule (fixed here, before the comparison), applied to the LIVE document.**
   Their findings doc publishes ten cells with a per-cell `compute s/tok`. The rule, mechanised:
   take every markdown row whose first cell is a single letter `A`-`J` followed by a word break;
   column 2 is `tok/s`, column 3 is `compute s/tok`; a cell whose label contains `drop` ran with
   `--drop-cold-experts`, which *skips routed experts* and therefore changes the very byte count
   being predicted, so it is excluded. The target is the **median compute s/token over the
   full-width cells**; `max_measured` for P-3 is the max `tok/s` over *all* cells. Exactly ten
   cells `A`-`J` must be found and at least five must survive the exclusion, or the run aborts
   rather than re-deriving the target from a table whose shape changed.
   At stake time: excluded `C`, `H`, `J`; included
   `A 0.116, B 0.116, D 0.107, E 0.115, F 0.117, G 0.113, I 0.115` -> median 0.115 s.
7. **The target is extracted, not confirmed.** `--score` downloads their findings doc and runs
   the rule above **on what it finds now**. It also prints a line-by-line diff against the
   transcription staked in §2. A divergence is reported loudly and **scored** — it does not
   abort. This is the change that gives the kill rule a live path (§0-bis). The run still aborts
   if the document is unreachable, if the ten-cell table cannot be parsed, if the `~N tok/s
   ceiling` headline is gone, or if the phrase "dual-channel DDR4" has disappeared (that phrase
   is the sole justification for the DDR4 preset). `--offline` is **refused** in `--score`:
   scoring our transcription against itself cannot fail by construction.

Raw output: `weights/data/exp51_external_retrodiction.json` and `.log`; the fetched header is
cached at `weights/data/exp51_gguf_header.bin` so re-runs are offline and byte-identical.

---

## 2. The staked numbers

```stake
prediction_tok_s          = 8.9195
active_gb_per_token       = 2.1728
target_measured_tok_s     = 8.6957
target_headline_tok_s     = 9.0000
max_measured_desktop_tok_s= 7.3300
kill_rel_error            = 0.1500
```

Derivation, entirely from their file's header:

| quantity | value | where it comes from |
|---|---:|---|
| total params | 35.5053 B | 753 tensors, summed |
| routed expert params | 33.0176 B | `ffn_*_exps`, 256 experts |
| `token_embd` | 0.5086 B | untied (`output.weight` exists) -> **#76 subtracts it** |
| always-active `ne` | 1.9791 B | total - routed - token_embd |
| active params/token | 3.0109 B | `ne + routed x 8/256` |
| effective bits | 5.02 | file bytes x 8 / total params |
| **active bytes/token** | **2.1728 GB** | `(ne x 5.02 + 1.0318 x 5.02) / 8 x 1.15` |
| eta_r (MoE, RAM tier) | 0.38 | `plan.evaluate`, shipped since `d590749`; re-derived from source at runtime |
| RAM bandwidth | 51.0 GB/s | `MACHINES["rtx-3060"]["rb"]`, DDR4-3200 dual-channel preset |
| **predicted ceiling** | **8.9195 tok/s** | `0.38 x 51.0 / 2.1728` |

The line-number citations that used to be in this table (`plan.py:674`, `plan.py:640-668`) were
**already stale** when they were written — `plan.py` has moved since. They are replaced by
symbol names, and the script now re-derives all four constants from the installed `quantprobe`
and aborts on any disagreement, so this table cannot silently drift from the shipped tool again.
`plan.evaluate`'s own `pure CPU (GPU idle)` row is checked to reproduce `8.919522 tok/s` exactly.

---

## 3. Predictions and the KILL RULE

- **P-1 (the one that counts).** `|predicted / 8.6957 - 1| < 15%` against the median compute
  floor of the seven full-width cells.
- **P-2.** `|predicted / 9.0 - 1| < 15%` against their stated headline ceiling.
- **P-3 (free falsification, parameter-free).** A ceiling that is exceeded is not a ceiling:
  `predicted > 7.33`, their best measured desktop tok/s. If our number came in *below* a speed
  they actually achieved on that machine, Law 4 is refuted outright on this arm regardless of
  P-1 and P-2. Note that 7.33 is cell `J`, a `--drop-cold-experts` cell that reads *fewer* bytes
  than the full-width configuration being predicted, so it is entitled to exceed a full-width
  ceiling. P-3 is therefore **stricter than the physics requires** — the defensible bound is the
  best full-width cell, 6.14. The stricter one is kept; the inconsistency is disclosed rather
  than quietly resolved in our favour.

**The conjunction, which is narrower than any single predicate.** P-1 alone admits
`[7.391, 10.000]`; P-2 alone admits `[7.650, 10.350]`; P-3 admits `> 7.33`. All three hold only
inside **`[7.650, 10.000]` tok/s** — floor set by P-2, ceiling by P-1. In bandwidth terms, at
2.1728 GB/token and eta 0.38, that is **43.74 to 57.18 GB/s**, i.e. dual-channel
**DDR4-2734 to DDR4-3573**. Every `--score` run recomputes and prints this band.

**Achieved margin, stated next to the band.** The staked 8.9195 misses the median cell by 2.57%
and the headline by 0.89%, against a 15% threshold — it clears the nearer band edge by 12.1%.
The 15% is not tuned to that: it was written into `U-33.predicted_effect` before this experiment
was drafted. But a reader should know the band is roughly 6x the achieved error, and §5 now
states exactly which alternative constant sets it does and does not exclude.

**KILL RULE.** The claim counts as the external replication C-06 asks for **only if P-1 AND P-2
AND P-3 all hold.** If any fails:

- **C-06 stays open.** No modern-hardware replication is claimed.
- **U-33 is rewritten** from "external corroboration" to state plainly that the apparent 7% match
  did *not* survive contact with their actual file, and that the earlier agreement came from two
  errors partially cancelling (see §4).
- The miss is published in `FINDINGS.md` at equal prominence to a hit, per protocol.
- No constant is retuned to recover it. `eta_r`, the bandwidth preset and the `1.15` multiplier
  are calibrated on other evidence; back-fitting them to one external row would destroy the only
  property that makes this test worth running.

---

## 4. Two errors in U-33's own prose, published before the result

Correcting the register text is a precondition of this experiment, not a consequence of it. Both
corrections were found while assembling the inputs and both are stated here up front so nobody
can read them as post-hoc:

1. **U-33 says "CPU-tier eta (0.30)". No such constant exists.** `quantprobe/plan.py` has shipped
   `eta_r = 0.38 if moe else 0.62` since v1.0 and contains no `0.30` anywhere on the decode path.
   The prose value was written from memory. **At 0.30 this experiment predicts 7.04 tok/s and the
   kill rule FIRES.** The value used here is the shipped one, and the fact that the *published*
   value would have failed is disclosed on the record.
2. **U-33's active-byte estimate was ~18% low.** It implied roughly 1.8 GB/token, from the
   `q35-A-shexp` sibling; their real file gives 2.1728 GB/token. So U-33's original "8.2-8.7 vs
   ~9" agreement came from an eta that was too low *and* a byte count that was too low, partly
   cancelling. Whatever this experiment scores, **U-33's stated arithmetic was wrong twice** and
   its register entry must be corrected in the same commit that scores this.

---

## 5. Disclosed sensitivities — and the one that DOES have kill power

Computed by the same script, in the same run, before scoring. These are the honest boundaries of
the claim, not alternatives to be selected from after the fact. **P-1/P-2/P-3 bind on the
8.9195 row and on nothing else.**

The joint pass band is `[7.650, 10.000]` tok/s (§3). "in/out" below is against **that**, not
against P-1 alone.

| variation | predicted tok/s | inside the joint band? |
|---|---:|:--|
| **staked: eta 0.38, 51.0 GB/s, #76 on, x1.15 on** | **8.920** | **in** |
| DDR4-2400 dual-channel (38.4 GB/s) | 6.716 | OUT (P-1, P-2) |
| DDR4-2666 dual-channel (42.7 GB/s) | 7.468 | **OUT (P-2)** |
| DDR4-2933 dual-channel (46.9 GB/s) | 8.202 | in |
| DDR4-3200 dual-channel (51.2 GB/s theoretical) | 8.955 | in |
| finding #76 (embedding gather) switched OFF | 7.631 | **OUT (P-2), by 0.019** |
| activation multiplier x1.15 removed | 10.257 | OUT (P-1) |
| eta = 0.30, the value U-33's prose quoted | 7.042 | OUT (P-1, P-2) |
| exact per-tensor bytes (U-29 convention), no x1.15 | 8.556 | in |
| plus L-19's CPU-attention term at ctx=256 | 7.789 | in |
| DDR4-2666 **and** the L-19 term together | 6.659 | OUT (P-1, P-2) |

**Two claims that stood in this section before the adversarial review, and were wrong.** Both
were wrong in the direction that made the experiment look more robust than it is; both are
corrected here rather than quietly deleted.

- ~~"Any claim made from a PASS is conditional on DDR4-2666 or faster."~~ **False.** DDR4-2666
  predicts 7.468, below P-2's floor of 7.650 — it **fails**. The true condition is
  **≥ 43.74 GB/s, i.e. DDR4-2734 or faster; among JEDEC grades, DDR4-2933 or DDR4-3200.** The
  error came from checking the sensitivity rows against P-1's band only and forgetting that the
  kill rule is a conjunction.
- ~~"This experiment cannot be cited as evidence for finding #76, because the #76-off variant
  also lands inside the band."~~ **False for the same reason.** #76-off predicts 7.631 and misses
  P-2's floor by 0.019 tok/s. Formally it fails — but a margin of 0.25% is not evidence of
  anything, and **the operational conclusion is unchanged: do not cite this experiment for #76.**
  It is stated correctly now so the next reader does not inherit the wrong arithmetic.

### 5-A. THE OPEN FALSIFIER: their DRAM grade. Staked, with the consequence pre-committed.

This is the one input in the experiment whose value is genuinely unknown today, and therefore the
only part of it that can still go either way. They publish "dual-channel DDR4" and nothing else.
DDR4-3200 is the JEDEC ceiling and a standard fit for a 2021-class 8-core laptop, which is why
the shipped preset is used — but **DDR4-2666 is at least as common in that machine class, and at
DDR4-2666 this retrodiction FAILS.**

**Staked now, before asking:** if the upstream author (or any other evidence: an SPD dump, a
`wmic memorychip get speed`, a photo of the SODIMM label) establishes that laptop's DDR4 grade,
the run is re-scored at that bandwidth with `--dram-mts <MT/s>` and:

- **≥ 2734 MT/s** (43.74 GB/s) and ≤ 3573: the retrodiction stands, and the conditional is
  discharged rather than carried.
- **≤ 2733 MT/s** — which includes **DDR4-2666 and DDR4-2400** — the **KILL RULE FIRES
  RETROACTIVELY.** P-1/P-2's PASS is withdrawn, C-06 stays open, U-33 is rewritten as a miss,
  and no constant is retuned to recover it. Verified: `--dram-mts 2666` prints
  `FAIL: P-1 hit | P-2 MISS | P-3 hit` and exits 1.
- **No answer:** the claim keeps the conditional in its register entry, permanently, in the form
  "conditional on DDR4-2734+ which we did not verify". It is never dropped for brevity.

That ask goes to the upstream author either way. It is one command.

Two smaller disclosures, both C-14-adjacent:

- **The L-19 CPU-attention row transfers a constant across machine states.** `1.55e-6 s per
  position per layer` was calibrated on the GTX 1060 box; applying it to their laptop is exactly
  the cross-state comparison C-14 forbids, and `qwen35moe` is mostly Gated Delta Net
  (`full_attention_interval = 4`) so the term does not even apply at full strength. It is
  published as a magnitude only and has **no** kill power. It must not be cited as a prediction.
- **`51.0` vs `51.2`.** The shipped preset rounds DDR4-3200's exact 51.2 GB/s down to 51.0. The
  staked number uses the shipped 51.0 (8.920); the exact figure gives 8.955. Both are inside the
  band and nothing turns on it, but the staked row is the *shipped* constant, not the better one.

---

## 6. Known defects in the inputs, listed before the result

- **`nextn` tensors are charged as always-active.** The file carries `block_count = 41` with
  `nextn_predict_layers = 1`, i.e. 40 real blocks plus an MTP head their engine does not run.
  Its 8.4 M params are 0.4% of `ne` — too small to matter here, but it is a real over-charge and
  it will matter on an MTP-heavy file.
- **`kvp` is over-counted ~4x for this architecture.** `qwen35moe` is hybrid: only every 4th
  block is full attention (`full_attention_interval = 4`), the rest are Gated Delta Net. Our
  `kvp` formula assumes full attention on all 41. It does not enter the staked number (the
  ceiling is the ctx->0 limit) but it would poison any depth-dependent use of this file.
- **Their Qwen3.6 numbers are single runs**, not their 256-token best-of protocol on the phone
  tables — their README says so explicitly. The desktop campaign *is* 256-token, but round 1's
  `--cache-mb auto` confound is theirs and acknowledged; we use the compute column, which they
  show is flat across all three rounds, precisely to sidestep it.
- **Their runtime is not llama.cpp.** It is their own streaming engine on top of it. The compute
  path is ggml GEMV either way, but this is not our stack and we did not audit theirs.
- **We are matching a compute floor they derived, not a tok/s they measured.** Their 0.107-0.117
  s/token is an instrumented split of a wall-clock they report separately; we take their
  instrumentation on faith.
- **The staked arithmetic omits the shipped codebook penalty.** `plan.evaluate` divides `eta_r`
  by `(1 + codebook_share * IQ_CPU_TG_PENALTY)` (U-17/C-13). This file is Q4_K_M with a codebook
  share of exactly 0, so the two agree — but the script never checked that, and on an IQ file it
  would have silently predicted something that is neither the shipped tool's number nor a stated
  alternative. It now aborts if `codebook_share > 0`.
- **P-1 and P-2 are not independent.** Their "~9 tok/s ceiling" is their own prose reading of the
  same ~0.11 s/token compute floor that P-1's median is computed from. Passing both is close to
  passing one test twice; the conjunction is still narrower than either predicate alone (§3),
  but it is not two pieces of evidence.

---

## 7. What would REFUTE this, and what a PASS does not prove

**Refuted by:** any of P-1/P-2/P-3 failing, i.e. a prediction outside **`[7.650, 10.000]`**
tok/s (§3). The concrete outcome that triggers it, spelled out because an adversarial reviewer is
entitled to demand one: *their laptop turns out to carry DDR4-2666, the standard alternative to
DDR4-3200 in that machine class. The prediction becomes 7.468 tok/s, P-2 misses by 17.1%, the
verdict is FAIL, C-06 stays open and U-33 becomes a published miss.* That path is live today,
costs one command from the upstream author, and is pre-committed in §5-A.

The second live path: they revise the published cells. The target is re-extracted from their
document on every score run, so a revision moves the target and can fire the kill rule. It used
to abort instead — see §0-bis.

**Aborted, not refuted (exit 2, no number):** the accounting identity in step 2 not closing, the
mirror disagreeing with `quantprobe.spec.from_gguf`, the self-test failing to cover the MoE or
#76 branches, our constants not matching what `quantprobe` ships, `plan.evaluate` not
reproducing our tok/s, any candidate failing to size, more than one candidate rounding to
22.3 GB, the file carrying IQ codebook tensors, or their document being unreachable or
unparseable. A wrong number is worse than no number — and an abort now **overwrites** any
previous scored result on disk, so a stale PASS cannot be mistaken for the current run's output.

Note the asymmetry this encodes, which is the correction §0-bis makes: **our** side moving voids
the comparison; **their** side moving is the comparison.

**A PASS does not prove:**
- that Law 4 holds on modern GPUs. This is the CPU/DRAM tier only. **C-06 is about a batched
  decode ceiling on a modern GPU and this experiment does not touch it.** Even a clean PASS
  closes only the "does the law transfer to hardware we do not own" half of the ask, and the
  register entry must say which half.
- that eta = 0.38 is *correct*. One agreement at one point is consistent with an eta that is
  wrong in a way that cancels against a byte convention that is also wrong — which is exactly
  what U-31/U-29 already established about this pair of constants, and exactly what §4 shows
  happened to U-33's own arithmetic.
- anything about prefill, about depth, or about their phone results.
- that we predicted anything. We retrodicted a public number. §0.

---

**Wired into:** nothing, deliberately. This scores an existing law against an external
observation; it introduces no term and changes no constant. The only artifacts it may produce
are register corrections (U-33's eta and byte errors, §4; and this document's own two arithmetic
errors, §5) and a data request to the upstream author (§5-A).

---

## 8. Change log — what the adversarial review of 2026-07-30 changed, and when

All of it landed **before** any register text or `FINDINGS.md` entry was written from this
result, and none of it moved the staked block in §2, which is byte-identical to what
`--stake` emitted. Recorded here so the edits are auditable rather than invisible.

| # | Defect | Fix |
|---|---|---|
| 1 | **The kill rule could not fire.** Both operands were frozen at stake time and a 0.5% drift gate sat upstream of a 15% threshold, so every run ended PASS or ABORT. | Drift gate restricted to our side; target re-extracted live and scored; every run prints a KILL-RULE REACHABILITY block and names the run as unreachable when it is. §0-bis |
| 2 | A revised external source **aborted**, closing the only live falsification path. | Divergence is diffed, reported and **scored**. §1.7 |
| 3 | The pass condition was stated as "DDR4-2666 or faster". It is **DDR4-2734 or faster** — 2666 fails P-2. | Corrected, and the band is now computed by the script, not by hand. §5 |
| 4 | "#76-off also lands inside the band" — it does not, it misses P-2. | Corrected; the operational conclusion (don't cite this for #76) is unchanged. §5 |
| 5 | The DRAM grade was filed as a no-kill-power sensitivity. | Promoted to a staked falsifier with a pre-committed consequence and a `--dram-mts` code path, verified to produce FAIL at 2666. §5-A |
| 6 | Constants were **copied literals** with stale line-number citations (`plan.py:674`; it is 904) that nothing verified. | Re-derived from the installed `quantprobe` at runtime; abort on mismatch. §1.4-bis |
| 7 | The bytes→tok/s half of the chain was an **unverified re-implementation**. | Checked against `plan.evaluate`'s own `pure CPU (GPU idle)` row to 1e-9. §1.4-bis |
| 8 | The self-test never checked that it **entered** the MoE or #76 branches. | Branch coverage is a precondition; unmet coverage widens the sample, then aborts. §1.4 |
| 9 | A candidate that failed to size was **skipped**, leaving a partial identification. | Aborts. §1.1 |
| 10 | `--offline` produced a full PASS verdict from an unverified transcription. | Refused in `--score`. §1.7 |
| 11 | An abort left a **stale PASS** JSON on disk as the apparent current result. | Aborts overwrite it with an abort record. §7 |
| 12 | `codebook_share` was computed and then ignored, though the shipped eta divides by it. | Aborts if non-zero. §6 |
| 13 | P-3 uses a `--drop-cold-experts` cell (7.33) as a full-width bound. | Kept (it is the stricter choice) but disclosed as inconsistent. §3 |
