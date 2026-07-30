# Pre-registration #86: score the decode law on a machine we have never touched

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, BEFORE the comparison cell was
evaluated. **STAKED.** · **Experiment #51** · **Register:** U-33 (open), C-06 (open)

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

2. **Fetch only the header.** One HTTP Range request pulls the first 16 MiB of the 22.3 GB file
   and a self-contained parser reads the 753-tensor table (`gguf.GGUFReader` cannot: it maps the
   whole file). The parse is proved correct by an accounting identity that must close exactly:
   `header bytes + alignment pad + sum(per-tensor bytes) == remote Content-Length`.
3. **Check it is the model we staked on:** `general.architecture == qwen35moe`,
   `expert_count == 256`, `expert_used_count == 8`, `block_count == 41`. Otherwise abort.
4. **Recompute active bytes with `quantprobe.spec` arithmetic**, including finding **#76**'s
   embedding-gather correction (`token_embd` leaves the always-active set iff a separate
   `output`/`lm_head` tensor exists — it does here, so the correction applies).
   The re-implementation is **proved faithful** by a self-test against the real
   `quantprobe.spec.from_gguf` on local GGUFs (a small dense, a tiny file, and a MoE); any
   disagreement on `t/a/ne/moe/bits/kvp/n_layer/arch/codebook_share` aborts the run.
5. **Predict the zero-I/O ceiling** as the weight-byte term of `plan.evaluate`'s pure-CPU row at
   `ctx -> 0`: `tok/s = eta_r * ram_bw / active_GB`. No KV term, no disk term, no context term —
   a ceiling is by definition the zero-I/O, zero-context limit, which is exactly the quantity
   they report.
6. **Target extraction rule (fixed here, before the comparison).** Their findings doc publishes
   ten cells with a per-cell `compute s/tok`. The target is the **median over the seven cells run
   at the model's own routing width**. Cells `C`, `H`, `J` use `--drop-cold-experts 0.75`, which
   *skips routed experts* and therefore changes the very byte count being predicted; they are
   excluded, and they are named here so the exclusion cannot be chosen later.
   Included: `A 0.116, B 0.116, D 0.107, E 0.115, F 0.117, G 0.113, I 0.115` -> median 0.115 s.
7. **Re-verify the transcription.** `--score` re-downloads their findings doc and asserts every
   one of the ten `| tok/s | compute |` pairs still appears verbatim, plus the phrases
   "9 tok/s ceiling" and "dual-channel DDR4". If the source moved, the run aborts.

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
| eta_r (MoE, RAM tier) | 0.38 | `plan.py:674`, shipped since `d590749` |
| RAM bandwidth | 51.0 GB/s | DDR4-3200 dual-channel preset |
| **predicted ceiling** | **8.9195 tok/s** | `0.38 x 51.0 / 2.1728` |

---

## 3. Predictions and the KILL RULE

- **P-1 (the one that counts).** `|predicted / 8.6957 - 1| < 15%` against the median compute
  floor of the seven full-width cells.
- **P-2.** `|predicted / 9.0 - 1| < 15%` against their stated headline ceiling.
- **P-3 (free falsification, parameter-free).** A ceiling that is exceeded is not a ceiling:
  `predicted > 7.33`, their best measured desktop tok/s. If our number came in *below* a speed
  they actually achieved on that machine, Law 4 is refuted outright on this arm regardless of
  P-1 and P-2.

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

## 5. Disclosed sensitivities — no kill power, published anyway

Computed by the same script, in the same run, before scoring. These are the honest boundaries of
the claim, not alternatives to be selected from after the fact. **P-1/P-2/P-3 bind on the
8.9195 row and on nothing else.**

| variation | predicted tok/s |
|---|---:|
| **staked: eta 0.38, 51.0 GB/s, #76 on, x1.15 on** | **8.920** |
| DDR4-2400 dual-channel (38.4 GB/s) | 6.716 |
| DDR4-2666 dual-channel (42.7 GB/s) | 7.468 |
| DDR4-2933 dual-channel (46.9 GB/s) | 8.202 |
| DDR4-3200 dual-channel (51.2 GB/s theoretical) | 8.955 |
| finding #76 (embedding gather) switched OFF | 7.631 |
| activation multiplier x1.15 removed | 10.257 |
| eta = 0.30, the value U-33's prose quoted | 7.042 |
| exact per-tensor bytes (U-29 convention), no x1.15 | 8.556 |
| plus L-19's CPU-attention term at ctx=256 | 7.789 |
| DDR4-2666 **and** the L-19 term together | 6.659 |

Read from this, stated now rather than after:

- **The single biggest hole is that we do not know their DRAM speed.** They write
  "dual-channel DDR4" and nothing else. DDR4-3200 is the JEDEC ceiling and the standard fit for a
  2021-class 8-core laptop, which is why the shipped preset is used — but at DDR4-2666 the
  margin against P-1 is thin, and **at DDR4-2400 this retrodiction fails outright**. Any claim
  made from a PASS is conditional on "DDR4-2666 or faster", and must say so.
- **This experiment cannot be cited as evidence for finding #76**, because the #76-off variant
  also lands inside the band. It discriminates the `x1.15` activation multiplier (which fails
  without it) and it discriminates eta, and that is all.
- **One command from them would harden or kill this**: the DDR4 grade of that laptop
  (`wmic memorychip get speed`). That ask goes in the register either way.

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

---

## 7. What would REFUTE this, and what a PASS does not prove

**Refuted by:** any of P-1/P-2/P-3 failing. Concretely, a prediction outside 7.39-10.00 tok/s
(P-1's band) or outside 7.65-10.35 (P-2's), or one at or below 7.33 tok/s (P-3). A PASS is
narrow: the staked number has ~0.8 points of margin at the bottom of the plausible DDR4 range.

**Refuted independently of the number:** if the accounting identity in step 2 does not close, if
the mirror disagrees with `quantprobe.spec.from_gguf`, if more than one candidate file rounds to
22.3 GB, or if their published cells no longer match our transcription. Each of those aborts the
run with exit code 2 and produces no number at all. A wrong number is worse than no number.

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
are register corrections (U-33's eta and byte errors, §4) and a data request to the upstream
author (§5).
