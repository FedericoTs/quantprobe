# Pre-registration #10: Law 6 candidate — speculation economics (decode as a bandwidth arbitrage)

**Author:** Federico Sciuca · **Date staked:** 2026-07-24, committed BEFORE any speculative-decoding
run has ever been executed on this machine.
**Status: STAKED. Scoring planned in public during the week of 2026-07-27.**

## The claim under test

Speculative decoding is a known trick (llama.cpp ships it stock). What nobody prices is its
**economics**: which setup pays, on which hardware, for which workload. Our laws say the answer is
computable. Decode is bandwidth-bound on commodity tiers (Law 4); verifying k drafted tokens in one
batch costs roughly ONE weight-read (Law 5's batch amortization applied to decode). Therefore the
speedup ≈ expected tokens emitted per target read, discounted by draft cost — and it should be
**largest exactly where decode is most bandwidth-starved**. If that holds, speculation becomes a
priced knob in the machine profile, not folklore.

## Conventions (fixed before any run)

- llama.cpp b10098, llama-server, `-np 1`, temperature 0; effective tok/s from server timings.
- GPU state (`nvidia-smi memory.used`) logged before every batch — binding since 2026-07-24
  (see LAW5_PROTOCOL.md correction).
- **W-code** workload: a real 2k-token source file from this repo (quantprobe/plan.py head) plus an
  instruction to make a small localized edit — high copy-rate, the agent case.
- **W-prose** workload: WikiText continuation — low copy-rate, the control.
- No speculative flag (`--spec-type`, draft models, ngram modes) has been run on this box before
  this commit; baselines cited below were measured 2026-07-24 without speculation.

## Stakes

- **S-a (ngram / prompt-lookup, dense).** qwen7b-Q4_K_M at ngl99, baseline decode 21.65 tok/s
  (measured, server path, short ctx): stock ngram speculation on W-code → **×1.25–2.0** effective;
  on W-prose → **≤ ×1.15**. The gap IS the mechanism (copyability-driven); if prose gains as much
  as code, the lookup model is wrong and gets published as such.
- **S-b (the MoE union tax transfers to verify batches).** Qwen3-30B-A3B Q2_K hybrid
  (`-ngl 99 -ot exps=CPU`, baseline 19.9–20.0 short-ctx): same ngram setup on W-code. Drafted-token
  batches union experts across the CPU boundary, eating the amortization — Law 5's mechanism
  cross-applied to decode. Staked: the MoE's speed gain fraction lands at **≤ 0.75×** the dense
  case's, i.e. (S_moe − 1) ≤ 0.75 × (S_dense − 1). If S_moe ≈ S_dense, the transfer claim is
  refuted.
- **S-c (tiny-draft pays on a same-family pair).** Qwen2.5-0.5B-Instruct-Q8_0 drafting
  Qwen2.5-7B-Instruct-Q4_K_M, both resident on the 6 GB card: staked **×1.2–1.8** on W-code vs the
  7B-Instruct's own no-spec baseline (baseline measured first, same session; the stake is the
  ratio). Below ×1.1 = draft overhead swamps the win on 6 GB-class hardware.
- **S-d (the tier signature — second-round stake).** The law predicts the relative-speedup ordering
  **CPU-pure ≥ hybrid ≥ full-GPU** at matched workload (the more bandwidth-bound the tier, the more
  a verified batch is worth). Deliberately staked as an ordering only; exact bands will be staked
  after S-a/S-b/S-c land and BEFORE the tier runs, protocol-style.

### Added arm S-e (staked 2026-07-26, before acquiring any MTP-capable model)

A community user reported **29-30 tok/s** on Qwen3.6-35B-A3B (5070 12GB / 32GB) where this
tool predicted ~16 for the placement alone — a ~1.84x multiplier from **multi-token prediction**.
No model currently on this machine carries MTP heads, so this is staked before it can be run.

The sign is genuinely not obvious, which is what makes it worth staking: our own measured Law-4
corollary found draft-model speculation **2.3x SLOWER** on MoE (verify batches union far more
experts than a single token does). MTP should escape that penalty because its heads reuse the
same forward pass rather than running a separate draft model — but "should" is not "does".

- **S-e1 (MTP helps MoE where draft models hurt).** On an MTP-capable MoE model, MTP-enabled
  decode lands **1.4-2.2x** its own MTP-disabled baseline on the same file and placement. Below
  1.15x means MTP inherits the expert-union tax after all and our corollary generalises further
  than we thought; above 2.2x means the acceptance rate is higher than any published figure.
- **S-e2 (the multiplier is placement-independent).** The MTP speedup ratio varies by **less
  than 25%** between all-in-VRAM and hybrid placements. MTP changes tokens-per-read, placement
  changes seconds-per-read; if they interact strongly, the law needs a joint term rather than
  a multiplier.
- **S-e3 (the law composes).** Predicted tok/s = (Law 4 placement prediction) x (measured MTP
  multiplier) lands within **±25%** of measured — the same band the law already claims. A miss
  here means MTP is not a clean multiplier on the placement identity.

If S-e3 holds, `plan` gains an `--mtp` flag applying the measured multiplier, and the scope
limit now stated in LAWS.md is replaced by a term. If it misses, the honest outcome is that we
keep telling MTP users we do not model their case.

## Refuted if

S-a lands outside its band or W-prose ≈ W-code; S-b ratio > 0.9; S-c < ×1.1; S-d ordering
inverted. Misses publish with the same prominence as hits — they are the point.

## If it survives

η_spec enters the planner as a priced column; speculation joins placement, format, KV-policy and
persistence as a decided atom in the per-machine profile; and the demo writes itself: the same
2016 desktop, measurably faster, **predicted first**.

---

### Added arm S-f (staked 2026-07-26, before downloading the model)

S-e measured MTP's sign flipping with placement on a **MoE** model, and attributed the loss to
the expert-union tax. That attribution is an inference, not a measurement — the confound is that
"MoE" and "bandwidth-bound" moved together. S-f separates them with a **dense** MTP model small
enough to sit entirely in 6 GB of VRAM: `unsloth/Qwen3.5-4B-MTP` at Q4_K_M (2.83 GB).

**The mechanistic claim being tested.** For a DENSE model, a verify batch of k tokens reads the
*same weights* as a single token — weights are shared across the batch, so there is no extra
traffic. For a MoE model, a verify batch unions the experts that each drafted token routes to,
so the traffic grows with k. If that is really the mechanism, then MTP should help a dense model
in **both** placements, and the S-e hybrid loss belongs to the expert union specifically — not
to bandwidth-boundedness in general.

Matrix, same file and prompt throughout, `--spec-type none` vs `draft-mtp`:

| | MTP off | MTP on |
|---|---|---|
| dense 4B, all-in-VRAM (compute-bound) | baseline | S-f1 |
| dense 4B, pure CPU (bandwidth-bound, no experts) | baseline | S-f2 |

- **S-f1 (dense, GPU-resident): 1.3–2.0×.** Weights are already fast to read; MTP saves passes
  and the extra batch work is nearly free on a GPU. Below 1.1× would mean MTP carries a fixed
  overhead that never pays on Pascal-class hardware regardless of tier — which would make the
  S-e "spilling" gain a paging artifact rather than a general property.
- **S-f2 (dense, CPU): 1.3–2.2×, i.e. it HELPS here too.** This is the sharp one. S-e showed a
  24% LOSS on bandwidth-bound MoE; if bandwidth-boundedness were the cause, dense-CPU should
  also lose. I predict it **gains**, because without experts there is no extra traffic to pay
  for. A loss here refutes the expert-union explanation and means MTP simply does not pay on
  slow tiers, whatever the architecture.
- **S-f3 (the discriminator).** dense-CPU gain > MoE-hybrid gain by at least **0.4×** in ratio
  terms (measured MoE-hybrid was 0.76×). This is the single number that decides whether the
  S-e finding is about *experts* or about *bandwidth*.

**If S-f2 and S-f3 hold**, the rule becomes precise and shippable: *MTP pays whenever a verify
batch does not increase weight traffic — always for dense models, and for MoE only when the
saved pass costs more than the unioned experts.* That is a joint term the planner can carry.
**If they miss**, the honest statement is simpler and weaker: MTP is unpredictable per machine
and users should measure it themselves — which the tool would then say plainly.

## Arm S-e scored (2026-07-26, log: weights/data/prereg_se_mtp.log)

Model: `mudler/Qwen3.6-35B-A3B-APEX-MTP-I-Nano` (11.7 GB, 41 layers incl. the MTP head) — the
same family the reporting user runs. MTP toggled with `--spec-type none` vs `draft-mtp` on the
**same file**, same prompt, same placement: nothing else moves. Warm-up request discarded.

| placement | MTP off | MTP on | effect |
|---|---|---|---|
| hybrid (attention→VRAM, experts→RAM) | 16.39 | 12.42 | **0.76× — a 24% LOSS** |
| all-GPU (11.7 GB model on a 6 GB card, spilling) | 3.03 | 5.89 | **1.94× — a 94% GAIN** |

- **S-e1: SPLIT, and my stated direction was wrong.** Staked 1.4–2.2× *faster* in both
  placements. On hybrid it is **0.76×** — below the 1.15 floor I set, which I wrote would mean
  "MTP inherits the expert-union tax after all". It does. On the spilling placement it is
  **1.94×**, inside the staked band. Same file, same model, opposite signs.
- **S-e2: DECISIVE MISS — and this is the real finding.** Staked the multiplier would vary by
  **less than 25%** across placements. Measured variation: **157%**. I also wrote what a miss
  would mean, before measuring: *"if they interact strongly, the law needs a joint term rather
  than a multiplier."* That is now the measured outcome.
- **S-e3: unscoreable as posed.** It assumed a single multiplier existed to compose with Law 4.
  None does.

### Why the sign flips (mechanism, consistent with an existing measured corollary)

MTP trades **more work per forward pass** for **fewer forward passes**. Whether that pays depends
entirely on which side is expensive on your tier:

- **Experts in RAM (hybrid):** a verify batch unions more experts than a single token needs, and
  every extra expert is a slow RAM read. The extra work costs more than the saved pass. This is
  the same mechanism as the already-published Law 4 corollary — draft-model speculation measured
  **2.3× slower** on MoE — and MTP does *not* escape it, despite reusing the forward pass.
- **Model spilling (all-GPU on too little VRAM):** an extra forward pass means re-paging the
  model across PCIe. That is enormously expensive, so halving the number of passes nearly doubles
  throughput.

**The rule for users, which is not what anyone would guess:** MTP is not free acceleration. It
pays when an extra *pass* is expensive, and costs when an extra *batch* is expensive.

### Scope limit — the regime our hardware cannot test

The all-GPU arm here is a **thrashing** regime, not a healthy compute-bound one: an 11.7 GB model
on a 6 GB card. The regime the reporting user is actually in — model comfortably GPU-resident on
12 GB — is **untested on this box** and is the one place a clean 1.8× would be expected. Nothing
here contradicts his measurement; the two sit in different regimes, and confirming the crossover
needs either a larger card or a model small enough to fit ours entirely.

### Consequence

`plan --mtp` is **not** shipped: a flag applying a single multiplier would encode exactly the
model this measurement refuted. The LAWS.md scope limit stands, now with a measured direction
attached rather than a guess. What users get today is the honest placement-conditional guidance
above.
