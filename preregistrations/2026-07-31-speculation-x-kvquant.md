# Pre-registration #93: speculation × KV-quant — two banked levers, same token, never measured together

**Author:** Federico Sciuca · **Date staked:** 2026-07-31, BEFORE any GPU run. **STAKED.**
No benchmark for this prereg has been executed; only the harness self-test and the
failing-input refusal check (both GPU-free) ran before this text was frozen.

> **Renumbered #92 → #93 on 2026-07-31 (pre-release audit), before any arm was run.** This
> document was staked at 03:34 with a number already taken by
> `2026-07-31-per-shape-calibration.md` (staked 03:32). Nothing here changed except the number:
> no kill rule, no stake, no threshold. The collision mattered because `findings.py` keys staked
> pre-registrations by integer, so the second file silently overwrote the first and the
> "every staked prereg is cited" gate could only ever see one of them. `findings.validate` now
> refuses a duplicate number outright.

## Why this cell, and why it is not free

Two levers are banked and have never shared a session:

- **Dense-split speculative draft** (prereg #69): Qwen2.5-14B Q4_K_M split + 0.5B CPU draft,
  K=2, **1.335×** — measured at *shallow* context (a short novel-code prompt, depth ≈ 0).
- **q8_0 KV cache** (prereg #25): **+37.0%** at d16384 — measured on *Qwen3-30B-A3B MoE*,
  no draft, and quality-cleared at ratio 1.0003 on generic prose (prereg #91, 4k chunks).

Naive composition multiplies the two banked headlines: 1.335 × 1.37 ≈ **1.83×**. That number
imports each lever's best-case foreign config (different model, different depth, different
placement) and I am staking now that it is NOT the number this box will print. But the two
levers do act through different resources — speculation amortizes WEIGHT reads across accepted
tokens, KV-quant shrinks CACHE reads — so *some* composition should survive. Three
interference channels, named before measuring:

- **(a)** the draft has its OWN KV cache. llama-speculative exposes `-ctkd/-ctvd` separately
  from `-ctk/-ctv`, so "q8_0 KV" is actually two decisions, and the 5th arm below splits them.
- **(b)** the verify step reads the target KV **once per round** for K+1 positions. At K=2 and
  #69's acceptance, ~2 accepted tokens/round already share one KV read — the verify batch
  amortizes the *same bytes* q8_0 shrinks. Two levers dividing one byte pool compose
  sub-multiplicatively, not independently.
- **(c)** a quantized draft KV could degrade the draft's deep-context predictions, cutting
  acceptance — which eats the speculation gain while tok/s alone would mis-attribute the loss
  to "interference". Therefore the acceptance rate is a STAKED measurement, not a footnote.

## Design

Model pair and placement (frozen for every arm, no per-arm renegotiation):

| | |
|---|---|
| target | `D:/evo-compress-data/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf` (bartowski, the #69 file) |
| draft | `D:/evo-compress-data/gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf` (bartowski 2026-07-28; the June file with identical size caused #69's crash — mtime-gated) |
| placement | `-ngl 16` target (NOT #69's 28: at 28, f16 KV for 16.9k ctx does not fit 6 GB — 28×4 KiB×16896 ≈ 1.35 GiB KV + 4.9 GiB weights; at 16 it is ~1.03 GiB KV + ~2.8 GiB weights), `-ngld 0` draft (CPU, per #69) |
| context | `-c 16896`, deep prompt of 14336–16384 wikitext-2 tokens (llama-tokenize-verified), `-n 256`, `--temp 0`, seed 42, `-t 4`, `-fa on`, no-mmap, K = `--spec-draft-n-max 2` |
| binaries | `llama-speculative.exe` and `llama-bench.exe` from the SAME build (`tools/llama.cpp-pristine/build/bin`); token counting only via b10098 `llama-tokenize` |

Arms (bench arms use `llama-bench -p 0 -n 128 -d 0,16384 -r 3 -o json`; spec arms are 3
independent llama-speculative runs each):

| arm | harness | target KV | draft KV | measures |
|---|---|---|---|---|
| T00 | bench | f16 | — | tg128 @ d0 and @ d16384 (d0 row feeds K-1) |
| T01 | bench | q8_0 | — | tg128 @ d0 and @ d16384 |
| T10 | spec K=2 | f16 | f16 | tok/s + acceptance at ~15.5k depth |
| T11 | spec K=2 | q8_0 | q8_0 | tok/s + acceptance |
| T11h | spec K=2 | q8_0 | **f16** | the disambiguation arm for channel (c) |

Derived quantities (deep values throughout; medians over reps):
`G = T01/T00` (KV lever, no draft) · `S = T10/T00` (speculation at depth; cross-harness pair,
same build — the harness constant is disclosed, and it CANCELS in I) ·
`C = T11/T00` (composed) · **`I = (T11/T10) / (T01/T00)`** (interaction ratio; both ratios
within-harness, so harness constants divide out).

## Stakes

- **P-1 (the KV lever transfers).** `G ≥ 1.06` at d16384. The +37% was MoE-at-16k on 30B;
  this is a dense split where CPU weight reads dominate the token, so the KV share is smaller.
  Modeled expectation ~1.10–1.15. Branch staked now: `G < 1.03` → the lever does NOT transfer
  to this placement and the banked +37% gets a scope note naming its config.
- **P-2 (speculation survives depth).** `S ≥ 1.15` with median acceptance ≥ 60%. The 1.335×
  was shallow; at 15.5k the target pays a KV read per round (amortized, channel b) but the
  serial CPU draft pays its own deep-KV read per drafted token, un-amortized.
- **P-3 (composition is worth shipping).** `C ≥ 0.90 × G × S` — the composed lever keeps at
  least 90% of the naive product. Its kill is **K-7** below.
- **P-4 (interaction direction — the headline).** `I ≤ 1.00`, staked point estimate **0.98**:
  sub-multiplicative, because the verify batch already amortizes the KV bytes q8_0 shrinks
  (b) and the draft adds un-shrinkable weight reads to every round, diluting the q8_0 saving
  inside the draft arms. Interpretive branches pre-written: `I < 0.97` → sub-multiplicative
  CONFIRMED, quantprobe must compose these levers with a measured interaction factor, not by
  multiplication; `0.97 ≤ I ≤ 1.03` → independent within resolution, multiplicative
  composition banked with ±3% error bar; `I > 1.03` → super-multiplicative, mechanism story
  WRONG, nothing banked until explained.
- **P-5 (the acceptance stake — mandatory, tok/s cannot hide the mechanism).**
  `|acc(T11) − acc(T10)| ≤ 3.0` absolute points (q8_0 KV was quality-neutral on prose at 4k in
  #91; I stake that it stays acceptance-neutral here). Attribution rule using the T11h arm,
  fixed in advance: if acc(T11) drops > 3 pts but acc(T11h) is within 1 pt of acc(T10), the
  damage is the DRAFT's quantized cache (channel c) and the shippable composition becomes
  target-q8_0 + draft-f16; if acc(T11h) drops too, the damage is target-side and no KV split
  rescues it. Expected ordering if P-5 misses: acc(T10) ≥ acc(T11h) > acc(T11).

## Kill rules (mechanical; the script enforces every one)

- **K-1 — the cannot-vary guard (depth gate).** From the T00 bench pair:
  `tg(d16384)/tg(d0) > 0.88` → the KV term is < 12% of the token at this placement, the KV
  factor of the 2×2 CANNOT move more than noise, and every downstream number would be a
  clean-looking null. Verdict **UNMEASURABLE-AT-THIS-PLACEMENT**, exit 3, nothing scored.
- **K-2 — prompt gate.** Deep prompt outside 14336–16384 tokens (`llama-tokenize` count) →
  exit 2 BEFORE any benchmark. A char-length floor (3.2 chars/token × 14336) fires even
  earlier, before the tokenizer or the GPU is touched.
- **K-3 — KV-type gate.** Every arm's parsed cache types must equal the staked assignment:
  bench arms from the `-o json` `type_k/type_v` fields; spec arms from the two per-context
  `K (<type>):` load lines in target-then-draft order, exactly two, fail-closed (a llama.cpp
  load-order change fires the gate rather than silently swapping target and draft). Any
  mismatch → exit 2, INVALID. Two arms whose logs show the same KV type are the #91 guard
  restated: identical operands cannot produce an informative ratio.
- **K-4 — placement gate.** Every log must show `offloaded 16/… layers` for the target and
  `0/…` for the draft (bench: `n_gpu_layers == 16`); any OOM, crash (the #69
  "invalid vector subscript" signature included), or timeout in any arm → the WHOLE run is
  INVALID, exit 2. No per-arm `-ngl` renegotiation: q8_0 arms could fit more layers, and
  letting them would conflate placement with cache format.
- **K-5 — acceptance gate.** Each spec rep must yield a parsable `accept = X%` AND ≥ 128
  decoded tokens; an arm with < 2 valid reps → exit 2. tok/s from a rep without acceptance is
  never scored.
- **K-6 — uninformative guard.** K-1 passed but `|G − 1| < 0.01` AND `|T11/T10 − 1| < 0.01` →
  verdict **UNINFORMATIVE, not a pass** for "no interaction" (the #85-arms-C/D shape), exit 3.
- **K-7 — composition kill.** `T11 < max(T01, T10)` → ACTIVE interference: composing the
  levers is worse than the better one alone, the composition must NOT ship, and the pair is
  registered anti-composable with the acceptance numbers attached.

## The failing input, constructed and executed before staking

**The trap this experiment is built around:** run the identical 2×2 with a short prompt — the
depth never reaches where the KV term binds, all four cells land within noise of each other,
and `I = 1.000` prints as a beautiful "levers are independent" conclusion that is actually a
measurement that cannot vary.

Constructed concrete input: `weights/data/exp57_FAILING_INPUT_shallow_prompt.txt` (first ~2000
chars of wikitext-2, ≈ 500 tokens — exactly the #69-style shallow prompt someone would
naturally reuse). Verified at staking time, no GPU touched:

```
python weights/exp57_spec_x_kvquant.py --prompt-file weights/data/exp57_FAILING_INPUT_shallow_prompt.txt
→ REFUSED (K-2 char floor), exit code 2
```

The residual case — a prompt that passes the token count while the KV term still does not bind
(e.g. someone reruns at `-ngl 28 -c 4096`, or a future placement hides KV cost) — is exactly
what K-1 catches at run time, from measured tg, not from assumptions. And if depth binds but
both paired ratios are flat anyway, K-6 refuses to convert flatness into a finding.

`--self-test` additionally proves each gate fires on fabricated inputs (short file, twin-f16
logs where q8_0 was staked, swapped target/draft KV order, a log with tok/s but no acceptance
line, a shallow tg pair) and that the I-arithmetic reproduces a hand-computed sub-multiplicative
case. A gate that cannot fail is the failure signature this repo hunts; every one of these has
now been watched failing.

## Honest risk, stated before the run

The staked point `I = 0.98` sits 2% from 1.00. Bench arms carry ~±1% (r=3 stddev printed);
#69's spec arms repeated to ~±1%. First-order noise on I is therefore ±2–3%, so the run may
land in the `0.97–1.03` "independent within resolution" branch — that outcome is pre-declared
informative (it banks multiplicative composition with an error bar), not a miss to be
narrated around. What would be a genuine miss: `I > 1.03`, or P-5 acceptance collapse.
Also disclosed: S compares llama-bench tg128 against llama-speculative's decoded-phase t/s
(same build, same session); the pair-ratio I is immune, S itself carries the harness constant,
same as #69 accepted.

## What ships on each branch

- P-3 holds and P-4 lands sub/independent → quantprobe's dense-split advice gains a composed
  row: `-ctk q8_0 -ctv q8_0` + K=2 CPU draft at depth, with the measured C (not 1.83×) and the
  measured I as the composition factor.
- P-5 misses via the draft channel → the composed row ships as target-q8_0 + draft-f16
  (`-ctkd f16 -ctvd f16`), and "quantize the draft's KV too" is registered as a trap.
- K-7 fires → anti-composition registered; the two levers stay separate rows with a warning.
- K-1 fires → this placement cannot host the test; re-stake at deeper ctx or different split
  in a NEW prereg (no silent parameter search inside this one).

**Explicitly NOT claimed:** generalization off this box, off this model pair, off K=2, or to
acceptance regimes other than deep-prose continuation (the #69 acceptance was shallow novel
code; the two numbers are not comparable and the script compares acceptance only ACROSS ARMS
of this run). Runtime estimate 1.5–2 h (nine 16k prefills + six bench depth-fills).

**Script:** `weights/exp57_spec_x_kvquant.py` · raw output `weights/data/exp57_*` ·
**Wired into (on a valid run):** the speculation×KV cell of `quantprobe/plan.py` composition
advice; FINDINGS entry either way, misses at equal prominence.
