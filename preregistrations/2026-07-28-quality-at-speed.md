# Pre-registration #40: does the 5× speculation multiplier survive HIGHER QUALITY?

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The goal, restated precisely

100+ tok/s for ONE person on consumer hardware, at **high quality** — not at the aggressive
quantization our 108 tok/s was measured on. Our headline runs `Qwen3-30B-A3B` at **Q2_K, 2.95
bits**, whose quality cost the planner prices at ×1.05–1.07 perplexity. The honest question is
whether the speculation multiplier is a property of the LEVER or of the low-bit model.

## The mechanism says it should survive, and that is a falsifiable prediction

#36/#37 established the cost unit is the **verify round** — one full weight read — not the token.
A higher-bit model makes every round proportionally more expensive but does not change how many
rounds a given draft pattern needs. So the multiplier should be roughly **invariant** in bits,
while the base rate falls with bytes. Concretely: `Qwen3-Coder-30B-A3B-Instruct-Q3_K_M` is
13.70 GB against Q2_K's 10.49 GB — 1.31× the bytes — so raw decode should fall ~1.3× and the
speculated figure should fall by about the same factor.

If that holds, 108 / 1.31 ≈ **83 tok/s at Q3_K_M**, and 100 tok/s at this quality is NOT reachable
by speculation alone on this box. If the multiplier instead RISES with bits (more bytes per round
means the fixed per-round overhead is amortised better), we clear 100.

Placement comes from the tool itself (dogfooding): `-ot "blk\.(7..47)\.ffn_.*_exps\.=CPU"`,
predicted 15.8 tok/s raw.

## Arms (llama-server, edit task, temp 0, fresh server per arm, request 1 = the result per #38)

| arm | model | speculation |
|---|---|---|
| Q2-raw | Qwen3-30B-A3B Q2_K (2.95 bits) | none |
| Q2-spec | same | `m 384 n 4` |
| **Q3-raw** | Qwen3-Coder-30B-A3B Q3_K_M (13.70 GB) | none |
| **Q3-spec** | same | `m 384 n 4` |

## Stakes

- **P-1 (the planner's own prediction holds).** Q3-raw lands within **±25%** of the tool's
  predicted 15.8 tok/s. This is a live test of quantprobe on a model it has never seen.
- **P-2 (the multiplier is a property of the lever, not the model).** Q3-spec/Q3-raw is within
  **±20%** of Q2-spec/Q2-raw. This is the scientific claim.
- **P-3 (the honest target check).** Q3-spec ≥ **100 tok/s**. I expect this to MISS at ~83, and I
  am staking it anyway because the goal is 100 at quality and a near-miss must be reported as a
  miss, not spun.
- **P-4 (identity).** Q3-spec is byte-identical to Q3-raw at matched request index.

## KILL RULE

If P-2 fails — the multiplier is not preserved — then speculation is entangled with quantization
and every number in #36–#38 is scoped to Q2_K only. That would be a significant narrowing of this
project's headline result and must be published as such.

## What ships

The quality-vs-speed frontier UNDER SPECULATION: for each bit-width, the achievable tok/s. That is
the table a person actually needs to choose a model for their machine, and no such table exists
anywhere — every published benchmark quotes raw decode, which we now know understates the
achievable rate by 5× on copy-regime work.

---

## Scored (2026-07-28, log: `weights/data/prereg40_quality_at_speed.log`)

**Verdict: P-1 HIT, P-2 MISS — the KILL RULE FIRES, P-3 MISS (predicted, and worse than
predicted), P-4 HIT. The speculation multiplier is NOT invariant in bits, and our headline is
hereby scoped.**

| model / quant | raw | speculated | multiplier | ms/token raw | ms/token spec |
|---|---|---|---|---|---|
| Qwen3-30B Q2_K (10.49 GB) | 20.83 | 99.06 | 4.76× | 48.0 | 10.1 |
| Qwen3-**Coder**-30B Q2_K_L (10.55 GB) | 21.32 | 98.80 | 4.63× | 46.9 | 10.1 |
| Qwen3-**Coder**-30B Q3_K_M (13.70 GB) | 17.45 | **59.17** | **3.39×** | 57.3 | 16.9 |

The first draft of this comparison was **confounded** — Qwen3-30B vs Qwen3-Coder-30B changes the
model AND the bits together. The third row plus a same-model control at Q2_K_L isolates it: across
two different models at ~10.5 GB the multiplier is 4.76× and 4.63×, so **bits is the variable**.

- **P-1 (planner within ±25% on an unseen model): HIT.** Predicted 15.8, measured 17.45 (+10.4%).
  quantprobe called a model it had never benchmarked, at a bit-width it had never seen, to within
  10%.
- **P-2 (multiplier invariant in bits, ±20%): MISS.** 4.63× → 3.39× = **0.73 of it**, well outside
  the band. The kill rule fires: **#36–#38's 5× is scoped to ~3-bit-class quantization**, and this
  project will state it that way from now on.
- **P-3 (≥100 tok/s at higher quality): MISS at 59.17.** I staked a predicted miss at ~83 and the
  reality is worse. Reported as a miss, not spun.
- **P-4 (identity): HIT.** Both Coder arms are `867ecdf1eee9` at matched request index, at both
  bit-widths.

### Why the multiplier shrinks — the mechanism, from our own ledger

Bytes rise 1.299× (10.55 → 13.70 GB), and:

| | ms/token cost ratio Q3/Q2 |
|---|---|
| raw decode | **1.222×** — slightly BELOW the byte ratio |
| speculated | **1.670×** — far ABOVE it |

Raw decode tracks bytes, as Law 4 says it must. Speculated decode degrades **1.37× faster than
bytes**. The reason is in the ledger we already built: a verify round processes ~50 tokens at once,
which is a *batched* operation, and #26 measured that batched work converges to a **compute**
ceiling (~405–445 t/s on this GPU) rather than a bandwidth one. Speculation converts a
bandwidth-bound problem into a compute-bound one — and once you are compute-bound, extra bits cost
extra *dequantisation work* on every one of those 50 tokens, not just extra bytes moved. C-05's
"a quantized byte is not a byte" appearing a fifth time, now on the speculation axis.

**Consequence, stated plainly: speculation and quantization are not independent levers.** The more
you speculate, the more the low-bit format pays for itself — and the more a high-bit model costs
you. Nobody publishes this because everyone benchmarks raw decode.

### The honest answer to "100 tok/s at high quality on this box"

**Not reachable at Q3_K_M: 59 tok/s is the measured ceiling with every lever we have.** The
quality-vs-speed frontier under speculation, on a 2016 desktop, single user:

| quantization | quality cost (planner) | achievable tok/s (copy regime) |
|---|---|---|
| Q2_K / Q2_K_L (~3 bits) | ×1.05–1.07 ppl | **99** |
| Q3_K_M (~3.9 bits) | ×1.02–1.05 ppl | **59** |

100 tok/s at ~3-bit quality is real and reproducible today. 100 tok/s at 4-bit quality is not, on
this hardware — the gap is ~1.7×, which is exactly the DRAM bandwidth a newer machine would buy.

**Wired into:** `findings/REGISTER.json:V-04` (the 5× scoped to ~3-bit) · `C-05` (fifth instance) ·
`findings/REGISTER.json:V-12` (the quality-vs-speed frontier table).
