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
