# Pre-registration #76: stop charging for bytes the machine never reads

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, corrections computed BEFORE the code. **STAKED.**

`ne = total - routed` puts `token_embd` in the always-active set, priced at >=4.5 bits every
token. Decode GATHERS one row of a ~150k-row matrix (~zero bytes). Correct when embeddings are
TIED (that tensor IS the output projection and is fully read); a double-charge when untied.

**Change:** subtract `token_embd` params from `ne` iff a separate `output`/`lm_head` tensor exists.

## Stakes (each arm's new error, computed from measured data before implementing)

| arm | now | STAKED after |
|---|---|---|
| Qwen3-Coder-30B | −11.6% | **−1.9%** |
| Qwen3.5-35B APEX-Mini | −9.0% | **+2.0%** |
| Qwen3-30B-A3B | −10.9% | **−5.5%** |
| Qwen2.5-7B Q4_K_M | −4.9% | **+0.9%** |
| Qwen2.5-14B split | +0.6% | **+5.1%** |
| DS-Lite 16B Q4_K_M | −22.4% | **−17.6%** |

- **P-1.** Each arm above lands within ±3 points of its staked value.
- **P-2 (THE FALSIFICATION TEST, free).** The four TIED models (0.5B, 0.6B, 4B, gemma-12B) must
  move by **< 0.5%** — they have no double charge, so a correct implementation cannot touch them.
- **P-3.** The MoE K-quant family's mean error moves from −9.1% to inside ±5%.

## KILL RULE
If P-2 fails the implementation is wrong (it is touching models with nothing to fix) and is
reverted regardless of what P-1 shows. If P-1 fails but P-2 holds, the physics is right and the
magnitude model is wrong: publish and re-derive, do not tune.

**Wired into:** pending; `spec.from_gguf` ne computation.
