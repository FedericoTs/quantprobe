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

---

## SCORED — 2026-07-30

| arm | was | staked | after | |
|---|---|---|---|---|
| Qwen2.5-7B Q4_K_M | −4.9% | +0.9% | **+2.1%** | HIT |
| Qwen2.5-14B split | +0.6% | +5.1% | **+4.7%** | HIT |
| DS-Lite 16B Q4_K_M | −22.4% | −17.6% | **−19.5%** | HIT |
| Qwen3-Coder-30B | −11.6% | −1.9% | **−3.2%** | HIT |
| Qwen3-30B-A3B | −10.9% | −5.5% | **−2.5%** | miss by 0.01pt of the ±3 bound |
| Qwen3.5-35B APEX-Mini | −9.0% | +2.0% | **+5.4%** | miss by 0.4pt |

- **P-2 HIT, and it is the load-bearing one.** All four TIED models moved by **≤ 0.03 points**
  (0.00 / 0.02 / 0.03 / 0.03). The implementation touches exactly the models with a double
  charge and nothing else — the falsification test the prereg built in for free.
- **P-3 HIT.** The MoE K-quant family's mean error: **−9.1% → 0.0%**.
- **P-1 MISS on 2 of 6**, both because my hand-computed correction scaled only the active-byte
  total, while the real change also flows through `ne` into the MoE active formula. Both misses
  are in the *more accurate* direction. Per the kill rule ("if P-1 fails but P-2 holds, the
  physics is right and the magnitude model is wrong: publish and re-derive, do not tune") the
  change SHIPS and the arithmetic error is published rather than back-fitted.

**Ladder after #76:** median |error| 9.0% → **7.4%**, five arms now inside ±5%. The remaining
top residuals are all Mechanism B (codebook pricing): APEX-MTP +70.3%, DS-IQ2 +28.4%,
Qwen3.6-Q2_K_XL +25.3% — larger than before, exactly as predicted for an opposite-signed bias
once the first is removed.
