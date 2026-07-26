# Pre-registration #16: the low-bit GPU collapse is a *format* effect, not a bit-width effect

**Author:** Federico Sciuca · **Date staked:** 2026-07-26, BEFORE the discriminating measurement.
**Status: STAKED.**

## How this surfaced

Chasing a different bug (the planner withholding the all-in-VRAM row for a dense 12B) exposed a
second one underneath it. `plan.py` gates GPU decode efficiency on bit-width:

```python
geta_w = geta if bits >= 4 else gl      # gl = 0.04 on this Pascal card
```

A model at 3.99 bits is therefore predicted **8.75× slower** than the same model at 4.00 bits.
For `gemma4-12b` at 3.51 effective bits the law predicts **1.0 tok/s** all-in-VRAM — so low that
the tool recommends *pure CPU at 3.9* instead. Measured reality for that file, all in VRAM:
**9.56 tok/s.** The tool is steering users to a placement 2.4× slower than the one it rejected.

**This was already known and never wired in.** LAWS.md (Law 2, 2026-07-25) states the collapse is
"**dequant-format-dependent, not bit-width-dependent**", citing Bonsai-27B at **Q1_0 — 1.13 bits —
running 11.94 ± 0.04 tok/s all-in-VRAM where the gl model predicted ~1.8.** A published finding
never reached the code that needed it. That gap is the actual defect here.

## Prior evidence, prefill only (`weights/data/law5_h12_formats.log`)

Same 7B model, same card, all-in-VRAM, pp2048 — only the quantization format differs:

| format | bits | pp2048 tok/s |
|---|---|---|
| Q4_K_M | 4.5 | 27.49 ± 0.04 |
| **Q2_K** | **2.8** | **17.71 ± 0.22** |
| IQ3_S | 3.66 | 3.74 ± 0.01 |
| IQ3_XS | 3.3 | 4.04 ± 0.00 |

The ordering is not monotone in bits. **Q2_K has the fewest bits and is 4.4× faster than IQ3_XS.**
K-quants dequantize with a scale and a min; IQ-quants walk a codebook lookup, which Pascal does
badly. Bit-width is a confound; the format is the cause.

But that is **prefill**, and `gl` governs **decode**. Prefill is compute-bound and decode is
bandwidth-bound, so the mechanism need not carry over. This stake tests decode directly.

## The discriminating stake

`Qwen2.5-7B-Instruct` in **Q2_K** (2.80 GiB) versus **IQ3_XS** (3.11 GiB), all in VRAM, tg128,
3 runs each. IQ3_XS is 11% *larger*, so pure bandwidth predicts it should be ~11% *slower* —
nothing more. The current `gl` model predicts both collapse identically (both are under 4 bits).

- **P-1 (K-quants do not collapse).** Q2_K measures **≥ 15 tok/s**. The `gl` model predicts
  192 × 0.04 / 3.46 ≈ **2.2 tok/s**. Anything above 15 refutes bit-width gating outright.
- **P-2 (IQ-quants do collapse).** IQ3_XS measures **< 8 tok/s**, i.e. the collapse is real and
  `gl` has a genuine domain — it is merely pointed at the wrong predicate.
- **P-3 (the split is not bandwidth).** Q2_K ÷ IQ3_XS **≥ 2.5×**, despite Q2_K carrying fewer
  bits per weight and the two files differing by 11% in size. Bandwidth alone predicts 1.11×.

## Refuted if

- P-1 fails (Q2_K really does collapse) → the current bit-width gate is right and the gemma
  measurement needs another explanation.
- P-2 fails (IQ3_XS runs fine too) → `gl` has no domain at all and must be deleted, not re-aimed.
- P-3 fails (ratio < 2.5×) → the difference is bandwidth, and format is not the mechanism.

Any of these publishes with equal prominence.

## What ships if it holds

The gate changes from bit-width to **format family**, read from the GGUF where `spec.py` already
parses the quant type. Every anchor must retrodict unchanged (P-4): the 30B hybrid 19.3, the 110B
disk-stream 0.19, the corrected 18.35 baseline. If an anchor moves, it does not ship.

**Who this affects:** Q2_K, Q3_K_M and Q3_K_L are among the most widely used quantizations in
local LLM communities — they are what people run when a model *nearly* fits. quantprobe has been
telling every one of those users their GPU is useless for it.
