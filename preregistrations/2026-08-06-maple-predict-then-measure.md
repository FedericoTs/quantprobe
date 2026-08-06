# MAPLE-PREVIEW 20B-A1B — the prediction, published before the download

**Date staked:** 2026-08-06. At stake time NOTHING from this model exists on this box - no
GGUF, no fork, no bytes. The spec comes from public metadata only: deepgrove/maple-preview
(MIT; 20B total, ~1B active, 24 layers, 256 experts / 8 active, SWA-512:GA hybrid attention,
ternary) and stamsam/maple-preview-gguf (TQ2_0 5.45 GB = 2.18 bits/param avg; tiered format:
168 ternary + 2 Q4_0 + **121 F32 tensors**; requires the stamsam/llama.cpp `prism` fork rev
9ee03ee - mainline cannot load arch `maple`).

## The two rival predictions for THIS box (GTX 1060 6GB / 16GB DDR4), staked

- **The law, as shipped** (`quantprobe plan --total 20 --active 1 --bits 2.18`, output
  committed verbatim in weights/data/maple_prediction.txt): **75.4 tok/s, hybrid
  attention->VRAM / experts->RAM**, active 0.43 GB/token, capacity-bound 0.6 GB over the
  VRAM boundary. Conditional recorded now: the real file is 5.45 GB (plan estimated 6.0 from
  bits) - if `plan --gguf` on the actual file fits all-in-VRAM, the AIV row's prediction
  becomes operative and BOTH are in the committed output.
- **The naive active-bytes rival:** eta x BW / (1B x 2.18/8) = **~352 tok/s** - absurd on its
  face, which is the point: A1B marketing arithmetic ignores the always-active path, and this
  model's own file carries 121 F32 tensors of it.

## The named unknown, stated before it bites

FORMAT_EBW has NO ternary entry (#49's gap). The RTX 4000 Ada external point (97 tok/s on a
360 GB/s card) implies TQ2_0 effective throughput far below K-quant class - consistent with
EITHER a heavy protected path OR compute-shaped ternary kernels (the IQ-codebook lesson,
possibly repeated). Our CPU-vs-GPU split measurements are designed to separate the two; the
first measured ternary FORMAT_EBW entry is a named goal of this experiment.

## External retrodiction targets (their card's numbers, our scoring at measure time)

| machine | their claim | note |
|---|---|---|
| H200 (F16 40.5 GB, AIV) | 385 tok/s | bandwidth-rich regime |
| RTX 4000 Ada (TQ2_0) | ~97 decode / ~111 prefill | the TQ-kernel discriminator |
| Jetson Xavier (TQ2_0, CPU) | 14.8 decode | shared-LPDDR regime |
| 96-core host (TQ2_0, CPU) | ~5.5 | the compute-shaped smoking gun if real |

## Measurement plan + kill rules

1. **KR-M1 (runtime honesty):** the prism fork is a NEW RUNTIME and a NEW MACHINE STATE -
   build it, re-anchor via `calibrate` on that build, and never compare its numbers against
   b10098 rows without saying so. Fork build failure on Windows/CUDA = precondition-block
   (exit-2 semantics), published as such.
2. **KR-M2:** predictions above are frozen at this commit; `plan --gguf` on the real file may
   REFINE placement but the staked numbers do not move.
3. Bench TQ2_0 (and Q4_K_M split if time): tg128 r=3, one state, locks, GPU logged.
4. **P-M1:** the law's operative row lands within its published band (all-in-VRAM floor
   semantics if AIV; +/-25% if split). **P-M2:** the naive 352 is refuted by >2x (it cannot
   be within 2x of measured, or the protected-path thesis takes real damage and we say so).

Verdict + the first ternary EBW entry appended here. Media: the predict-then-measure card.
