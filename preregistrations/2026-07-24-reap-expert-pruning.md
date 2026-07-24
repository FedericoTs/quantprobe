# Pre-registration: expert pruning (REAP-50%) vs the placement laws

**Author:** Federico Sciuca ([quantprobe](https://github.com/FedericoTs/quantprobe))
**Date staked:** 2026-07-24 — committed before the model downloads completed
**Target:** [cerebras Qwen3-Coder-REAP-25B-A3B](https://huggingface.co/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF)
(50% of experts pruned, coding-calibrated) vs unpruned Qwen3-Coder-30B-A3B, both Q3_K_M, same box (2016 desktop).
**Context:** community claims circulate that expert deletion "costs nothing" (same in-domain benchmark score)
while low-bit quantization "costs 15 points". Both halves deserve a held-out, out-of-domain test.

## Staked predictions

- **P-a (quality, out-of-domain):** the REAP-50 model shows **+8–20% WikiText-2 perplexity** vs the
  unpruned parent at matched quant. Pruning is calibrated on coding traffic; its cost hides out-of-domain.
  **If it comes back <3%, Law 2's "trained networks are dense everywhere" phrasing needs revision and
  I will publish that revision with the same prominence.**
- **P-b (speed):** decode tok/s **identical within ±5%** between the two, both RAM-resident with the same
  placement — active-params-per-token is unchanged by pruning (~3.3B either way). Pruning buys FIT
  (capacity), not per-token speed; the law says bytes-per-token rule decode, and pruning doesn't touch them.
- **P-c (the real value, priced by Law 4):** the pruned file's smaller footprint promotes it a memory
  tier on mid-range hardware (e.g. fits 12 GB VRAM where the parent cannot) — and THAT, not the pruning
  itself, is where any speedup lives. `quantprobe plan` output for both files attached after measurement.

## Refuted if

P-a: out-of-domain delta <3% (refutes my density claim) or >30% (my band was too kind to pruning).
P-b: RAM-resident decode differs by >10% at equal placement.
Protocol: llama-perplexity on WikiText-2 test (same chunks both), llama-bench tg32 same flags both,
raw logs committed in-tree. Misses published with the same prominence as hits, per house rules.
