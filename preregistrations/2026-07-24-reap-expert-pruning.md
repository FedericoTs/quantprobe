# Pre-registration: expert pruning (REAP-50%) vs the placement laws

**Author:** Federico Sciuca ([quantprobe](https://github.com/FedericoTs/quantprobe))
**Date staked:** 2026-07-24 — committed before the model downloads completed
**Status:** SCORED same day — one hit, one miss against my own band, one not-scoreable-here. Verbatim:

- **P-a: MISS (band too kind to pruning).** Measured **+39.0%** WikiText-2 perplexity (7.9852 ± 0.26
  parent vs 11.0990 ± 0.37 REAP-50, identical 32 chunks, identical flags) vs my staked +8–20%. My own
  refutation clause (">30%: my band was too kind to pruning") triggers. The qualitative thesis —
  calibrated pruning is NOT free; the cost hides exactly where in-domain benchmarks don't look — is
  confirmed at roughly double the strength I predicted. Law 2 stands reinforced; my band is published as wrong.
- **P-b: not scoreable on the reference box.** The parent's 14.7 GB Q3_K_M exceeds 16 GB RAM →
  partial disk paging, tg32 error bars of ±79%/±66% (3.38 ± 2.66 vs 4.74 ± 3.15) — the measurement
  noise is an order of magnitude larger than the ±5% question. A clean test needs ≥24 GB RAM;
  `bench --contribute` from such a box settles it. (The active-parameter arithmetic is unchanged by
  pruning — 3.3B either way — so the prediction stands, unmeasured here.)
- **P-c: HIT — the value IS tier promotion.** On a 16 GB-VRAM card the pruned file fits all-in-VRAM
  (**61.7 tok/s predicted**) while the parent must run hybrid (**15.7**): **×3.9**, all of it from
  the tier jump the smaller file enables — none from pruning per se. Raw log: [`weights/data/reap_verdict.log`](../weights/data/reap_verdict.log).
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
