# Pre-registration #73: the KV depth term past 16k — where the users actually live

**Author:** Federico Sciuca · **Date staked:** 2026-07-30, predictions captured BEFORE any run. **STAKED.**

## Why

Law 4 v2's context term is validated to **d16384** and no further (#66: the 4k arm landed −0.9%,
the 16k dense-split arm missed −58% and opened C-11, since fixed). Meanwhile every user this
project met this week runs **100k context**: E-08's agentic setup, Attackwave at 131k, the
6 GB-laptop trio at ~102k, MoneroApe's fork. Our own public audit mis-graded an RTX 2060 report
partly because I compared a 100k-context measurement against a d0 efficiency band. The term is
either right out there or it is not, and we ship advice into that regime daily.

## Predictions, captured from the shipped tool before measuring

`plan --gguf ... --ctx D`, this box, calibrated+anchored path (its printed winner each time):

| arm | d0 | d4096 | d16384 | d32768 |
|---|---|---|---|---|
| Qwen2.5-7B Q4_K_M | 19.4 AIV | 18.8 AIV | 9.7 (split 20/28) | 7.4 (split 17/28) |
| Qwen3-30B-A3B Q2_K (flagship) | 17.0 (split 31%) | — | 12.3 (split 15%) | 9.6 (hybrid) |

Note the tool switches PLACEMENT as depth grows (KV crowds VRAM). That is the C-11 machinery
doing its job, and it means the arms below must be measured **as emitted**, not on a fixed
config — comparing a d0 config at d32768 would test nothing the tool claims.

## Protocol

`llama-bench -d D` (its own depth mode: prefill D tokens, then time tg) with the emitted
placement per depth, `-n 128 -r 3`, one session, clocks logged per C-10. KV left at f16 for the
7B AIV arm at d0/d4096; where the emitted config is a split, the emitted flags are used verbatim.
If an arm cannot fit at all (VRAM+KV over budget), that is recorded as **"emitted config does not
run"** — which would itself be a C-11-class defect, not a skipped arm.

## Stakes

- **P-1 (the term holds out to 32k).** Each measured arm lands within the tool's printed band of
  its own captured prediction above (±25% off-VRAM; the all-in-VRAM rows keep the one-sided
  ≥0.90× floor semantics, L-18).
- **P-2 (the SHAPE is right, not just the points).** The measured decay from d0 to d32768 is
  **monotonic** and within ±30% of the predicted decay ratio on both arms (7B predicted
  19.4→7.4 = 0.38×; flagship 17.0→9.6 = 0.56×). This tests the KV term's slope, which is the
  actual physics claim — point accuracy can be right for the wrong reason.
- **P-3 (no emit is unrunnable).** Every emitted depth config actually loads and runs. C-11 was
  supposed to end the "predicted a config that thrashes" class; 32k is its first real test.

## KILL RULE

**If P-2 fails — the measured slope departs the predicted slope by more than ±30% — the context
term does NOT extend past 16k**, and `plan --ctx` prints a validated-to-16k scope line above that
depth rather than a number we cannot stand behind. If P-3 fails, C-11's fix is incomplete and
reopens with the failing config named. Misses publish at the same prominence as hits.

**Wired into:** pending; Law 4 v2's scope line, `plan --ctx` behaviour, and the C-11 closure
either way.

---

## SCORED — 2026-07-30, same session

Raw log: `weights/data/prereg73_kv_depth.log`. Clocks 139 MHz idle → 1847 MHz loaded (healthy).

| arm | depth | predicted | measured | error |
|---|---|---|---|---|
| 7B dense, all-in-VRAM | 0 | 19.4 (floor) | **22.89** | 1.18× above the floor ✓ |
| 7B dense, all-in-VRAM | 4096 | 18.8 | **18.98** | **−0.9%** ✓ |
| 7B **dense split** 20/28 | 16384 | 9.7 | **3.44** | **+182% over** ✗ |
| 7B **dense split** 17/28 | 32768 | 7.4 | **1.54** | **+381% over** ✗ |
| 30B MoE split 31% | 0 | 17.0 | **21.45** | −20.7% (in band) ✓ |
| 30B MoE split 15% | 16384 | 12.3 | **15.95** | −22.9% (in band) ✓ |
| 30B MoE hybrid | 32768 | 9.6 | **10.70** ± 2.52 | −10.3% (in band) ✓ |

- **P-1 SPLIT BY PLACEMENT CLASS.** Every arm that keeps attention on the GPU (all-in-VRAM, MoE
  expert-split, MoE hybrid) lands in band at every depth — including 32k, twice our previously
  validated ceiling. Every arm that puts whole dense layers (attention included) on the CPU
  misses catastrophically.
- **P-2 MISS for dense splits, HIT for MoE.** Predicted decay vs measured: 7B dense ×0.381
  predicted against ×0.067 measured (ratio 0.18 — far outside ±30%); 30B MoE ×0.565 against
  ×0.499 (ratio 0.88 — inside). **The kill rule fires for dense splits only.**
- **P-3 HIT.** Every emitted config loaded and ran; C-11's fit math is sound. The defect is
  physics, not fit — which also re-explains #66's −58% at 16k as this same class, not a
  budgeting bug.
- **THE MECHANISM, and it is quantitative.** The unexplained time per CPU-resident layer:
  **23.5 ms at d16384 and 46.7 ms at d32768 — 1.43 and 1.42 µs per position per layer.** Two
  independent depths agreeing to 1%: CPU-side attention reads and multiplies the whole KV cache
  every token, which is COMPUTE on 4 threads, not the bandwidth read our KV term prices. A
  GPU-resident attention block does the same work with ~100× the parallelism, which is why the
  same term is exact (−0.9%) on the all-in-VRAM arm.

**Actions taken:** the kill rule's scope line ships now (dense splits at depth print the
warning instead of a number we cannot stand behind). The measured constant is NOT shipped as a
term in this prereg — it was derived from the two arms that failed here, and fitting on failures
is the thing this project refuses. It becomes **prereg #74**, tested out-of-sample on a different
model before any prediction depends on it. Register: **L-19** (the law's scope, measured),
**U-25** (the term, staked separately).
