# Pre-registration #69: the dense-SPLIT draft cell — where verify batches amortize instead of multiply

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the download completes. **STAKED.**

## The mechanism bet

#67/#68 mapped speculation's economics into two measured regimes: dense all-in-VRAM (+11% — the
verify batch is nearly free on GPU) and MoE split (0.74–0.81× — the verify batch multiplies CPU
expert bytes via the union tax). This cell is the third and last regime: a DENSE model split
across GPU/CPU. Mechanism prediction: a K+1-token verify batch reads each CPU-resident layer's
weights ONCE for the whole batch — the same host-transfer amortization measured for `-ub`
(#19/#20, +73–75%) — so the CPU share of the token (the majority, for a 14B on this box)
divides by the accepted-tokens-per-round. **Dense split should be the BEST speculation cell,
not the worst**, the mirror image of the MoE result.

Model: Qwen2.5-14B-Instruct Q4_K_M (~9 GB, dense, same family as the 0.5B drafter — vocab ✓).
Downloading via `quantprobe fetch` as this is staked.

## Stakes

- **P-1 (baseline sanity).** The no-draft baseline lands within the tool's printed ±25% band of
  whatever `plan --gguf` predicts for its emitted placement (captured before the bench).
- **P-2 (THE MECHANISM).** With the 0.5B draft, novel code, best K in {2,3,4,6}: **≥ 1.30×**
  the same-config no-draft baseline. (The 7B-AIV cell managed 1.11×; the amortization mechanism
  predicts MORE here despite the slower target, because the CPU share is what amortizes.)
- **P-3 (the K-shift).** The optimal K is **HIGHER than 2** (the AIV optimum): amortization
  rewards longer drafts, so the curve should peak at K=3–6, not K=1–2. This tests the mechanism
  shape, not just the magnitude.
- **P-4 (the quality point, not a speed record).** The drafted 14B stays BELOW the 7B's raw
  22.6 tok/s — stated in advance so nobody reads this as a frontier claim. The prize is 14B
  intelligence at usable speed: a new row for the `target` speed-intelligence ladder.

## KILL RULE

**If P-2 fails (< 1.30×), the amortization mechanism does not survive contact with the dense
split** and the speculation map closes with all three regimes measured ≤ 1.11× — novel-text
speculation on this box is then bounded at +11% everywhere, full stop, recorded as a law-grade
scope statement. If P-3 fails but P-2 holds, the win is real but the mechanism story needs
revision — say so, don't hand-wave.

**Wired into:** `dense_draft_note` (plan.py) — the dense-split branch carries the measured +33%
K=2 CPU-draft advice; the speculation map in D-09 now has all three regimes measured.

---

## SCORED — 2026-07-28, same day

Raw log: `weights/data/prereg69_dense_split.log`. Baseline (llama-bench, emitted config
`-ngl 28 -t 4`): **5.54 ± 0.03 tok/s**. All speculative arms: llama-speculative, temp 0, novel
code prompt, draft **on CPU** (`-ngld 0` — VRAM had no room, which turned out to matter).

| K | acceptance | tok/s | vs baseline |
|---|-----------|-------|-------------|
| 1 | 85.6% | 7.19 | 1.30× |
| **2** | **76.3%** | **7.31 / 7.46 / 7.43 (mean 7.40)** | **1.335×** |
| 3 | 69.8% | 3.54 / 3.54 | 0.64× |
| 4 | 64.4% | 4.43 | 0.80× |
| 6 | 56.3% | 3.92 | 0.71× |
| 8 | 46.4% | 5.57 | 1.01× |

- **P-1 HIT.** Tool predicted 4.8 for the emitted placement; measured 5.54 (tool −13%, inside ±25%).
- **P-2 HIT.** Best staked K (2): **1.335×** ≥ 1.30×, robust across three runs (1.32/1.35/1.34).
  Dense split IS the best speculation cell, as the mechanism bet predicted: 1.335× > 1.11× (AIV,
  #67) > 0.74–0.81× (MoE split, #68).
- **P-3 MISS — mechanism story revised, per the kill rule's own terms.** The optimum is K=2, not
  K=3–6; every K≥3 arm lands at or below baseline. The staked amortization story ignored two
  costs. (1) On a split config the drafter has nowhere to live in VRAM, so it runs on CPU — and
  each drafted token is a serial 0.5B pass paid from the SAME DDR4 bandwidth pocket the verify
  amortization saves. (2) Acceptance decays fast on this pair (85.6% → 46.4% from K=1→8), so
  accepted-per-round saturates while per-round cost keeps climbing. The amortization is real —
  it is why the split cell beats the AIV cell at all — but it shares its pocket with the draft.
- **P-4 HIT.** 7.40 < 22.6 (the 7B's raw). This is a quality point: 14B intelligence at 7.4 tok/s
  (+33% over its own best no-draft config), a new speed-intelligence ladder row, at zero VRAM cost.
- **Open anomaly (logged, not load-bearing):** the K≥3 region is deterministically non-monotonic
  (K=3 3.54 < K=4 4.43 < K=8 5.57; K=3 reproduces to ±0.003). Temp-0 speculative decode is
  greedy-equivalent, so this is round-structure cost, not output drift. Unexplained; the
  conclusion (peak at K≤2, cliff after) does not depend on it.
- **Incident, disclosed:** the first three spec runs crashed at draft load ("invalid vector
  subscript") — a tokenizer-metadata mismatch between the bartowski 14B and a June-era 0.5B file;
  `quantprobe fetch` had silently skipped the re-download because a same-named, same-size file
  existed. Fixed by fetching the same-lineage (bartowski) 0.5B. Tool gap noted: fetch has no
  force/verify-hash path (registered U-18).

**Kill rule outcome:** P-2 held, so the speculation map does NOT close at +11% — it closes at
**+33.5%, in the dense-split cell, at K=2, with a CPU-resident same-family draft.**
