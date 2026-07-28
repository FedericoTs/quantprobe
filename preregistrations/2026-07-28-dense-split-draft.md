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

**Wired into:** pending; `dense_draft_note` / the speculation map scores either way.
