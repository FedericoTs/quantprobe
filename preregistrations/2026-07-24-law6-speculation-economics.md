# Pre-registration #10: Law 6 candidate — speculation economics (decode as a bandwidth arbitrage)

**Author:** Federico Sciuca · **Date staked:** 2026-07-24, committed BEFORE any speculative-decoding
run has ever been executed on this machine.
**Status: STAKED. Scoring planned in public during the week of 2026-07-27.**

## The claim under test

Speculative decoding is a known trick (llama.cpp ships it stock). What nobody prices is its
**economics**: which setup pays, on which hardware, for which workload. Our laws say the answer is
computable. Decode is bandwidth-bound on commodity tiers (Law 4); verifying k drafted tokens in one
batch costs roughly ONE weight-read (Law 5's batch amortization applied to decode). Therefore the
speedup ≈ expected tokens emitted per target read, discounted by draft cost — and it should be
**largest exactly where decode is most bandwidth-starved**. If that holds, speculation becomes a
priced knob in the machine profile, not folklore.

## Conventions (fixed before any run)

- llama.cpp b10098, llama-server, `-np 1`, temperature 0; effective tok/s from server timings.
- GPU state (`nvidia-smi memory.used`) logged before every batch — binding since 2026-07-24
  (see LAW5_PROTOCOL.md correction).
- **W-code** workload: a real 2k-token source file from this repo (quantprobe/plan.py head) plus an
  instruction to make a small localized edit — high copy-rate, the agent case.
- **W-prose** workload: WikiText continuation — low copy-rate, the control.
- No speculative flag (`--spec-type`, draft models, ngram modes) has been run on this box before
  this commit; baselines cited below were measured 2026-07-24 without speculation.

## Stakes

- **S-a (ngram / prompt-lookup, dense).** qwen7b-Q4_K_M at ngl99, baseline decode 21.65 tok/s
  (measured, server path, short ctx): stock ngram speculation on W-code → **×1.25–2.0** effective;
  on W-prose → **≤ ×1.15**. The gap IS the mechanism (copyability-driven); if prose gains as much
  as code, the lookup model is wrong and gets published as such.
- **S-b (the MoE union tax transfers to verify batches).** Qwen3-30B-A3B Q2_K hybrid
  (`-ngl 99 -ot exps=CPU`, baseline 19.9–20.0 short-ctx): same ngram setup on W-code. Drafted-token
  batches union experts across the CPU boundary, eating the amortization — Law 5's mechanism
  cross-applied to decode. Staked: the MoE's speed gain fraction lands at **≤ 0.75×** the dense
  case's, i.e. (S_moe − 1) ≤ 0.75 × (S_dense − 1). If S_moe ≈ S_dense, the transfer claim is
  refuted.
- **S-c (tiny-draft pays on a same-family pair).** Qwen2.5-0.5B-Instruct-Q8_0 drafting
  Qwen2.5-7B-Instruct-Q4_K_M, both resident on the 6 GB card: staked **×1.2–1.8** on W-code vs the
  7B-Instruct's own no-spec baseline (baseline measured first, same session; the stake is the
  ratio). Below ×1.1 = draft overhead swamps the win on 6 GB-class hardware.
- **S-d (the tier signature — second-round stake).** The law predicts the relative-speedup ordering
  **CPU-pure ≥ hybrid ≥ full-GPU** at matched workload (the more bandwidth-bound the tier, the more
  a verified batch is worth). Deliberately staked as an ordering only; exact bands will be staked
  after S-a/S-b/S-c land and BEFORE the tier runs, protocol-style.

## Refuted if

S-a lands outside its band or W-prose ≈ W-code; S-b ratio > 0.9; S-c < ×1.1; S-d ordering
inverted. Misses publish with the same prominence as hits — they are the point.

## If it survives

η_spec enters the planner as a priced column; speculation joins placement, format, KV-policy and
persistence as a decided atom in the per-machine profile; and the demo writes itself: the same
2016 desktop, measurably faster, **predicted first**.
