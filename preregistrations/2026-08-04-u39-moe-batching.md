# U-39 — does batching survive expert offload? (staked before the sweep)

**Date staked:** 2026-08-04, before any batched measurement of a MoE on this box.
**Context:** U-38 measured dense-7B-in-VRAM aggregate decode jumping from ~54 tok/s (N=8) to
219.4 tok/s (N=32) — 9.5× single-stream — overturning C-06. That result is one model on one
placement. This stake asks whether the headline 30B MoE, run the way this repo actually
recommends it (attention+KV in VRAM, experts in system RAM), shows the same behaviour.

## Why it plausibly does NOT

On the dense-in-VRAM placement, batching amortises one weight-read across N streams and the
freed resource is idle compute. On the expert-offload placement, ~half or more of every token is
**system-RAM reads of routed experts**, and different streams route to different experts each
step — so expert bytes may scale with N instead of amortising, and DDR4 has no idle headroom to
absorb it. If that mechanism is right, the MoE's aggregate curve should be far flatter than the
dense one.

## Protocol

`llama-batched-bench` on Qwen3-30B-A3B Q2_K (11.3 GB), placement `-ngl 99 -ot
"blk\..*\.ffn_.*_exps\.=CPU" --no-mmap --threads 4`, `-c 8192 -b 2048 -ub 512`, `-npp 64 -ntg
128`, `-npl 1,2,4,8,16,32`. Quiet box, one session, no other llama processes (C-14).

Sanity anchor in the same session: dense 7B N=16 must reproduce ≥150 aggregate tok/s, or the
session cannot score (harness must be able to show the dense jump before claiming the MoE lacks
one).

Secondary, exploratory, NOT staked: dense 7B at N=9,10,11,12 to locate the jump edge.

## Staked bands

- **P1 (central):** MoE aggregate(8)/aggregate(1) lands in **[1.0, 2.5]** — batching helps
  weakly or not at all under expert offload.
- **P2:** no dense-style jump: MoE aggregate(32)/aggregate(16) **< 1.5**.
- **KR-A (refutation, the happy miss):** if MoE aggregate(32)/aggregate(1) **> 6**, the
  "RAM-tier expert reads kill batching" mechanism is wrong, MoE batches like dense, and a 2016
  desktop is a legitimate multi-user 30B server. That outcome overturns this stake's central
  model and must be published as such.
- **KR-B (validity):** the dense anchor must hit ≥150 aggregate at N=16 in the same session,
  else no verdict either way.
- **KR-C (validity):** VRAM must not spill (no partial offload of attention); if the KV for 32
  streams forces layers off-GPU, score only the N range where placement held.

## What each outcome changes

- P1+P2 hold → the planner's future batch axis needs a *placement-dependent* batching term, and
  serving advice inverts with user count: one user → 30B MoE; many users → dense-in-VRAM.
- KR-A fires → the 1060 headline becomes "serves N users on a 30B", and the batching term may be
  placement-independent above the kernel switch.
