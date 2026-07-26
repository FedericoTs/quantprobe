# Pre-registration #19: batch size is a placement variable, and our search is blind to it

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## Why this is the first research task

`plan.evaluate()` takes `(t, a, ne, moe, bits, vc, vb, rc, rb, db, geta, ctx, kvp, n_layer)`.
**Batch size is not an input.** The law answers "which placement wins for single-stream decode",
while presenting itself as "the fastest way to run this model" — and those are different questions
whenever a prompt is longer than one token.

We already have measured evidence that batch changes the answer, we just never wired it in:
partial expert offload gives **~2–3× on prefill and ~+12% on decode** from the *same flags on the
same file* (pre-registration #13). The only thing that differs is how many tokens a weight read
serves.

## The mechanism

Reading `ggml/src/ggml-backend.cpp` and `ggml-cuda.cu` in our pinned build (b10098): an op whose
weight `src` lives in a **host** buffer is offered to the GPU, and CUDA accepts it when the op's
batch size clears a threshold (32 by default). For `MUL_MAT_ID` — the MoE expert matmul — that
batch size is the number of tokens in the **ubatch**.

So with `-ot exps=CPU`, expert weights cross PCIe **once per ubatch**, not once per token. The
per-token transfer cost is therefore `CPU_bytes / (BW · ub)` — it falls as `ub` rises. That is a
different mechanism from anything the law currently models, and it is why PCIe can matter at all
on this box (single-token decode over PCIe loses to CPU-side compute by 4.3×; see today's
analysis).

## Why our own prior miss does not settle this

Law 5's **H3b** tested `-ub 1024` and missed — but it tested it at **`-ngl 99` on a dense 7B fully
resident in VRAM** (`LAW5_PROTOCOL.md:118`). With zero host-resident weights the offload path
never fires, so H3b measured buffer-squeeze, not transfer amortization. The cell where this
mechanism lives — **CPU-resident experts** — has only ever been run at the default `ub=512`.

Stating this explicitly because "we already tested that and it missed" is exactly the kind of
recollection that has been wrong before in this project.

## Stakes

`Qwen3-30B-A3B-Q2_K` (10.48 GiB, 2.95 bpw), `-ngl 99 -ot "exps=CPU" -mmp 0`, on the reference box.
`llama-bench`, warm-up discarded, r=3, GPU memory **and** temperature logged before and after
(the thermal convention added yesterday).

- **P-1 (the lever).** `pp2048` at `-ub 2048` beats `-ub 512` by **≥1.5×**.
- **P-2 (the double dissociation — the real test of the mechanism).** The same sweep on a model
  **fully resident in VRAM** (`Qwen2.5-7B Q4_K_M`, `-ngl 99`, no `-ot`) shows **≤1.05×** — no
  meaningful gain, and plausibly a regression. If raising `ub` helps *both*, the explanation is
  generic compute batching rather than host-to-device amortization, and the mechanism claim fails
  even if P-1 holds.
- **P-3 (it is a prefill lever, not a decode one).** `tg128` at `-ub 2048` lands within ±5% of
  `-ub 512`, or worse. A gain here would mean I have misunderstood the mechanism, since decode
  generates one token per step and cannot fill a ubatch.
- **P-4 (no anchor moves).** This adds a flag; it does not touch the law. All four published
  anchors must retrodict bit-identically.

## Refuted if

P-1 fails, or P-2 shows the same gain on the VRAM-resident control. Either outcome means batch is
not the placement variable I claim it is on this hardware.

## What ships if it holds

Not a law change — **a search dimension.** `plan.evaluate()` stays frozen, so no published anchor
can move (the `optimize.py` pattern: a search layer over a frozen law). The planner would emit
`-b`/`-ub` **only** on placements that leave weights in host RAM, never as a blanket default,
and only up to a VRAM headroom ceiling: raising `ub` grows the compute buffer, and
pre-registration #13 measured a **−29% cliff** one step past the VRAM ceiling. A prefill win that
trips that cliff is a net loss for a chat user, which is what P-3 exists to check.

**Known blocker to fix first:** `runtime.py` forwards only `-ngl`, `-ot` and `--mmap` into
`llama-bench`. Any new flag added to the plan's flag string is silently dropped from `bench`, so
predicted-vs-measured would drift by exactly the size of the new lever. The forwarder must be
fixed before this ships, or the validation loop stops validating.
