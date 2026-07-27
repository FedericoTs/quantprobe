# Pre-registration #21: KV placement is separable from weight placement — and shares the same budget

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## The claim

The law welds KV to the layers it serves: `kv_gb` is added to whichever tier holds the weights,
and `ETA_KV = 0.70` prices it there. But llama.cpp exposes `-nkvo/--no-kv-offload`, which keeps
the KV cache in host memory **independently of where the weights live**. That is a placement
dimension the search does not have.

It matters more now than it did yesterday. Pre-registration #20 established that **placement and
batch compete for one VRAM budget** — the expert split fills VRAM, which starves the large-ubatch
compute buffer and costs 42% of prompt processing. KV is a third claimant on that same budget, and
at depth it is not small: 96 KB/token on this model, so 16k of context is ~1.5 GB — comparable to
the entire compute-buffer headroom the ubatch lever needs.

So the interesting question is not "is CPU-side KV slower" (it must be). It is whether **giving up
KV residency buys back enough VRAM to unlock a lever worth more than it costs.**

## Configurations, `Qwen3-30B-A3B-Q2_K`, reference box

Split placement (`-ot "blk\.(16..47)\.ffn_.*_exps\.=CPU"`), which #20 showed is VRAM-starved.

## Stakes

- **P-1 (the cost is real).** At `-d 16384`, `-nkvo 1` **hurts decode by ≥10%** versus `-nkvo 0`.
  KV moves to a tier ~4× slower and is re-read every token; if this does not show, either the KV
  term or my understanding of the flag is wrong.
- **P-2 (the cost is depth-dependent).** At `-d 0` the same flag is **neutral, within ±3%**. With
  no context there is no cache to read, so placement of an empty thing cannot matter.
- **P-3 (the joint-budget test — the reason this is worth doing).** On the split placement at
  `-ub 2048`, `-nkvo 1` **recovers ≥20% of prompt processing** versus `-nkvo 0`, because the VRAM
  released by evicting KV is exactly what the compute buffer was short of. This is the claim that
  the three claimants are **fungible**: if it holds, the search must optimise them jointly rather
  than pick a placement and then bolt levers on.
- **P-4 (no anchor moves).** A flag, not a law change. All four published anchors bit-identical.

## Refuted if

**P-3 fails.** That would mean VRAM freed from KV does not convert into usable compute-buffer
headroom — the budget is shared but not fungible — and the dimensions can go on being optimised
one at a time. That is a genuinely useful negative result and would simplify the search.

P-1 failing would instead mean I have misread what `-nkvo` does, and P-3 would be uninterpretable.

## What ships if it holds

Nothing automatic. A KV-placement recommendation is **depth-dependent and workload-dependent** —
it trades generation speed at long context for prompt speed — so at most it becomes a disclosed
option alongside the phase advice, in the same shape as v1.13.1: state the trade, name the
alternative command, let the user choose. `plan.evaluate()` stays frozen either way.
