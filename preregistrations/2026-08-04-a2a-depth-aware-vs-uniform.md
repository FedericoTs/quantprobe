# A2A — depth-aware vs uniform quantization, apples to apples (staked before the build)

**Date staked:** 2026-08-04, before the fp16 source finished downloading and before any
quantize, eval, or benchmark run of either arm.
**Why:** a Reddit commenter asked for exactly the right thing: *"same model, same hardware,
same context, compared against llama.cpp"*. The tool's core quality claim (−9% perplexity at
equal bytes; byte-identical files 2.25 ppl apart) predates the current instrument stack. This
prereg packages the claim as one reproducible table with today's instruments, on the exact
model class the community runs.

## Arms

One model, one box, one session (C-14), one context:

- **U (uniform):** `Qwen2.5-7B-Instruct-Q2_K.gguf` — the community-standard llama.cpp uniform
  low-bit quant already on disk (3,015,940,800 bytes).
- **D (depth-aware):** built by `quantprobe probe` (fragile-band measurement on the fp16
  source against `wiki.test.raw`) + `quantprobe quantize`, targeted to land within **±2% of
  U's file size**. Bytes are the budget; placement of protection is the treatment.

Both arms measured identically: perplexity (llama.cpp `--perplexity`), KL divergence vs the
fp16 teacher (`--kl-divergence`, full-distribution + "Same top p"), decode tok/s (same
placement flags), and the 40-task staked business suite via llama-server at 16k.

**exllamav2 arm: not runnable on Pascal.** Stated, not faked — it goes on the
`bench --contribute` ask for an Ampere volunteer.

## Staked predictions

- **P1 (the headline claim, now falsifiable on this model):** D beats U on perplexity at equal
  bytes by **≥4%** (the −9% was measured on a different family; staking the full −9% here
  would be pretending transfer we have not measured).
- **P2:** D beats U on KL divergence (median KLD strictly lower; "Same top p" strictly
  higher). Perplexity can hide token flips (L-27); KLD is the sharp instrument, so P2 is the
  one that matters.
- **P3 (speed invariance):** D and U decode within **±3%** of each other at the same
  placement — speed comes from placement and format class, not from where protection went.
  If D is slower by >3%, the recipe pays a hidden speed tax and the README table gets a
  speed column with the loss printed.
- **P4 (business delta, exploratory — no kill attached):** suite pass-rates recorded for both
  arms. At 7B/2-bit both may fail broadly; the per-cluster split is the data of interest.
- **KR-1 (kill):** if D does NOT beat U on BOTH ppl and median KLD, the depth-aware recipe
  does not transfer to Qwen2.5-7B at ~2 bits, and the README's −9% claim gets a scope
  qualifier naming this miss at equal prominence.
- **KR-2 (validity):** both arms within ±2% file size, same session, same flags, box quiet —
  else no verdict.

## Protocol notes

fp16 source: `Qwen/Qwen2.5-7B-Instruct-GGUF` fp16 4-way split, merged if llama.cpp requires.
Probe/quantize commands and all raw outputs land under `weights/data/a2a_*`. The table ships
in the README's measured-results section win or lose.
