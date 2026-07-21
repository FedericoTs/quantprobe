# Depth-aware 2-bit GGUF quantization (llama.cpp recipes + the fragility atlas)

**TL;DR:** at ~2-bit, *where* you spend your extra bits matters more than spending them — but the right
placement is **model-specific and unpredictable from the config**: probe it (30 min), then quantize. On
Gemma 4 12B, protecting the **last 12 layers'** FFNs beats protecting the first 12 — same file size — by
2.25 ppl. On Mistral-7B the fragility is at the **opposite end** (early, by ~25×). Stock `llama-quantize`,
no code changes, no imatrix, data-free.

## The fragility atlas (measured, one 12/8-layer FFN band → Q2_K at a time)

| model | head | MLP | fragile end | band Δppl (early → late) |
|---|---|---|---|---|
| Gemma 4 12B | tied | GeGLU | **late** (~4×) | +2.14 / +3.22 / +3.16 / **+7.98** |
| Qwen2.5-7B | untied | SwiGLU | **late** (~2–3×, U-shape) | +0.88 / +0.54 / +0.58 / **+1.85** |
| Mistral-7B-v0.3 | untied | SwiGLU | **EARLY** (~25×) | **+6.53** / +0.15 / +0.22 / +0.26 |

Qwen and Mistral are architectural near-twins yet point in opposite directions: **no config-file rule
predicts the direction** (not head-tying, not MLP type, not family). Weight statistics don't predict it
either (Gemma's kurtosis points the wrong way). **Only a functional band probe decides** — and it's cheap:
quantize one band's FFNs to Q2_K at a time (rest high-precision), measure perplexity on ~32 chunks, protect
the band that spikes.

## Measured result (llama.cpp b9596, GTX 1060 6GB, WikiText-2 test, 32 chunks, `-ngl 99`)

| variant | flags (FFN placement) | size | PPL |
|---|---|---:|---:|
| A uniform | all FFN Q2_K | 4.73 GB | 14.41 ± 0.43 |
| **B late-protected** | **blk 36–47 FFN @ Q4_K** | 5.22 GB | **10.02 ± 0.28** |
| C early-protected (control) | blk 0–11 FFN @ Q4_K | 5.22 GB | 12.27 ± 0.36 |

B and C are byte-identical — only the *placement* of the protected band differs. The same +0.5 GB buys
**4.39 ppl placed late vs 2.14 placed early**: placement is worth ~2× the budget.

## The recipe (variant B)

```
llama-quantize \
  --tensor-type "blk\.([0-9]|[12][0-9]|3[0-5])\.ffn_.*=q2_k" \
  --tensor-type "blk\.(3[6-9]|4[0-7])\.ffn_.*=q4_k" \
  --tensor-type "attn_.*=q4_k" \
  --token-embedding-type q4_k \
  gemma-4-12B-f16.gguf gemma4-12b-depthaware.gguf Q2_K 8
```

## Why late layers? (the science)

Measured on the fp16 model (trellis codec, then confirmed above in k-quants): Gemma 4's 2-bit fragility is
**~4× higher in the late layers** (+7.98 ppl for 2-bit in blk 36–47 vs +2.14 in blk 0–11, everything else
fp16). Mechanism: at moderate (2-bit) error, late-layer noise feeds the tied-embedding logit head with no
downstream layers to wash it out. This is the *opposite* of the common intuition (and of MoE models in the
catastrophic 1-bit regime, where early error compounds through depth) — and weight statistics (kurtosis)
point the wrong way. **Depth-fragility direction is architecture- and regime-dependent; a 30-minute
functional band-probe finds it, weight statistics do not.**

Part of a larger study: data-free 2-bit quantization laws across MoE and dense architectures
(rank-conditional incoherence, the density of trained networks, depth-fragility inversion). Paper: PAPER_MOE
(md/tex in this repo). All numbers measured on a single 6 GB GTX 1060.

## Reproduce

1. Convert: `python convert_hf_to_gguf.py <gemma-4-12b> --outfile gemma-4-12B-f16.gguf --outtype f16`
2. Quantize with the flags above (llama.cpp ≥ b9596 for `--tensor-type` regex support).
3. Evaluate: `llama-perplexity -m <gguf> -f wiki.test.raw --chunks 32 -ngl 99`

Full logs: `weights/data/gemma_ggml_abc.log`. Band-curve measurement: `weights/data/gemma_band_*.log`.
