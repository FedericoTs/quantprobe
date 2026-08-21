# The Atlas — every machine the law has been scored on

quantprobe predicts tok/s from bandwidth and bytes. Every prediction is falsifiable, and this
page is where predictions meet machines. **One command turns your box into a datapoint:**

```bash
pip install quantprobe
quantprobe bench --gguf your-model.gguf --contribute
```

It prints predicted-vs-measured and a pre-filled issue link. **You review and submit — nothing
is ever sent automatically.** Points that land *outside* the predicted band are the most
valuable ones; misses are published at the same size as hits, and several of the tool's best
corrections started as someone else's contradicting measurement.

## Scored so far

| machine | source | result |
|---|---|---|
| GTX 1060 6GB + 16GB DDR4 (2016 desktop) | ours, reference box | 14-model ladder, **8.4% median error**; every number in the repo reproduces here on demand |
| RTX 3090 + 117.6B MoE | community replication (register E-06) | exposed **five real defects**, all fixed in v1.19 with tests named after the report; `calibrate` exists because of it |
| DGX Spark (third-party reports) | out-of-sample retrodiction | 32B dense **1.27× low**, 30B MoE **1.09× low** — inside the documented floor band; the Gemma-26B "violation" dissolved once its 480 KB/pos KV was priced at the reporter's context |
| airllm hosts (third-party reports) | retrodiction | the unexplained **30× spread** (0.07–2 tok/s) resolves as a RAM/disk tier boundary |
| 1.56 TB Kimi K3 rig (third-party) | retrodiction | laptop preset lands within **0.3%** under the corrected trunk-dominated byte model; the relayed "10 tok/s" was a 200–320× unit inversion |
| **RTX 3090 @ 250W, Qwen3.8-27B W4A16 on vLLM** | third-party ([syv-ai](https://github.com/syv-ai/qwen38-27b-rtx3090)) | out-of-sample retrodiction, **and the floor holds**: we predict **25.0 tok/s** all-in-VRAM for 27B at 5.78 effective bits; their published non-speculative single-stream baseline is **46 tok/s** — **1.84×**, against a documented band of "≥0.90× every time, typically 1.1–1.8× higher". The 14th test of that one-sided claim, and the first on a **different inference stack** (vLLM, not llama.cpp) and a **hybrid linear-attention** model. 1.84× is a new maximum for the observed range — noted, not smoothed. Separately **confirms D-10 across implementations**: they measure **381 tok/s on document reproduction vs ~133 on ordinary chat** with the same stack, which is the copyability mechanism we measured with ngram-simple on a GTX 1060, reproduced with DFlash2 on Ampere |
| **AMD RX 5700 XT 8GB** (RDNA1/gfx1010, Vulkan, Win11) | community, `--contribute` ([issue #1](https://github.com/FedericoTs/quantprobe/issues/1)) | **the most accurate prediction in this table: 73.1 predicted vs 73.18 ± 0.16 measured (+0.1%)** on Qwen2.5-7B Q4_0, all in VRAM — and the first non-NVIDIA point. Carried two independent confirmations for free: **Q4_0 +18.4% over Q5_K_M** (61.8 → 73.18), extending the format lever (L-15/V-17) from Pascal to RDNA1, and ngram-simple measuring **no gain on novel generation** exactly as D-10 predicts. Also exposed the bug in its own title — the payload reported `total=None active=None` — now fixed and pinned by two tests |

## What your datapoint settles

The register names its open questions, and they are hardware-starved, not idea-starved:

- **Ampere+ kernel boundary** — the batch-width cliff (8→9 on Pascal: 54 → 176 aggregate
  tok/s) sets both serving advice and the speculation draft rule. One `llama-batched-bench`
  sweep on a 30/40/50-series card locates your generation's boundary.
- **GPU-resident η** — all-in-VRAM predictions are a documented floor (real speed ≥0.90× on
  13/13, typically 1.1–1.8× higher). Your calibrated bench turns the floor into a point.
- **Mac / 50-series presets** — currently extrapolated, honestly labelled. One bench each.
- **exllamav2 A2A arm** — our depth-aware-vs-uniform benchmark (−13.2% ppl at equal bytes,
  staked) needs its exllamav2 comparison on hardware that runs it; Pascal does not.

Every contribution gets a row here, linked to its issue.
