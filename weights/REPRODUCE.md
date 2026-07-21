# Reproducing the results

All perplexities are on the **full WikiText-2 test set** (standard GPTQ protocol: full
`wikitext-2-raw-v1` test split, the model's own tokenizer, non-overlapping 2048-token
windows, PPL = exp(ΣNLL / Σtokens)).

## Environment
- **GPU**: any CUDA GPU with ≥6 GB. Developed on an NVIDIA GTX 1060 6 GB (Pascal, no tensor
  cores); the streaming evaluation is bounded to fit 6 GB.
- **Python** with `torch`, `transformers`, `safetensors`, `numpy`, `matplotlib` (see `.venv/`).
- **Models** (HuggingFace fp16/bf16 safetensors): DeepSeek-V2-Lite and Qwen1.5-MoE-A2.7B.
  Paths are `MDIR` in `evoq_moe.py` and `EVOQ_QWEN_DIR` for Qwen.
- **Data-free**: no calibration set is required. The optional AWQ variant uses 512
  WikiText-2-*train* tokens (making it *data-light*, not data-free).

## Key files
| file | role |
|---|---|
| `evoq_moe.py` | DeepSeek harness: trellis codec, streaming eval, mixed-precision `k_for()`, modes `baseline`/`measure`/`validate` |
| `evoq_moe_qwen.py` | Qwen1.5-MoE port (reuses the codec via monkeypatch) |
| `forced_routing.py` | §4: force fp16 routing in the uniform-2bit model |
| `forced_output.py` | §4: keep a tensor group at fp16 (writers vs internals) |
| `m1_kvlatent.py` | confirmatory: drop only KV-latent to 2-bit *inside* the carve-out |
| `route_locality.py` | §7: routing temporal-locality measurement |
| `make_rd_figs.py`, `make_routing_fig.py`, `make_generality_fig.py`, `make_vram_fig.py` | figures F1/F4, F2, F5, F3 |

## Environment variables
`EVOQ_NWIN=200` → evaluate the full test set (clamps to ~151 windows DeepSeek / ~146 Qwen).
`EVOQ_CACHE=1` → reuse the per-layer quantized-weight cache (omit for a from-scratch re-quant).
Bit-widths: `EVOQ_ATTN_K`, `EVOQ_SHARED_K`, `EVOQ_DENSE_K` (attention/shared/dense MLP),
`EVOQ_DOWN_K` (routed-expert down_proj); routed gate/up are fixed at 2-bit. `EVOQ_INT8_GS=1`
enables int8 group side-info. `EVOQ_AWQ=1 EVOQ_AWQ_STATIC=1` enables one-pass static AWQ.

## Commands (→ expected full-set ppl)
```
# fp16 baseline                                              -> 6.307  (matches SINQ's BF16 6.31)
EVOQ_NWIN=200 python -m weights.evoq_moe baseline

# carve-out (headline): 4-bit attn+shared, 2-bit experts, 3-bit down  -> 6.962 (gap-ratio 1.104x)
EVOQ_DOWN_K=3 EVOQ_ATTN_K=4 EVOQ_SHARED_K=4 EVOQ_DENSE_K=4 EVOQ_INT8_GS=1 EVOQ_CACHE=1 EVOQ_NWIN=200 \
  python -m weights.evoq_moe measure

# uniform 2-bit (all tensors)                                -> 18.315
EVOQ_DOWN_K=2 EVOQ_ATTN_K=2 EVOQ_SHARED_K=2 EVOQ_DENSE_K=2 EVOQ_INT8_GS=1 EVOQ_CACHE=1 EVOQ_NWIN=200 \
  python -m weights.evoq_moe measure

# data-light AWQ                                             -> 6.768 (gap-ratio 1.073x)
EVOQ_AWQ=1 EVOQ_AWQ_STATIC=1 EVOQ_DOWN_K=3 EVOQ_ATTN_K=4 EVOQ_SHARED_K=4 EVOQ_DENSE_K=4 \
  EVOQ_INT8_GS=1 EVOQ_CACHE=1 EVOQ_NWIN=200 python -m weights.evoq_moe measure

# §4 causal decomposition (recompute % vs full-set fp16=6.307 / uniform=18.315)
EVOQ_NWIN=200 python -m weights.forced_routing                                              # routing  -> 21%
EVOQ_KEEP_FP16="o_proj,down_proj" EVOQ_NWIN=200 python -m weights.forced_output             # writers  -> 49%
EVOQ_KEEP_FP16="q_proj,kv_a_proj_with_mqa,kv_b_proj,gate_proj,up_proj" EVOQ_NWIN=200 \
  python -m weights.forced_output                                                           # internal -> 95%
EVOQ_KEEP_FP16="kv_a_proj_with_mqa,kv_b_proj" EVOQ_NWIN=200 python -m weights.forced_output # kv-latent-> 87%

# Qwen generality
EVOQ_NWIN=200 python -m weights.evoq_moe_qwen baseline                                       # fp16     -> 7.217
EVOQ_DOWN_K=3 EVOQ_ATTN_K=4 EVOQ_SHARED_K=4 EVOQ_INT8_GS=1 EVOQ_CACHE=1 EVOQ_NWIN=200 \
  python -m weights.evoq_moe_qwen measure                                                    # carve-out-> 7.749
```

> **Note on `forced_output.py`**: it prints *hardcoded* subset baselines (fp16=5.66, uniform=15.38);
> its intervention ppl is at the requested `EVOQ_NWIN`. Recompute the recovered-% against the full-set
> baselines (fp16=6.307, uniform=18.315): `% = (uniform − intervention) / (uniform − fp16)`.

## Verification performed
- **Audit** of every paper number against `data/moe_results.txt` and `data/full_*.log`.
- **From-scratch re-quant** of the carve-out (no cache) → 6.2483 (NWIN=8), bit-identical to cache.
- **Gap-ratio invariance** verified across the 16–32k-token subset → full test set (1.105×→1.104×).

## Cross-paper comparison
MxMoE (arXiv:2505.05799) is the only prior work reporting a ~2-bit DeepSeek-V2-Lite WikiText-2 number
(fp16 5.92 / 2.25-bit 7.01), but at **seqlen 4096** (its harness default). We compare via gap-ratio,
which cancels the context-length difference. MC-MoE evaluates Mixtral; EAQuant evaluates DeepSeek-MoE-16B
(a different model); SINQ reports no ~2-bit point (its BF16 6.31 corroborates our full-precision baseline).

---

# Campaign manifest (round 2–4): every headline claim → its script and log

| claim | script | log |
|---|---|---|
| Rank-conditional rotation (270,000×) | m1_gauge.py, m1_dichotomy.py | gauge_*.log, dichotomy_*.log |
| Density: expert RD floor, 1-bit collapse | expert_bits.py | expertbits_*.log |
| Density: routing flat / domain-flat (Jaccard 1.00) | router_confidence.py, task_trim_probe.py | task_trim.log |
| Density: activations diffuse | gemma_semantic_probe.py | gemma_semantic.log |
| Gemma depth inversion + 1.91×→1.45× | evoq_gemma.py (bands + flipped) | gemma_band_*.log, gemma_flipped.log |
| Fragility atlas (Qwen late, Mistral early 25×) | dense_2bit_gate.py, atlas chains | qwen_mech_probe.log, mistral_atlas.log |
| Byte-identical A/B/C in llama.cpp | GGUF_DEPTH_RECIPE.md commands | gemma_ggml_abc.log |
| 30B depth win, data-free vs calibrated | q30 chains | q30_bands.log, q30_UD.log |
| Tiered decode law + η fit | scaling_law.py | colibri_probes.log, world_chain.log |
| exps=CPU +54%; 30B 12.6 CPU-only; batch 4.5× | llama-bench commands in logs | colibri_probes.log, qwen30b_t4.log |
| Lookahead 91% / prefetch 99% | lookahead_probe.py | world_chain.log |
| 110B from SATA at 0.19 (pre-registered) | hf_fetch.py + llama-bench | glm_bench.log |
| XMP ×1.52 dense (pre-registered), constraint migration | llama-bench (warm-up protocol) | xmp_definitive.log |
| Speculation × MoE antagonism (2.3× slower) | llama-server A/B | spec30.log |
| E2c refutation + memory-system diagnosis | fork patch (GGML_MMID_EM) | e2c_ab.log, thread_scaling.log |

**Bench protocol:** warm-up pass discarded (Defender first-read scans + cold cache), then ≥3 reps;
never bench two >8 GB models back-to-back on a 16 GB box; post-reboot numbers lie for ~10 minutes.
