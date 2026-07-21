# §4 Causal decomposition — canonical record (full WikiText-2 test set, 151 windows)

The clean, reviewer-facing record of the §4 decomposition. All recovery percentages are computed
against the **full-set** baselines, matching the paper:

```
fp16 = 6.3070   uniform-2bit (all tensors) = 18.3145   collapse = 12.0075
recovery% = (uniform - intervention) / (uniform - fp16)
```

| Intervention (keep this group at fp16 in the all-2-bit model) | intervention ppl (151 win) | recovery % |
|---|---:|---:|
| force fp16 routing (replay fp16's expert choices) | 15.7445 | **21%** |
| residual writers (o_proj, down_proj) | 12.4564 | **49%** |
| internal projections (q/kv, gate/up) | 6.8951 | **95%** |
| &nbsp;&nbsp;— MLA KV-latent alone (kv_a, kv_b) | 7.8825 | **87%** |
| gate/up | 14.4950 | 32% |
| q_proj | 15.5876 | 23% |

**Stability across evaluation length (KV-latent):** 86% on the 16k-token subset (NWIN=8) → **87%** on the
full test set (NWIN=151). The ranking is identical at both lengths; the finding is not eval-length-dependent.

**The load-bearing evidence is the in-regime control, which does NOT depend on this marginal
decomposition or its baseline:** dropping the two KV-latent tensors to 2-bit *inside* the deployed
carve-out costs **+5.27 ppl** (6.9616 → 12.2320) — vs +0.64 for the residual writers, ≈8× more, ≈46% of
the carve-out's entire advantage over uniform, from two tiny tensors. (`m1_kvlatent.py`)

**Parameter-matched control** (rules out "small/low-rank tensors are just fragile"): dropping a
KV-*parameter-sized* row-slice of the full-rank q_proj to 2-bit inside the carve-out — see
`m1_param_control.py` and `moe_results.txt` for the measured cost vs the KV-latent's +5.27.

---

### ⚠ Note on the raw `forced_output.py` logs
Historically `forced_output.py` printed a **hardcoded NWIN=8 baseline** (uniform 15.3798, fp16 5.6570),
so its `-> recovers X%` line is **wrong for a full-set run** (it mixes a 151-window intervention ppl with
an 8-window baseline — e.g. it printed kv-latent "77%" where the correct full-set value is 87%). The
table above recomputes every row against the correct full-set baseline. The script now reads
`EVOQ_UNI_PPL` / `EVOQ_FP16_PPL`, so re-running with `EVOQ_UNI_PPL=18.3145 EVOQ_FP16_PPL=6.3070` prints
the correct percentages directly. **This is a logging artifact only — the paper's numbers are correct.**

### Reproduce
```
EVOQ_NWIN=200 python -m weights.forced_routing                                              # routing  -> 21%
EVOQ_UNI_PPL=18.3145 EVOQ_FP16_PPL=6.3070 EVOQ_KEEP_FP16="o_proj,down_proj" EVOQ_NWIN=200 \
  python -m weights.forced_output                                                            # writers  -> 49%
EVOQ_UNI_PPL=18.3145 EVOQ_FP16_PPL=6.3070 EVOQ_KEEP_FP16="q_proj,kv_a_proj_with_mqa,kv_b_proj,gate_proj,up_proj" \
  EVOQ_NWIN=200 python -m weights.forced_output                                              # internal -> 95%
EVOQ_UNI_PPL=18.3145 EVOQ_FP16_PPL=6.3070 EVOQ_KEEP_FP16="kv_a_proj_with_mqa,kv_b_proj" EVOQ_NWIN=200 \
  python -m weights.forced_output                                                            # kv-latent-> 87%
EVOQ_INT8_GS=1 EVOQ_NWIN=200 python -m weights.m1_kvlatent                                    # in-regime drop -> +5.27
EVOQ_INT8_GS=1 EVOQ_NWIN=200 python -m weights.m1_param_control                               # param-matched control
```
