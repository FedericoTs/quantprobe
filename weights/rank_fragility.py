"""rank_fragility.py -- test the RANK-ROBUSTNESS DUALITY (the candidate master law).

Hypothesis: a tensor's quantization fragility is governed by the SPECTRAL CONCENTRATION of the map it
implements -- low effective-rank (a bottleneck, few singular directions) => quantization noise has no
redundant directions to average into => fragile; high effective-rank (a flat spectrum, e.g. an expert)
=> noise averages out => robust AND incompressible (maximal entropy). If the data-free spectral-flatness
H/n anti-correlates with the MEASURED fragility (KV-latent 87% / +5.27 in-regime; q-slice +0.12; experts
~free), then effective rank is the master variable and the rho^2 saturation law has its data-free predictor.
Data-free, CPU-only (a few SVDs).
"""
from __future__ import annotations
import sys
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import weights.evoq_moe as em


def spectrum(W):
    s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
    s2 = s ** 2
    p = s2 / (s2.sum() + 1e-30)
    H = float(np.exp(-(p * np.log(p + 1e-30)).sum()))   # entropy effective rank (flat spectrum -> ~n)
    stable = float(s2.sum() / (s2.max() + 1e-30))        # stable rank
    return H, stable, len(s)


def main():
    smap = em.shard_map()
    li = 13
    def g(k):
        return em.read_tensor(smap, f"model.layers.{li}.{k}.weight").float().numpy()
    items = [
        ("q_proj (full-rank attn)", "self_attn.q_proj"),
        ("o_proj (residual writer)", "self_attn.o_proj"),
        ("kv_a (down->latent)", "self_attn.kv_a_proj_with_mqa"),
        ("kv_b (latent->heads)", "self_attn.kv_b_proj"),
        ("expert0.gate_proj", "mlp.experts.0.gate_proj"),
        ("expert0.down_proj", "mlp.experts.0.down_proj"),
    ]
    T = {}
    for name, k in items:
        try:
            T[name] = g(k)
        except Exception as e:
            print(f"  (skip {name}: {e})")
    ka, kb = T.get("kv_a (down->latent)"), T.get("kv_b (latent->heads)")
    if ka is not None and kb is not None:
        T["kv COMPOSED (the bottleneck)"] = kb @ ka[:512]    # effective K/V map through the 512 latent

    print(f"RANK-FRAGILITY (DeepSeek-V2-Lite layer {li}) -- does spectral flatness predict fragility?\n")
    print(f"  {'tensor':30s}  eff_rank  stable_r  nominal   H/n (flatness)")
    for name, W in T.items():
        H, st, n = spectrum(W)
        print(f"  {name:30s}  {H:7.1f}  {st:7.1f}  {n:6d}   {H/n:6.3f}")
    print("\n  MEASURED fragility (lower flatness should mean higher fragility):")
    print("    KV-latent (kv_a+kv_b / composed) = 87% of collapse, +5.27 ppl in-regime")
    print("    q_proj KV-sized slice            = +0.12 ppl (43x less, same param count)")
    print("    residual writers (o/down)        = 49%")
    print("    routed experts                   = ~free (rel-MSE = D(R=2), maximal entropy)")


if __name__ == "__main__":
    main()
