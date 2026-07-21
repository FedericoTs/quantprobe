"""dense_axiom_probe.py -- pre-test AXIOM A1 for the Gemma-4-12B port WITHOUT the gated weights.

A1: a DENSE model's MLP bulk is 2-bit-compressible near the Gaussian RD floor (rel-MSE = D(R=2) = 0.069),
the property that made the MoE carve-out work. We test it on the DENSE tensors already local in
DeepSeek-V2-Lite -- the shared-expert MLP, the dense layer-0 MLP, and attention -- as a proxy for dense
Gemma. Same trellis codec as the paper. Data-free, a handful of tensors (fast).

VALIDATE: dense MLP rel-MSE ~0.07-0.09 with kurtosis ~0 -> A1 holds, green-light the Gemma weights.
          rel-MSE >> 0.12 or heavy tails / low flatness -> dense 2-bit is at risk, reconsider before the port.
"""
from __future__ import annotations
import sys
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import weights.evoq_moe as em

FLOOR = 0.069  # D(R=2), the routed-expert measured floor


def probe(name, W):
    try:
        W = np.ascontiguousarray(W.astype(np.float32))
        wh, _ = em.trellis_quant(W, K=2)
        wh = np.asarray(wh, np.float32).reshape(W.shape)
        relmse = float(((W - wh) ** 2).sum() / (W ** 2).sum())
        f = W.reshape(-1)
        kurt = float(((f - f.mean()) ** 4).mean() / (f.var() ** 2) - 3)
        s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
        p = s ** 2 / (s ** 2).sum()
        flat = float(np.exp(-(p * np.log(p + 1e-30)).sum()) / len(s))
        tag = "AT FLOOR" if relmse < 0.09 else ("ok" if relmse < 0.12 else "FRAGILE")
        print(f"  {name:34s}: 2bit rel-MSE={relmse:.4f}  kurt={kurt:+.2f}  flatness={flat:.3f}  [{tag}]", flush=True)
        return relmse
    except Exception as e:
        print(f"  {name:34s}: FAILED ({e})", flush=True)
        return None


def main():
    smap = em.shard_map()
    g13 = lambda k: em.read_tensor(smap, f"model.layers.13.{k}.weight").float().numpy()
    g0 = lambda k: em.read_tensor(smap, f"model.layers.0.{k}.weight").float().numpy()
    print(f"AXIOM A1 PROBE -- is the DENSE bulk 2-bit-compressible near the RD floor ({FLOOR})?\n")
    print("  [MoE baseline -- known at the floor]")
    probe("routed expert gate (MoE)", g13("mlp.experts.0.gate_proj"))
    probe("routed expert down (MoE)", g13("mlp.experts.0.down_proj"))
    print("  [DENSE proxies for Gemma-4-12B]")
    probe("shared expert gate (DENSE MLP)", g13("mlp.shared_experts.gate_proj"))
    probe("shared expert up   (DENSE MLP)", g13("mlp.shared_experts.up_proj"))
    probe("shared expert down (DENSE MLP)", g13("mlp.shared_experts.down_proj"))
    probe("layer-0 dense MLP gate", g0("mlp.gate_proj"))
    probe("layer-0 dense MLP down", g0("mlp.down_proj"))
    probe("attention q_proj (dense)", g13("self_attn.q_proj"))
    probe("attention o_proj (dense writer)", g13("self_attn.o_proj"))
    print("\n  VALIDATE: dense MLP near the floor -> A1 holds for dense Gemma; else dense 2-bit is at risk.")


if __name__ == "__main__":
    main()
