"""expert_subspace.py -- the empirical hinge of the "rearrange the model" idea.

Hypothesis: the 64 routed experts of a layer are NOT 64 independent matrices but live near a
low-dimensional affine subspace -- expert_i ~= backbone (shared mean) + sum_k c_ik * B_k (a few shared
basis matrices). If true, the MoE factorizes as {resident backbone + tiny per-expert coefficients}, which
(a) collapses the per-token read from full experts to a handful of scalars -> raises the bandwidth ceiling
far past the naive active-byte bound (the lever that beats Pascal's memory wall), and (b) unifies with the
project's low-rank-delta codec thread. If FALSE (experts are full-rank / independent), the rearrangement
is lossy and dead -- so this SVD settles it. Data-free, CPU-only (weight loading + a 64x64 eigendecomp).
"""
from __future__ import annotations
import sys
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import weights.evoq_moe as em


def analyze(smap, li, proj, n_exp=64):
    Ws = []
    for e in range(n_exp):
        key = f"model.layers.{li}.mlp.experts.{e}.{proj}.weight"
        Ws.append(em.read_tensor(smap, key).float().numpy().reshape(-1))
    M = np.stack(Ws)                                   # [64, D]
    total = float((M ** 2).sum())
    mean = M.mean(0, keepdims=True)
    backbone = float((mean ** 2).sum()) * n_exp        # energy carried by the shared mean
    Mc = M - mean                                      # centered (expert variation)
    G = Mc @ Mc.T                                      # [64,64] gram of the variation
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    cum = np.cumsum(ev) / (ev.sum() + 1e-12)
    print(f"  L{li:2d} {proj:10s}: backbone(mean)={100*backbone/total:5.1f}% of total energy | "
          f"of the variation, top-1/4/8/16/32 of 64 modes = "
          f"{100*cum[0]:4.0f}/{100*cum[3]:4.0f}/{100*cum[7]:4.0f}/{100*cum[15]:4.0f}/{100*cum[31]:4.0f}%",
          flush=True)
    del Ws, M, Mc, G


def main():
    smap = em.shard_map()
    print("EXPERT-SUBSPACE TEST (DeepSeek-V2-Lite, 64 routed experts/layer)", flush=True)
    print("If 'backbone' is high AND a few variation-modes capture most -> experts factorize "
          "(backbone + low-rank delta) -> the rearrangement is real.\n", flush=True)
    for li in [3, 13, 24]:                              # early / mid / late MoE layers
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            analyze(smap, li, proj)


if __name__ == "__main__":
    main()
