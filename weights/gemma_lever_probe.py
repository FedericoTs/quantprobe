"""gemma_lever_probe.py -- data-free, minutes: which compression lever is REAL for Gemma 4 12B's MLP bulk,
before committing to a 6-8h run? Probes each tensor for:
  - low-rank exploitability (energy in the top 10% / 25% singular values) -> factorization lever
  - outlier concentration (L2 energy in the top 0.1% / 0.01% weights)     -> outlier-protection lever
  - kurtosis / dynamic range                                             -> heavy-tail lever
If MLPs are low-rank -> low-rank + 2-bit-residual could beat plain 2-bit. If outlier-heavy -> keep top-k
fp16. If flat/white -> only bit-ALLOCATION (depth-aware) helps. Run with the MAIN venv (numpy+safetensors).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from safetensors import safe_open

MDIR = "D:/evo-compress-data/gemma-4-12b"
ST = MDIR + "/model.safetensors"
LP = "model.language_model."


def probe(name, W):
    W = np.asarray(W, np.float32)
    a = np.abs(W).ravel()
    e2 = float((a ** 2).sum())
    thr999 = np.quantile(a, 0.999); thr9999 = np.quantile(a, 0.9999)
    out_1000 = float((a[a >= thr999] ** 2).sum() / e2)      # top 0.1% of weights
    out_10000 = float((a[a >= thr9999] ** 2).sum() / e2)    # top 0.01%
    f = W.ravel(); kurt = float(((f - f.mean()) ** 4).mean() / (f.var() ** 2 + 1e-30) - 3)
    octaves = float(np.log2(a.max() / (np.median(a) + 1e-12)))
    M = W if W.shape[0] <= W.shape[1] else W.T
    if M.shape[0] > 3000:
        M = M[np.random.default_rng(0).choice(M.shape[0], 3000, replace=False)]
    s = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    en = s ** 2 / (s ** 2).sum()
    n = len(en)
    top10 = float(en[:max(1, n // 10)].sum()); top25 = float(en[:max(1, n // 4)].sum())
    print(f"  {name:20s}: SVenergy top10%={top10:.2f} top25%={top25:.2f} | "
          f"outlier top0.1%={out_1000:.3f} top0.01%={out_10000:.3f} | kurt={kurt:+.2f} range={octaves:.0f}oct", flush=True)


def main():
    f = safe_open(ST, framework="pt")
    tensors = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.q_proj", "self_attn.o_proj"]
    print("GEMMA 4 12B lever probe (low-rank? outlier? flat?) -- data-free\n"
          "  low-rank lever if top10%SV >> 0.6 | outlier lever if top0.1% >> 0.10 | else only allocation\n")
    for li in [3, 24, 44]:
        print(f"layer {li}:")
        for t in tensors:
            try:
                probe(f"{t.split('.')[-1]}", f.get_tensor(f"{LP}layers.{li}.{t}.weight").float().numpy())
            except Exception as e:
                print(f"  {t}: FAIL {e}")


if __name__ == "__main__":
    main()
