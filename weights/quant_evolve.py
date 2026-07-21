"""Evolution loop, round 1 - attack the 2-3 bit frontier where scalar methods collapsed.

The lever the SOTA sub-4-bit codecs (QuIP#, AQLM) use and our scalar baseline lacks:
VECTOR QUANTIZATION - quantize d-dim GROUPS of (incoherence-rotated) weights jointly
to a learned codebook, instead of one scalar at a time. VQ's rate-distortion advantage
grows with dimension d, which is exactly what makes 2-bit survivable.

Head-to-head verifier (held-out perplexity + bits/weight) vs the scalar incumbent
(Hadamard+NF+outlier) at ~2-bit and ~3-bit. If VQ makes 2-bit usable where scalar
gave gibberish, that is the result worth chasing.
"""
from __future__ import annotations

import numpy as np

from weights.quant_smoke import evaluate, q_none
from weights.quant_sota import _fwht_rows, hadamard_nf_outlier


def _assign(X, C, chunk=100000):
    out = np.empty(len(X), np.int32)
    c2 = (C * C).sum(1)
    for i in range(0, len(X), chunk):
        x = X[i:i + chunk]
        d = (x * x).sum(1, keepdims=True) - 2.0 * (x @ C.T) + c2[None, :]
        out[i:i + chunk] = d.argmin(1)
    return out


def _kmeans(X, K, iters=8, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=len(X) < K)].astype(np.float32).copy()
    for _ in range(iters):
        a = _assign(X, C)
        for k in range(K):
            m = a == k
            if m.any():
                C[k] = X[m].mean(0)
    return C


def hadamard_vq(a, bits=2.0, d=2, g=128, p=0.0, seed=0):
    """Incoherence-rotate per g-block, then VECTOR-quantize d-dim groups to a learned
    codebook. bits/weight = log2(K)/d (+ scale + codebook + outliers). p = outlier frac."""
    rows, cols = a.shape
    mask = np.abs(a) >= np.quantile(np.abs(a), 1.0 - p) if p > 0 else np.zeros(a.shape, bool)
    base = a.copy()
    base[mask] = 0.0

    pad = (-cols) % g
    A = np.pad(base, ((0, 0), (0, pad))) if pad else base
    N = A.reshape(rows, -1, g).reshape(-1, g)
    signs = np.random.default_rng(seed).integers(0, 2, g).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(g)
    amax = np.abs(R).max(1, keepdims=True)
    amax[amax == 0] = 1.0
    Rn = R / amax

    flat = Rn.ravel().astype(np.float32)
    padv = (-len(flat)) % d
    if padv:
        flat = np.concatenate([flat, np.zeros(padv, np.float32)])
    V = flat.reshape(-1, d)
    K = int(2 ** round(bits * d))
    rng = np.random.default_rng(seed + 1)
    sub = V[rng.choice(len(V), min(16384, len(V)), replace=False)]
    C = _kmeans(sub, K, iters=8, seed=seed)
    Vh = C[_assign(V, C)]

    Rnh = Vh.ravel()[:Rn.size].reshape(Rn.shape)
    back = (_fwht_rows(Rnh * amax) / np.sqrt(g)) * signs
    Wh = back.reshape(rows, -1)[:, :cols].astype(np.float32)
    Wh[mask] = a[mask]

    nout = int(mask.sum())
    total = len(V) * np.log2(K) + 16.0 * N.shape[0] + K * d * 16 + nout * 32
    return Wh, total


def main():
    schemes = [
        ("fp16 (reference)", q_none),
        ("~3b scalar: Had+NF+outlier", lambda a: hadamard_nf_outlier(a, 3, 128, 0.005)),
        ("~3b VQ d=2 (K=64)", lambda a: hadamard_vq(a, 3.0, 2, 128, 0.005)),
        ("~2b scalar: Had+NF+outlier", lambda a: hadamard_nf_outlier(a, 2, 128, 0.005)),
        ("~2b VQ d=2 (K=16)", lambda a: hadamard_vq(a, 2.0, 2, 128, 0.005)),
        ("~2b VQ d=4 (K=256)", lambda a: hadamard_vq(a, 2.0, 4, 128, 0.005)),
    ]
    print(f"{'scheme':<30}{'bits/wt':>9}{'ppl':>12}")
    print("-" * 52)
    rows = []
    for name, q in schemes:
        bpw, ppl = evaluate(name, q)
        rows.append((name, bpw, ppl))
        print(f"{name:<30}{bpw:>9.3f}{ppl:>12.3f}", flush=True)

    ref = rows[0][2]
    print(f"\nfp16 ref ppl = {ref:.3f}")
    for tag in ("3b", "2b"):
        grp = [r for r in rows if tag in r[0]]
        sca = [r for r in grp if "scalar" in r[0]][0]
        best = min(grp, key=lambda r: r[2])
        print(f"  {tag}: scalar {sca[2]:.2f} ppl  ->  best {best[0].split(':')[0].strip()} "
              f"{best[2]:.3f} ppl @ {best[1]:.2f} bits")
        if best[2] < sca[2] * 0.5:
            print(f"       => VQ RESCUES the frontier (>=2x better than scalar)")


if __name__ == "__main__":
    main()
