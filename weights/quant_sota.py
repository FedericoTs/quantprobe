"""Aim at the real SOTA on a small model: how low can we push bits/weight while
keeping Qwen2.5-0.5B smart? We reproduce the CORE ideas of the SOTA codecs in numpy
(the published kernels are GPU-bound, but the techniques are portable):

  - INCOHERENCE ROTATION (QuIP / QuIP# core): a fast block Walsh-Hadamard rotation
    with random signs spreads outliers so weights quantize cleanly. ~free (a seed).
  - NORMAL-FLOAT codebook (NF4-style): non-uniform levels matched to the weight
    distribution instead of uniform RTN levels.
  - OUTLIER preservation: keep the top fraction in fp16.

Verifier (cheap, CPU): held-out perplexity + average bits/weight. Builds the
bits-vs-quality frontier at 4/3/2-bit so we can see the real shrink limit.
"""
from __future__ import annotations

import numpy as np

from weights.quant_smoke import evaluate, q_none, rtn, rtn_group


def _fwht_rows(a):
    """In-place fast Walsh-Hadamard transform along last axis (size = power of 2)."""
    a = a.astype(np.float32).copy()
    g = a.shape[-1]
    h = 1
    while h < g:
        for i in range(0, g, 2 * h):
            x = a[:, i:i + h].copy()
            y = a[:, i + h:i + 2 * h].copy()
            a[:, i:i + h] = x + y
            a[:, i + h:i + 2 * h] = x - y
        h *= 2
    return a


def _levels(bits, kind, seed=0):
    n = 2 ** bits
    if kind == "uniform":
        lv = np.linspace(-1.0, 1.0, n)
    else:  # normal-float: Gaussian quantiles at MIDPOINT probabilities (no sample-extreme
        # outlier on the end levels), normalized so the outermost level is +/-1.
        s = np.random.default_rng(seed).standard_normal(2_000_000)
        p = (np.arange(n) + 0.5) / n
        lv = np.quantile(s, p)
        lv = lv / np.max(np.abs(lv))
    return np.sort(lv).astype(np.float32)


def _nearest(vals, levels):
    idx = np.searchsorted(levels, vals)
    idx = np.clip(idx, 1, len(levels) - 1)
    pick_left = np.abs(vals - levels[idx - 1]) <= np.abs(vals - levels[idx])
    idx = np.where(pick_left, idx - 1, idx)
    return levels[idx]


def _group_codebook(a, bits, g, levels):
    """Per-(row,group) absmax-scaled nearest-level quantization. Returns (W_hat, total_bits)."""
    rows, cols = a.shape
    pad = (-cols) % g
    A = np.pad(a, ((0, 0), (0, pad))) if pad else a
    G = A.reshape(rows, -1, g)                       # [rows, ngrp, g]
    amax = np.abs(G).max(axis=2, keepdims=True)
    amax[amax == 0] = 1.0
    Gh = _nearest((G / amax).ravel(), levels).reshape(G.shape) * amax
    Wh = Gh.reshape(rows, -1)[:, :cols].astype(np.float32)
    nscale = rows * G.shape[1]
    return Wh, bits * a.size + 16.0 * nscale


def nf_group(a, bits=4, g=128):
    return _group_codebook(a, bits, g, _levels(bits, "normal"))


def hadamard_nf(a, bits=4, g=128, seed=0):
    """Rotate each g-block by random-sign Walsh-Hadamard, NF-quantize, rotate back."""
    rows, cols = a.shape
    pad = (-cols) % g
    A = np.pad(a, ((0, 0), (0, pad))) if pad else a
    N = A.reshape(rows, -1, g).reshape(-1, g)        # [rows*ngrp, g]
    signs = np.random.default_rng(seed).integers(0, 2, g).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(g)           # rotate into incoherent basis
    lv = _levels(bits, "normal")
    amax = np.abs(R).max(axis=1, keepdims=True)
    amax[amax == 0] = 1.0
    Rh = _nearest((R / amax).ravel(), lv).reshape(R.shape) * amax
    back = (_fwht_rows(Rh) / np.sqrt(g)) * signs      # inverse rotation
    Wh = back.reshape(rows, -1)[:, :cols].astype(np.float32)
    return Wh, bits * a.size + 16.0 * N.shape[0]


def hadamard_nf_outlier(a, bits=4, g=128, p=0.005, seed=0):
    thr = np.quantile(np.abs(a), 1.0 - p)
    mask = np.abs(a) >= thr
    base = a.copy()
    base[mask] = 0.0
    Wh, bits_base = hadamard_nf(base, bits, g, seed)
    Wh[mask] = a[mask]
    nout = int(mask.sum())
    return Wh.astype(np.float32), bits_base + nout * (16 + 16) - nout * bits


def main():
    schemes = [("fp16 (reference)", q_none), ("RTN int8", lambda a: rtn(a, 8))]
    for b in (4, 3, 2):
        schemes.append((f"RTN-group {b}b", lambda a, b=b: rtn_group(a, b, 128)))
        schemes.append((f"NF-group {b}b", lambda a, b=b: nf_group(a, b, 128)))
        schemes.append((f"Hadamard+NF {b}b", lambda a, b=b: hadamard_nf(a, b, 128)))
        schemes.append((f"Hadamard+NF+outlier {b}b", lambda a, b=b: hadamard_nf_outlier(a, b, 128, 0.005)))

    print(f"{'scheme':<30}{'bits/wt':>9}{'ppl':>10}")
    print("-" * 50)
    rows = []
    for name, q in schemes:
        bpw, ppl = evaluate(name, q)
        rows.append((name, bpw, ppl))
        print(f"{name:<30}{bpw:>9.3f}{ppl:>10.3f}", flush=True)

    ref = rows[0][2]
    print(f"\nfp16 reference ppl = {ref:.3f}")
    print("lowest-bit scheme staying within +0.30 ppl of fp16:")
    ok = [r for r in rows if r[2] <= ref + 0.30 and r[1] < 16]
    if ok:
        best = min(ok, key=lambda r: r[1])
        print(f"  {best[0]}: {best[1]:.3f} bits/wt, ppl {best[2]:.3f} (+{best[2]-ref:.3f})  "
              f"=> ~{16/best[1]:.1f}x smaller than fp16, near-lossless")
    for b in (4, 3, 2):
        grp = [r for r in rows if r[0] == f"RTN-group {b}b"]
        best_b = min([r for r in rows if r[0].endswith(f"{b}b")], key=lambda r: r[2])
        base = grp[0][2] if grp else float('nan')
        print(f"  {b}-bit: best = {best_b[0]} ppl {best_b[2]:.3f}  (vs RTN-group {base:.3f})")


if __name__ == "__main__":
    main()
