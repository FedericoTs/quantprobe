"""R25-A4 PRE-FLIGHT (gate before building ECTCQ). A trellis/VQ can only recover the SPACE-FILLING
loss of scalar quantization, which at high rate is 0.5*log2(2*pi*e/12) ~ 0.254 bits/sample above
the rate-distortion bound. This measures the champion's ACTUAL gap to the Gaussian R(D) bound in
the space where it quantizes (per-group rotated, amax-normalized). If the gap is already < ~0.08
b/w (entropy coding + the bounded, non-Gaussian distribution leave little on the table), ECTCQ has
no prize and the loop pivots. If the gap is ~0.15-0.25, ECTCQ is worth a round.

For each tensor: rotate+normalize -> Rn; run champion ECVQ -> (reconstructed Rn_hat, index rate R,
distortion D); compare R to R_gauss(D)=0.5*log2(sigma^2/D) and to the empirical Shannon lower
bound R_slb = h(Rn) - 0.5*log2(2*pi*e*D) (h = differential entropy of the actual source).

Run:  python -m weights.trellis_preflight
"""
from __future__ import annotations

import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors import safe_open
from weights.codec_zoo import _ecvq_levels, _nearest_idx, _entropy_bits
from weights.quant_lab import WPATH, quant_keys
from weights.quant_sota import _fwht_rows

G = 128
LAM = 0.008   # champion ECVQ.008 operating point


def diff_entropy(x, bins=512):
    """Differential entropy estimate (bits) of a 1-D sample via histogram."""
    lo, hi = np.percentile(x, [0.01, 99.99])
    h, edges = np.histogram(np.clip(x, lo, hi), bins=bins, density=True)
    w = edges[1] - edges[0]
    p = h * w
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() + np.log2(w))  # H_bits(bins) + log2(width) = differential


def main():
    tot = 0
    sumR = sumD = sumS2 = 0.0
    sumRg = sumSlb = 0.0
    rn_pool = []
    with safe_open(WPATH, framework="pt") as f:
        for k in sorted(quant_keys(f)):
            W = f.get_tensor(k).float().numpy()
            rows, cols = W.shape
            pad = (-cols) % G
            A = np.pad(W, ((0, 0), (0, pad))) if pad else W
            N = A.reshape(rows, -1, G).reshape(-1, G)
            signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
            R = _fwht_rows(N * signs) / np.sqrt(G)
            amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
            Rn = (R / amax).ravel().astype(np.float32)
            # champion ECVQ at LAM
            rng = np.random.default_rng(1)
            samp = Rn[rng.integers(0, len(Rn), min(20000, len(Rn)))]
            lv = _ecvq_levels(samp, 64, LAM)
            idx = _nearest_idx(Rn, lv)
            Rn_hat = lv[idx]
            D = float(np.mean((Rn - Rn_hat) ** 2))
            s2 = float(np.var(Rn))
            R_rate = _entropy_bits(idx, len(lv))           # bits/sample (achievable index entropy)
            Rg = 0.5 * np.log2(s2 / D) if D > 0 else float('inf')
            h = diff_entropy(Rn)
            Rslb = h - 0.5 * np.log2(2 * np.pi * np.e * D)  # Shannon lower bound rate
            n = Rn.size
            sumR += R_rate * n; sumD += D * n; sumS2 += s2 * n
            sumRg += Rg * n; sumSlb += Rslb * n; tot += n
            if len(rn_pool) < 5:
                rn_pool.append((k, R_rate, D, s2, Rg, Rslb))

    R = sumR / tot; D = sumD / tot; s2 = sumS2 / tot; Rg = sumRg / tot; Rslb = sumSlb / tot
    print(f"Champion ECVQ.{int(LAM*1000):03d} in the rotated-normalized space (pooled over weights):\n")
    print(f"  achieved index rate  R       = {R:.4f} bits/sample")
    print(f"  achieved distortion  D       = {D:.6f}   (Rn variance sigma^2 = {s2:.4f})")
    print(f"  Gaussian R-D bound   R(D)    = {Rg:.4f} bits   -> gap to Gaussian = {R-Rg:.4f} b/w")
    print(f"  Shannon lower bound  R_slb   = {Rslb:.4f} bits -> gap to SLB      = {R-Rslb:.4f} b/w")
    print(f"  high-rate scalar space-filling loss (ceiling a trellis can recover) = 0.2546 b/w\n")
    gap = R - Rslb
    recoverable = min(gap, 0.2546)
    print(f"=== PRE-FLIGHT VERDICT ===")
    print(f"  Gap to the empirical R-D floor (Shannon LB) = {gap:.4f} b/w.")
    if gap < 0.08:
        print(f"  -> < 0.08 b/w: NO prize. ECVQ already near the achievable floor (entropy coding + "
              f"bounded non-Gaussian source ate the space-filling gain). PIVOT, do not build ECTCQ.")
    elif gap < 0.15:
        print(f"  -> 0.08-0.15 b/w: marginal. ECTCQ might recover ~{recoverable*0.5:.3f}; "
              f"build only if A1-A3 didn't already move the frontier.")
    else:
        print(f"  -> >= 0.15 b/w: real room. ECTCQ could recover ~{recoverable*0.6:.3f} b/w; BUILD it.")


if __name__ == "__main__":
    main()
