"""Codec zoo: the registry of evolvable quantization codecs. I (Claude, the LLM
mutation operator) add/rewrite codec functions here each generation -- genuine
code-level mutation, not flag-toggling.

Each codec:  fn(W, key, calib) -> (W_hat: np.ndarray float32, total_bits: float)
and a decode-cost tag (realizability gate): "low" | "med" | "high".

Round 1 seeds the 4.934 champion + a NEW operator invented this round:
ADDITIVE / RESIDUAL quantization (the AQLM core lever) -- quantize coarse, then
quantize the residual, spending the bit budget as coarse+residual passes.
"""
from __future__ import annotations

import numpy as np

from weights.quant_lab import _awq_scale
from weights.quant_sota import hadamard_nf, hadamard_nf_outlier

G = 128


def _act(W, key, calib, alpha=0.5):
    s = _awq_scale(calib, key, alpha)
    return W * s[None, :], s


def champ(W, key, calib):
    """rotate + act-aware + NF + outlier  == the 4.934 wall (~3.30 bits)."""
    Ws, s = _act(W, key, calib)
    wh, b = hadamard_nf_outlier(Ws, 3, G, 0.005)
    return (wh / s[None, :]).astype(np.float32), b + 16 * W.shape[1]


def resid21(W, key, calib):
    """NEW operator: additive/residual quant -- 2-bit coarse + 1-bit residual (~3.3 bits)."""
    Ws, s = _act(W, key, calib)
    w1, b1 = hadamard_nf_outlier(Ws, 2, G, 0.005)
    w2, b2 = hadamard_nf(Ws - w1, 1, G)
    return ((w1 + w2) / s[None, :]).astype(np.float32), b1 + b2 + 16 * W.shape[1]


def resid21_noact(W, key, calib):
    """Ablation: additive 2+1 without activation scaling (isolates the residual gain)."""
    w1, b1 = hadamard_nf_outlier(W, 2, G, 0.005)
    w2, b2 = hadamard_nf(W - w1, 1, G)
    return (w1 + w2).astype(np.float32), b1 + b2


def resid21_bigout(W, key, calib):
    """Additive 2+1 with 1% outliers (more fp16 outliers protect the heavy tail)."""
    Ws, s = _act(W, key, calib)
    w1, b1 = hadamard_nf_outlier(Ws, 2, G, 0.01)
    w2, b2 = hadamard_nf(Ws - w1, 1, G)
    return ((w1 + w2) / s[None, :]).astype(np.float32), b1 + b2 + 16 * W.shape[1]


################################################################################
# ROUND 2 mutation: replace FIXED normal-float levels with LEARNED (Lloyd-Max)
# levels fit to each layer's actual rotated-weight distribution. Clean A/B vs champ
# (identical pipeline; only the codebook levels change from generic-normal -> optimal).
################################################################################
from weights.quant_sota import _fwht_rows, _nearest  # noqa: E402


def _lloyd_levels(sample, n, iters=10):
    c = np.quantile(sample, (np.arange(n) + 0.5) / n).astype(np.float64)
    for _ in range(iters):
        idx = np.abs(sample[:, None] - c[None, :]).argmin(1)
        for k in range(n):
            m = idx == k
            if m.any():
                c[k] = sample[m].mean()
    return np.sort(c).astype(np.float32)


def _had_lloyd_out(W, bits, g, p, seed=0):
    rows, cols = W.shape
    mask = np.abs(W) >= np.quantile(np.abs(W), 1.0 - p) if p > 0 else np.zeros(W.shape, bool)
    base = W.copy(); base[mask] = 0.0
    pad = (-cols) % g
    A = np.pad(base, ((0, 0), (0, pad))) if pad else base
    N = A.reshape(rows, -1, g).reshape(-1, g)
    signs = np.random.default_rng(seed).integers(0, 2, g).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(g)
    amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
    Rn = R / amax
    rng = np.random.default_rng(seed + 1)
    samp = Rn.ravel()
    samp = samp[rng.integers(0, len(samp), min(20000, len(samp)))]
    lv = _lloyd_levels(samp, 2 ** bits)
    Rh = _nearest(Rn.ravel(), lv).reshape(Rn.shape) * amax
    back = (_fwht_rows(Rh) / np.sqrt(g)) * signs
    wh = back.reshape(rows, -1)[:, :cols]
    wh[mask] = W[mask]
    return wh.astype(np.float32), bits * W.size + 16 * N.shape[0] + int(mask.sum()) * 32


def lloyd_champ(W, key, calib):
    """champ pipeline but with LEARNED per-layer optimal levels instead of fixed NF."""
    Ws, s = _act(W, key, calib)
    wh, b = _had_lloyd_out(Ws, 3, G, 0.005)
    return (wh / s[None, :]).astype(np.float32), b + 16 * W.shape[1]


def lloyd_4lev_resid(W, key, calib):
    """Learned-level 3-bit base, then a learned 4-level (incl near-zero) residual pass."""
    Ws, s = _act(W, key, calib)
    w1, b1 = _had_lloyd_out(Ws, 3, G, 0.005)
    # residual quantized per-group with learned 4 levels (Lloyd naturally places one near 0)
    R = Ws - w1
    rows, cols = R.shape
    samp = R.ravel()
    samp = samp[np.random.default_rng(2).integers(0, len(samp), min(20000, len(samp)))]
    samp = samp / (np.abs(samp).max() + 1e-9)
    lv = _lloyd_levels(samp, 4)
    g = G
    pad = (-cols) % g
    A = np.pad(R, ((0, 0), (0, pad))) if pad else R
    Gp = A.reshape(rows, -1, g)
    amax = np.abs(Gp).max(2, keepdims=True); amax[amax == 0] = 1.0
    w2 = (_nearest((Gp / amax).ravel(), lv).reshape(Gp.shape) * amax).reshape(rows, -1)[:, :cols]
    return ((w1 + w2) / s[None, :]).astype(np.float32), b1 + 2.0 * R.size + 16 * Gp.shape[0] * Gp.shape[1] + 16 * W.shape[1]


################################################################################
# ROUND 3 mutation: SENSITIVITY-WEIGHTED MIXED PRECISION on top of learned levels.
# Use calibration activation magnitude to give the most-important input channels
# more bits (4) and the rest fewer (2) -- spend the budget where it matters.
################################################################################
def lloyd_mixed(W, key, calib, f=0.5, bhi=4, blo=2):
    imp = calib[key]
    hi = imp >= np.quantile(imp, 1.0 - f)
    Ws, sc = _act(W, key, calib)
    Wh = np.zeros_like(Ws)
    bh = bl = 0.0
    if hi.any():
        w, bh = _had_lloyd_out(Ws[:, hi], bhi, G, 0.005); Wh[:, hi] = w
    if (~hi).any():
        w, bl = _had_lloyd_out(Ws[:, ~hi], blo, G, 0.005); Wh[:, ~hi] = w
    return (Wh / sc[None, :]).astype(np.float32), bh + bl + 16 * W.shape[1]


def lloyd_mixed_50(W, key, calib):  # ~3.0 bits avg (half hi, half lo)
    return lloyd_mixed(W, key, calib, 0.5, 4, 2)


def lloyd_mixed_25(W, key, calib):  # ~2.7 bits avg (quarter hi)
    return lloyd_mixed(W, key, calib, 0.25, 4, 2)


def lloyd_mixed_33(W, key, calib):  # ~3.0 bits avg via 3/3 split (top third 4-bit, rest ~2.5)
    return lloyd_mixed(W, key, calib, 0.33, 4, 3)


################################################################################
# ROUND 4 - a GENUINELY NEW strategy (not recombination): ENTROPY-CONSTRAINED
# quantization. Use MANY learned levels (high quality) but pay only the measured
# ENTROPY of the index stream in storage (lossless arithmetic coding achieves it).
# Decouples quality (#levels) from storage (entropy). No deployed LLM quantizer
# does this; it leverages this project's lossless-compression DNA.
################################################################################
def _nearest_idx(vals, levels):
    idx = np.searchsorted(levels, vals)
    idx = np.clip(idx, 1, len(levels) - 1)
    left = np.abs(vals - levels[idx - 1]) <= np.abs(vals - levels[idx])
    return np.where(left, idx - 1, idx)


def _entropy_bits(idx, K):
    counts = np.bincount(idx.ravel(), minlength=K).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _had_entropy(W, nlev, g, p, seed=0):
    rows, cols = W.shape
    mask = np.abs(W) >= np.quantile(np.abs(W), 1.0 - p) if p > 0 else np.zeros(W.shape, bool)
    base = W.copy(); base[mask] = 0.0
    pad = (-cols) % g
    A = np.pad(base, ((0, 0), (0, pad))) if pad else base
    N = A.reshape(rows, -1, g).reshape(-1, g)
    signs = np.random.default_rng(seed).integers(0, 2, g).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(g)
    amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
    Rn = R / amax
    rng = np.random.default_rng(seed + 1)
    samp = Rn.ravel()
    samp = samp[rng.integers(0, len(samp), min(20000, len(samp)))]
    lv = _lloyd_levels(samp, nlev)
    idx = _nearest_idx(Rn.ravel(), lv)
    Rh = lv[idx].reshape(Rn.shape) * amax
    back = (_fwht_rows(Rh) / np.sqrt(g)) * signs
    wh = back.reshape(rows, -1)[:, :cols]
    wh[mask] = W[mask]
    ent = _entropy_bits(idx, nlev)                      # bits/weight an arithmetic coder achieves
    bits = ent * W.size + 16 * N.shape[0] + nlev * 16 + int(mask.sum()) * 32
    return wh.astype(np.float32), bits


def entropy_q(W, key, calib, nlev):
    Ws, s = _act(W, key, calib)
    wh, b = _had_entropy(Ws, nlev, G, 0.005)
    return (wh / s[None, :]).astype(np.float32), b + 16 * W.shape[1]


def entropy8(W, key, calib):
    return entropy_q(W, key, calib, 8)


def entropy16(W, key, calib):
    return entropy_q(W, key, calib, 16)


def entropy32(W, key, calib):
    return entropy_q(W, key, calib, 32)


################################################################################
# ROUND 5 - ECVQ (Entropy-Constrained quantization). Jointly optimize levels for
# distortion AND rate: assign x to argmin_k (x-c_k)^2 - lambda*log2(p_k), so the
# quantizer CONCENTRATES probability (low entropy) at controlled distortion. Use a
# big level pool (K=64) and let lambda prune it via the rate term. The storage-optimal
# scalar quantizer -- never used in LLM quant; pure lossless-coding territory.
################################################################################
def _ecvq_levels(sample, K, lam, iters=14):
    c = np.quantile(sample, (np.arange(K) + 0.5) / K).astype(np.float64)
    p = np.full(K, 1.0 / K)
    for _ in range(iters):
        d = (sample[:, None] - c[None, :]) ** 2 - lam * np.log2(np.maximum(p[None, :], 1e-12))
        idx = d.argmin(1)
        cnt = np.bincount(idx, minlength=K).astype(np.float64)
        p = (cnt + 1e-9) / (cnt.sum() + 1e-9 * K)
        for k in range(K):
            m = idx == k
            if m.any():
                c[k] = sample[m].mean()
    keep = np.bincount(idx, minlength=K) > 0       # drop unused levels
    return np.sort(c[keep]).astype(np.float32)


def _had_ecvq(W, lam, g, p, seed=0, Kpool=64):
    rows, cols = W.shape
    mask = np.abs(W) >= np.quantile(np.abs(W), 1.0 - p) if p > 0 else np.zeros(W.shape, bool)
    base = W.copy(); base[mask] = 0.0
    pad = (-cols) % g
    A = np.pad(base, ((0, 0), (0, pad))) if pad else base
    N = A.reshape(rows, -1, g).reshape(-1, g)
    signs = np.random.default_rng(seed).integers(0, 2, g).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(g)
    amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
    Rn = R / amax
    rng = np.random.default_rng(seed + 1)
    samp = Rn.ravel()
    samp = samp[rng.integers(0, len(samp), min(20000, len(samp)))]
    lv = _ecvq_levels(samp, Kpool, lam)
    idx = _nearest_idx(Rn.ravel(), lv)
    Rh = lv[idx].reshape(Rn.shape) * amax
    back = (_fwht_rows(Rh) / np.sqrt(g)) * signs
    wh = back.reshape(rows, -1)[:, :cols]
    wh[mask] = W[mask]
    ent = _entropy_bits(idx, len(lv))
    bits = ent * W.size + 16 * N.shape[0] + len(lv) * 16 + int(mask.sum()) * 32
    return wh.astype(np.float32), bits


def ecvq(W, key, calib, lam):
    Ws, s = _act(W, key, calib)
    wh, b = _had_ecvq(Ws, lam, G, 0.005)
    return (wh / s[None, :]).astype(np.float32), b + 16 * W.shape[1]


def ecvq_lo(W, key, calib):   # high rate penalty -> very low entropy
    return ecvq(W, key, calib, 0.02)


def ecvq_mid(W, key, calib):
    return ecvq(W, key, calib, 0.008)


def ecvq_hi(W, key, calib):   # low penalty -> more levels, higher quality
    return ecvq(W, key, calib, 0.003)


################################################################################
# ROUND 6 - push ECVQ (fine sweep + additive) and invent SIGMA-DELTA noise shaping:
# quantize along the input dim carrying rounding error forward so consecutive errors
# cancel in the matmul sum (GPTQ's feedback idea WITHOUT the Hessian; from DSP).
################################################################################
def ecvq_005(W, key, calib):
    return ecvq(W, key, calib, 0.005)


def ecvq_006(W, key, calib):
    return ecvq(W, key, calib, 0.006)


def ecvq_additive(W, key, calib):
    """ECVQ base + ECVQ residual -- entropy-optimal refinement toward near-lossless."""
    Ws, s = _act(W, key, calib)
    w1, b1 = _had_ecvq(Ws, 0.008, G, 0.005)
    w2, b2 = _had_ecvq(Ws - w1, 0.02, G, 0.0)
    return ((w1 + w2) / s[None, :]).astype(np.float32), b1 + b2 + 16 * W.shape[1]


def sigma_delta(W, key, calib):
    """NEW: noise-shaping quantization -- carry the rounding error forward along the
    input dimension within each group so adjacent errors cancel in the matmul sum."""
    Ws, s = _act(W, key, calib)
    rows, cols = Ws.shape
    rng = np.random.default_rng(1)
    samp = Ws.ravel()
    samp = samp / (np.abs(samp).max() + 1e-9)
    samp = samp[rng.integers(0, len(samp), min(20000, len(samp)))]
    lv = _lloyd_levels(samp, 8)
    out = np.empty_like(Ws)
    for c0 in range(0, cols, G):
        blk = Ws[:, c0:c0 + G]
        amax = np.abs(blk).max(1, keepdims=True); amax[amax == 0] = 1.0
        n = blk / amax
        carry = np.zeros((rows, 1), np.float32)
        qb = np.empty_like(n)
        for j in range(blk.shape[1]):
            v = n[:, j:j + 1] + carry
            q = _nearest(v.ravel(), lv).reshape(-1, 1)
            carry = v - q
            qb[:, j:j + 1] = q
        out[:, c0:c0 + blk.shape[1]] = qb * amax
    return (out / s[None, :]).astype(np.float32), 3.0 * Ws.size + 16 * (cols // G + 1) * rows + 16 * W.shape[1]


################################################################################
# ROUND 7 - cross-layer entropy ALLOCATION (spend bits across layers by sensitivity),
# a leaky sigma-delta rescue, and an aggressive ECVQ point.
################################################################################
def ecvq_adaptive(W, key, calib, base_lam=0.008, power=0.6):
    """Per-layer lambda: allocate MORE bits (lower lambda) to high-activation layers,
    fewer to tolerant ones -- automatic cross-layer mixed precision via the rate knob."""
    a_this = float(calib[key].mean())
    a_med = float(np.median([float(v.mean()) for v in calib.values()]))
    lam = base_lam * (a_this / max(a_med, 1e-9)) ** power   # high act -> larger lam? invert below
    lam = base_lam * (a_med / max(a_this, 1e-9)) ** power    # high act -> smaller lam -> more bits
    return ecvq(W, key, calib, float(np.clip(lam, 0.001, 0.05)))


def ecvq_012(W, key, calib):
    return ecvq(W, key, calib, 0.012)


def sigma_delta_leaky(W, key, calib, decay=0.8):
    """Sigma-delta with a LEAKY integrator (decay the carry) to stop the blow-up."""
    Ws, s = _act(W, key, calib)
    rows, cols = Ws.shape
    rng = np.random.default_rng(1)
    samp = Ws.ravel()
    samp = samp / (np.abs(samp).max() + 1e-9)
    samp = samp[rng.integers(0, len(samp), min(20000, len(samp)))]
    lv = _lloyd_levels(samp, 8)
    out = np.empty_like(Ws)
    for c0 in range(0, cols, G):
        blk = Ws[:, c0:c0 + G]
        amax = np.abs(blk).max(1, keepdims=True); amax[amax == 0] = 1.0
        n = blk / amax
        carry = np.zeros((rows, 1), np.float32)
        qb = np.empty_like(n)
        for j in range(blk.shape[1]):
            v = n[:, j:j + 1] + carry
            q = _nearest(np.clip(v, -1, 1).ravel(), lv).reshape(-1, 1)
            carry = decay * (v - q)
            qb[:, j:j + 1] = q
        out[:, c0:c0 + blk.shape[1]] = qb * amax
    return (out / s[None, :]).astype(np.float32), 3.0 * Ws.size + 16 * (cols // G + 1) * rows + 16 * W.shape[1]


################################################################################
# ROUND 8 - deeper math: ECVQ + DATA-AWARE LOW-RANK RESIDUAL. Aggressive ECVQ base
# (~2.8b) + the optimal rank-r correction (SVD of the residual in the activation-
# aware space) that mops up the dominant error directions for a fractional bit cost.
################################################################################
def _rsvd(E, r, p=6, seed=0):
    """Randomized rank-r SVD -- top-r components only, low memory + fast."""
    out, inn = E.shape
    k = min(r + p, out, inn)
    Om = np.random.default_rng(seed).standard_normal((inn, k)).astype(np.float32)
    Y = E @ Om
    Q_, _ = np.linalg.qr(Y)
    B = Q_.T @ E
    Ub, S_, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q_ @ Ub)[:, :r], S_[:r], Vt[:r]


def _ecvq_lr(W, key, calib, lam, r):
    Ws, s = _act(W, key, calib)
    Q, bq = _had_ecvq(Ws, lam, G, 0.005)
    E = (Ws - Q).astype(np.float32)
    U, S_, Vt = _rsvd(E, r)
    L = (U * S_) @ Vt
    Wh = (Q + L) / s[None, :]
    out, inn = W.shape
    return Wh.astype(np.float32), bq + r * (out + inn) * 16 + 16 * inn


def ecvq_lr8(W, key, calib):
    return _ecvq_lr(W, key, calib, 0.012, 8)


def ecvq_lr16(W, key, calib):
    return _ecvq_lr(W, key, calib, 0.012, 16)


def ecvq_lr32(W, key, calib):
    return _ecvq_lr(W, key, calib, 0.012, 32)


def ecvq_lr16_aggr(W, key, calib):  # even more aggressive ECVQ base + rank-16
    return _ecvq_lr(W, key, calib, 0.020, 16)


################################################################################
# ROUND 9 - deeper math: D4 LATTICE quantization + entropy coding. Quantize rotated
# weights to the nearest point of the D4 lattice (fast nearest-point, no codebook
# search) to capture the "space-filling gain" scalar ECVQ leaves on the table, then
# entropy-code the lattice indices. The principled step past scalar quantization.
################################################################################
def _d4_nearest(X):
    """Nearest D4 lattice point (integer 4-vectors with EVEN coordinate sum)."""
    f = np.round(X)
    s = f.sum(1).astype(np.int64)
    odd = (s % 2) != 0
    if odd.any():
        err = (X - f)[odd]
        j = np.abs(err).argmax(1)
        rows = np.where(odd)[0]
        f[rows, j] = f[rows, j] + np.where(err[np.arange(len(rows)), j] >= 0, 1.0, -1.0)
    return f


def _coord_entropy(P):
    """HONEST lattice rate: sum of per-coordinate entropies (each coord is 1-D and
    well-sampled). A valid achievable coding rate -- unlike the joint empirical
    entropy, which under-counts when points are near-unique at fine resolution."""
    tot = 0.0
    for d in range(P.shape[1]):
        _, counts = np.unique(P[:, d], return_counts=True)
        pr = counts / counts.sum()
        tot += float(-(pr * np.log2(pr)).sum())
    return tot                                       # bits per d-vector


def _lattice_entropy(P):
    return _coord_entropy(P)


def d4_lattice(W, key, calib, q):
    Ws, s = _act(W, key, calib)
    rows, cols = Ws.shape
    pad = (-cols) % G
    A = np.pad(Ws, ((0, 0), (0, pad))) if pad else Ws
    N = A.reshape(rows, -1, G).reshape(-1, G)
    signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(G)
    amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
    Rn = R / amax
    flat = Rn.ravel().astype(np.float32)
    padv = (-len(flat)) % 4
    if padv:
        flat = np.concatenate([flat, np.zeros(padv, np.float32)])
    V = flat.reshape(-1, 4) / q
    P = _d4_nearest(V)
    Rh = (P * q).ravel()[:Rn.size].reshape(Rn.shape) * amax
    back = (_fwht_rows(Rh) / np.sqrt(G)) * signs
    wh = back.reshape(rows, -1)[:, :cols]
    ent = _lattice_entropy(P)
    bits = (ent / 4.0) * Ws.size + 16 * N.shape[0] + 16 * W.shape[1]
    return (wh / s[None, :]).astype(np.float32), bits


def d4_q12(W, key, calib):
    return d4_lattice(W, key, calib, 0.12)


def d4_q16(W, key, calib):
    return d4_lattice(W, key, calib, 0.16)


def d4_q20(W, key, calib):
    return d4_lattice(W, key, calib, 0.20)


################################################################################
# ROUND 10 - E8 LATTICE (optimal 8-D sphere packing, the QuIP# core) + entropy.
# E8 = D8 union (D8 + 1/2). Bigger space-filling gain (~0.5 bit/dim) than D4.
################################################################################
def _d8_nearest(X):
    f = np.round(X)
    s = f.sum(1).astype(np.int64)
    odd = (s % 2) != 0
    if odd.any():
        err = (X - f)[odd]
        j = np.abs(err).argmax(1)
        rows = np.where(odd)[0]
        f[rows, j] = f[rows, j] + np.where(err[np.arange(len(rows)), j] >= 0, 1.0, -1.0)
    return f


def _e8_nearest(X):
    a = _d8_nearest(X)
    b = _d8_nearest(X - 0.5) + 0.5
    da = ((X - a) ** 2).sum(1)
    db = ((X - b) ** 2).sum(1)
    return np.where((da <= db)[:, None], a, b)


def e8_lattice(W, key, calib, q):
    Ws, s = _act(W, key, calib)
    rows, cols = Ws.shape
    pad = (-cols) % G
    A = np.pad(Ws, ((0, 0), (0, pad))) if pad else Ws
    N = A.reshape(rows, -1, G).reshape(-1, G)
    signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(G)
    amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
    Rn = R / amax
    flat = Rn.ravel().astype(np.float32)
    padv = (-len(flat)) % 8
    if padv:
        flat = np.concatenate([flat, np.zeros(padv, np.float32)])
    V = flat.reshape(-1, 8) / q
    P = _e8_nearest(V)
    Rh = (P * q).ravel()[:Rn.size].reshape(Rn.shape) * amax
    back = (_fwht_rows(Rh) / np.sqrt(G)) * signs
    wh = back.reshape(rows, -1)[:, :cols]
    ent = _coord_entropy(P)                          # HONEST per-coordinate rate (bits/8-vec)
    bits = (ent / 8.0) * Ws.size + 16 * N.shape[0] + 16 * W.shape[1]
    return (wh / s[None, :]).astype(np.float32), bits


def e8_q14(W, key, calib):
    return e8_lattice(W, key, calib, 0.14)


def e8_q18(W, key, calib):
    return e8_lattice(W, key, calib, 0.18)


def e8_q22(W, key, calib):
    return e8_lattice(W, key, calib, 0.22)


################################################################################
# ROUND 11 - map the full E8 rate-distortion curve (finer + coarser q) to establish
# E8+entropy as the dominant frontier across ALL bit budgets.
################################################################################
def e8_q08(W, key, calib):
    return e8_lattice(W, key, calib, 0.08)


def e8_q10(W, key, calib):
    return e8_lattice(W, key, calib, 0.10)


def e8_q28(W, key, calib):
    return e8_lattice(W, key, calib, 0.28)


def e8_q40(W, key, calib):
    return e8_lattice(W, key, calib, 0.40)


CODECS = {
    "champ (4.934 wall)": (champ, "med"),
    "E8-lattice q.08": (e8_q08, "high"),
    "E8-lattice q.14": (e8_q14, "high"),
    "E8-lattice q.22": (e8_q22, "high"),
    "E8-lattice q.28": (e8_q28, "high"),
    "D4-lattice q.16": (d4_q16, "high"),
    "D4-lattice q.20": (d4_q20, "high"),
}
