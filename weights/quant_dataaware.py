"""ROUND 13 -- the DATA-AWARE axis (GPU). Every codec in codec_zoo so far minimizes
WEIGHT-space MSE ||W - W_hat||^2. The held-out-PPL verifier actually rewards OUTPUT
fidelity ||(W - W_hat) x||^2. GPTQ (Frantar et al. 2023) closes that gap: it uses the
calibration Hessian H = X X^T to round each weight column so the resulting OUTPUT error
is compensated by adjusting not-yet-quantized columns (sequential error feedback). It is
the deployed data-aware SOTA we never actually ran in this arena.

This module:
  * captures per-layer input Hessians on GPU from a calibration stream,
  * implements blocked GPTQ (Cholesky error feedback, group-asymmetric levels),
  * adds the NOVEL fusion: GPTQ feedback on a fine grid, paying only the INDEX ENTROPY
    (the project's lossless-coding edge bolted onto output-error awareness),
  * scores every scheme through the SAME held-out-PPL + honest-bits verifier, on CUDA.

Run:  python -m weights.quant_dataaware
"""
from __future__ import annotations

import gc
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoTokenizer

from weights.quant_lab import (CALIB_TEXT, CFG, EVAL_TEXT, WPATH, build_model,
                               quant_keys)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
G = 128


# ----------------------------------------------------------------------------- model on GPU
def gpu_model():
    m = build_model().to(DEV).float().eval()
    return m


def load_fp16_gpu(model):
    msd = model.state_dict()
    with safe_open(WPATH, framework="pt") as f:
        for k in f.keys():
            if k in msd:
                msd[k].copy_(f.get_tensor(k).to(msd[k].dtype).to(DEV))
    model.tie_weights()


def ppl_gpu(model, tok):
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :1024].to(DEV)
    with torch.no_grad():
        return float(torch.exp(model(ids, labels=ids).loss))


# ----------------------------------------------------------------------------- Hessian capture
def capture_hessians(model, tok, n_tokens=16000, win=512):
    """Accumulate H = sum_t x_t x_t^T (input second moment) per target Linear, on GPU,
    over a calibration stream long enough (>> max in-dim 4864) for a well-conditioned H."""
    raw = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data/corpora/generic-text/enwik8_256k"), "rb").read()
    calib = raw[:96000].decode("latin-1")  # disjoint from EVAL_TEXT (bytes 120000+)
    ids_all = tok(calib, return_tensors="pt").input_ids[0][:n_tokens]

    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
    H, cnt = {}, {}
    hooks = []

    def mk(key):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            g = x.t() @ x
            if key in H:
                H[key] += g
                cnt[key] += x.shape[0]
            else:
                H[key] = g
                cnt[key] = x.shape[0]
        return h

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (name + ".weight") in qk:
            hooks.append(mod.register_forward_pre_hook(mk(name + ".weight")))

    with torch.no_grad():
        for s in range(0, len(ids_all) - 1, win):
            chunk = ids_all[s:s + win].unsqueeze(0).to(DEV)
            if chunk.shape[1] < 8:
                break
            model(chunk)
    for h in hooks:
        h.remove()
    for k in H:
        H[k] /= max(cnt[k], 1)
    return H


# ----------------------------------------------------------------------------- GPTQ core
def _group_params(W, maxq, g):
    """Asymmetric per-(row, group) scale & zero-point from the ORIGINAL weights."""
    out, inn = W.shape
    ng = (inn + g - 1) // g
    scale = torch.zeros(out, ng, device=W.device)
    zp = torch.zeros(out, ng, device=W.device)
    for gi in range(ng):
        blk = W[:, gi * g:(gi + 1) * g]
        mn = blk.min(1).values
        mx = blk.max(1).values
        s = (mx - mn).clamp(min=1e-8) / maxq
        scale[:, gi] = s
        zp[:, gi] = torch.round(-mn / s)
    return scale, zp


def gptq_layer(W, H, bits, g=G, blocksize=128, percdamp=0.01):
    """Blocked GPTQ with Cholesky error feedback. Returns (Wq float32, int_codes int16)."""
    W = W.clone().float()
    out, inn = W.shape
    maxq = (1 << bits) - 1
    H = H.clone().float()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    scale, zp = _group_params(W, maxq, g)

    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(inn, device=W.device)
    H[idx, idx] += damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper-triangular factor

    Q = torch.zeros_like(W)
    codes = torch.zeros(out, inn, dtype=torch.int16, device=W.device)

    for i1 in range(0, inn, blocksize):
        i2 = min(i1 + blocksize, inn)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            w = W1[:, i]
            gi = col // g
            s = scale[:, gi]
            z = zp[:, gi]
            qi = torch.clamp(torch.round(w / s) + z, 0, maxq)
            q = (qi - z) * s
            Q1[:, i] = q
            codes[:, col] = qi.to(torch.int16)
            d = Hinv1[i, i]
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < inn:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    ng = scale.shape[1]
    return Q, codes, 32 * out * ng  # scale+zp overhead bits


def _entropy_bits(codes, maxq):
    c = codes.reshape(-1).to(torch.int64).cpu().numpy()
    counts = np.bincount(c - c.min(), minlength=1).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------------------------------------- eval harness
def run(schemes):
    tok = AutoTokenizer.from_pretrained(CFG)
    model = gpu_model()
    load_fp16_gpu(model)
    fp16 = ppl_gpu(model, tok)
    print(f"fp16 held-out ppl = {fp16:.3f}  (device {DEV})\n", flush=True)

    print("capturing Hessians ...", flush=True)
    H = capture_hessians(model, tok)
    print(f"  captured {len(H)} layer Hessians\n", flush=True)

    # cache original fp32 target weights on CPU
    Worig = {}
    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
        for k in qk:
            Worig[k] = f.get_tensor(k).float()

    msd = model.state_dict()
    print(f"{'scheme':<28}{'bits/wt':>9}{'ppl':>10}")
    print("-" * 47)
    results = []
    for name, (bits, mode) in schemes.items():
        load_fp16_gpu(model)  # reset
        bits_tot, elem = 0.0, 0
        for k in Worig:
            W = Worig[k].to(DEV)
            Hk = H[k]
            Q, codes, ov = gptq_layer(W, Hk, bits, G)
            if mode == "uniform":
                b = bits * W.numel() + ov
            else:  # entropy
                ent = _entropy_bits(codes, (1 << bits) - 1)
                b = ent * W.numel() + ov
            msd[k].copy_(Q.to(msd[k].dtype))
            bits_tot += b
            elem += W.numel()
            del W, Q, codes
        model.tie_weights()
        p = ppl_gpu(model, tok)
        bpw = bits_tot / elem
        results.append((name, bpw, p))
        print(f"{name:<28}{bpw:>9.3f}{p:>10.3f}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nfp16 = {fp16:.3f} | frontier refs: ECVQ.008 4.483@3.13b | entropy32 3.971@5.27b")
    for n, bpw, p in sorted(results, key=lambda r: r[2]):
        print(f"  {p:8.3f} @ {bpw:.3f}b   {n}")
    return results


SCHEMES = {
    "GPTQ 4b (data-aware SOTA)": (4, "uniform"),
    "GPTQ 3b (data-aware SOTA)": (3, "uniform"),
    "GPTQ-entropy 4b grid":      (4, "entropy"),
    "GPTQ-entropy 5b grid":      (5, "entropy"),
}


# =============================================================================
# ROUND 14 -- THE FRONTIER FUSION: incoherence ROTATION (QuIP) + GPTQ Hessian
# error-feedback + ECVQ entropy-coded levels. No published method unifies all
# three. Rotate the input space by a block-diagonal signed-Hadamard R (whitens
# the weight distribution); transform the Hessian H' = R^T H R so feedback runs
# in the rotated coordinates; quantize columns left-to-right with GPTQ feedback
# to ENTROPY-CONSTRAINED levels (ECVQ) and pay only the index entropy.
# =============================================================================
def _hadamard(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / math.sqrt(n)


def block_R(inn, g, seed=0):
    """Dense block-diagonal orthogonal rotation: per block = diag(signs) @ Hadamard."""
    Hb = _hadamard(g)
    rng = np.random.default_rng(seed)
    R = np.zeros((inn, inn), np.float32)
    for b in range(0, inn, g):
        e = min(b + g, inn)
        m = e - b
        s = (rng.integers(0, 2, m).astype(np.float32) * 2 - 1)
        R[b:e, b:e] = (s[:, None] * Hb[:m, :m])
    return torch.from_numpy(R).to(DEV)


def _ecvq_levels_np(sample, K, lam, iters=14):
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
    keep = np.bincount(idx, minlength=K) > 0
    return np.sort(c[keep]).astype(np.float32)


def gptq_layer_levels(W, H, levels, g=G, blocksize=128, percdamp=0.01):
    """GPTQ feedback quantizing each column to the nearest of `levels` (a fixed sorted
    codebook in normalized space), per-(row,group) amax scaling. Returns (Wq, codes, lv)."""
    W = W.clone().float()
    out, inn = W.shape
    H = H.clone().float()
    lv = torch.from_numpy(levels).to(W.device)
    K = lv.numel()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    ng = (inn + g - 1) // g
    amax = torch.zeros(out, ng, device=W.device)
    for gi in range(ng):
        blk = W[:, gi * g:(gi + 1) * g]
        a = blk.abs().max(1).values
        amax[:, gi] = torch.clamp(a, min=1e-8)

    damp = percdamp * torch.mean(torch.diag(H))
    di = torch.arange(inn, device=W.device)
    H[di, di] += damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    codes = torch.zeros(out, inn, dtype=torch.int16, device=W.device)
    lo, hi = float(levels[0]), float(levels[-1])   # hoist: NO .item() in the hot loop

    for i1 in range(0, inn, blocksize):
        i2 = min(i1 + blocksize, inn)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            w = W1[:, i]
            a = amax[:, col // g]
            nrm = (w / a).clamp(lo, hi)
            ci = torch.bucketize(nrm, lv)
            ci = torch.clamp(ci, 1, K - 1)
            left = (nrm - lv[ci - 1]).abs() <= (nrm - lv[ci]).abs()
            ci = torch.where(left, ci - 1, ci)
            q = lv[ci] * a
            Q1[:, i] = q
            codes[:, col] = ci.to(torch.int16)
            d = Hinv1[i, i]
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < inn:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, codes, 16 * out * ng  # amax overhead


def run_fusion(points):
    """points: list of (name, lam, Kpool). Rotated + GPTQ-feedback + ECVQ-entropy."""
    tok = AutoTokenizer.from_pretrained(CFG)
    model = gpu_model()
    load_fp16_gpu(model)
    fp16 = ppl_gpu(model, tok)
    print(f"fp16 held-out ppl = {fp16:.3f}  (device {DEV})\n", flush=True)

    print("capturing Hessians ...", flush=True)
    H = capture_hessians(model, tok)
    print(f"  captured {len(H)} layer Hessians\n", flush=True)

    Worig = {}
    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
        for k in qk:
            Worig[k] = f.get_tensor(k).float()

    # cache rotations + rotated Hessians per distinct in-dim
    Rcache, Hrot = {}, {}
    for k in Worig:
        inn = Worig[k].shape[1]
        if inn not in Rcache:
            Rcache[inn] = block_R(inn, G, seed=0)
        R = Rcache[inn]
        Hrot[k] = R.t() @ H[k] @ R

    msd = model.state_dict()
    print(f"{'scheme':<34}{'bits/wt':>9}{'ppl':>10}")
    print("-" * 53)
    results = []
    rng = np.random.default_rng(1)
    for name, lam, Kpool in points:
        t0 = time.time()
        load_fp16_gpu(model)
        bits_tot, elem = 0.0, 0
        for k in Worig:
            W = Worig[k].to(DEV)
            R = Rcache[W.shape[1]]
            Wr = W @ R                       # rotate input space
            # ECVQ levels from a rotated, amax-normalized sample
            ng = (Wr.shape[1] + G - 1) // G
            samp = (Wr / Wr.abs().max()).reshape(-1).cpu().numpy()
            samp = samp[rng.integers(0, len(samp), min(20000, len(samp)))]
            lv = _ecvq_levels_np(samp / (np.abs(samp).max() + 1e-9), Kpool, lam)
            Qr, codes, ov = gptq_layer_levels(Wr, Hrot[k], lv)
            Q = Qr @ R.t()                   # un-rotate
            ent = _entropy_bits(codes, len(lv))
            b = ent * W.numel() + ov
            msd[k].copy_(Q.to(msd[k].dtype))
            bits_tot += b
            elem += W.numel()
            del W, Wr, Qr, Q, codes
        model.tie_weights()
        p = ppl_gpu(model, tok)
        bpw = bits_tot / elem
        results.append((name, bpw, p))
        print(f"{name:<34}{bpw:>9.3f}{p:>10.3f}   ({time.time()-t0:.0f}s)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nfp16 = {fp16:.3f} | frontier refs: ECVQ.008 4.483@3.13b | ECVQ.003 4.169@3.91b | entropy16 4.109@4.24b")
    for n, bpw, p in sorted(results, key=lambda r: r[2]):
        print(f"  {p:8.3f} @ {bpw:.3f}b   {n}")
    return results


FUSION = [
    ("rot+GPTQ+ECVQ lam.008", 0.008, 64),
    ("rot+GPTQ+ECVQ lam.005", 0.005, 64),
    ("rot+GPTQ+ECVQ lam.003", 0.003, 64),
    ("rot+GPTQ+ECVQ lam.012", 0.012, 64),
]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fusion":
        run_fusion(FUSION)
    else:
        run(SCHEMES)
