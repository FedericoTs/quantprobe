"""ROUND 15 -- FAST data-aware quantization (fully vectorized; NO per-column Python loop).

Lesson from Round 14: GPTQ's sequential column loop is CPU-bound on this GPU (~20 min/scheme).
The original 30 rounds were fast because every quantizer was vectorized. So we recover the
data-aware signal WITHOUT the loop:

  * capture the cheap BLOCK-DIAGONAL Hessian B[gi] (g x g per input group, ~2 MB/layer, not
    the full 95 MB in x in matrix). Under a pure diagonal-H approximation the incoherence
    rotation makes every in-group coordinate equally sensitive (|R[j,c]|^2 = 1/g), collapsing
    to the AWQ signal that already failed in R7 -- so the NOVEL signal lives in the within-group
    OFF-diagonals, which the block Hessian keeps.
  * rotate the block Hessian: D[gi,c] = diag(R^T B[gi] R) = the true output sensitivity of each
    rotated coordinate.
  * weight a fully-vectorized ECVQ (level fit + assignment) by w = amax^2 * D so the codebook
    minimizes OUTPUT distortion, then pay only the index entropy.

Run:  python -m weights.quant_fast
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

from weights.quant_dataaware import (DEV, G, _hadamard, gpu_model,
                                     load_fp16_gpu, ppl_gpu)
from weights.quant_lab import CFG, WPATH, quant_keys

HAD = torch.from_numpy(_hadamard(G)).to(DEV)              # [G,G] orthonormal
SIGNS = (torch.from_numpy(np.random.default_rng(0).integers(0, 2, G).astype(np.float32))
         * 2 - 1).to(DEV)                                  # fixed sign flips (seed 0)
ROT = SIGNS[:, None] * HAD                                 # rotation block R (apply: v @ ROT)


def capture_block_hessian(model, tok, g=G, n_tokens=16000, win=512):
    """Per target Linear, accumulate the block-diagonal input Hessian B[key] : [ng, g, g]."""
    raw = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data/corpora/generic-text/enwik8_256k"), "rb").read()
    calib = raw[:96000].decode("latin-1")
    ids = tok(calib, return_tensors="pt").input_ids[0][:n_tokens]
    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
    B, cnt = {}, {}

    def mk(key):
        def h(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            inn = x.shape[1]
            ng = (inn + g - 1) // g
            if inn % g:
                x = torch.cat([x, x.new_zeros(x.shape[0], ng * g - inn)], 1)
            xr = x.reshape(x.shape[0], ng, g).permute(1, 2, 0)     # [ng, g, N]
            bb = torch.bmm(xr, xr.transpose(1, 2))                 # [ng, g, g]
            B[key] = bb if key not in B else B[key] + bb
            cnt[key] = x.shape[0] + cnt.get(key, 0)
        return h

    hooks = [mod.register_forward_pre_hook(mk(name + ".weight"))
             for name, mod in model.named_modules()
             if isinstance(mod, nn.Linear) and (name + ".weight") in qk]
    with torch.no_grad():
        for s in range(0, len(ids) - 1, win):
            ch = ids[s:s + win].unsqueeze(0).to(DEV)
            if ch.shape[1] < 8:
                break
            model(ch)
    for h in hooks:
        h.remove()
    for k in B:
        B[k] = B[k] / max(cnt[k], 1)
    return B


def rotated_sensitivity(B):
    """D[gi,c] = diag(R^T B[gi] R), the output sensitivity of each rotated coordinate."""
    ng = B.shape[0]
    Rb = ROT.expand(ng, G, G)
    M = torch.bmm(torch.bmm(Rb.transpose(1, 2), B), Rb)            # [ng,g,g]
    D = torch.diagonal(M, dim1=1, dim2=2).clamp(min=0)            # [ng,g]
    return D


def _w_ecvq_levels(vals, wts, K, lam, iters=12, cap=40000):
    """Weighted entropy-constrained levels (torch, GPU). vals,wts: 1-D."""
    if vals.numel() > cap:
        sel = torch.randint(0, vals.numel(), (cap,), device=vals.device)
        vals, wts = vals[sel], wts[sel]
    q = torch.linspace(0, 1, K, device=vals.device)
    c = torch.quantile(vals, q).double()
    p = torch.full((K,), 1.0 / K, device=vals.device, dtype=torch.float64)
    v = vals.double()
    w = wts.double()
    for _ in range(iters):
        d = w[:, None] * (v[:, None] - c[None, :]) ** 2 - lam * torch.log2(p.clamp_min(1e-12))[None, :]
        idx = d.argmin(1)
        cnt = torch.bincount(idx, minlength=K).double()
        p = (cnt + 1e-9) / (cnt.sum() + 1e-9 * K)
        for k in range(K):
            m = idx == k
            if m.any():
                c[k] = (w[m] * v[m]).sum() / w[m].sum().clamp_min(1e-12)
    keep = torch.bincount(idx, minlength=K) > 0
    return torch.sort(c[keep].float()).values


def _entropy(idx, K):
    cnt = torch.bincount(idx.reshape(-1), minlength=K).double()
    p = cnt[cnt > 0] / cnt.sum()
    return float(-(p * torch.log2(p)).sum())


def daware_ecvq(W, B, lam, Kpool=64, g=G, mode="amax2d"):
    """Vectorized output-aware ECVQ. W:[out,inn] gpu; B:[ng,g,g] block Hessian. Returns (Wq, bits).
    mode: 'none' (control = plain ECVQ) | 'd' (rotated-coord sensitivity) | 'amax2d' (full output MSE)."""
    out, inn = W.shape
    ng = (inn + g - 1) // g
    pad = ng * g - inn
    A = torch.cat([W, W.new_zeros(out, pad)], 1) if pad else W
    N = A.reshape(out, ng, g).reshape(-1, g)                       # [out*ng, g], rowvec m=row*ng+gi
    R = (N * SIGNS) @ HAD                                          # rotate (== FWHT/sqrt g)
    amax = R.abs().max(1, keepdim=True).values.clamp_min(1e-8)     # [M,1] per row-vector
    Rn = R / amax                                                  # normalized

    if mode == "none":
        Wt = torch.ones_like(Rn)
    else:
        D = rotated_sensitivity(B)                                 # [ng,g]
        gi = (torch.arange(N.shape[0], device=W.device) % ng)
        Dm = D[gi]                                                 # [M,g]
        Wt = (amax ** 2) * Dm if mode == "amax2d" else Dm.expand(N.shape[0], g).clone()
    Wt = Wt / Wt.mean().clamp_min(1e-12)                           # normalize scale (lam comparable)

    lv = _w_ecvq_levels(Rn.reshape(-1), Wt.reshape(-1), Kpool, lam)
    K = lv.numel()
    # assignment: argmin_k Wt*(x-c)^2 - lam*log p_k, per element, chunked
    flatv = Rn.reshape(-1)
    flatw = Wt.reshape(-1)
    logp = torch.zeros(K, device=W.device)                        # filled after a first pass
    # first pass: nearest (unweighted) to seed p, then one weighted refine for codes
    codes = torch.empty(flatv.numel(), dtype=torch.long, device=W.device)
    CH = 2_000_000
    # seed p with nearest
    for s in range(0, flatv.numel(), CH):
        e = min(s + CH, flatv.numel())
        codes[s:e] = (flatv[s:e, None] - lv[None, :]).abs().argmin(1)
    cnt = torch.bincount(codes, minlength=K).double()
    p = (cnt + 1e-9) / cnt.sum()
    logp = (lam * torch.log2(p.clamp_min(1e-12))).float()
    for s in range(0, flatv.numel(), CH):
        e = min(s + CH, flatv.numel())
        d = flatw[s:e, None] * (flatv[s:e, None] - lv[None, :]) ** 2 - logp[None, :]
        codes[s:e] = d.argmin(1)

    Rh = (lv[codes].reshape(-1, g) * amax)                        # de-normalize
    back = (Rh @ HAD.T) * SIGNS                                   # inverse rotation
    Wq = back.reshape(out, ng * g)[:, :inn]
    ent = _entropy(codes, K)
    bits = ent * (out * inn) + 16 * N.shape[0] + 16 * K
    return Wq, bits


def run(points):
    tok = AutoTokenizer.from_pretrained(CFG)
    model = gpu_model()
    load_fp16_gpu(model)
    fp16 = ppl_gpu(model, tok)
    print(f"fp16 held-out ppl = {fp16:.3f}  (device {DEV})\n", flush=True)
    print("capturing block Hessians ...", flush=True)
    t = time.time()
    B = capture_block_hessian(model, tok)
    print(f"  {len(B)} block Hessians in {time.time()-t:.0f}s\n", flush=True)

    Worig = {}
    with safe_open(WPATH, framework="pt") as f:
        for k in quant_keys(f):
            Worig[k] = f.get_tensor(k).float()

    msd = model.state_dict()
    print(f"{'scheme':<26}{'bits/wt':>9}{'ppl':>10}{'time':>8}")
    print("-" * 53)
    results = []
    for name, lam, K, mode in points:
        t0 = time.time()
        load_fp16_gpu(model)
        bt, el = 0.0, 0
        for k in Worig:
            W = Worig[k].to(DEV)
            Wq, b = daware_ecvq(W, B[k], lam, K, mode=mode)
            msd[k].copy_(Wq.to(msd[k].dtype))
            bt += b
            el += W.numel()
            del W, Wq
        model.tie_weights()
        p = ppl_gpu(model, tok)
        bpw = bt / el
        results.append((name, bpw, p))
        print(f"{name:<26}{bpw:>9.3f}{p:>10.3f}{time.time()-t0:>7.0f}s", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    print(f"\nfp16 {fp16:.3f} | refs: ECVQ.008 4.483@3.13b | ECVQ.003 4.169@3.91b | "
          f"R14 fusion 4.272@3.02b")
    for n, bpw, p in sorted(results, key=lambda r: r[2]):
        print(f"  {p:8.3f} @ {bpw:.3f}b   {n}")
    return results


POINTS = [
    ("daware-ECVQ lam.004", 0.004, 64, "amax2d"),
    ("daware-ECVQ lam.008", 0.008, 64, "amax2d"),
    ("daware-ECVQ lam.015", 0.015, 64, "amax2d"),
    ("daware-ECVQ lam.030", 0.030, 64, "amax2d"),
]

# Diagnostic set: validate the fast path (control == plain ECVQ, must hit ~4.48@3.13b)
# and isolate the off-diagonal-rotation signal (D-only, no amax^2 domination).
DIAG = [
    ("control(none) lam.003", 0.003, 64, "none"),
    ("control(none) lam.008", 0.008, 64, "none"),
    ("Donly lam.003",         0.003, 64, "d"),
    ("Donly lam.008",         0.008, 64, "d"),
]


if __name__ == "__main__":
    import sys
    run(DIAG if len(sys.argv) > 1 and sys.argv[1] == "diag" else POINTS)
