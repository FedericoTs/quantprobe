"""ROUND 16 -- TRAINED entropy-constrained VECTOR quantization (the space-filling-gain axis).

The fixed D4/E8 lattices failed once counted honestly (joint entropy of near-unique
fine-resolution points saturates -> undercount). A TRAINED finite codebook avoids that:
with K<=256 codewords the index stream's entropy is genuinely achievable (a real range
coder hits it), AND the codebook ADAPTS to the actual rotated-weight distribution rather
than imposing a fixed lattice. We train a d-dim entropy-constrained VQ (ECVQ in R^d:
assign x to argmin_k ||x-c_k||^2 - lam*log2 p_k; rate term prunes codewords) and pay the
index entropy / d per weight. Fully vectorized (chunked [P,K] argmin); no Python inner loop.

Run:  python -m weights.quant_vq
"""
from __future__ import annotations

import gc
import os
import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from weights.quant_dataaware import DEV, G, gpu_model, load_fp16_gpu, ppl_gpu
from weights.quant_fast import HAD, SIGNS, _entropy
from weights.quant_lab import CFG, WPATH, quant_keys


def _assign(x, c, logp, chunk=1_000_000):
    """argmin_k ||x-c_k||^2 - logp_k, chunked. x:[P,d], c:[K,d], logp:[K]. Returns idx:[P]."""
    P = x.shape[0]
    csq = (c * c).sum(1)                              # [K]
    out = torch.empty(P, dtype=torch.long, device=x.device)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        xc = x[s:e] @ c.t()                           # [n,K]
        d = (x[s:e] * x[s:e]).sum(1, keepdim=True) - 2 * xc + csq[None, :] - logp[None, :]
        out[s:e] = d.argmin(1)
    return out


def vq_ecvq(W, lam, K=256, d=2, g=G, iters=12, cap=60000, p_out=0.0):
    out, inn = W.shape
    # outlier preservation: keep top-p |W| as fp16, zero before rotation
    if p_out > 0:
        thr = torch.quantile(W.abs().reshape(-1)[torch.randint(0, W.numel(), (200000,), device=W.device)],
                             1.0 - p_out)
        mask = W.abs() >= thr
        Wb = torch.where(mask, torch.zeros_like(W), W)
        n_out = int(mask.sum())
    else:
        Wb, mask, n_out = W, None, 0
    ng = (inn + g - 1) // g
    pad = ng * g - inn
    A = torch.cat([Wb, Wb.new_zeros(out, pad)], 1) if pad else Wb
    N = A.reshape(out, ng, g).reshape(-1, g)
    R = (N * SIGNS) @ HAD
    amax = R.abs().max(1, keepdim=True).values.clamp_min(1e-8)
    Rn = R / amax                                     # [M, g] normalized
    pts = Rn.reshape(-1, d)                           # [P, d]
    P = pts.shape[0]

    sub = pts[torch.randint(0, P, (min(cap, P),), device=W.device)]
    c = sub[torch.randperm(sub.shape[0], device=W.device)[:K]].clone()   # data-init
    logp = torch.zeros(K, device=W.device)
    for _ in range(iters):
        idx = _assign(sub, c, logp)
        cnt = torch.bincount(idx, minlength=K).float()
        p = (cnt + 1e-9) / (cnt.sum() + 1e-9 * K)
        logp = lam * torch.log2(p.clamp_min(1e-12))
        sums = torch.zeros(K, d, device=W.device).index_add_(0, idx, sub)
        c = torch.where(cnt[:, None] > 0, sums / cnt[:, None].clamp_min(1), c)

    idx_all = _assign(pts, c, logp)
    rec = c[idx_all].reshape(-1, g) * amax            # de-normalize
    back = (rec @ HAD.t()) * SIGNS
    Wq = back.reshape(out, ng * g)[:, :inn].contiguous()
    if p_out > 0:
        Wq[mask] = W[mask]
    used = torch.bincount(idx_all, minlength=K) > 0
    ent = _entropy(idx_all, K)                        # bits per d-vector
    bits = (ent / d) * (out * inn) + 16 * N.shape[0] + int(used.sum()) * d * 16 + n_out * 32
    return Wq, bits


def run(points):
    tok = AutoTokenizer.from_pretrained(CFG)
    model = gpu_model()
    load_fp16_gpu(model)
    fp16 = ppl_gpu(model, tok)
    print(f"fp16 held-out ppl = {fp16:.3f}  (device {DEV})\n", flush=True)

    Worig = {}
    with safe_open(WPATH, framework="pt") as f:
        for k in quant_keys(f):
            Worig[k] = f.get_tensor(k).float()

    msd = model.state_dict()
    print(f"{'scheme':<26}{'bits/wt':>9}{'ppl':>10}{'time':>8}")
    print("-" * 53)
    results = []
    for name, lam, K, d, p_out in points:
        t0 = time.time()
        load_fp16_gpu(model)
        bt, el = 0.0, 0
        for k in Worig:
            W = Worig[k].to(DEV)
            Wq, b = vq_ecvq(W, lam, K, d, p_out=p_out)
            msd[k].copy_(Wq.to(msd[k].dtype))
            bt += b; el += W.numel()
            del W, Wq
        model.tie_weights()
        p = ppl_gpu(model, tok)
        bpw = bt / el
        results.append((name, bpw, p))
        print(f"{name:<26}{bpw:>9.3f}{p:>10.3f}{time.time()-t0:>7.0f}s", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    print(f"\nfp16 {fp16:.3f} | refs: ECVQ.008 4.483@3.13b | ECVQ.003 4.169@3.91b | entropy16 4.109@4.24b")
    for n, bpw, p in sorted(results, key=lambda r: r[2]):
        print(f"  {p:8.3f} @ {bpw:.3f}b   {n}")
    return results


POINTS = [
    ("VQ2d K256 lam.002", 0.002, 256, 2, 0.0),
    ("VQ2d K256 lam.004", 0.004, 256, 2, 0.0),
    ("VQ2d K256 lam.006", 0.006, 256, 2, 0.0),
]

# R24: RECOMBINE confirmed winners -- ECVQ + vector-quant + 0.5% OUTLIERS.
# Clean d=1 (scalar) vs d=2 (vector) head-to-head, BOTH with outliers, matched bits.
# The GA-spirit move: does vector quant edge scalar ECVQ once both have the outlier lever?
POINTS3 = [
    ("d1+out lam.003", 0.003, 64, 1, 0.005),
    ("d1+out lam.006", 0.006, 64, 1, 0.005),
    ("d2+out lam.002", 0.002, 256, 2, 0.005),
    ("d2+out lam.004", 0.004, 256, 2, 0.005),
    ("d2+out lam.006", 0.006, 256, 2, 0.005),
]


if __name__ == "__main__":
    import sys
    run(POINTS3 if len(sys.argv) > 1 and sys.argv[1] == "p3" else POINTS)
