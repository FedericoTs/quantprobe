"""ROUND 17 -- TWO-SIDED incoherence rotation (the full-QuIP lever we're missing).

So far we rotate only the INPUT dim (per-group Hadamard). QuIP/QuIP# rotate BOTH sides:
W' = U W V with U (out-side), V (in-side) orthogonal -> W' is maximally incoherent
(flat, sub-Gaussian) so scalar quant error is smaller. Because we UNDO both rotations in
reconstruction (W_hat = U^T Q V^T), we don't need to propagate rotations through the network
-- this is a clean test of "does double rotation make the weights quantize better?".

Fully vectorized (block-grid einsums + bucketize assignment); no Python inner loop.
Control = single-sided (input-only) in the SAME harness for a clean A/B.

Run:  python -m weights.quant_rot
"""
from __future__ import annotations

import gc
import time

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from weights.quant_dataaware import DEV, G, _hadamard, gpu_model, load_fp16_gpu, ppl_gpu
from weights.quant_fast import HAD, SIGNS, _entropy, _w_ecvq_levels
from weights.quant_lab import CFG, WPATH, quant_keys

SIGNS_O = (torch.from_numpy(np.random.default_rng(1).integers(0, 2, G).astype(np.float32))
           * 2 - 1).to(DEV)

_OCACHE = {}


def _out_rot(go):
    """Hadamard + sign block for an output group of size `go` (cached)."""
    if go not in _OCACHE:
        had = torch.from_numpy(_hadamard(go)).to(DEV)
        s = (torch.from_numpy(np.random.default_rng(1).integers(0, 2, go).astype(np.float32))
             * 2 - 1).to(DEV)
        _OCACHE[go] = (had, s)
    return _OCACHE[go]


def _assign_nearest(x, lv, chunk=4_000_000):
    K = lv.numel()
    out = torch.empty(x.numel(), dtype=torch.long, device=x.device)
    xf = x.reshape(-1)
    for s in range(0, xf.numel(), chunk):
        e = min(s + chunk, xf.numel())
        ci = torch.bucketize(xf[s:e], lv).clamp(1, K - 1)
        left = (xf[s:e] - lv[ci - 1]).abs() <= (xf[s:e] - lv[ci]).abs()
        out[s:e] = torch.where(left, ci - 1, ci)
    return out


def rot_ecvq(W, lam, K=64, g=G, two=True, go=G, p_out=0.0):
    out, inn = W.shape
    HADO, SO = _out_rot(go)

    # outlier preservation: keep top-p |W| in fp16, zero them before rotation
    if p_out > 0:
        thr = torch.quantile(W.abs().reshape(-1)[torch.randint(0, W.numel(), (200000,), device=W.device)],
                             1.0 - p_out)
        mask = W.abs() >= thr
        base = torch.where(mask, torch.zeros_like(W), W)
        n_out = int(mask.sum())
    else:
        base, mask, n_out = W, None, 0

    no = (out + go - 1) // go
    ni = (inn + g - 1) // g
    Wp = torch.zeros(no * go, ni * g, device=W.device)
    Wp[:out, :inn] = base
    X = Wp.reshape(no, go, ni, g)                                # [O, a, I, b]

    Xi = torch.einsum('Oaib,bc->Oaic', X * SIGNS, HAD)           # input rotation
    Xo = (torch.einsum('Oaic,ad->Odic', Xi * SO[None, :, None, None], HADO)
          if two else Xi)                                       # output rotation
    amax = Xo.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-8)  # [O,1,I,1] per block
    Xn = Xo / amax

    lv = _w_ecvq_levels(Xn.reshape(-1), torch.ones(Xn.numel(), device=W.device), K, lam)
    codes = _assign_nearest(Xn, lv)
    Kk = lv.numel()
    Xq = lv[codes].reshape(Xo.shape) * amax

    if two:
        Xi2 = torch.einsum('Odic,ad->Oaic', Xq, HADO) * SO[None, :, None, None]
    else:
        Xi2 = Xq
    Y = torch.einsum('Oaic,bc->Oaib', Xi2, HAD) * SIGNS
    Wq = Y.reshape(no * go, ni * g)[:out, :inn].contiguous()
    if p_out > 0:
        Wq[mask] = W[mask]

    ent = _entropy(codes, Kk)
    bits = ent * (out * inn) + 16 * (no * ni) + 16 * Kk + n_out * 32
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
    for name, lam, K, two, go, p_out in points:
        t0 = time.time()
        load_fp16_gpu(model)
        bt, el = 0.0, 0
        for k in Worig:
            W = Worig[k].to(DEV)
            Wq, b = rot_ecvq(W, lam, K, two=two, go=go, p_out=p_out)
            msd[k].copy_(Wq.to(msd[k].dtype))
            bt += b; el += W.numel()
            del W, Wq
        model.tie_weights()
        p = ppl_gpu(model, tok)
        bpw = bt / el
        results.append((name, bpw, p))
        print(f"{name:<26}{bpw:>9.3f}{p:>10.3f}{time.time()-t0:>7.0f}s", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    print(f"\nfp16 {fp16:.3f} | refs: ECVQ.008 4.483@3.13b | ECVQ.003 4.169@3.91b")
    for n, bpw, p in sorted(results, key=lambda r: r[2]):
        print(f"  {p:8.3f} @ {bpw:.3f}b   {n}")
    return results


POINTS = [
    ("1-sided lam.003", 0.003, 64, False, G, 0.0),
    ("2-sided lam.003", 0.003, 64, True, G, 0.0),
    ("1-sided lam.006", 0.006, 64, False, G, 0.0),
    ("2-sided lam.006", 0.006, 64, True, G, 0.0),
    ("1-sided lam.010", 0.010, 64, False, G, 0.0),
    ("2-sided lam.010", 0.010, 64, True, G, 0.0),
]

# R18: two-sided + 0.5% outliers, vs the tuned codec_zoo ECVQ frontier (4.483@3.13b, 4.169@3.91b).
# go32 = finer output-group scales; 1-sided+out (block) as in-harness reference.
POINTS_OUT = [
    ("2sided+out go128 l.003", 0.003, 64, True, 128, 0.005),
    ("2sided+out go128 l.005", 0.005, 64, True, 128, 0.005),
    ("2sided+out go128 l.008", 0.008, 64, True, 128, 0.005),
    ("2sided+out go32  l.005", 0.005, 64, True, 32, 0.005),
    ("2sided+out go32  l.008", 0.008, 64, True, 32, 0.005),
    ("1sided+out go128 l.005", 0.005, 64, False, 128, 0.005),
]


if __name__ == "__main__":
    import sys
    run(POINTS_OUT if len(sys.argv) > 1 and sys.argv[1] == "out" else POINTS)
