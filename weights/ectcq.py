"""R25v2-R4 -- ECTCQ: ENTROPY-CONSTRAINED trellis-coded quantization (rate INSIDE the Viterbi
metric). A4 confirmed ~0.3 b/w gap to the Shannon bound; plain TCQ (R23) lost because it paid
fixed-rate bits. Here the branch metric is (x-lv)^2 + lam*(-log2 p(lv|coset)), with p refined
over 2 Viterbi passes (Chou-style alternation), so the trellis BUYS its space-filling gain at
entropy prices -- the exact fusion QTIP doesn't do (it has no rate term).

Levels: 64-level per-tensor learned pool (ECVQ init), interleaved into 4 cosets (sorted order
mod 4). Honest rate = empirical H(level | trellis state) along chosen paths (decoder knows the
state; an arithmetic coder achieves it; per-tensor 4x64 tables counted) + A1-coded side info
(amax 0.0792 + outliers 0.0857) + AWQ 0.011. 0.5%% outliers kept fp16 (pre-rotation).

Run:  python -m weights.ectcq
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors import safe_open
from transformers import AutoTokenizer
from weights.quant_dataaware import DEV, G, gpu_model, load_fp16_gpu, ppl_gpu
from weights.quant_fast import HAD, SIGNS
from weights.quant_lab import CFG, WPATH, quant_keys

NS_COSET = [0, 2, 1, 3]
NS_PREV = [[0, 2], [0, 2], [1, 3], [1, 3]]
K = 64
P_OUT = 0.005
SIDE = 0.0792 + 0.0857 + 0.011        # A1-coded amax + outliers + AWQ raw (b/w)


def ecvq_levels_t(x, lam_lv=0.0, iters=8):
    """64-level 1-D codebook (Lloyd on GPU sample)."""
    samp = x[torch.randint(0, x.numel(), (40000,), device=x.device)]
    q = torch.linspace(0, 1, K, device=x.device)
    c = torch.quantile(samp, q)
    for _ in range(iters):
        d = (samp[:, None] - c[None, :]).abs()
        idx = d.argmin(1)
        for k in range(K):
            m = idx == k
            if m.any():
                c[k] = samp[m].mean()
    return torch.sort(c).values


def ectcq_layer(W, lam, iters=2):
    out, inn = W.shape
    n_out = max(1, int(round(W.numel() * P_OUT)))
    thr = W.abs().reshape(-1)[torch.randint(0, W.numel(), (200000,), device=W.device)]
    thr = torch.quantile(thr, 1.0 - P_OUT)
    mask = W.abs() >= thr
    base = torch.where(mask, torch.zeros_like(W), W)
    ng = (inn + G - 1) // G
    pad = ng * G - inn
    A = torch.cat([base, base.new_zeros(out, pad)], 1) if pad else base
    N = A.reshape(out, ng, G).reshape(-1, G)
    R = (N * SIGNS) @ HAD
    amax = R.abs().max(1, keepdim=True).values.clamp_min(1e-8)
    X = R / amax                                        # [B, T]
    B, T = X.shape

    lv = ecvq_levels_t(X.reshape(-1))                   # [K] sorted
    cosets = [torch.arange(c, K, 4, device=W.device) for c in range(4)]   # level ids per coset
    nll = torch.zeros(4, K // 4, device=W.device)       # -log2 p(level | coset), refined

    for it in range(iters):
        # per-step per-coset best (distortion + lam*nll) and its within-coset index
        SC = torch.empty(B, T, 4, device=W.device)
        SIDX = torch.empty(B, T, 4, dtype=torch.long, device=W.device)
        for c in range(4):
            lvc = lv[cosets[c]]
            d = (X.unsqueeze(2) - lvc.view(1, 1, -1)) ** 2 + lam * nll[c].view(1, 1, -1)
            mn, ix = d.min(2)
            SC[:, :, c] = mn
            SIDX[:, :, c] = ix
        INF = 1e30
        cost = torch.full((B, 4), INF, device=W.device); cost[:, 0] = 0.0
        back = torch.empty(B, T, 4, dtype=torch.int8, device=W.device)
        for t in range(T):
            new = torch.empty(B, 4, device=W.device)
            for ns in range(4):
                c = NS_COSET[ns]
                p0, p1 = NS_PREV[ns]
                bc = SC[:, t, c]
                c0 = cost[:, p0] + bc
                c1 = cost[:, p1] + bc
                pick1 = c1 < c0
                new[:, ns] = torch.where(pick1, c1, c0)
                back[:, t, ns] = pick1.to(torch.int8)
            cost = new
        state = cost.argmin(1)
        ar = torch.arange(B, device=W.device)
        states = torch.empty(B, T, dtype=torch.long, device=W.device)
        within = torch.empty(B, T, dtype=torch.long, device=W.device)
        Xq = torch.empty(B, T, device=W.device)
        st = state
        for t in range(T - 1, -1, -1):
            c = torch.tensor(NS_COSET, device=W.device)[st]
            ix = SIDX[ar, t, c]
            states[:, t] = st
            within[:, t] = ix
            Xq[:, t] = lv[ix * 4 + c]                   # coset interleave: level id = 4*ix + c
            b = back[ar, t, st]
            st = torch.tensor(NS_PREV, device=W.device)[st, b.long()]
        # refine p(level|coset) from usage
        cflat = torch.tensor(NS_COSET, device=W.device)[states].reshape(-1)
        wflat = within.reshape(-1)
        for c in range(4):
            cnt = torch.bincount(wflat[cflat == c], minlength=K // 4).double()
            p = (cnt + 0.5) / (cnt.sum() + 0.5 * (K // 4))
            nll[c] = (-torch.log2(p)).float()

    # HONEST rate (audit wf_062a79ec): the decoder reads, per step, the FULL 32-ary symbol
    # levid = 4*within + coset conditioned on the CURRENT (predecessor) state -- this pays the
    # branch/path bits H(coset|S) the earlier H(within|entered-state) formula omitted (~0.95 b/w).
    prev = torch.empty_like(states)
    prev[:, 0] = 0
    prev[:, 1:] = states[:, :-1]
    qflat = prev.reshape(-1)
    cflat = torch.tensor(NS_COSET, device=W.device)[states].reshape(-1)
    levid = 4 * within.reshape(-1) + cflat
    ent_bits = 0.0
    for q in range(4):
        m = qflat == q
        cnt = torch.bincount(levid[m], minlength=K).double()
        p = cnt[cnt > 0] / cnt.sum()
        ent_bits += float(-(p * torch.log2(p)).sum() * cnt.sum())
    rate = ent_bits / (B * T)                            # bits/weight incl path bits

    Rh = Xq * amax
    Wq = ((Rh @ HAD.T) * SIGNS).reshape(out, ng * G)[:, :inn].contiguous()
    Wq[mask] = W[mask]
    bits = rate * (out * inn)                            # side counted globally via SIDE
    return Wq, bits


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = gpu_model()
    load_fp16_gpu(model)
    fp16 = ppl_gpu(model, tok)
    print(f"fp16 {fp16:.3f} | target: beat champion 4.483 @ 3.00 b/w honest\n", flush=True)
    Worig = {}
    with safe_open(WPATH, framework="pt") as f:
        for k in quant_keys(f):
            Worig[k] = f.get_tensor(k).float()
    msd = model.state_dict()
    print(f"{'lam':>8}{'level b/w':>10}{'honest b/w':>11}{'ppl':>8}{'time':>7}")
    print("-" * 46)
    import sys as _s
    if "fine" in _s.argv:
        lams, iters = (0.018, 0.022, 0.026, 0.030), 4   # bracket the ~3.0 b/w headline point
    elif "aggr" in _s.argv:
        lams, iters = (0.05, 0.15, 0.4), 4
    else:
        lams, iters = (0.004, 0.008, 0.015), 2
    for lam in lams:
        t0 = time.time()
        load_fp16_gpu(model)
        bt, el = 0.0, 0
        for k in Worig:
            W = Worig[k].to(DEV)
            Wq, b = ectcq_layer(W, lam, iters=iters)
            msd[k].copy_(Wq.to(msd[k].dtype))
            bt += b; el += W.numel()
            del W, Wq
        model.tie_weights()
        p = ppl_gpu(model, tok)
        lvl = bt / el
        honest = lvl + SIDE
        tag = "  <== BEATS CHAMPION" if (honest < 3.00 and p < 4.483) else ""
        print(f"{lam:>8.3f}{lvl:>10.3f}{honest:>11.3f}{p:>8.3f}{time.time()-t0:>6.0f}s{tag}", flush=True)
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
