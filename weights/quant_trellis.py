"""ROUND 23 -- TRELLIS-CODED QUANTIZATION (TCQ, Marcellin-Fischer 1990; the structure behind
QTIP, the 2024 sub-4-bit SOTA). The last untried codec-space SOTA lever. A 4-state trellis with
a 2^(R+1)-level codebook split into 4 cosets (D0..D3); Viterbi picks the min-distortion coset
sequence -> ~1.3 dB coding gain over scalar quant at the same rate. Fully vectorized: Viterbi is
batched over ALL (row x group) sequences in parallel, looping only over the 128-length sequence
dimension (no per-element Python loop). Optionally entropy-codes the level indices (our edge).

Run:  python -m weights.quant_trellis
"""
from __future__ import annotations

import gc
import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from weights.quant_dataaware import DEV, G, gpu_model, load_fp16_gpu, ppl_gpu
from weights.quant_fast import HAD, SIGNS, _entropy
from weights.quant_lab import CFG, WPATH, quant_keys

# next_state -> coset index (which of the 4 interleaved subsets labels transitions INTO it)
NS_COSET = torch.tensor([0, 2, 1, 3], device=DEV)
# next_state -> the two possible previous states
NS_PREV = torch.tensor([[0, 2], [0, 2], [1, 3], [1, 3]], device=DEV)


def tcq(W, bits=3, g=G, entropy=True):
    """4-state TCQ on per-(row,group) rotated, amax-normalized weights. R=bits.
    Codebook = 2^(bits+1) uniform levels in [-1,1], cosets = levels[c::4]."""
    out, inn = W.shape
    ng = (inn + g - 1) // g
    pad = ng * g - inn
    A = torch.cat([W, W.new_zeros(out, pad)], 1) if pad else W
    N = A.reshape(out, ng, g).reshape(-1, g)               # [B, T=g]
    R = (N * SIGNS) @ HAD
    amax = R.abs().max(1, keepdim=True).values.clamp_min(1e-8)
    X = (R / amax)                                         # [B, T] in ~[-1,1]
    B, T = X.shape

    K = 1 << (bits + 1)                                    # codebook size (2x rate)
    lv = torch.linspace(-1.0, 1.0, K, device=W.device)     # uniform levels
    cosets = [lv[c::4] for c in range(4)]                  # 4 subsets, K/4 levels each

    # per-step per-coset best level cost + within-coset index
    SC = torch.empty(B, T, 4, device=W.device)
    SIDX = torch.empty(B, T, 4, dtype=torch.long, device=W.device)
    for c in range(4):
        d = (X.unsqueeze(2) - cosets[c].view(1, 1, -1)) ** 2   # [B,T,|coset|]
        mn, ix = d.min(2)
        SC[:, :, c] = mn
        SIDX[:, :, c] = ix

    INF = 1e30
    cost = torch.full((B, 4), INF, device=W.device)
    cost[:, 0] = 0.0
    back = torch.empty(B, T, 4, dtype=torch.int8, device=W.device)   # which of 2 prev states
    for t in range(T):
        new = torch.empty(B, 4, device=W.device)
        for ns in range(4):
            c = int(NS_COSET[ns])
            p0, p1 = int(NS_PREV[ns, 0]), int(NS_PREV[ns, 1])
            bc = SC[:, t, c]
            cand0 = cost[:, p0] + bc
            cand1 = cost[:, p1] + bc
            pick1 = cand1 < cand0
            new[:, ns] = torch.where(pick1, cand1, cand0)
            back[:, t, ns] = pick1.to(torch.int8)
        cost = new

    # backtrack
    state = cost.argmin(1)                                  # [B]
    codes = torch.empty(B, T, dtype=torch.long, device=W.device)
    Xq = torch.empty(B, T, device=W.device)
    ar = torch.arange(B, device=W.device)
    for t in range(T - 1, -1, -1):
        c = NS_COSET[state]                                 # [B] coset at this step
        idx_in = SIDX[ar, t, c]                             # within-coset level idx
        glob = c + 4 * idx_in                               # global level index (c::4 indexing)
        codes[:, t] = glob
        Xq[:, t] = lv[glob]
        b = back[ar, t, state]                              # 0/1 -> which prev
        state = NS_PREV[state, b.long()]

    Rh = Xq * amax
    Wq = ((Rh @ HAD.T) * SIGNS).reshape(out, ng * g)[:, :inn]
    if entropy:
        ent = _entropy(codes, K)                            # bits/weight (achievable)
    else:
        ent = float(bits)                                   # fixed-rate: R bits/weight
    total = ent * (out * inn) + 16 * B
    return Wq, total


def run(points):
    tok = AutoTokenizer.from_pretrained(CFG)
    model = gpu_model()
    load_fp16_gpu(model)
    fp16 = ppl_gpu(model, tok)
    print(f"fp16 held-out ppl = {fp16:.3f}\n", flush=True)

    Worig = {}
    with safe_open(WPATH, framework="pt") as f:
        for k in quant_keys(f):
            Worig[k] = f.get_tensor(k).float()

    msd = model.state_dict()
    print(f"{'scheme':<26}{'bits/wt':>9}{'ppl':>10}{'time':>8}")
    print("-" * 53)
    results = []
    for name, bits, ent in points:
        t0 = time.time()
        load_fp16_gpu(model)
        bt, el = 0.0, 0
        for k in Worig:
            W = Worig[k].to(DEV)
            Wq, b = tcq(W, bits, entropy=ent)
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
    ("TCQ R2 fixed",   2, False),
    ("TCQ R2 entropy", 2, True),
    ("TCQ R3 fixed",   3, False),
    ("TCQ R3 entropy", 3, True),
    ("TCQ R4 entropy", 4, True),
]


if __name__ == "__main__":
    run(POINTS)
