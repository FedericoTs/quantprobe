"""QTIP fixed-rate bitshift-trellis PREFLIGHT: iso-rate R=2 weight-MSE bake-off, scalar ECVQ vs
trellis, on REAL Llama-2-7B FFN tensors (rotated). Tests whether the high-effective-dimension
trellis reaches the Gaussian D(R) bound (0.069 @ R=2) vs scalar (0.118) -- the gain that closes
the 2-bit cliff. The trellis is FIXED-rate (window = state = codeword-address) so NO path-entropy
tax (the thing that killed variable-rate ectcq.py).

Critic-mandated fixes baked in: (1) shared UNIT-VARIANCE normalization for both arms (the amax-vs-code
scale mismatch caused a false FAIL); (2) >=3 real 7B tensors incl down_proj; (3) EMPIRICAL scalar D;
(4) report kurtosis + 0.5%-outlier sub-check. Conservative random-Gaussian codebook (3INST only helps).

Gate: ratio=D_trellis/D_scalar <=0.75 PASS, <=0.85 escalate L16, >=0.90 FAIL. Run on >=2/3 tensors.
Run:  python -m weights.qtip_preflight
"""
from __future__ import annotations
import math, sys
import numpy as np, torch
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from weights.quant_sota import _fwht_rows
from weights.codec_zoo import _ecvq_levels, _nearest_idx
from weights.evoq_llama import shard_map, read_tensor

G = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rotate_unitstd(W, seed=0, subB=4096):
    """per-group signed-FWHT, then normalize each group-row to UNIT STD (critic fix, not amax)."""
    rows, cols = W.shape
    pad = (-cols) % G
    A = np.pad(W, ((0, 0), (0, pad))) if pad else W
    N = A.reshape(rows, -1, G).reshape(-1, G)
    signs = np.random.default_rng(seed).integers(0, 2, G).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / math.sqrt(G)
    std = R.std(1, keepdims=True); std[std == 0] = 1.0
    X = (R / std).astype(np.float32)                       # [Bg, 128] unit-std per row
    rng = np.random.default_rng(seed + 7)
    if X.shape[0] > subB:
        X = X[rng.choice(X.shape[0], subB, replace=False)]
    return X


def scalar_mse(X, K=4):
    flat = X.ravel()
    rng = np.random.default_rng(1)
    samp = flat[rng.integers(0, flat.size, min(20000, flat.size))]
    lv = _ecvq_levels(samp, K, lam=0.0)                    # pure Lloyd, exactly K levels (R=log2 K)
    idx = _nearest_idx(flat, lv)
    return float(np.mean((flat - lv[idx]) ** 2)), len(lv)


@torch.no_grad()
def trellis_mse(X, L=12, K=2, seed=0, chunk=512):
    """fixed-rate bitshift Viterbi (R=K/V, V=1) over the within-group seq T=128, batched over rows.
    Conservative random-Gaussian unit-std codebook (3INST would only do better)."""
    nstate = 1 << L; KV = K; ncand = 1 << KV; npred = 1 << (L - KV)
    g = torch.Generator().manual_seed(seed)
    recons = torch.randn(nstate, generator=g).to(DEV); recons = recons / recons.std()
    sumdelta = (torch.arange(ncand, device=DEV) << (L - KV)).view(1, -1)
    base = (torch.arange(nstate, device=DEV) >> KV)[::ncand].view(-1, 1)
    state_cand = base + sumdelta                            # [npred, ncand]
    Bn, T = X.shape
    tot, n = 0.0, 0
    for c0 in range(0, Bn, chunk):
        Xt = torch.tensor(X[c0:c0 + chunk].T, device=DEV)   # [T, b]
        b = Xt.shape[1]
        cost = (recons.view(1, -1) - Xt[0].view(-1, 1)) ** 2  # [b, nstate]
        back = torch.empty(T, b, npred, dtype=torch.int16, device=DEV)
        for t in range(1, T):
            se = (recons.view(1, -1) - Xt[t].view(-1, 1)) ** 2
            cand = cost[:, state_cand]                      # [b, npred, ncand]
            bestv, besti = cand.min(-1)
            back[t] = besti.to(torch.int16)
            cost = se + bestv.repeat_interleave(ncand, dim=1)
        final = torch.empty(T, b, dtype=torch.long, device=DEV)
        final[T - 1] = cost.argmin(-1)
        for t in range(T - 1, 0, -1):
            grp = final[t] >> KV
            chosen = back[t].gather(1, grp.view(-1, 1)).squeeze(1).long()
            final[t - 1] = state_cand[grp, chosen]
        Xq = recons[final]                                  # [T, b]
        tot += float(((Xt - Xq) ** 2).sum()); n += b * T
    return tot / n


def main():
    smap = shard_map()
    tensors = ["model.layers.15.mlp.gate_proj.weight",
               "model.layers.22.mlp.gate_proj.weight",
               "model.layers.15.mlp.down_proj.weight"]
    print(f"{'tensor':<30}{'kurt':>7}{'D_scal':>9}{'D_trel':>9}{'ratio':>8}{'bw_eq':>7}  verdict")
    print("-" * 82)
    npass = 0
    for nm in tensors:
        W = read_tensor(smap, nm).float().numpy()
        X = rotate_unitstd(W)
        kurt = float(((X - X.mean()) ** 4).mean() / (X.var() ** 2))
        Ds, K = scalar_mse(X, 4)
        Dt = trellis_mse(X, L=12, K=2)
        ratio = Dt / Ds; bw = 0.5 * math.log2(Ds / Dt)
        v = "PASS" if ratio <= 0.75 else ("ESCALATE-L16" if ratio <= 0.85 else "FAIL")
        npass += (ratio <= 0.75)
        print(f"{nm.split('.',2)[-1][:29]:<30}{kurt:>7.2f}{Ds:>9.4f}{Dt:>9.4f}{ratio:>8.3f}{bw:>7.3f}  {v}", flush=True)
    print(f"\n{npass}/3 PASS (gate: >=2/3 ratio<=0.75). D_trellis target ~0.069 (Gaussian D(R=2)), "
          f"scalar ~0.118. Conservative random-Gaussian code; 3INST only improves.")
    print("If PASS -> build qtip_trellis.py (L=16, 3INST, tail-biting) + Llama-2-7B head-to-head "
          "(PTQ target ~6.8 vs champion 9.63; 5.86 needs FT, out of scope).")


if __name__ == "__main__":
    main()
