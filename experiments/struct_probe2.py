"""Probe 2: chance baseline for greedy matching; SVD spectra; head-level alignment."""
import numpy as np
from safetensors import safe_open

WP = "C:/Users/Federico/Documents/evo-compress/weights/data/qwen/base.safetensors"

def rowcos(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return An @ Bn.T

def greedy_match_mean(C):
    C = np.abs(C.copy())
    n = min(C.shape)
    tot = 0.0
    for _ in range(n):
        i, j = np.unravel_index(C.argmax(), C.shape)
        tot += C[i, j]
        C[i, :] = -1
        C[:, j] = -1
    return tot / n

rng = np.random.default_rng(1)
print("=== chance baseline: greedy matched |cos| of pure-noise matrices ===")
for shape in [(512, 896), (128, 896)]:
    A = rng.standard_normal(shape).astype(np.float32)
    B = rng.standard_normal(shape).astype(np.float32)
    print(f"shape {shape}: chance greedy matched|cos| = {greedy_match_mean(rowcos(A,B)):.4f}")

with safe_open(WP, framework="pt") as f:
    pre = "model.layers.{}.{}.weight"
    g = lambda L, t: f.get_tensor(pre.format(L, t)).float().numpy()

    print()
    print("=== SVD spectra: energy fraction in top-r singular values ===")
    for typ in ["self_attn.q_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.down_proj"]:
        W = g(5, typ)
        s = np.linalg.svd(W, compute_uv=False)
        e = np.cumsum(s ** 2) / (s ** 2).sum()
        n = len(s)
        print(f"{typ:22s} rank={n}: top-32={e[31]:.3f} top-64={e[63]:.3f} "
              f"top-10%={e[n//10-1]:.3f} stable-rank={(s**2).sum()/s[0]**2:.0f}")

    print()
    print("=== head-level cross-layer alignment (q_proj heads, 64x896 blocks) ===")
    Q5 = g(5, "self_attn.q_proj").reshape(14, 64, 896)
    Q6 = g(6, "self_attn.q_proj").reshape(14, 64, 896)
    # head-to-head similarity: |cos| of flattened head matrices + subspace overlap
    Hc = rowcos(Q5.reshape(14, -1), Q6.reshape(14, -1))
    print("flattened-head |cos| matrix max:", np.abs(Hc).max().round(4))
    # subspace overlap: principal angles via top-16 row-space bases
    def rowbasis(M, k=16):
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        return Vt[:k]
    ov = np.zeros((14, 14))
    B5 = [rowbasis(Q5[i]) for i in range(14)]
    B6 = [rowbasis(Q6[j]) for j in range(14)]
    for i in range(14):
        for j in range(14):
            ov[i, j] = (np.linalg.svd(B5[i] @ B6[j].T, compute_uv=False) ** 2).mean()
    print(f"subspace overlap (top-16 row bases): mean={ov.mean():.3f} max={ov.max():.3f} "
          f"chance~{16/896:.3f}")
    # within-layer: q heads vs each other (head redundancy within a layer)
    ow = np.zeros((14, 14))
    for i in range(14):
        for j in range(14):
            if i != j:
                ow[i, j] = (np.linalg.svd(B5[i] @ B5[j].T, compute_uv=False) ** 2).mean()
    print(f"within-layer L5 q-head subspace overlap: mean(offdiag)={ow.sum()/(14*13):.3f}")
