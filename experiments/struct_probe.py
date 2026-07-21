"""Quick empirical probe of weight-space structure in Qwen2.5-0.5B (CPU, numpy).
Q1 cross-layer redundancy: raw + permutation-aligned row correlation, adjacent layers.
Q2 column (input-channel) variance heterogeneity -> permutation-grouping headroom.
Q3 scale side-channel: entropy of per-block amax stream vs the 16 bits currently paid.
Q4 kurtosis by tensor type (statistical niches).
"""
import numpy as np
from safetensors import safe_open

WP = "C:/Users/Federico/Documents/evo-compress/weights/data/qwen/base.safetensors"
G = 128

def get(f, name):
    return f.get_tensor(name).float().numpy()

def rowcos(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return An @ Bn.T

def greedy_match_mean(C):
    """Greedy max |cos| matching (upper bound proxy for Hungarian)."""
    C = np.abs(C.copy())
    n = min(C.shape)
    tot = 0.0
    for _ in range(n):
        i, j = np.unravel_index(C.argmax(), C.shape)
        tot += C[i, j]
        C[i, :] = -1
        C[:, j] = -1
    return tot / n

def fwht(a):
    h = 1
    a = a.copy()
    n = a.shape[-1]
    while h < n:
        a = a.reshape(*a.shape[:-1], n // (2 * h), 2, h)
        x, y = a[..., 0, :], a[..., 1, :]
        a = np.concatenate([x + y, x - y], axis=-1).reshape(*a.shape[:-3], -1)
        h *= 2
    return a

def ent_bits(idx):
    c = np.bincount(idx)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log2(p)).sum())

with safe_open(WP, framework="pt") as f:
    keys = list(f.keys())
    pre = "model.layers.{}.{}.weight"

    print("=== Q1: cross-layer redundancy (adjacent layers) ===")
    for typ in ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.down_proj"]:
        A = get(f, pre.format(5, typ))
        B = get(f, pre.format(6, typ))
        flat = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
        # row-space alignment (sample 512 rows if big)
        ra, rb = A, B
        if A.shape[0] > 1024:
            sel = np.random.default_rng(0).choice(A.shape[0], 512, replace=False)
            ra, rb = A[sel], B[sel]
        C = rowcos(ra, rb)
        m = greedy_match_mean(C)
        rnd = 1.0 / np.sqrt(A.shape[1])
        print(f"{typ:22s} flat-corr={flat:+.4f}  matched|cos|={m:.4f}  (random~{rnd:.4f})  shape={A.shape}")

    # also: same tensor, distant layers + delta compressibility proxy
    A = get(f, pre.format(5, "self_attn.q_proj")); B = get(f, pre.format(6, "self_attn.q_proj"))
    d = B - A
    print(f"q_proj L6-L5 delta std ratio: std(delta)/std(B) = {d.std()/B.std():.3f} (1.41=independent)")

    print()
    print("=== Q2: input-channel (column) std heterogeneity ===")
    for typ in ["self_attn.q_proj", "mlp.down_proj", "self_attn.o_proj"]:
        W = get(f, pre.format(5, typ))
        cs = W.std(0)
        srt = np.sort(cs)
        # grouping-headroom proxy: bits saved = 0.5*log2(AMvar/GMvar) within groups,
        # contiguous vs sorted columns
        def group_loss(c):
            v = (c ** 2).reshape(-1, G) if len(c) % G == 0 else (c[: len(c)//G*G] ** 2).reshape(-1, G)
            am = v.mean(1)
            gm = np.exp(np.log(v + 1e-12).mean(1))
            return float(np.mean(0.5 * np.log2(am / gm)))
        print(f"{typ:22s} col-std p99/p50={np.percentile(cs,99)/np.percentile(cs,50):.2f}  "
              f"mix-loss contiguous={group_loss(cs):.3f}b sorted={group_loss(srt):.3f}b")

    print()
    print("=== Q3: scale side-channel (per-block amax) entropy ===")
    W = get(f, pre.format(5, "mlp.down_proj"))
    rows, cols = W.shape
    pad = (-cols) % G
    A2 = np.pad(W, ((0, 0), (0, pad))) if pad else W
    N = A2.reshape(rows, -1, G).reshape(-1, G)
    signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
    R = fwht(N * signs) / np.sqrt(G)
    amax = np.abs(R).max(1)
    la = np.log2(amax + 1e-12)
    print(f"blocks={len(amax)}  log2(amax): std={la.std():.3f} range={la.max()-la.min():.2f}")
    # quantize log-amax with step .04 (max scale error ~1.4% -> negligible distortion)
    q = np.round((la - la.min()) / 0.04).astype(np.int64)
    h0 = ent_bits(q)
    # predictive: delta vs previous block in same row
    q2 = q.reshape(rows, -1)
    dq = np.diff(q2, axis=1).ravel()
    hd = ent_bits(dq - dq.min())
    print(f"amax entropy: raw h0={h0:.2f}b  row-delta={hd:.2f}b  (currently paying 16b) "
          f"-> saving ~{(16-min(h0,hd))/G:.3f} bits/weight")

    print()
    print("=== Q4: kurtosis by tensor type (layer 5 + layer 20) ===")
    for L in [5, 20]:
        out = []
        for typ in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            W = get(f, pre.format(L, typ))
            x = W.ravel()
            k = float(((x - x.mean()) ** 4).mean() / (x.var() ** 2))
            out.append(f"{typ.split('.')[-1]}={k:.1f}")
        print(f"L{L}: " + "  ".join(out))

    print()
    print("=== embedding row-norm structure (untouched by codec) ===")
    E = get(f, "model.embed_tokens.weight")
    rn = np.linalg.norm(E, axis=1)
    print(f"embed {E.shape}: row-norm p1={np.percentile(rn,1):.2f} p50={np.percentile(rn,50):.2f} "
          f"p99={np.percentile(rn,99):.2f}; frac rows with norm < 0.5*median: {(rn < 0.5*np.median(rn)).mean():.3f}")
