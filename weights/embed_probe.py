import json, numpy as np, torch
from safetensors import safe_open
mdir = "D:/evo-compress-data/DeepSeek-V2-Lite"
idx = json.load(open(mdir + "/model.safetensors.index.json"))["weight_map"]

def load(name):
    with safe_open(mdir + "/" + idx[name], framework="pt") as f:
        return f.get_tensor(name).float().numpy()

print("loading...", flush=True)
emb = load("model.embed_tokens.weight")
head = load("lm_head.weight")
print("embed", emb.shape, "lm_head", head.shape, flush=True)

def stats(W, name):
    rn = np.linalg.norm(W, axis=1); flat = W.ravel()
    kurt = ((flat - flat.mean())**4).mean() / flat.var()**2 - 3
    print(f"[{name}] rownorm min{rn.min():.3f} med{np.median(rn):.3f} max{rn.max():.3f} "
          f"max/med{rn.max()/np.median(rn):.1f}  std{flat.std():.4f} kurt{kurt:.2f}", flush=True)

stats(emb, "embed"); stats(head, "lm_head")

def kmeans_residual(W, name, k=256, iters=12, sample=20000):
    rng = np.random.default_rng(0); n = W.shape[0]
    X = torch.tensor(W[rng.choice(n, min(sample, n), replace=False)], dtype=torch.float32)
    c = X[torch.randperm(len(X))[:k]].clone()
    x2 = (X*X).sum(1, keepdim=True)
    for _ in range(iters):
        d = x2 - 2*X@c.T + (c*c).sum(1)[None, :]
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any(): c[j] = X[m].mean(0)
    frac = ((X - c[a]).var() / X.var()).item()
    print(f"  [{name}] k={k}: resid/glob={frac:.3f} -> bits_saved/wt(Gauss)~{-0.5*np.log2(frac):.3f}; "
          f"id_overhead {np.log2(k)/W.shape[1]:.4f} b/w", flush=True)

kmeans_residual(emb, "embed"); kmeans_residual(head, "lm_head")

def relmse_2bit(W, name, groups=128):
    n, d = W.shape; err = 0.0; tot = 0.0
    for i in range(0, d, groups):
        blk = W[:, i:i+groups]; a = np.abs(blk).max(1, keepdims=True) + 1e-9
        q = np.round(np.clip(blk/a, -1, 1) * 1.5) / 1.5 * a
        err += ((blk - q)**2).sum(); tot += (blk**2).sum()
    print(f"  [{name}] uniform per-{groups} 2bit rel-MSE={err/tot:.4f} (expert floor 0.069)", flush=True)

relmse_2bit(emb, "embed"); relmse_2bit(head, "lm_head")

# Cluster-conditional 2bit: subtract per-cluster centroid, then 2bit the residual. Does it beat plain 2bit?
def cluster_then_2bit(W, name, k=256, iters=12, groups=128):
    Wt = torch.tensor(W, dtype=torch.float32)
    c = Wt[torch.randperm(len(Wt))[:k]].clone()
    x2 = (Wt*Wt).sum(1, keepdim=True)
    for _ in range(iters):
        # chunked assign over full vocab
        a = torch.empty(len(Wt), dtype=torch.long)
        cc = (c*c).sum(1)
        for i in range(0, len(Wt), 8192):
            xb = Wt[i:i+8192]
            d = (xb*xb).sum(1, keepdim=True) - 2*xb@c.T + cc[None, :]
            a[i:i+8192] = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any(): c[j] = Wt[m].mean(0)
    R = (Wt - c[a]).numpy()  # residual
    # 2bit the residual
    n, d = R.shape; err = 0.0
    for i in range(0, d, groups):
        blk = R[:, i:i+groups]; aa = np.abs(blk).max(1, keepdims=True) + 1e-9
        q = np.round(np.clip(blk/aa, -1, 1) * 1.5) / 1.5 * aa
        err += ((blk - q)**2).sum()
    tot = (W**2).sum()  # vs ORIGINAL energy (this is what matters for the layer output)
    centroid_cost = k * d * 16 / (n * d)  # bits/weight to store centroids fp16
    print(f"  [{name}] cluster(k={k})+2bit_residual: rel-MSE vs original={err/tot:.4f}  "
          f"centroid_overhead={centroid_cost:.4f} b/w + id {np.log2(k)/d:.4f}", flush=True)

cluster_then_2bit(emb, "embed"); cluster_then_2bit(head, "lm_head")
print("DONE", flush=True)
