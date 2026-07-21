"""gemma_probe.py -- data-free Phase-1 probe for the Gemma-4-12B port. Run once the (gated) weights are local:
    python -m weights.gemma_probe D:/path/to/gemma-4-12b

LESSON from dense_axiom_probe.py: the trellis codec equalizes WEIGHT rel-MSE to ~0.069 on every tensor, so
weight-MSE is NOT a fragility discriminator. The real data-free discriminators are effective-rank (flatness)
and kurtosis. This probe ranks Gemma's tensor types by those -> the carve-out candidate set (protect the
low-flatness / heavy-tailed ones) -- and computes the PLE-aware 6 GB memory budget. No eval, no codec.
"""
from __future__ import annotations
import json, os, sys, glob
import numpy as np
try:
    from safetensors import safe_open
except Exception:
    safe_open = None


def stats(W):
    W = np.asarray(W, np.float32)
    if W.ndim != 2:
        W = W.reshape(W.shape[0], -1)
    f = W.reshape(-1)
    kurt = float(((f - f.mean()) ** 4).mean() / (f.var() ** 2 + 1e-30) - 3)
    # effective rank on the smaller dimension for speed
    M = W if W.shape[0] <= W.shape[1] else W.T
    if M.shape[0] > 4096:
        M = M[:4096]
    s = np.linalg.svd(M.astype(np.float64), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    flat = float(np.exp(-(p * np.log(p + 1e-30)).sum()) / len(s))
    return kurt, flat, tuple(W.shape)


def find_tensor(mdir, needle):
    idx_path = os.path.join(mdir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        wmap = json.load(open(idx_path))["weight_map"]
        for name, shard in wmap.items():
            if needle in name:
                with safe_open(os.path.join(mdir, shard), framework="np") as f:
                    return name, f.get_tensor(name)
    for shard in glob.glob(os.path.join(mdir, "*.safetensors")):
        with safe_open(shard, framework="np") as f:
            for name in f.keys():
                if needle in name:
                    return name, f.get_tensor(name)
    return None, None


def main(mdir):
    cfg = json.load(open(os.path.join(mdir, "config.json")))
    tc = cfg.get("text_config", cfg)
    H = tc.get("hidden_size"); L = tc.get("num_hidden_layers"); V = tc.get("vocab_size")
    ffn = tc.get("intermediate_size"); ple = tc.get("hidden_size_per_layer_input", 256)
    tied = cfg.get("tie_word_embeddings", True)
    print(f"Gemma config: hidden={H} layers={L} vocab={V} ffn={ffn} PLE_dim={ple} tied_embed={tied}\n")

    # PLE-aware memory budget (GB) at candidate allocations
    core = 0  # transformer core params (attn + mlp) -- estimate from dims if present
    if H and L and ffn:
        attn = L * (4 * H * H)                    # rough: q,k,v,o (GQA makes k,v smaller; upper bound)
        mlp = L * (3 * H * ffn)
        core = attn + mlp
    emb = (V * H) if (V and H) else 0
    ple_params = (V * L * ple) if (V and L) else 0
    def gb(p, bits): return p * bits / 8 / 1e9
    print("PLE-aware 6 GB budget (PLE streamed from CPU, not resident):")
    print(f"  transformer core ~{core/1e9:.1f}B @2bit = {gb(core,2):.2f} GB | @2.5bit = {gb(core,2.5):.2f} GB")
    print(f"  tied embed ~{emb/1e9:.2f}B: fp16={gb(emb,16):.2f} / 4bit={gb(emb,4):.2f} GB")
    print(f"  PLE ~{ple_params/1e9:.1f}B: STREAMED (0 GB resident) | else fp16={gb(ple_params,16):.1f} GB")
    print(f"  => resident @ (core 2bit + embed 4bit + PLE streamed) ~= {gb(core,2)+gb(emb,4):.2f} GB\n")

    print("Tensor fragility ranking (LOW flatness / HIGH |kurtosis| = protect):")
    print(f"  {'tensor':28s}  kurtosis  flatness  shape")
    targets = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "embed_tokens"]
    for t in targets:
        name, W = find_tensor(mdir, f"layers.13.{t}" if "embed" not in t else t)
        if W is None:
            print(f"  {t:28s}  (not found -- check tensor names in this checkpoint)")
            continue
        k, fl, sh = stats(W)
        tag = "PROTECT" if (fl < 0.35 or abs(k) > 2.0) else "2-bit ok"
        print(f"  {t:28s}  {k:+7.2f}  {fl:7.3f}  {str(sh):18s} [{tag}]")
    print("\n  Next: Phase 2 (port + fp16 sanity), then Phase 3 (2-bit dense MLP functional gate). See GEMMA_PORT.md.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or safe_open is None:
        print("usage: python -m weights.gemma_probe <gemma_model_dir>   (needs safetensors)")
    else:
        main(sys.argv[1])
