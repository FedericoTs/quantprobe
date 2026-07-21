"""evoq_run -- KERNEL-BACKED runtime. Replaces QuantLinear's dequant_t+F.linear (which
materializes the full fp32 weight every forward) with the activation-side F0 GEMV kernel:
weights stay PACKED at 4-bit resident; per token we FWHT-prep the activation once and call
the fused decode-LUT GEMV. This is the path whose util we measured (~0.39 on real 7B shapes).

KQuantLinear.forward(x):
  z   = (x / awq_s) reshaped to groups, * signs           # per-token activation rotation
  xrr = FWHT(z) / sqrt(G)                                  # [B, in]
  y   = f0_gemv3(packed4, xrr_b, lv16, amax2d)  per token  # dense decode-LUT GEMV (kernel)
  y  += scatter(out_val * x[out_cols], out_rows)           # exact outlier sidecar (sparse)

Gate: KQuantLinear forward == evoq.dequant_t+F.linear (bit-near-exact) on real tensors.
Then: load the real model with KQuantLinears, measure decode tok/s + VRAM on GPU.

Run (in the CUDA build env, via tools/run_kernel.cmd ... or):  python -m weights.evoq_run gate | bench05 | bench7b
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors import safe_open
from weights.evoq import unpack6_t, fwht_inplace_t, dequant_t, load_container
from weights.evoq_kernel import build, pack4

G = 128
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        _EXT = build()
    return _EXT


def pack4_t(idx2d: torch.Tensor) -> torch.Tensor:
    """idx [out,in] (int, <16) -> packed [out, in/2] uint8 (low nibble = even col)."""
    out, inn = idx2d.shape
    a = idx2d.view(out, inn // 2, 2).to(torch.int32)
    return (a[:, :, 0] | (a[:, :, 1] << 4)).to(torch.uint8)


class KQuantLinear(nn.Module):
    def __init__(self, comp, bias, device="cuda"):
        super().__init__()
        self.rows, self.cols = int(comp["rows"]), int(comp["cols"])
        self.ng = self.cols // G
        as_t = lambda v: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        idx = unpack6_t(as_t(comp["packed"]), int(comp["n_idx"])).view(self.rows, self.cols)
        packed4 = pack4_t(idx).contiguous()
        lv16 = torch.zeros(16, dtype=torch.float32)
        lvv = as_t(comp["lv"]).float()
        lv16[: lvv.numel()] = lvv
        amax2d = as_t(comp["amax"]).float().view(self.rows, self.ng).contiguous()
        signs = as_t(comp["signs"]).float()
        awq_s = as_t(comp["awq_s"]).float()
        out_pos = as_t(comp["out_pos"]).long()
        out_val = as_t(comp["out_val"]).float()
        # The dense kernel reconstructs the ZEROED-outlier tensor; dequant_t REPLACES at
        # outlier positions. To match (replace, not add), store out_corr = out_val - w_dense[pos]
        # so a single sparse scatter-add converts the dense value to the exact one.
        comp_dense = {k: v for k, v in comp.items()}
        comp_dense["out_pos"] = (out_pos.new_zeros(0)).numpy() if isinstance(comp["out_pos"], np.ndarray) else out_pos.new_zeros(0)
        comp_dense["out_val"] = (out_val.new_zeros(0)).numpy() if isinstance(comp["out_val"], np.ndarray) else out_val.new_zeros(0)
        Wd = dequant_t({k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in comp_dense.items()},
                       torch.device("cpu"), torch.float32, bf16_round=False)
        w_dense_at = Wd.reshape(-1)[out_pos]
        out_corr = out_val - w_dense_at
        self.register_buffer("packed4", packed4.to(device), persistent=False)
        self.register_buffer("lv16", lv16.to(device), persistent=False)
        self.register_buffer("amax2d", amax2d.to(device), persistent=False)
        self.register_buffer("signs", signs.to(device), persistent=False)
        self.register_buffer("awq_s", awq_s.to(device), persistent=False)
        # CSR by output row (sorted) for the atomic-free outlier kernel
        orows = (out_pos // self.cols)
        ocols = (out_pos % self.cols).to(torch.int32)
        order = torch.argsort(orows)
        orows_s = orows[order]
        self.register_buffer("csr_col", ocols[order].contiguous().to(device), persistent=False)
        self.register_buffer("csr_val", out_corr[order].contiguous().to(device), persistent=False)
        row_ptr = torch.zeros(self.rows + 1, dtype=torch.int32)
        row_ptr[1:] = torch.bincount(orows_s, minlength=self.rows).cumsum(0).to(torch.int32)
        self.register_buffer("csr_ptr", row_ptr.contiguous().to(device), persistent=False)
        self.bias = bias.to(device).float() if bias is not None else None
        self.device = device

    def forward(self, x):
        orig = x.shape
        in_dtype = x.dtype
        xf = x.reshape(-1, self.cols).float().to(self.device)        # [B, in]
        B = xf.shape[0]
        e = ext()
        xrr = e.fwht_prep(xf, self.awq_s, self.signs).view(B, self.cols)  # fused prep, 1 launch
        y = xf.new_empty(B, self.rows)
        for b in range(B):                                           # decode: B==1 (one kernel call)
            yb = e.f0_gemv3(self.packed4, xrr[b].contiguous(), self.lv16, self.amax2d)
            e.csr_outlier(xf[b].contiguous(), self.csr_ptr, self.csr_col, self.csr_val, yb)  # atomic-free
            y[b] = yb
        if self.bias is not None:
            y = y + self.bias.view(1, -1)
        return y.reshape(*orig[:-1], self.rows).to(in_dtype)


def gate():
    """KQuantLinear forward == dequant_t+F.linear on real 0.5B tensors (bit-near-exact)."""
    meta, comps = load_container("weights/data/qwen05b.evoq")
    names = list(comps)[:6] + [n for n in comps if "down_proj" in n][:1]
    rng = np.random.default_rng(0)
    worst = 0.0
    for nm in names:
        c = comps[nm]
        rows, cols = int(c["rows"]), int(c["cols"])
        kq = KQuantLinear(c, None, device="cuda")
        x = torch.from_numpy(rng.standard_normal((1, cols)).astype(np.float32))
        y_k = kq(x.cuda()).cpu()
        W = dequant_t({k: v for k, v in c.items()}, torch.device("cpu"), torch.float32, bf16_round=False)
        y_ref = x @ W.T
        rel = (y_k - y_ref).abs().max().item() / (y_ref.abs().max().item() + 1e-9)
        worst = max(worst, rel)
        print(f"  {nm.split('.')[-2]:<10} {rows:>5}x{cols:<6}  rel err {rel:.2e}  {'OK' if rel < 2e-2 else 'FAIL'}")
    print(f"\nworst rel err = {worst:.2e}  ({'PASS' if worst < 2e-2 else 'FAIL'}) "
          f"-- kernel path matches the champion dequant (int8-free F0, Q8 not used here)")


def _swap_quant_layers(model, comps, device):
    name2mod = dict(model.named_modules())
    nbytes = 0
    for k, c in comps.items():
        mod_name = k[: -len(".weight")] if k.endswith(".weight") else k
        if mod_name not in name2mod:
            continue
        mod = name2mod[mod_name]
        bias = mod.bias.detach() if getattr(mod, "bias", None) is not None else None
        kq = KQuantLinear(c, bias, device=device)
        nbytes += kq.packed4.numel() + kq.amax2d.numel() * 4 + kq.csr_val.numel() * 6
        parent = name2mod[mod_name.rsplit(".", 1)[0]] if "." in mod_name else model
        setattr(parent, mod_name.rsplit(".", 1)[-1], kq)
    return nbytes


def _decode_toks(model, tok, n_new=64, in_dev="cuda"):
    ids = tok("The history of science is", return_tensors="pt").input_ids.to(in_dev)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1:].argmax(-1).to(in_dev)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(n_new):
            o = model(nxt, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[:, -1:].argmax(-1).to(in_dev)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return n_new / dt


def bench05():
    from transformers import AutoTokenizer
    from weights.quant_lab import CFG, build_model, load_fp16, ppl, EVAL_TEXT
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model(); load_fp16(model)
    meta, comps = load_container("weights/data/qwen05b.evoq")
    torch.cuda.reset_peak_memory_stats()
    qbytes = _swap_quant_layers(model, comps, dev)
    model = model.to(dev)
    print(f"quantized matmul resident = {qbytes/1e6:.0f} MB (F0 4-bit) | "
          f"fp16 would be {qbytes/0.5*2/1e6:.0f} MB", flush=True)
    # correctness: held-out ppl vs champion 4.63
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :1024].to(dev)
    with torch.no_grad():
        p = float(torch.exp(model(ids, labels=ids).loss))
    print(f"held-out ppl = {p:.4f}  (champion 4.6302; fp16 3.9332)", flush=True)
    tps = _decode_toks(model, tok)
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"decode = {tps:.1f} tok/s | peak VRAM = {peak:.0f} MB", flush=True)


class HeadShim(nn.Module):
    """lm_head in bf16 on GPU (saves 1.1GB vs fp32); casts the hidden state to match."""
    def __init__(self, W):
        super().__init__()
        self.register_buffer("W", W.to(torch.bfloat16), persistent=False)
    def forward(self, x):
        return (x.to(torch.bfloat16) @ self.W.t()).float()


class EmbShim(nn.Module):
    """embed in bf16 on GPU (1.1GB vs 2.2GB fp32); returns fp32 for consistent compute."""
    def __init__(self, W):
        super().__init__()
        self.register_buffer("W", W.to(torch.bfloat16), persistent=False)
    def forward(self, ids):
        return torch.embedding(self.W, ids).float()


def _quant_rows_int8(W):
    """W [out,in] fp -> (int8 [out,in], scale [out] fp32). Per-row symmetric int8."""
    W = W.float()
    scale = (W.abs().amax(1) / 127.0).clamp_min(1e-12)
    Wq = torch.round(W / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return Wq.contiguous(), scale.contiguous()


class QEmbShim(nn.Module):
    """int8 embed (0.55GB vs 1.1GB bf16): gather row + dequant by per-row scale -> fp32."""
    def __init__(self, W):
        super().__init__()
        Wq, scale = _quant_rows_int8(W)
        self.register_buffer("Wq", Wq, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
    def forward(self, ids):
        rows = torch.embedding(self.Wq, ids)                 # int8 rows
        sc = torch.embedding(self.scale.unsqueeze(1), ids)   # [.,1]
        return rows.float() * sc


class Int8Linear(nn.Module):
    """int8 lm_head (0.55GB vs 1.1GB bf16): per-row int8 W + per-tensor Q8 activation + dp4a GEMV."""
    def __init__(self, W, ext_):
        super().__init__()
        Wq, scale = _quant_rows_int8(W)
        self.register_buffer("Wq", Wq, persistent=False)
        self.register_buffer("wscale", scale, persistent=False)
        self.ext = ext_
    def forward(self, x):
        orig = x.shape
        xf = x.reshape(-1, orig[-1]).float()
        B = xf.shape[0]
        out = self.Wq.shape[0]
        y = xf.new_empty(B, out)
        for b in range(B):
            xb = xf[b]
            xscale = (xb.abs().max() / 127.0).clamp_min(1e-12)   # GPU scalar (no .item() -> graph-safe)
            xq = torch.round(xb / xscale).clamp(-127, 127).to(torch.int8).contiguous()
            yb = self.ext.int8_gemv(self.Wq, xq, self.wscale, 1.0)   # xscale applied below (tensor, no sync)
            y[b] = yb * xscale
        return y.reshape(*orig[:-1], out)


def build_7b_allgpu(embed_gpu=False, quant_io=False):
    from transformers import AutoConfig, AutoTokenizer
    import torch.nn as nn
    from weights.evoq_7b import meta_model, MDIR, ODIR, LayerShim
    cfg = AutoConfig.from_pretrained(MDIR); cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MDIR)
    model = meta_model(cfg)
    L = cfg.num_hidden_layers
    with safe_open(os.path.join(ODIR, "residual.safetensors"), framework="pt") as rf:
        res = {k: rf.get_tensor(k) for k in rf.keys()}
    if quant_io:
        model.model.embed_tokens = QEmbShim(res["model.embed_tokens.weight"]).cuda()   # int8 GPU
    elif embed_gpu:
        model.model.embed_tokens = EmbShim(res["model.embed_tokens.weight"]).cuda()   # bf16 GPU
    else:
        model.model.embed_tokens = nn.Embedding.from_pretrained(
            res["model.embed_tokens.weight"].to(torch.float32), freeze=True)    # CPU
    model.model.norm.to_empty(device="cuda")
    model.model.norm.weight.data.copy_(res["model.norm.weight"].to(torch.float32).cuda())
    model.lm_head = (Int8Linear(res["lm_head.weight"], ext()) if quant_io
                     else HeadShim(res["lm_head.weight"])).cuda()               # int8 or bf16 GPU
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    model.model.rotary_emb = Qwen2RotaryEmbedding(config=cfg).cuda()
    t0 = time.perf_counter()
    for li in range(L):
        prefix = f"model.layers.{li}."
        layer = model.model.layers[li]; layer.to_empty(device="cpu")
        layer.input_layernorm.weight.data.copy_(res[prefix + "input_layernorm.weight"].to(torch.float32))
        layer.post_attention_layernorm.weight.data.copy_(res[prefix + "post_attention_layernorm.weight"].to(torch.float32))
        meta, comps = load_container(os.path.join(ODIR, f"layer{li:02d}.evoq"))
        for name, mod in list(layer.named_modules()):
            for cn, child in list(mod.named_children()):
                full = prefix + (f"{name}.{cn}" if name else cn) + ".weight"
                if isinstance(child, nn.Linear) and full in comps:
                    bk = full[:-len(".weight")] + ".bias"
                    bias = res[bk].to(torch.float32) if bk in res else None
                    setattr(mod, cn, KQuantLinear(comps[full], bias, device="cuda"))
        del comps
        # non-quant bits of the layer (norms) to cuda; KQuantLinears already cuda
        layer.input_layernorm.cuda(); layer.post_attention_layernorm.cuda()
        if hasattr(layer.self_attn, "q_norm"): pass
        model.model.layers[li] = LayerShim(layer, "cuda")
        gc.collect()
        if li % 7 == 0:
            print(f"  built layer {li}  ({torch.cuda.memory_allocated()/1e9:.2f} GB)", flush=True)
    model.eval()
    print(f"  built 7B in {time.perf_counter()-t0:.0f}s", flush=True)
    return model, tok


def bench7b(quant_io=False):
    from weights.evoq_7b import EVAL_TEXT
    torch.cuda.reset_peak_memory_stats()
    model, tok = build_7b_allgpu(embed_gpu=quant_io, quant_io=quant_io)
    in_dev = "cuda" if quant_io else "cpu"
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :512].to(in_dev)
    with torch.no_grad():
        p = float(torch.exp(model(ids, labels=ids).loss))
    print(f"\n7B held-out ppl = {p:.4f}  (fp16 2.7510; champion bf16 2.9226) | quant_io={quant_io}", flush=True)
    after_w = torch.cuda.memory_allocated() / 1e9
    tps = _decode_toks(model, tok, n_new=48, in_dev=in_dev)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"7B decode = {tps:.1f} tok/s | weights+io resident {after_w:.2f} GB | peak {peak:.2f} GB  "
          f"(Q4_K_M: 21.97 t/s @ 4.36 GiB)", flush=True)


def profile_ops():
    """Per-op decode-step time breakdown across the REAL 7B shapes (B=1), via cuda events.
    Identifies whether fwht_prep, the GEMV, or the outlier scatter dominates -> next target."""
    dev = "cuda"
    e = ext()
    L = 28
    shapes = [("q", 3584, 3584), ("k", 512, 3584), ("v", 512, 3584), ("o", 3584, 3584),
              ("gate", 18944, 3584), ("up", 18944, 3584), ("down", 3584, 18944)]
    rng = np.random.default_rng(0)
    ev = lambda: torch.cuda.Event(enable_timing=True)

    def t_op(fn, warm=5, n=100):
        for _ in range(warm): fn()
        torch.cuda.synchronize()
        a, b = ev(), ev(); a.record()
        for _ in range(n): fn()
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / n

    tot = {"prep": 0.0, "gemv": 0.0, "outl": 0.0, "launch_overhead_est": 0.0}
    print(f"{'shape':<8}{'prep ms':>9}{'gemv ms':>9}{'outl ms':>9}{'sum ms':>9}")
    for nm, out, inn in shapes:
        ng = inn // G
        K = 12
        lv16 = torch.zeros(16, device=dev); lv16[:K] = torch.sort(torch.randn(K, device=dev))[0]
        idx = torch.from_numpy(rng.integers(0, K, (out, inn)).astype(np.uint8)).to(dev)
        packed = pack4_t(idx).contiguous()
        amax = (0.5 + torch.rand(out, ng, device=dev))
        awq = torch.rand(inn, device=dev) + 0.5
        signs = (torch.randint(0, 2, (G,), device=dev).float() * 2 - 1)
        x = torch.randn(1, inn, device=dev)
        nnz = int(out * inn * 0.005)
        ocols = torch.randint(0, inn, (nnz,), device=dev)
        orows = torch.randint(0, out, (nnz,), device=dev)
        ovals = torch.randn(nnz, device=dev)
        xrr = e.fwht_prep(x, awq, signs).view(1, inn)
        t_prep = t_op(lambda: e.fwht_prep(x, awq, signs))
        t_gemv = t_op(lambda: e.f0_gemv3(packed, xrr[0].contiguous(), lv16, amax))
        y_buf = torch.zeros(1, out, device=dev)
        def outl():
            y_buf.index_add_(1, orows, x[:, ocols] * ovals.view(1, -1))
        t_outl = t_op(outl)
        s = t_prep + t_gemv + t_outl
        tot["prep"] += t_prep * L; tot["gemv"] += t_gemv * L; tot["outl"] += t_outl * L
        print(f"{nm:<8}{t_prep:>9.3f}{t_gemv:>9.3f}{t_outl:>9.3f}{s:>9.3f}")
    one_tok = tot["prep"] + tot["gemv"] + tot["outl"]
    print(f"\nPER-TOKEN (x{L} layers): prep {tot['prep']:.1f}ms  gemv {tot['gemv']:.1f}ms  "
          f"outl {tot['outl']:.1f}ms  = {one_tok:.1f}ms -> {1000/one_tok:.1f} tok/s (matmul path only)")
    print("  shares: prep %.0f%%  gemv %.0f%%  outl %.0f%%" %
          (100*tot['prep']/one_tok, 100*tot['gemv']/one_tok, 100*tot['outl']/one_tok))
    print("  (196 GEMV + 196 prep + 196 outl = ~588 kernel launches/token; +attn/norm/lm_head/Python).")


def graphbench(which="05"):
    """Measure decode tok/s with a CUDA GRAPH (zero Python/launch overhead) vs eager.
    Captures one decode step (fixed cache slot) and replays -- timing is valid even though
    the replayed output is positionally static. Proves the kernel-ceiling is reachable and
    that our custom ops are graph-capturable (the C++/no-Python headroom, quantified)."""
    from transformers import AutoTokenizer
    from transformers.cache_utils import StaticCache
    if which == "05":
        from weights.quant_lab import CFG, build_model, load_fp16
        tok = AutoTokenizer.from_pretrained(CFG)
        model = build_model(); load_fp16(model)
        meta, comps = load_container("weights/data/qwen05b.evoq")
        _swap_quant_layers(model, comps, "cuda"); model = model.to("cuda").float(); in_dev = "cuda"
    else:
        model, tok = build_7b_allgpu(embed_gpu=True, quant_io=True); in_dev = "cuda"
    model.eval()
    cfg = model.config
    maxlen = 64
    cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=maxlen,
                        device="cuda", dtype=torch.float32)
    prompt = tok("The history of science is", return_tensors="pt").input_ids.to(in_dev)
    with torch.no_grad():
        out = model(prompt, use_cache=True, past_key_values=cache)
        nxt = out.logits[:, -1:].argmax(-1)
    # static buffers for one-token decode at a fixed position
    pos = torch.tensor([prompt.shape[1]], device="cuda")
    cache_pos = pos.clone()
    static_in = nxt.to(in_dev).clone()

    def step():
        return model(static_in, use_cache=True, past_key_values=cache,
                     cache_position=cache_pos).logits

    with torch.no_grad():
        for _ in range(3):  # warmup
            step()
        torch.cuda.synchronize()
        # eager timing
        N = 40; t0 = time.perf_counter()
        for _ in range(N):
            step()
        torch.cuda.synchronize()
        eager = N / (time.perf_counter() - t0)
        # graph capture
        try:
            eager_logits = step().clone()                      # reference for correctness
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                logits = step()
            g.replay(); torch.cuda.synchronize()
            # CORRECTNESS: replayed logits must match eager (same static input) -> not degenerate
            match = torch.allclose(logits, eager_logits, atol=1e-2, rtol=1e-2)
            same_argmax = (logits.argmax(-1) == eager_logits.argmax(-1)).all().item()
            t0 = time.perf_counter()
            for _ in range(N):
                g.replay()
            torch.cuda.synchronize()
            graphed = N / (time.perf_counter() - t0)
            # PHYSICAL CEILING: tok/s <= bandwidth/resident_bytes (192GB/s / ~4.5GB ~= 43 for 7B)
            phys = 192e9 / (4.5e9 if which == "7b" else 0.25e9)
            flag = "" if graphed <= phys * 1.3 else f"  !! EXCEEDS PHYSICAL CEILING {phys:.0f} -> ARTIFACT"
            ok = "OK" if (match and same_argmax) else f"DEGENERATE (match={match}, argmax={same_argmax})"
            print(f"\n{which}: eager decode = {eager:.1f} tok/s | CUDA-graph = {graphed:.1f} tok/s "
                  f"({graphed/eager:.2f}x) | replay-correctness {ok}{flag}")
        except Exception as ex:
            print(f"\n{which}: eager = {eager:.1f} tok/s | GRAPH CAPTURE FAILED: {type(ex).__name__}: {ex}")
            print("  (HF forward not capture-safe here -> C++ runtime / llama.cpp integration is the robust path)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if mode == "gate":
        gate()
    elif mode == "profile":
        profile_ops()
    elif mode == "graph05":
        graphbench("05")
    elif mode == "graph7b":
        graphbench("7b")
    elif mode == "bench05":
        bench05()
    elif mode == "bench7b":
        bench7b(quant_io="--qio" in sys.argv)
