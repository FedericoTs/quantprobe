"""Campaign-4 FRONTIER: champion-codec encoder for Llama-2-7B (the field-standard benchmark,
to compare WikiText-2 ppl head-to-head vs QTIP 2b=5.86 / QuIP# 6.19 / AQLM ~6.2).

Reuses the shared codec (rotation + AWQ + scalar ECVQ + entropy + outliers) via evoq.encode_tensor.
Streaming per-layer (6GB-safe). lambda (EVOQ_LAM) sweeps the rate -> b/w; push it up to reach ~2-bit.

Usage (in the venv):
  EVOQ_LAM=0.008 python -m weights.evoq_llama encode      # -> weights/data/llama2_7b_evoq_L<lam>/
  python -m weights.evoq_llama baseline                    # fp16 WikiText-2 ppl (validate ~5.47)
"""
from __future__ import annotations

import gc
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
from safetensors.torch import save_file as torch_save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weights.evoq import encode_tensor, save_container, pack6, P_OUT, KPOOL
from weights.quant_lab import _awq_scale


def gptq_encode_tensor(W, awq_s, Hg, lam=None, seed=0):
    """champion encode_tensor but with output-aware GPTQ assignment (uses rotated-activation Gram Hg).
    Mirrors evoq.encode_tensor's container exactly; only the index ASSIGNMENT changes."""
    import math
    from weights.codec_zoo import _ecvq_levels
    from weights.noise_shaping import _gptq_block
    from weights.quant_sota import _fwht_rows
    lam = LAM if lam is None else lam
    rows, cols = W.shape; ng = cols // G
    Ws = (W * awq_s[None, :]).astype(np.float32)
    thr = np.quantile(np.abs(Ws), 1.0 - P_OUT)
    mask = np.abs(Ws) >= thr
    base = Ws.copy(); base[mask] = 0.0
    signs = (np.random.default_rng(seed).integers(0, 2, G).astype(np.float32) * 2 - 1)
    N = base.reshape(rows, -1, G).reshape(-1, G)
    R = _fwht_rows(N * signs) / math.sqrt(G)
    amax = np.abs(R).max(1); amax[amax == 0] = 1.0
    Rn = R / amax[:, None]                                   # [rows*ng, G]
    rng = np.random.default_rng(seed + 1)
    samp = Rn.ravel()[rng.integers(0, Rn.size, min(20000, Rn.size))]
    lv = _ecvq_levels(samp, KPOOL, lam).astype(np.float32)
    idx_flat = np.empty((rows * ng, G), np.uint8)
    for gi in range(ng):                                     # per input-group GPTQ with its Gram
        block = Rn[gi::ng]                                   # [rows, G] all rows of input-group gi
        idx_flat[gi::ng] = _gptq_block(block, Hg[gi], lv).astype(np.uint8)
    idx = idx_flat.reshape(-1)
    out_pos = np.flatnonzero(mask.ravel()).astype(np.int32)
    out_val = W.ravel()[out_pos].astype(np.float32)
    return dict(packed=pack6(idx), n_idx=idx.size, K=len(lv),
                lv=lv.astype(np.float32), amax=amax.astype(np.float32),
                signs=signs.astype(np.int8), awq_s=awq_s.astype(np.float32),
                out_pos=out_pos, out_val=out_val, rows=rows, cols=cols)

MDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "llama2_7b_base")
LAM = float(os.environ.get("EVOQ_LAM", "0.008"))
ODIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"llama2_7b_evoq_L{LAM}")
GPTQ = bool(int(os.environ.get("EVOQ_GPTQ", "0")))
if GPTQ:
    ODIR = ODIR + "_gptq"
LINEAR7 = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CALIB_TOKENS = int(os.environ.get("EVOQ_CALIB", "512"))
G = 128


def shard_map():
    import json
    with open(os.path.join(MDIR, "model.safetensors.index.json")) as fh:
        wm = json.load(fh)["weight_map"]
    return {k: os.path.join(MDIR, v) for k, v in wm.items()}


def read_tensor(smap, key):
    with safe_open(smap[key], framework="pt") as f:
        return f.get_tensor(key)


def meta_model(cfg):
    cfg.torch_dtype = torch.float32
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    return m.to(torch.float32)


def materialize_layer(layer, smap, prefix):
    layer.to_empty(device="cpu")
    sd = layer.state_dict()
    for name in list(sd.keys()):
        sd[name].copy_(read_tensor(smap, prefix + name).to(torch.float32))


def causal_mask(s, dtype=torch.float32):
    m = torch.full((s, s), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(m, diagonal=1)[None, None]


def _rotary(cfg):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    return LlamaRotaryEmbedding(config=cfg)


def _calib_text(tok):
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2_train.txt")
    if os.path.exists(cache):
        txt = open(cache, encoding="utf-8").read()
    else:
        from datasets import load_dataset
        txt = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"][:1500])
    return tok(txt, return_tensors="pt").input_ids[:, :CALIB_TOKENS]


def encode():
    os.makedirs(ODIR, exist_ok=True)
    cfg = AutoConfig.from_pretrained(MDIR)
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    smap = shard_map()
    model = meta_model(cfg)
    L = cfg.num_hidden_layers
    rot = _rotary(cfg)

    ids = _calib_text(tok)
    emb_w = read_tensor(smap, "model.embed_tokens.weight").to(torch.float32)
    h = emb_w[ids[0]].unsqueeze(0)
    del emb_w
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    cos_sin = rot(h, pos)
    mask = causal_mask(ids.shape[1])

    residual = {}
    for k in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        residual[k] = read_tensor(smap, k).contiguous()

    t0 = time.time()
    for li in range(L):
        prefix = f"model.layers.{li}."
        layer = model.model.layers[li]
        materialize_layer(layer, smap, prefix)
        calib, rawX, hooks = {}, {}, []

        def mk(key):
            def hkfn(mod, inp):
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                calib[key] = x.abs().mean(0).numpy() if key not in calib else calib[key] + x.abs().mean(0).numpy()
                if GPTQ:
                    rawX[key] = x.numpy() if key not in rawX else np.concatenate([rawX[key], x.numpy()], 0)
            return hkfn
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in LINEAR7:
                hooks.append(mod.register_forward_pre_hook(mk(name.split(".")[-1])))
        with torch.no_grad():
            h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        for hk in hooks:
            hk.remove()

        tensors = {}
        for name, mod in layer.named_modules():
            short = name.split(".")[-1]
            if isinstance(mod, nn.Linear) and short in LINEAR7:
                W = mod.weight.detach().numpy()
                s = _awq_scale(calib, short, 0.5).astype(np.float32)
                if GPTQ:
                    from weights.noise_shaping import xrr_of
                    xr = xrr_of(rawX[short], s, (np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1))
                    Hg = np.einsum("ngk,ngl->gkl", xr, xr) / xr.shape[0]
                    tensors[prefix + name + ".weight"] = gptq_encode_tensor(W, s, Hg, lam=LAM)
                else:
                    tensors[prefix + name + ".weight"] = encode_tensor(W, s, lam=LAM,
                                                                        self_check=(short == "q_proj"))
        for nname in ("input_layernorm", "post_attention_layernorm"):
            residual[prefix + nname + ".weight"] = getattr(layer, nname).weight.detach().to(torch.float32).contiguous()
        save_container(os.path.join(ODIR, f"layer{li:02d}.evoq"), tensors,
                       dict(model="Llama-2-7b-hf", layer=li, lam=LAM, seed=0))
        layer.to_empty(device="meta")
        del tensors
        gc.collect()
        print(f"  layer {li+1}/{L} encoded  ({time.time()-t0:.0f}s)", flush=True)

    torch_save_file(residual, os.path.join(ODIR, "residual.safetensors"))
    total = sum(os.path.getsize(os.path.join(ODIR, f)) for f in os.listdir(ODIR))
    print(f"DONE: {ODIR}  total {total/1e9:.2f} GB  lam={LAM}", flush=True)


def baseline():
    """fp16 WikiText-2 ppl via streaming layers (6GB-safe). Validates protocol == ~5.47."""
    from weights.wikitext2_ppl import get_wikitext2_test_ids
    cfg = AutoConfig.from_pretrained(MDIR)
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    smap = shard_map()
    model = meta_model(cfg)
    rot = _rotary(cfg)
    ids_all = get_wikitext2_test_ids(tok)
    seqlen, nwin = 2048, int(os.environ.get("EVOQ_NWIN", "4"))
    import torch.nn.functional as F
    nlls, ntok = [], 0
    emb_w = read_tensor(smap, "model.embed_tokens.weight").to(torch.float32)
    headw = read_tensor(smap, "lm_head.weight").to(torch.float32)
    normw = read_tensor(smap, "model.norm.weight").to(torch.float32)
    t0 = time.time()
    for w in range(nwin):
        ids = ids_all[:, w * seqlen:(w + 1) * seqlen]
        h = emb_w[ids[0]].unsqueeze(0)
        pos = torch.arange(ids.shape[1]).unsqueeze(0)
        cos_sin = rot(h, pos)
        mask = causal_mask(ids.shape[1])
        for li in range(cfg.num_hidden_layers):
            layer = model.model.layers[li]
            materialize_layer(layer, smap, f"model.layers.{li}.")
            with torch.no_grad():
                h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
            layer.to_empty(device="meta")
            gc.collect()
        v = h.pow(2).mean(-1, keepdim=True)
        h = (h * torch.rsqrt(v + cfg.rms_norm_eps)) * normw
        logits = h[0] @ headw.T
        loss = F.cross_entropy(logits[:-1].float(), ids[0, 1:])
        nlls.append(loss * (seqlen - 1)); ntok += seqlen - 1
        print(f"  win {w+1}/{nwin} running ppl = {torch.exp(torch.stack(nlls).sum()/ntok):.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nLlama-2-7B fp16 WikiText-2 ppl ({nwin} win) = {torch.exp(torch.stack(nlls).sum()/ntok):.4f}  (ref 5.47)")


def _bw_of_container(odir, L):
    """Honest resident b/w from JSON metadata only: 4-bit idx + fp16 amax(/128) + 0.5% outliers + lv."""
    import json
    bits = 0.0; nW = 0
    for li in range(L):
        with open(os.path.join(odir, f"layer{li:02d}.evoq.json")) as fh:
            tm = json.load(fh)["tensors"]
        for nm, t in tm.items():
            n = t["rows"] * t["cols"]; nW += n
            nout = int(round(n * 0.005))
            bits += 4.0 * n + 16.0 * (n // 128) + nout * (32 + 16) + t["K"] * 16
    return bits / nW


@torch.no_grad()
def measure():
    """WikiText-2 ppl of the champion-quantized Llama-2-7B via streaming dequant layers (GEMM).
    EVOQ_LAM picks the container; EVOQ_NWIN windows of 2048 (default 16 = 32k tok, close to full)."""
    from weights.evoq import load_container, dequant_t
    from weights.wikitext2_ppl import get_wikitext2_test_ids
    import torch.nn.functional as F
    cfg = AutoConfig.from_pretrained(MDIR)
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    model = meta_model(cfg)
    rot = _rotary(cfg).cuda()
    L = cfg.num_hidden_layers
    seqlen, nwin = 2048, int(os.environ.get("EVOQ_NWIN", "16"))
    ids_all = get_wikitext2_test_ids(tok)
    nwin = min(nwin, ids_all.numel() // seqlen)
    ids = ids_all[:, :nwin * seqlen].reshape(nwin, seqlen)              # [B, 2048]

    with safe_open(os.path.join(ODIR, "residual.safetensors"), framework="pt") as rf:
        emb = rf.get_tensor("model.embed_tokens.weight").float()
        normw = rf.get_tensor("model.norm.weight").float().cuda()
        headw = rf.get_tensor("lm_head.weight").float().cuda()
    h = F.embedding(ids, emb).cuda()                                    # [B, 2048, 4096] fp32
    pos = torch.arange(seqlen, device="cuda").unsqueeze(0)
    cos_sin = rot(h, pos)
    mask = causal_mask(seqlen).cuda()
    t0 = time.time()
    for li in range(L):
        prefix = f"model.layers.{li}."
        layer = model.model.layers[li]; layer.to_empty(device="cuda")
        meta, comps = load_container(os.path.join(ODIR, f"layer{li:02d}.evoq"))
        name2mod = dict(layer.named_modules())
        with safe_open(os.path.join(ODIR, "residual.safetensors"), framework="pt") as rf:
            for nn_ in ("input_layernorm", "post_attention_layernorm"):
                getattr(layer, nn_).weight.data.copy_(rf.get_tensor(prefix + nn_ + ".weight").float().cuda())
        for k, c in comps.items():
            mod_name = k[len(prefix):-len(".weight")]
            W = dequant_t({kk: vv for kk, vv in c.items()}, torch.device("cuda"), torch.float32)
            name2mod[mod_name].weight.data.copy_(W)
        h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        layer.to_empty(device="meta")                                  # FREE this layer's weights
        del comps; gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)
    v = h.pow(2).mean(-1, keepdim=True)
    h = (h * torch.rsqrt(v + cfg.rms_norm_eps)) * normw
    nlls, ntok = [], 0
    for b in range(nwin):                                              # per-window lm_head+loss (avoid 4GB logits)
        logits = h[b] @ headw.T
        loss = F.cross_entropy(logits[:-1].float(), ids[b, 1:].cuda())
        nlls.append(loss * (seqlen - 1)); ntok += seqlen - 1
    ppl = torch.exp(torch.stack(nlls).sum() / ntok).item()
    bw = _bw_of_container(ODIR, L)
    print(f"\nLlama-2-7B champion lam={LAM}: WikiText-2 ppl = {ppl:.4f} @ {bw:.2f} b/w "
          f"({nwin} win)\n  FRONTIER: fp16 5.47 | QTIP 2b 5.86 | QuIP# 2b 6.19 | AQLM 2b ~6.2")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "encode"
    {"encode": encode, "baseline": baseline, "measure": measure}[mode]()
