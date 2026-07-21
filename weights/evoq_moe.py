"""evoq MoE: streaming 2-bit trellis quantization + WikiText-2 ppl for DeepSeek-V2-Lite
(15.7B total / 2.4B active, 27 layers, MLA attention, 64 routed + 2 shared experts/layer,
layer 0 dense) on a GTX 1060 6GB.

WHY THIS MODEL: at 2-bit (~2.5 b/w incl side-info) the quantized weights are ~4.8 GB and the
whole 15.7B MoE becomes RESIDENT in 6 GB -- where bf16 needs ~29 GB and even 4-bit (~7.3 GB)
will not fit. The only quantization that crosses the 6 GB line.

WHY DATA-FREE TRELLIS: experts see calibration data only when routed, so data-DEPENDENT methods
(GPTQ/AWQ) starve rarely-activated experts. Our QTIP bitshift-trellis codec is per-tensor and
data-FREE (codebook + per-group incoherence rotation, no Hessian, no activations), so every
expert is quantized equally well regardless of routing frequency. That is the MoE thesis.

We never hold the dense model. Streaming eval (measure7b pattern): embed all eval tokens, then
per layer -> materialize bf16 from shards (CPU fp32) -> quantize each big Linear in place with
trellis_quant -> move layer to GPU -> forward every eval sequence (bsz=1, caps MLA attention
memory) -> free. The router (mlp.gate), embeddings, lm_head and norms stay fp16.

Quantized linears (by suffix): q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj   (MLA attention)
                               gate_proj, up_proj, down_proj                    (dense MLP layer0,
                                                                   each expert, shared experts)
Mixed-precision (residual-writer rule): down_proj/o_proj write the residual stream unattenuated
-> optionally K=3 (EVOQ_DOWN_K / EVOQ_O_K). Default uniform K=2 for the clean 2-bit headline.

Run:  .venv/Scripts/python -m weights.evoq_moe baseline | validate | measure
Env:  EVOQ_MOE_DIR (model dir) EVOQ_NWIN (eval windows, def 8) EVOQ_DOWN_K EVOQ_O_K
      EVOQ_INT8_GS=1 (int8 group side-info) EVOQ_VAL_LAYER (validate: which MoE layer, def 5)
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weights.qtip_trellis import trellis_quant, K_RATE, G
from weights.quant_lab import _awq_scale

MDIR = os.environ.get("EVOQ_MOE_DIR", r"D:\evo-compress-data\DeepSeek-V2-Lite")
TARGETS = ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "moe_results.txt")


# --------------------------------------------------------------------------- model plumbing
def shard_map():
    with open(os.path.join(MDIR, "model.safetensors.index.json")) as fh:
        wm = json.load(fh)["weight_map"]
    return {k: os.path.join(MDIR, v) for k, v in wm.items()}


def read_tensor(smap, key):
    with safe_open(smap[key], framework="pt") as f:
        return f.get_tensor(key)


def cache_dir(cfgstr):
    """Per-config cache of dequantized layer weights on D: -> resumable runs + cheap re-eval/demo
    without re-quantizing (8h). Keyed by the exact config string so different runs never collide."""
    import hashlib
    key = hashlib.md5(cfgstr.encode()).hexdigest()[:12]
    d = os.path.join(os.environ.get("EVOQ_CACHE_ROOT", r"D:\evo-compress-data\moe_cache"), key)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "config.txt"), "w").write(cfgstr)   # human-readable key -> config
    return d


def meta_model(cfg):
    cfg.torch_dtype = torch.float32
    cfg._attn_implementation = "eager"          # only eager is defined in modeling_deepseek
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    return m.to(torch.float32)


def build_rotary(cfg, model, device="cuda"):
    """Rebuild the attention rotary fresh on a REAL device. Meta construction leaves the cos/sin
    cache uninitialized; we re-instantiate the SAME class the model already chose (from cfg's
    remote code) so the cache is computed with real values. One shared instance, reused per layer."""
    RotCls = type(model.model.layers[0].self_attn.rotary_emb)   # the cfg-selected YaRN/RoPE class
    qk_rope = cfg.qk_rope_head_dim
    cache_len = 4096                              # >= eval seqlen; YaRN values are position-local
    rs = cfg.rope_scaling
    if rs is None:
        rot = RotCls(qk_rope, max_position_embeddings=cache_len, base=cfg.rope_theta)
    else:
        kwargs = {k: rs[k] for k in ("original_max_position_embeddings", "beta_fast",
                                     "beta_slow", "mscale", "mscale_all_dim") if k in rs}
        rot = RotCls(qk_rope, max_position_embeddings=cache_len,
                     scaling_factor=rs["factor"], base=cfg.rope_theta, **kwargs)
    return rot.to(device)


def materialize_cpu(layer, smap, prefix):
    """Fill one decoder layer's params (fp32 on CPU) from the bf16 shards."""
    layer.to_empty(device="cpu")
    sd = layer.state_dict()
    for name in list(sd.keys()):
        key = prefix + name
        if key in smap:
            sd[name].copy_(read_tensor(smap, key).to(torch.float32))
        # rotary buffers (cos_cached/sin_cached) are not in shards -> rebuilt separately


def causal_mask(s, dtype=torch.float32, device="cuda"):
    m = torch.full((s, s), torch.finfo(dtype).min, dtype=dtype, device=device)
    return torch.triu(m, diagonal=1)[None, None]


def free_layer(layer):
    """Free a processed layer back to meta. MUST detach the SHARED rotary first: to_empty(meta)
    recurses into self_attn.rotary_emb, and clobbering the one shared cuda rotary to meta would
    break every subsequent layer's .cuda()."""
    layer.self_attn.rotary_emb = None
    layer.to_empty(device="meta")


def k_for(name):
    """Per-component trellis rate (bits) from the FULL module path -> literature-backed mixed precision:
    MLA attention + shared experts + dense layer-0 MLP are a small fraction of params but a large
    fraction of low-bit error (QuantMoE-Bench: attn needs 4-8b, shared>routed, early>late; EAQuant:
    router protection mandatory -- router is fp16 already). Routed experts (the bulk) stay 2-bit; their
    down_proj (residual writer) gets DOWN_K. Defaults reproduce MxMoE-style carve-out at ~2.5 b/w."""
    wk = os.environ.get("EVOQ_WRITERS_K")                     # refinement: force residual-writers (o/down) low
    if wk and name.endswith(("o_proj", "down_proj")):         # (causal decomp: writers barely matter once
        return int(wk)                                        #  the internal kv-latent/gate-up are protected)
    if "self_attn" in name:                                   # MLA q/kv_a/kv_b/o
        return int(os.environ.get("EVOQ_ATTN_K", str(K_RATE)))
    if "shared_experts" in name:                              # always-active shared MLP
        return int(os.environ.get("EVOQ_SHARED_K", str(K_RATE)))
    if ".experts." in name:                                   # routed expert (the bulk -> stay low)
        if name.endswith("down_proj"):
            return int(os.environ.get("EVOQ_DOWN_K", str(K_RATE)))
        return K_RATE                                         # routed gate/up
    return int(os.environ.get("EVOQ_DENSE_K", str(K_RATE)))   # dense layer-0 MLP gate/up


# --------------------------------------------------------------------------- eval driver
def _eval_setup(nwin_default):
    from weights.wikitext2_ppl import get_wikitext2_test_ids
    cfg = AutoConfig.from_pretrained(MDIR, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    smap = shard_map()
    model = meta_model(cfg)
    # NOTE on the MoE forward branch: we deliberately leave the model in train mode so DeepseekV2MoE
    # takes the VECTORIZED branch (repeat_interleave by 6 + masked scatter) instead of moe_infer.
    # At bsz=1/seqlen-2048 the 6x expansion is ~100MB (fits the 6GB card), it is NUMERICALLY
    # IDENTICAL to moe_infer here (norm_topk_prob=false, attention_dropout=0), and it is ~10x FASTER
    # (moe_infer's per-expert Python loop does a .cpu() sync each layer: 61s/layer vs 12s/layer).
    # The 6GB-resident claim is computed analytically from quantized payload + fp16-kept, so it is
    # independent of which forward branch runs during measurement. (aux_loss is computed but unused.)
    L = cfg.num_hidden_layers
    seqlen = 2048
    nwin = int(os.environ.get("EVOQ_NWIN", str(nwin_default)))
    ids_all = get_wikitext2_test_ids(tok)
    nwin = min(nwin, ids_all.numel() // seqlen)
    ids = ids_all[:, :nwin * seqlen].reshape(nwin, seqlen)
    emb = read_tensor(smap, "model.embed_tokens.weight").float()
    h = F.embedding(ids, emb)                          # [nwin, seqlen, H] on CPU (kept off-GPU)
    del emb
    rot = build_rotary(cfg, model, "cuda")
    mask = causal_mask(seqlen)
    pos = torch.arange(seqlen, device="cuda").unsqueeze(0)
    return cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos


def _run_layer(layer, h, mask, pos):
    """Forward one decoder layer over all eval windows, bsz=1 at a time (caps MLA attn memory).
    h is [nwin, seqlen, H] on CPU; returns updated h on CPU."""
    out = torch.empty_like(h)
    for b in range(h.shape[0]):
        hb = h[b:b + 1].cuda()
        with torch.no_grad():
            yb = layer(hb, attention_mask=mask, position_ids=pos)[0]
        out[b:b + 1] = yb.cpu()
        del hb, yb
    return out


def _finish_ppl(h, smap, cfg, ids, seqlen, nwin):
    normw = read_tensor(smap, "model.norm.weight").float().cuda()
    headw = read_tensor(smap, "lm_head.weight").float().cuda()
    nlls, ntok = [], 0
    for b in range(nwin):
        hb = h[b].cuda()
        v = hb.pow(2).mean(-1, keepdim=True)
        hb = (hb * torch.rsqrt(v + cfg.rms_norm_eps)) * normw
        logits = hb @ headw.T
        nlls.append(F.cross_entropy(logits[:-1].float(), ids[b, 1:].cuda()) * (seqlen - 1))
        ntok += seqlen - 1
        del hb, logits
    return torch.exp(torch.stack(nlls).sum() / ntok).item()


def _capture_eabs(layer, hc, mask, pos):
    """Run a calib forward through the (still-fp16) layer with pre-hooks on every TARGET Linear,
    capturing per-input-channel E|x| -> the AWQ scale stat. Returns {name: eabs[in_features]} and
    the layer output (to carry the calib state forward through quantized-prev layers, sequential)."""
    eabs, hooks = {}, []

    def mk(nm):
        def hk(m, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            e = torch.nan_to_num(x.abs().mean(0), nan=0.0, posinf=0.0, neginf=0.0)   # guard drifted calib
            eabs[nm] = e                                  # [in_features] (one call per expert's routed tokens)
        return hk
    for name, mod in layer.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGETS:
            hooks.append(mod.register_forward_pre_hook(mk(name)))
    with torch.no_grad():
        hc_out = layer(hc, attention_mask=mask, position_ids=pos)[0]
    for hk in hooks:
        hk.remove()
    return eabs, hc_out


def _prepass_eabs(model, L, smap, rot, hc0, cmask, cpos):
    """Static AWQ: ONE fp16 streaming pass over the calib tokens, capturing per-input-channel E|x|
    for every TARGET linear in every layer -> {layer_idx: {name: eabs}}. No quantization and no
    sequential feedback (the calib state flows through FP16 layers only) -> finite where the
    sequential variant drifts to NaN ~27 layers deep."""
    hc = hc0
    out = {}
    for li in range(L):
        layer = model.model.layers[li]
        materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        eabs, hc = _capture_eabs(layer, hc, cmask, cpos)   # E|x| + fp16 layer output (carried forward)
        out[li] = eabs
        free_layer(layer)
        gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0:
            print(f"  [awq-prepass] layer {li}/{L} captured", flush=True)
    return out


def _quantize_layer(layer, report, awq_eabs=None, awq_alpha=0.5):
    """Quantize all TARGET linears in this (CPU-resident) layer in place. If awq_eabs is given,
    apply AWQ per-channel activation-aware scaling (awq_s from E|x|). Accumulates (bw*size)+size for
    the weighted-average b/w, plus fp16-KEPT params (router, norms, skipped) for honest accounting."""
    qhere = 0
    for name, mod in layer.named_modules():
        short = name.split(".")[-1]
        if isinstance(mod, nn.Linear) and short in TARGETS:
            if mod.weight.shape[1] % G != 0:        # trellis groups over cols in blocks of G=128;
                continue                            # only the dense layer-0 down_proj (10944) fails
            W = mod.weight.detach().float().cpu().numpy()  # device-agnostic (weight may be on cuda)
            s = None                                        # else keep fp16 (residual-writer, 0.15%)
            if awq_eabs is not None and name in awq_eabs:
                s = _awq_scale({name: awq_eabs[name].cpu().numpy()}, name, awq_alpha).astype(np.float32)
                if not np.isfinite(s).all():               # never let a bad AWQ scale poison the weight
                    s = None; report["awq_skip"] = report.get("awq_skip", 0) + 1
            wh, bw = trellis_quant(W, K=k_for(name), awq_s=s)
            if not np.isfinite(wh).all():                  # AWQ-quantized weight is NaN/inf -> retry data-free
                report["nan_fallback"] = report.get("nan_fallback", 0) + 1
                wh, bw = trellis_quant(W, K=k_for(name), awq_s=None)
            if not np.isfinite(wh).all():                  # still bad -> keep fp16 (last resort, never NaN)
                report["fp16_fallback"] = report.get("fp16_fallback", 0) + 1
                wh, bw = W, 16.0
            mod.weight.data = torch.from_numpy(wh).to(mod.weight.device, torch.float32)
            report["bits"] += bw * W.size
            report["q"] += W.size
            qhere += W.size
    report["kept"] += sum(p.numel() for p in layer.parameters()) - qhere   # router + norms + skipped


# --------------------------------------------------------------------------- modes
def baseline():
    """fp16 (fp32-from-bf16) WikiText-2 ppl via the streaming layer loop -- the honest reference."""
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = _eval_setup(8)
    print(f"DeepSeek-V2-Lite baseline: {nwin} windows x {seqlen} tok, {L} layers", flush=True)
    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        h = _run_layer(layer, h, mask, pos)
        free_layer(layer)
        gc.collect(); torch.cuda.empty_cache()
        if li % 4 == 0:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)
    ppl = _finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    line = f"DeepSeek-V2-Lite fp16 baseline: WikiText-2 ppl = {ppl:.4f} ({nwin} win, {time.time()-t0:.0f}s)"
    print("\n" + line, flush=True)
    _save(line)


def validate():
    """CHEAP sanity before the multi-hour full run: (1) single-expert round-trip MSE, and
    (2) quantize ONE MoE layer (EVOQ_VAL_LAYER, default 5) to 2-bit, rest fp16 -> ppl delta."""
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = _eval_setup(4)
    vlayer = int(os.environ.get("EVOQ_VAL_LAYER", "5"))

    # (1) single-expert round-trip MSE (data-free, the codec's intrinsic fidelity on an expert)
    print("=== single-expert round-trip MSE (layer 5, expert 0) ===", flush=True)
    for short in ("gate_proj", "up_proj", "down_proj"):
        W = read_tensor(smap, f"model.layers.5.mlp.experts.0.{short}.weight").float().numpy()
        wh, bw = trellis_quant(W, K=k_for(short))
        rel = float(((W - wh) ** 2).mean() / (W ** 2).mean())
        print(f"  expert0.{short:<10} {W.shape}  rel-MSE={rel:.4e}  @ {bw:.3f}b", flush=True)

    # (2) one MoE layer quantized, rest fp16
    print(f"\n=== one-layer ablation: quantize MoE layer {vlayer} only ===", flush=True)
    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        if li == vlayer:
            rep = {"bits": 0.0, "q": 0, "kept": 0}
            _quantize_layer(layer, rep)
        layer.cuda()
        h = _run_layer(layer, h, mask, pos)
        free_layer(layer)
        gc.collect(); torch.cuda.empty_cache()
    ppl = _finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    line = (f"DeepSeek-V2-Lite ONE-LAYER({vlayer}) 2-bit, rest fp16: WikiText-2 ppl = {ppl:.4f} "
            f"({nwin} win, {time.time()-t0:.0f}s)")
    print("\n" + line, flush=True)
    _save(line)


def measure():
    """Full run: quantize EVERY target linear (all 27 layers) -> 2-bit WikiText-2 ppl + avg b/w.
    EVOQ_AWQ=1 enables activation-aware scaling: a 512-tok calib state (disjoint wikitext train) is
    streamed through quantized-prev layers; per layer we capture E|x| from the fp16 calib forward and
    quantize with the AWQ scale (sequential AWQ, the measure7b pattern)."""
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = _eval_setup(8)
    down_k = os.environ.get("EVOQ_DOWN_K", str(K_RATE)); attn_k = os.environ.get("EVOQ_ATTN_K", str(K_RATE))
    shared_k = os.environ.get("EVOQ_SHARED_K", str(K_RATE)); dense_k = os.environ.get("EVOQ_DENSE_K", str(K_RATE))
    int8 = bool(os.environ.get("EVOQ_INT8_GS"))
    awq = bool(int(os.environ.get("EVOQ_AWQ", "0"))); awq_a = float(os.environ.get("EVOQ_AWQ_ALPHA", "0.5"))
    awq_static = bool(int(os.environ.get("EVOQ_AWQ_STATIC", "0")))   # one-pass E|x|, no sequential feedback
    if awq_static:
        awq = True
    writers_k = os.environ.get("EVOQ_WRITERS_K")
    cfgstr = (f"routed-gate/up=2 DOWN_K={down_k} ATTN_K={attn_k} SHARED_K={shared_k} DENSE_K={dense_k} "
              f"INT8_GS={int8} AWQ={awq}(a={awq_a}){' STATIC' if awq_static else ''}"
              f"{' WRITERS_K=' + writers_k if writers_k else ''}")
    print(f"DeepSeek-V2-Lite carve-out 2-bit: {nwin} win, {cfgstr}", flush=True)
    hc = cmask = cpos = None
    if awq:                                              # calib hidden state (disjoint wikitext train)
        ctxt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                 "wikitext2_train.txt"), encoding="utf-8").read()
        cids = tok(ctxt, return_tensors="pt").input_ids[:, :512]
        cemb = read_tensor(smap, "model.embed_tokens.weight").float()
        hc = F.embedding(cids, cemb).cuda(); del cemb
        cmask = causal_mask(512); cpos = torch.arange(512, device="cuda").unsqueeze(0)
    static_eabs = None
    if awq_static:                                       # ONE fp16 pre-pass: capture all E|x|, no feedback
        static_eabs = _prepass_eabs(model, L, smap, rot, hc, cmask, cpos)
    import json as _json
    from safetensors.torch import save_file as _sf_save, load_file as _sf_load
    cdir = cache_dir(cfgstr) if os.environ.get("EVOQ_CACHE") else None  # resumable + cheap re-eval
    if cdir:
        print(f"  cache: {cdir}", flush=True)
    rep = {"bits": 0.0, "q": 0, "kept": 0}
    maxl = int(os.environ.get("EVOQ_MAXL", str(L)))      # cap layers for cheap diagnostics (default all)
    t0 = time.time(); nan_layer = -1
    for li in range(min(L, maxl)):
        layer = model.model.layers[li]
        materialize_cpu(layer, smap, f"model.layers.{li}.")  # always: structure + fp16-kept (norms/router)
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        lc = os.path.join(cdir, f"layer_{li:02d}.safetensors") if cdir else None
        lm = os.path.join(cdir, f"layer_{li:02d}.json") if cdir else None
        if lc and os.path.exists(lc) and os.path.exists(lm):   # CACHE HIT: load dequantized weights
            cw = _sf_load(lc)
            for name, mod in layer.named_modules():
                if name in cw:
                    mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
            d = _json.load(open(lm)); rep["bits"] += d["bits"]; rep["q"] += d["q"]; rep["kept"] += d["kept"]
        else:                                                  # CACHE MISS: quantize (+ save)
            eabs = None
            if awq_static:                               # static: reuse the pre-captured fp16 E|x|
                eabs = static_eabs.get(li)
            elif awq:                                    # sequential: capture through quantized-prev layers
                eabs, _ = _capture_eabs(layer, hc, cmask, cpos)
            b0 = dict(rep)
            _quantize_layer(layer, rep, awq_eabs=eabs, awq_alpha=awq_a)
            if cdir:
                cw = {name: mod.weight.detach().half().cpu() for name, mod in layer.named_modules()
                      if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGETS}
                _sf_save(cw, lc)
                _json.dump({"bits": rep["bits"] - b0["bits"], "q": rep["q"] - b0["q"],
                            "kept": rep["kept"] - b0["kept"]}, open(lm, "w"))
        h = _run_layer(layer, h, mask, pos)
        if awq and not awq_static:                       # sequential only: propagate calib through quantized layer
            with torch.no_grad():
                hc = layer(hc, attention_mask=cmask, position_ids=cpos)[0]
            if not torch.isfinite(hc).all() and nan_layer < 0:
                nan_layer = li; print(f"  !! calib hc NON-FINITE first at layer {li}", flush=True)
        if not torch.isfinite(h).all() and nan_layer < 0:
            nan_layer = li; print(f"  !! eval h NON-FINITE first at layer {li}", flush=True)
        free_layer(layer)
        gc.collect(); torch.cuda.empty_cache()
        if li % 2 == 0 or li == min(L, maxl) - 1:
            print(f"  layer {li}/{L} quantized+fwd ({time.time()-t0:.0f}s) "
                  f"fallbacks: awq_skip={rep.get('awq_skip',0)} nan={rep.get('nan_fallback',0)} "
                  f"fp16={rep.get('fp16_fallback',0)} hfin={bool(torch.isfinite(h).all())}", flush=True)
    ppl = _finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    bw = rep["bits"] / rep["q"]
    _save(f"[raw] ppl={ppl:.4f} bw={bw:.4f} q={rep['q']} kept={rep['kept']} {cfgstr} {nwin}win {time.time()-t0:.0f}s")
    print(f"\n[raw] ppl={ppl:.4f} (saved defensively before formatting)", flush=True)   # never lose it to a print bug
    # HONEST resident size: quantized payload (incl per-group side-info) + ALL fp16-kept tensors:
    #   in-layer kept (routers mlp.gate, every layernorm) + embed + lm_head + final norm.
    qbytes = rep["bits"] / 8.0
    kept = rep["kept"] + sum(read_tensor(smap, k).numel() for k in
                             ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"))
    kept_bytes = kept * 2.0
    total_params = rep["q"] + kept
    eff_bw = (qbytes + kept_bytes) * 8.0 / total_params      # effective bits/weight over the WHOLE model
    line = (f"DeepSeek-V2-Lite carve-out 2-bit ({cfgstr}): "
            f"WikiText-2 ppl = {ppl:.4f} | quant {bw:.3f} b/w over {rep['q']/1e9:.2f}B params, "
            f"{kept/1e6:.0f}M kept fp16 | resident = {(qbytes+kept_bytes)/1e9:.2f} GB "
            f"(quant {qbytes/1e9:.2f} + fp16 {kept_bytes/1e9:.2f}) = {eff_bw:.3f} b/w whole-model | "
            f"{nwin} win, {time.time()-t0:.0f}s")
    print("\n" + line, flush=True)
    _save(line)


def _save(line):
    try:
        open(RESULTS, "a", encoding="utf-8").write(line + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    {"baseline": baseline, "validate": validate, "measure": measure}.get(mode, baseline)()
