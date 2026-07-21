"""evoq_moe_qwen.py -- GENERALITY 2nd model: the data-free carve-out harness ported to
Qwen1.5-MoE-A2.7B (24 layers, standard MHA attention q/k/v/o [NO MLA], 60 routed + 1 shared expert
per layer, EVERY layer MoE [no dense layer-0], plain top-4 softmax routing). Tests whether the
"protect attention + shared experts" rule generalizes beyond DeepSeek-V2-Lite's MLA architecture.

Reuses the VALIDATED codec/quant logic from evoq_moe (trellis_quant + _quantize_layer + _finish_ppl +
cache) via a monkeypatch of the per-tensor TARGETS / k_for to Qwen's names. Only the model plumbing is
Qwen-specific: transformers 4.49 computes rotary (cos,sin) once at the model level and passes it to each
layer as position_embeddings (DeepSeek's remote code computed it per-layer), so the streaming forward
must supply position_embeddings.

Run:  EVOQ_*_K ... .venv/Scripts/python -m weights.evoq_moe_qwen baseline | measure
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import weights.evoq_moe as em                         # reuse validated codec + cache + ppl
from weights.qtip_trellis import K_RATE

MDIR_Q = os.environ.get("EVOQ_QWEN_DIR", r"D:\evo-compress-data\qwen_moe")
# Qwen2MoE target linears: standard attention (q/k/v/o) + expert/shared MLP (gate/up/down).
TARGETS_Q = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def k_for_qwen(name):
    """Per-tensor trellis rate from the full module path -- the SAME carve-out rule as DeepSeek:
    attention + shared expert at EVOQ_ATTN_K / EVOQ_SHARED_K (4-bit), routed experts at 2-bit with
    their down_proj at EVOQ_DOWN_K (3-bit). Router (mlp.gate, mlp.shared_expert_gate), embeddings,
    lm_head, norms are not TARGETS -> kept fp16. No dense layer in Qwen1.5-MoE (all layers MoE)."""
    if "self_attn" in name:
        return int(os.environ.get("EVOQ_ATTN_K", str(K_RATE)))
    if "shared_expert" in name:                        # shared_expert.{gate,up,down}_proj
        return int(os.environ.get("EVOQ_SHARED_K", str(K_RATE)))
    if ".experts." in name:                            # routed expert (the bulk -> stay low)
        if name.endswith("down_proj"):
            return int(os.environ.get("EVOQ_DOWN_K", str(K_RATE)))
        return K_RATE
    return K_RATE


# monkeypatch the reused codec to Qwen's per-tensor scheme (this process only; real evoq_moe runs
# are separate processes). em._quantize_layer / _capture_eabs resolve TARGETS / k_for as module
# globals at call time, so patching them here makes the reused logic Qwen-aware.
em.TARGETS = TARGETS_Q
em.k_for = k_for_qwen


def shard_map_q():
    with open(os.path.join(MDIR_Q, "model.safetensors.index.json")) as fh:
        wm = json.load(fh)["weight_map"]
    return {k: os.path.join(MDIR_Q, v) for k, v in wm.items()}


def _eval_setup_qwen(nwin_default):
    from weights.wikitext2_ppl import get_wikitext2_test_ids
    from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeRotaryEmbedding
    cfg = AutoConfig.from_pretrained(MDIR_Q)
    cfg.torch_dtype = torch.float32                     # Qwen config defaults to bf16; force fp32 weights
    cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(MDIR_Q)
    smap = shard_map_q()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model = model.to(torch.float32)
    L = cfg.num_hidden_layers
    seqlen = 2048
    nwin = int(os.environ.get("EVOQ_NWIN", str(nwin_default)))
    ids_all = get_wikitext2_test_ids(tok)
    nwin = min(nwin, ids_all.numel() // seqlen)
    ids = ids_all[:, :nwin * seqlen].reshape(nwin, seqlen)
    emb = em.read_tensor(smap, "model.embed_tokens.weight").float()
    h = F.embedding(ids, emb)                           # [nwin, seqlen, H] on CPU
    del emb
    # rotary cos/sin computed ONCE (position-only, shared across windows/layers) -> position_embeddings
    rot = Qwen2MoeRotaryEmbedding(cfg).to("cuda")
    pos = torch.arange(seqlen, device="cuda").unsqueeze(0)
    dummy = torch.zeros(1, seqlen, cfg.hidden_size, device="cuda")
    with torch.no_grad():
        cos, sin = rot(dummy, pos)
    mask = em.causal_mask(seqlen)
    return cfg, tok, smap, model, L, seqlen, nwin, ids, h, (cos, sin), mask, pos


def _run_layer_qwen(layer, h, posemb, mask, pos):
    """Forward one decoder layer over all eval windows (bsz=1), passing the precomputed rotary."""
    cos, sin = posemb
    out = torch.empty_like(h)
    for b in range(h.shape[0]):
        hb = h[b:b + 1].cuda()
        with torch.no_grad():
            yb = layer(hb, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))[0]
        out[b:b + 1] = yb.cpu(); del hb, yb
    return out


def _loop(model, L, smap, posemb, mask, pos, h, quant, cdir, rep):
    maxl = int(os.environ.get("EVOQ_MAXL", str(L)))
    t0 = time.time()
    for li in range(min(L, maxl)):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.cuda()
        if quant:
            lc = os.path.join(cdir, f"layer_{li:02d}.safetensors") if cdir else None
            lm = os.path.join(cdir, f"layer_{li:02d}.json") if cdir else None
            if lc and os.path.exists(lc) and os.path.exists(lm):
                from safetensors.torch import load_file as _sf_load
                cw = _sf_load(lc)
                for name, mod in layer.named_modules():
                    if name in cw:
                        mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
                d = json.load(open(lm)); rep["bits"] += d["bits"]; rep["q"] += d["q"]; rep["kept"] += d["kept"]
            else:
                b0 = dict(rep)
                em._quantize_layer(layer, rep)          # data-free (TARGETS/k_for monkeypatched)
                if cdir:
                    from safetensors.torch import save_file as _sf_save
                    cw = {name: mod.weight.detach().half().cpu() for name, mod in layer.named_modules()
                          if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGETS_Q}
                    _sf_save(cw, lc)
                    json.dump({"bits": rep["bits"] - b0["bits"], "q": rep["q"] - b0["q"],
                               "kept": rep["kept"] - b0["kept"]}, open(lm, "w"))
        h = _run_layer_qwen(layer, h, posemb, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 2 == 0 or li == min(L, maxl) - 1:
            print(f"  layer {li}/{L} {'quant+' if quant else ''}fwd ({time.time()-t0:.0f}s) "
                  f"hfin={bool(torch.isfinite(h).all())}", flush=True)
    return h, time.time() - t0


def baseline():
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, posemb, mask, pos = _eval_setup_qwen(8)
    print(f"Qwen1.5-MoE-A2.7B fp16 baseline: {nwin} win x {seqlen} tok, {L} layers", flush=True)
    rep = {"bits": 0.0, "q": 0, "kept": 0}
    h, dt = _loop(model, L, smap, posemb, mask, pos, h, quant=False, cdir=None, rep=rep)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    line = f"Qwen1.5-MoE-A2.7B fp16 baseline: WikiText-2 ppl = {ppl:.4f} ({nwin} win, {dt:.0f}s)"
    print("\n" + line, flush=True); em._save(line)


def measure():
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, posemb, mask, pos = _eval_setup_qwen(8)
    down_k = os.environ.get("EVOQ_DOWN_K", str(K_RATE)); attn_k = os.environ.get("EVOQ_ATTN_K", str(K_RATE))
    shared_k = os.environ.get("EVOQ_SHARED_K", str(K_RATE)); int8 = bool(os.environ.get("EVOQ_INT8_GS"))
    cfgstr = f"qwen-moe routed-gate/up=2 DOWN_K={down_k} ATTN_K={attn_k} SHARED_K={shared_k} INT8_GS={int8}"
    print(f"Qwen1.5-MoE-A2.7B carve-out 2-bit: {nwin} win, {cfgstr}", flush=True)
    cdir = em.cache_dir(cfgstr) if os.environ.get("EVOQ_CACHE") else None
    if cdir:
        print(f"  cache: {cdir}", flush=True)
    rep = {"bits": 0.0, "q": 0, "kept": 0}
    h, dt = _loop(model, L, smap, posemb, mask, pos, h, quant=True, cdir=cdir, rep=rep)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    bw = rep["bits"] / rep["q"]
    em._save(f"[raw qwen] ppl={ppl:.4f} bw={bw:.4f} q={rep['q']} {cfgstr} {nwin}win {dt:.0f}s")
    print(f"\n[raw] ppl={ppl:.4f} (saved defensively)", flush=True)
    qbytes = rep["bits"] / 8.0
    kept = rep["kept"] + sum(em.read_tensor(smap, k).numel() for k in
                             ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"))
    eff = (qbytes + kept * 2.0) * 8.0 / (rep["q"] + kept)
    line = (f"Qwen1.5-MoE-A2.7B carve-out 2-bit ({cfgstr}): WikiText-2 ppl = {ppl:.4f} | "
            f"quant {bw:.3f} b/w over {rep['q']/1e9:.2f}B params, {kept/1e6:.0f}M kept fp16 | "
            f"resident = {(qbytes+kept*2.0)/1e9:.2f} GB = {eff:.3f} b/w whole-model | {nwin} win, {dt:.0f}s")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    {"baseline": baseline, "measure": measure}.get(mode, baseline)()
