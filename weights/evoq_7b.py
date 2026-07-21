"""evoq 7B: streaming encoder + split runtime for Qwen2.5-7B-Instruct on 16GB RAM / GTX 1060 6GB.

ENCODE (never materializes the dense model):
  meta-device skeleton; per decoder layer: materialize ORIGINAL bf16 weights from shards (~0.9GB),
  capture per-channel |x| means via pre-hooks on its 7 linears (in-flight calibration -- exact,
  since layer-i inputs depend only on layers < i), forward the ORIGINAL layer, encode the 7 tensors
  (champion codec, self_check on 1/layer), write qwen7b.layerNN.evoq, free the layer. Also writes
  a residual safetensors (embed, untied lm_head, norms, q/k/v biases) so the runtime needs no
  original shards.

RUN: meta skeleton; QuantLinears built streaming and placed per device map (layers < EVOQ_GPU_LAYERS
  on cuda, fp32 compute on sm_61; rest + embed + lm_head on CPU); per-layer DeviceShim moves the
  hidden state once per boundary. Modes: ppl (the 7B scale point) | chat (the demo).

Usage:  python -m weights.evoq_7b encode | ppl | chat   [EVOQ_GPU_LAYERS=18]
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors import safe_open
from safetensors.torch import save_file as torch_save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weights.evoq import QuantLinear, encode_tensor, load_container, save_container
from weights.quant_lab import _awq_scale

MDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qwen7b_base")
ODIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qwen7b_evoq")
LINEAR7 = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CALIB_TOKENS = 512
_RAW = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "corpora", "generic-text", "enwik8_256k"), "rb").read()
CALIB_TEXT = _RAW[:8000].decode("latin-1")
EVAL_TEXT = _RAW[120000:128000].decode("latin-1")


def shard_map():
    with open(os.path.join(MDIR, "model.safetensors.index.json")) as fh:
        wm = json.load(fh)["weight_map"]
    return {k: os.path.join(MDIR, v) for k, v in wm.items()}


def read_tensor(smap, key):
    with safe_open(smap[key], framework="pt") as f:
        return f.get_tensor(key)


def meta_model(cfg):
    cfg.torch_dtype = torch.float32          # fp32 skeleton (encoder math + sm_61 runtime)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    return m.to(torch.float32)


def materialize_layer(layer, smap, prefix):
    """Fill one decoder layer's params (fp32 on CPU) from the bf16 shards."""
    layer.to_empty(device="cpu")
    sd = layer.state_dict()
    for name in list(sd.keys()):
        sd[name].copy_(read_tensor(smap, prefix + name).to(torch.float32))


def causal_mask(s, dtype=torch.float32):
    m = torch.full((s, s), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(m, diagonal=1)[None, None]


# ----------------------------------------------------------------------------- ENCODE
def encode():
    os.makedirs(ODIR, exist_ok=True)
    cfg = AutoConfig.from_pretrained(MDIR)
    assert not cfg.tie_word_embeddings, "expected untied lm_head for 7B"
    tok = AutoTokenizer.from_pretrained(MDIR)
    smap = shard_map()
    model = meta_model(cfg)
    L = cfg.num_hidden_layers

    # rotary on CPU (fresh, not meta)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    rot = Qwen2RotaryEmbedding(config=cfg)

    # embed calib tokens
    emb_w = read_tensor(smap, "model.embed_tokens.weight").to(torch.float32)
    ids = tok(CALIB_TEXT, return_tensors="pt").input_ids[:, :CALIB_TOKENS]
    h = emb_w[ids[0]].unsqueeze(0)                                   # [1, T, H] fp32
    del emb_w
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    cos_sin = rot(h, pos)
    mask = causal_mask(ids.shape[1])

    residual = {}                                                    # non-quantized tensors (torch)
    residual["model.embed_tokens.weight"] = read_tensor(smap, "model.embed_tokens.weight").contiguous()
    residual["lm_head.weight"] = read_tensor(smap, "lm_head.weight").contiguous()
    residual["model.norm.weight"] = read_tensor(smap, "model.norm.weight").to(torch.float32).contiguous()

    t0 = time.time()
    for li in range(L):
        prefix = f"model.layers.{li}."
        layer = model.model.layers[li]
        materialize_layer(layer, smap, prefix)

        # in-flight calibration hooks on the 7 linears
        calib = {}
        hooks = []

        def mk(key):
            def hkfn(mod, inp):
                x = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).mean(0)
                calib[key] = x.numpy() if key not in calib else calib[key] + x.numpy()
            return hkfn

        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in LINEAR7:
                hooks.append(mod.register_forward_pre_hook(mk(name.split(".")[-1])))
        with torch.no_grad():
            h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        for hk in hooks:
            hk.remove()

        # encode the 7 projections
        tensors = {}
        for name, mod in layer.named_modules():
            short = name.split(".")[-1]
            if isinstance(mod, nn.Linear) and short in LINEAR7:
                W = mod.weight.detach().numpy()
                s = _awq_scale(calib, short, 0.5).astype(np.float32)
                tensors[prefix + name + ".weight"] = encode_tensor(
                    W, s, self_check=(short == "q_proj"))
                if mod.bias is not None:
                    residual[prefix + name + ".bias"] = mod.bias.detach().to(torch.bfloat16).contiguous()
        for nname in ("input_layernorm", "post_attention_layernorm"):
            residual[prefix + nname + ".weight"] = getattr(layer, nname).weight.detach().to(torch.float32).contiguous()

        save_container(os.path.join(ODIR, f"layer{li:02d}.evoq"), tensors,
                       dict(model="Qwen2.5-7B-Instruct", layer=li, lam=0.008, seed=0))
        layer.to_empty(device="meta")
        del tensors
        gc.collect()
        print(f"  layer {li+1}/{L} encoded  ({time.time()-t0:.0f}s)", flush=True)

    torch_save_file(residual, os.path.join(ODIR, "residual.safetensors"))
    total = sum(os.path.getsize(os.path.join(ODIR, f)) for f in os.listdir(ODIR))
    print(f"DONE: {ODIR}  total {total/1e9:.2f} GB", flush=True)


# ----------------------------------------------------------------------------- FP16 BASELINE
def baseline():
    """Held-out ppl of the ORIGINAL 7B via the same streaming layer loop (no dense model)."""
    cfg = AutoConfig.from_pretrained(MDIR)
    tok = AutoTokenizer.from_pretrained(MDIR)
    smap = shard_map()
    model = meta_model(cfg)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    rot = Qwen2RotaryEmbedding(config=cfg)
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :1024]
    emb_w = read_tensor(smap, "model.embed_tokens.weight").to(torch.float32)
    h = emb_w[ids[0]].unsqueeze(0)
    del emb_w
    pos = torch.arange(ids.shape[1]).unsqueeze(0)
    cos_sin = rot(h, pos)
    mask = causal_mask(ids.shape[1])
    t0 = time.time()
    for li in range(cfg.num_hidden_layers):
        layer = model.model.layers[li]
        materialize_layer(layer, smap, f"model.layers.{li}.")
        with torch.no_grad():
            h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        layer.to_empty(device="meta")
        gc.collect()
    nw = read_tensor(smap, "model.norm.weight").to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = (h * torch.rsqrt(var + cfg.rms_norm_eps)) * nw
    head = read_tensor(smap, "lm_head.weight").to(torch.float32)
    logits = h[0] @ head.T
    import torch.nn.functional as F
    loss = F.cross_entropy(logits[:-1], ids[0, 1:])
    print(f"\nQwen2.5-7B ORIGINAL (fp32-from-bf16) held-out ppl = {float(torch.exp(loss)):.4f}  "
          f"({time.time()-t0:.0f}s)")


# ----------------------------------------------------------------------------- RUNTIME
class LayerShim(nn.Module):
    """Moves the hidden state (and rotary/mask tensors) to this layer's device, once."""

    def __init__(self, layer, device):
        super().__init__()
        self.layer = layer
        self.device = torch.device(device)

    def forward(self, hidden_states, **kw):
        hidden_states = hidden_states.to(self.device)
        for k in ("attention_mask", "position_ids", "cache_position"):
            if kw.get(k) is not None and isinstance(kw[k], torch.Tensor):
                kw[k] = kw[k].to(self.device)
        if kw.get("position_embeddings") is not None:
            kw["position_embeddings"] = tuple(t.to(self.device) for t in kw["position_embeddings"])
        return self.layer(hidden_states, **kw)


def build_runtime():
    cfg = AutoConfig.from_pretrained(MDIR)
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MDIR)
    model = meta_model(cfg)
    L = cfg.num_hidden_layers
    n_gpu = int(os.environ.get("EVOQ_GPU_LAYERS", "18")) if torch.cuda.is_available() else 0
    print(f"runtime: layers 0-{n_gpu-1} on cuda (fp32), {n_gpu}-{L-1} + embed/head on cpu", flush=True)

    with safe_open(os.path.join(ODIR, "residual.safetensors"), framework="pt") as rf:
        res = {k: rf.get_tensor(k) for k in rf.keys()}

    # embed / final norm / lm_head on CPU
    model.model.embed_tokens = nn.Embedding.from_pretrained(
        res["model.embed_tokens.weight"].to(torch.float32), freeze=True)
    model.model.norm.to_empty(device="cpu")
    model.model.norm.weight.data.copy_(res["model.norm.weight"].to(torch.float32))
    model.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device="cpu", dtype=torch.float32)
    model.lm_head.weight.data.copy_(res["lm_head.weight"].to(torch.float32))
    model.model.rotary_emb.to_empty(device="cpu")
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    model.model.rotary_emb = Qwen2RotaryEmbedding(config=cfg)

    for li in range(L):
        dev = "cuda" if li < n_gpu else "cpu"
        prefix = f"model.layers.{li}."
        layer = model.model.layers[li]
        layer.to_empty(device="cpu")
        # norms + biases from residual
        layer.input_layernorm.weight.data.copy_(res[prefix + "input_layernorm.weight"].to(torch.float32))
        layer.post_attention_layernorm.weight.data.copy_(res[prefix + "post_attention_layernorm.weight"].to(torch.float32))
        meta, comps = load_container(os.path.join(ODIR, f"layer{li:02d}.evoq"))
        for name, mod in list(layer.named_modules()):
            for cn, child in list(mod.named_children()):
                full = prefix + (f"{name}.{cn}" if name else cn) + ".weight"
                if isinstance(child, nn.Linear) and full in comps:
                    bk = full[:-len(".weight")] + ".bias"
                    bias = res[bk].to(torch.float32) if bk in res else None
                    ql = QuantLinear(comps[full], bias, compute_dtype=torch.float32)
                    setattr(mod, cn, ql)
        del comps
        layer.to(dev)                       # moves packed buffers + norms to the mapped device
        model.model.layers[li] = LayerShim(layer, dev)
        gc.collect()
        if li % 7 == 0:
            print(f"  built layer {li}  ({dev})", flush=True)
    model.eval()
    return model, tok


def ppl_eval(model, tok):
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :1024]
    t0 = time.time()
    with torch.no_grad():
        out = model(ids, labels=ids)
    print(f"\nQwen2.5-7B evoq runtime held-out ppl = {float(torch.exp(out.loss)):.4f}   "
          f"({time.time()-t0:.0f}s)\n(prediction from the scale law: gap ~ +0.17 over fp16)")


def chat(model, tok):
    msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    n = out.shape[1] - ids.shape[1]
    print(f"\n=== Qwen2.5-7B @ ~6.6 bits/weight resident, GTX 1060 6GB + CPU split ===\n{txt}\n"
          f"({n} tokens in {time.time()-t0:.0f}s = {n/(time.time()-t0):.2f} tok/s)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ppl"
    if mode == "encode":
        encode()
    elif mode == "baseline":
        baseline()
    else:
        model, tok = build_runtime()
        if mode == "chat":
            chat(model, tok)
        else:
            ppl_eval(model, tok)
