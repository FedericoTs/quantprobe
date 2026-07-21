"""dense_2bit_gate.py -- the make-or-break functional gate for the dense-Gemma port, run on a DENSE model
we already have locally (Qwen2.5-7B base) -- NO download, NO gated weights.

Question A1': does 2-bit hold on a DENSE model's MLP bulk (which is what we'd 2-bit in Gemma)? Streams the
dense 7B layer-by-layer (evoq_7b skeleton), quantizes a chosen tensor group to 2-bit, and reports held-out
ppl vs fp16. If MLP@2bit is cheap -> dense 2-bit viable (protect attention, 2-bit the MLP bulk = dense
carve-out) -> green-light Gemma. If it collapses -> dense 2-bit is not viable, don't download Gemma.

EVOQ_DENSE_MODE in {fp16, mlp2, attn2, all2}; EVOQ_DENSE_CODEC in {fast, trellis}.
"""
from __future__ import annotations
import gc, os, sys, time
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
import weights.evoq_moe as em
from weights.lut_decode_gemv import pack_2bit, dequant

HERE = os.path.dirname(os.path.abspath(__file__))
MDIR = os.path.join(HERE, "data", "qwen7b_base")
LINEAR7 = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
WIN = 2048


def bits_for(mode, short):
    """bit-width per tensor for each mode (16 = keep fp16)."""
    if mode == "fp16":
        return 16
    if mode == "carveout":                # the deployment recipe: 2-bit MLP bulk + 4-bit attention
        return 2 if short in MLP else 4
    if mode == "mlp2":
        return 2 if short in MLP else 16
    if mode == "attn2":
        return 2 if short in ATTN else 16
    if mode == "all2":
        return 2
    return 16


def shard_map():
    import json
    with open(os.path.join(MDIR, "model.safetensors.index.json")) as fh:
        wm = json.load(fh)["weight_map"]
    return {k: os.path.join(MDIR, v) for k, v in wm.items()}


def read_tensor(smap, key):
    with safe_open(smap[key], framework="pt") as f:
        return f.get_tensor(key)


def materialize_layer(layer, smap, prefix):
    layer.to_empty(device="cpu")
    sd = layer.state_dict()
    for name in list(sd.keys()):
        sd[name].copy_(read_tensor(smap, prefix + name).to(torch.float32))


def causal_mask(s):
    m = torch.full((s, s), torch.finfo(torch.float32).min)
    return torch.triu(m, diagonal=1)[None, None]


def q_tensor(W, K, codec):
    if K >= 16:
        return W
    W = np.ascontiguousarray(W.astype(np.float32))
    if codec == "trellis" or K != 2:                     # fast path only implements 2-bit
        wh, _ = em.trellis_quant(W, K=K)
        return np.asarray(wh, np.float32).reshape(W.shape)
    packed, sc = pack_2bit(W)
    return dequant(packed, sc, W.shape[1]).astype(np.float32)


def run():
    mode = os.environ.get("EVOQ_DENSE_MODE", "fp16")
    codec = os.environ.get("EVOQ_DENSE_CODEC", "trellis")
    nwin = int(os.environ.get("EVOQ_NWIN", "16"))
    cfg = AutoConfig.from_pretrained(MDIR)
    cfg.torch_dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(MDIR)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model = model.to(torch.float32)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    rot = Qwen2RotaryEmbedding(config=cfg).to("cuda")
    smap = shard_map()
    L = cfg.num_hidden_layers

    from weights.wikitext2_ppl import get_wikitext2_test_ids
    ids_all = get_wikitext2_test_ids(tok)                # WikiText-2 test (paper metric)
    nwin = min(nwin, ids_all.numel() // WIN)
    ids = ids_all[:, :nwin * WIN].reshape(nwin, WIN)
    emb_w = read_tensor(smap, "model.embed_tokens.weight").to(torch.float32)
    h = emb_w[ids]                                        # [nwin, WIN, H]
    del emb_w
    pos = torch.arange(WIN).unsqueeze(0)
    mask = causal_mask(WIN)
    print(f"DENSE GATE (Qwen2.5-7B, {mode}, codec={codec}): {nwin}x{WIN} WikiText-2", flush=True)

    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        materialize_layer(layer, smap, f"model.layers.{li}.")
        nq = 0
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear):
                K = bits_for(mode, name.split(".")[-1])
                if K < 16:
                    mod.weight.data = torch.from_numpy(q_tensor(mod.weight.detach().numpy(), K, codec))
                    nq += 1
        layer.cuda()
        cos_sin = rot(h[:1].cuda(), pos.cuda())
        outs = []
        for b in range(nwin):
            with torch.no_grad():
                yb = layer(h[b:b + 1].cuda(), attention_mask=mask.cuda(), position_embeddings=cos_sin)[0]
            outs.append(yb.cpu())
        h = torch.cat(outs, 0)
        layer.to_empty(device="meta"); gc.collect(); torch.cuda.empty_cache()
        if li % 7 == 0 or li == L - 1:
            print(f"  layer {li}/{L} (q={nq}) ({time.time()-t0:.0f}s)", flush=True)

    nw = read_tensor(smap, "model.norm.weight").to(torch.float32)
    head = read_tensor(smap, "lm_head.weight").to(torch.float32)
    nlls, ntok = [], 0
    for b in range(nwin):
        hb = h[b]
        var = hb.pow(2).mean(-1, keepdim=True)
        hb = (hb * torch.rsqrt(var + cfg.rms_norm_eps)) * nw
        logits = hb @ head.T
        nlls.append(F.cross_entropy(logits[:-1].float(), ids[b, 1:]) * (WIN - 1)); ntok += WIN - 1
    ppl = float(torch.exp(torch.stack(nlls).sum() / ntok))
    line = f"DENSE-GATE [{mode} codec={codec}]: ppl={ppl:.4f} | Qwen2.5-7B dense, {nwin}x{WIN} WikiText-2"
    print("\n" + line, flush=True)
    open(os.path.join(HERE, "data", "dense_gate.log"), "a", encoding="utf-8").write(line + "\n")


if __name__ == "__main__":
    run()
