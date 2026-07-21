"""gemma_semantic_probe.py -- T1 go/no-go for SEMANTIC CARVE-OUT (contextual precision) on Gemma 4 12B.

Idea: keep the whole model 2-bit resident, but page the CONTEXT-relevant neurons to high precision.
Works only if GeGLU neuron activation is (a) CONCENTRATED per domain and (b) DIFFERENT across domains.
Measures, per layer, per domain: fraction of the 15360 intermediate neurons carrying 90/99% of activation
energy (hook = down_proj pre-input = act(gate)*up), and the cross-domain Jaccard of top-10% neuron sets.
PASS: <25% of neurons carry 90% AND cross-domain Jaccard low (<0.5). FAIL: flat or universal -> idea dead.
RUN WITH .venv-gemma. ~15 min (3 domains x 1 window x 1024 tok, fp16 streaming).
"""
from __future__ import annotations
import gc, os, sys
import numpy as np
import torch
from collections import UserDict
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

MDIR = "D:/evo-compress-data/gemma-4-12b"
LP = "model.language_model."
CAP_LAYERS = {3, 13, 24, 35, 44}
SEQ = 1024
HERE = os.path.dirname(os.path.abspath(__file__))


def domain_texts():
    # three genuinely different domains, all local
    wiki = open(os.path.join(HERE, "data", "wikitext2_train.txt"), encoding="utf-8").read()[:12000]
    code = open(os.path.join(HERE, "evoq_moe.py"), encoding="utf-8").read()[:12000]
    enc = open(os.path.join(os.path.dirname(HERE), "data", "corpora", "generic-text", "enwik8_256k"), "rb").read()[60000:72000].decode("latin-1")
    return {"prose": wiki, "code": code, "encyc": enc}


def run():
    cfg = AutoConfig.from_pretrained(MDIR)
    tcfg = cfg.text_config
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model = model.to(torch.float32).eval()
    lm = model.model.language_model
    tok = AutoTokenizer.from_pretrained(MDIR)
    f = safe_open(os.path.join(MDIR, "model.safetensors"), framework="pt")
    g = lambda k: f.get_tensor(k)

    lm.embed_tokens.to_empty(device="cpu")
    lm.embed_tokens.weight.data.copy_(g(LP + "embed_tokens.weight").float())
    lm.embed_tokens.embed_scale = torch.tensor(tcfg.hidden_size ** 0.5)
    rot = Gemma4TextRotaryEmbedding(tcfg).cuda()
    pos = torch.arange(SEQ, device="cuda").unsqueeze(0)

    doms = domain_texts()
    hs = {}
    for d, txt in doms.items():
        ids = tok(txt, return_tensors="pt").input_ids[:, :SEQ]
        with torch.no_grad():
            hs[d] = lm.embed_tokens(ids)
    sample = next(iter(hs.values()))[:1].cuda()
    mk = dict(config=tcfg, inputs_embeds=sample, attention_mask=None, past_key_values=None, position_ids=pos)
    masks = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
    pe = {lt: rot(sample, pos, lt) for lt in set(tcfg.layer_types)}
    del sample; torch.cuda.empty_cache()

    energy = {d: {} for d in doms}                       # energy[domain][layer] = per-neuron mean |act|
    L = tcfg.num_hidden_layers
    for li in range(L):
        lt = tcfg.layer_types[li]
        layer = lm.layers[li]
        layer.to_empty(device="cpu")
        sd = layer.state_dict()
        for name in list(sd.keys()):
            sd[name].copy_(g(f"{LP}layers.{li}.{name}").float())
        layer.cuda()
        hooks = []
        if li in CAP_LAYERS:
            store = {}
            def mkhook(dom_store):
                def hk(mod, inp):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    dom_store["e"] = x.abs().mean(0).cpu().numpy()   # per-neuron mean |act(gate)*up|
                return hk
            hooks.append((store, layer.mlp.down_proj.register_forward_pre_hook(mkhook(store))))
        for d in doms:
            if hooks:
                hooks[0][0].clear()
            with torch.no_grad():
                out = layer(hs[d][:1].cuda(), shared_kv_states=UserDict(), position_embeddings=pe[lt],
                            attention_mask=masks[lt], position_ids=pos, past_key_values=None)
            hs[d] = (out[0] if isinstance(out, tuple) else out).cpu()
            if hooks:
                energy[d][li] = hooks[0][0].get("e")
        for _, hk in hooks:
            hk.remove()
        layer.to_empty(device="meta"); gc.collect(); torch.cuda.empty_cache()
        if li % 12 == 0:
            print(f"  layer {li}/{L}", flush=True)

    print("\nSEMANTIC LOCALITY (per-domain neuron concentration + cross-domain overlap)")
    print(f"  {'layer':6s} {'dom':6s}  n90%   n99%   | top10% Jaccard vs other domains")
    for li in sorted(CAP_LAYERS):
        tops = {}
        for d in doms:
            e = energy[d][li]
            if e is None:
                continue
            order = np.argsort(-e); c = np.cumsum(e[order]); c /= c[-1]
            n90 = float(np.searchsorted(c, 0.90) + 1) / len(e)
            n99 = float(np.searchsorted(c, 0.99) + 1) / len(e)
            tops[d] = set(order[:len(e) // 10].tolist())
            print(f"  L{li:<5d} {d:6s}  {n90:5.1%} {n99:5.1%}", flush=True)
        ds = list(tops)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                jac = len(tops[ds[i]] & tops[ds[j]]) / max(1, len(tops[ds[i]] | tops[ds[j]]))
                print(f"         {ds[i]}~{ds[j]}: Jaccard={jac:.2f}", flush=True)
    print("\n  PASS if n90 < ~25% and cross-domain Jaccard < ~0.5 -> semantic carve-out is real; run T2 oracle.")


if __name__ == "__main__":
    run()
