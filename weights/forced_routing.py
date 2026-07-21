"""forced_routing.py -- a CAUSAL test of the routing mechanism (DeepSeek-V2-Lite).

We have shown CORRELATION: 2-bit quantization diverges expert routing, and quality collapses. This
tests CAUSATION: take the uniform-2bit model (ppl 15.38) but FORCE every MoE layer to select the same
experts (and weights) that the fp16 model chose for the same tokens. If perplexity drops back toward
fp16, routing divergence *causes* the collapse; the residual is the experts' own quantization error.

Three streaming passes over the same tokens (off the uniform cache, no re-quant):
  A  fp16                      -> capture (topk_idx, topk_weight) per layer per window  (sanity ~5.66)
  B  uniform-2bit, normal      -> reproduces the collapse                                (sanity ~15.38)
  C  uniform-2bit, FORCED fp16 routing (hook mlp.gate to replay A's selection)          -> the result
Report how much of the (B - A) gap the forced routing recovers.
"""
from __future__ import annotations

import gc
import os
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors.torch import load_file as _sf_load

import weights.evoq_moe as em

# the DeepSeek uniform-2bit cache (built by route_diverge.py b1ey9i329) -> md5 a5877fc5e500
UNIFORM_CFG = "routed-gate/up=2 DOWN_K=2 ATTN_K=2 SHARED_K=2 DENSE_K=2 INT8_GS=True AWQ=False(a=0.5)"
_STATE = {"li": -1, "w": -1}                                   # current (layer, window) for the force hook


def _stream(model, L, smap, rot, mask, pos, h0, nwin, quant, hook, captures, cdir):
    h = h0.clone()
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        if quant:                                              # load uniform-2bit cached weights
            cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))
            for name, mod in layer.named_modules():
                if name in cw:
                    mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        is_moe = hasattr(layer.mlp, "gate"); hk = None
        if is_moe and hook == "capture":
            buf = captures.setdefault(li, [])
            hk = layer.mlp.gate.register_forward_hook(
                lambda m, inp, out, _b=buf: _b.append((out[0].detach().cpu(), out[1].detach().cpu())))
        elif is_moe and hook == "force":
            _STATE["li"] = li
            def _fh(m, inp, out):                              # replace gate output with fp16's routing
                idx, wt = captures[_STATE["li"]][_STATE["w"]]
                return (idx.to(out[0].device), wt.to(out[1].device), out[2])
            hk = layer.mlp.gate.register_forward_hook(_fh)
        out = torch.empty_like(h)
        for b in range(nwin):
            _STATE["w"] = b
            hb = h[b:b + 1].cuda()
            with torch.no_grad():
                yb = layer(hb, attention_mask=mask, position_ids=pos)[0]
            out[b:b + 1] = yb.cpu(); del hb, yb
        h = out
        if hk:
            hk.remove()
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 6 == 0 or li == L - 1:
            print(f"  [{hook or 'plain'}] layer {li}/{L}", flush=True)
    return h


def run():
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(8)
    cdir = em.cache_dir(UNIFORM_CFG)
    print(f"forced-routing causal test: {nwin} win, uniform cache {cdir}", flush=True)
    h0 = h.clone(); captures = {}
    hA = _stream(model, L, smap, rot, mask, pos, h0, nwin, False, "capture", captures, cdir)
    pplA = em._finish_ppl(hA, smap, cfg, ids, seqlen, nwin)
    print(f"  fp16 ppl = {pplA:.4f} (sanity ~5.66)", flush=True)
    hB = _stream(model, L, smap, rot, mask, pos, h0, nwin, True, None, captures, cdir)
    pplB = em._finish_ppl(hB, smap, cfg, ids, seqlen, nwin)
    print(f"  uniform-2bit ppl = {pplB:.4f} (sanity ~15.38)", flush=True)
    hC = _stream(model, L, smap, rot, mask, pos, h0, nwin, True, "force", captures, cdir)
    pplC = em._finish_ppl(hC, smap, cfg, ids, seqlen, nwin)
    rec = 100.0 * (pplB - pplC) / (pplB - pplA + 1e-9)
    line = (f"FORCED-ROUTING (DeepSeek): fp16={pplA:.4f} | uniform-2bit={pplB:.4f} | "
            f"uniform+fp16-routing={pplC:.4f} -> recovers {rec:.0f}% of the collapse gap")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
