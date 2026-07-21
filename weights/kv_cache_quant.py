"""kv_cache_quant.py -- the weight<->KV-cache UNIFICATION test (round-2 physics).

In MLA the fragile bottleneck (the 512-dim c_KV latent) IS the KV cache. The rank-robustness law predicts
that quantizing the c_KV *activation* (the cache) should be just as fragile as quantizing the kv weights,
and for the same reason -- one low-rank channel. We keep the kv WEIGHTS at fp16 and quantize only the
c_KV ACTIVATION (per-token absmax, b bits) via a forward hook, sweeping b. If the cache-quant damage curve
mirrors the weight-quant curve (and shows the same critical-bit-width), the weight finding TRANSFERS to
KV-cache compression -- one law, two literatures.

EVOQ_CACHE_BITS in {16,8,4,3,2}. Rest of model = carve-out (cached). Full WikiText-2 set.
"""
from __future__ import annotations
import gc, os, sys, time
import numpy as np
import torch
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from safetensors.torch import load_file as _sf_load
import weights.evoq_moe as em

CARVE_CFG = "routed-gate/up=2 DOWN_K=3 ATTN_K=4 SHARED_K=4 DENSE_K=4 INT8_GS=True AWQ=False(a=0.5)"
FP16, CARVE = 6.3070, 6.9616
D_C = 512


def quantize_latent(c, bits):
    if bits >= 16:
        return c
    qmax = 2 ** (bits - 1) - 1
    scale = c.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax     # per-token absmax
    return (c / scale).round().clamp(-qmax - 1, qmax) * scale


def run():
    bits = float(os.environ.get("EVOQ_CACHE_BITS", "16"))
    nwin = int(os.environ.get("EVOQ_NWIN", "151"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(nwin)
    cdir = em.cache_dir(CARVE_CFG)
    print(f"KV-CACHE QUANT TEST: c_KV latent (the cache) -> {bits}-bit | kv weights fp16 | {nwin} win", flush=True)
    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))
        for name, mod in layer.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if name.endswith(("kv_a_proj_with_mqa", "kv_b_proj")):
                pass                                          # keep KV weights at fp16
            elif name in cw:
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        # hook: quantize the c_KV latent (first 512 of kv_a output) = the cache
        hk = None
        if bits < 16:
            def _hook(m, inp, out):
                cq = quantize_latent(out[..., :D_C], bits)
                return torch.cat([cq, out[..., D_C:]], dim=-1)
            hk = layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(_hook)
        h = em._run_layer(layer, h, mask, pos)
        if hk is not None:
            hk.remove()
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 9 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    line = (f"KV-CACHE QUANT [{bits}-bit c_KV]: ppl={ppl:.4f} (delta vs fp16-kv-cache baseline) | "
            f"carve-out(weights)={CARVE:.4f} fp16={FP16:.4f} | compare to WEIGHT-quant kv@2bit=+5.27")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
