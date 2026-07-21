"""m1_kvprobe.py -- break the bottleneck below the tensor: which HALF carries the +5.27?

Quantize only kv_a (down->latent, the WRITE) or only kv_b (latent->heads, the READ) to K bits, keeping
the other half fp16. Localizes the fragility to the write vs read of the latent channel. Also serves as a
weight-precision ladder (EVOQ_KV_K) complementing the cache-precision sweep.

EVOQ_KV_WHICH in {both, kv_a, kv_b}; EVOQ_KV_K bit-width. Full WikiText-2 set, int8 gs.
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
SETS = {"both": ("kv_a_proj_with_mqa", "kv_b_proj"),
        "kv_a": ("kv_a_proj_with_mqa",),
        "kv_b": ("kv_b_proj",)}


def run():
    which = os.environ.get("EVOQ_KV_WHICH", "both")
    K = int(os.environ.get("EVOQ_KV_K", "2"))
    targets = SETS[which]
    nwin = int(os.environ.get("EVOQ_NWIN", "151"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(nwin)
    cdir = em.cache_dir(CARVE_CFG)
    print(f"KV-PROBE: quantize {which} -> {K}-bit (other kv half fp16) | {nwin} win", flush=True)
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
                if name.endswith(targets):
                    W = mod.weight.detach().float().cpu().numpy()
                    wh, _ = em.trellis_quant(W, K=K)
                    mod.weight.data = torch.from_numpy(wh).to(mod.weight.device)
                # else: keep fp16 (materialized)
            elif name in cw:
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        h = em._run_layer(layer, h, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 9 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    line = (f"KV-PROBE [{which}@{K}bit]: ppl={ppl:.4f} (delta vs carve-out {ppl-CARVE:+.4f}) | "
            f"carve-out={CARVE:.4f} fp16={FP16:.4f} | ref: both@2bit=+5.27")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
