"""m1_kvlatent.py -- M1 confirmatory test (red-team): does the KV-latent's 87% share (measured on the
degenerate UNIFORM baseline) transfer to the low-error CARVE-OUT operating regime?

We take the deployed carve-out (attention/shared at 4-bit, experts at 2-bit; full-set ppl 6.96) and drop
ONLY the two MLA KV-latent tensors (kv_a_proj_with_mqa, kv_b_proj) from 4-bit to 2-bit, leaving everything
else at the carve-out. If ppl jumps sharply, protecting the KV-latent at 4-bit is what the carve-out is
buying -- the 87% finding transfers to the regime that matters. Reuses the carve-out cache (4-bit weights)
for all other tensors and re-quantizes just kv_a/kv_b from the original weights to 2-bit (so ~minutes, not
a 9h full re-quant). Full WikiText-2 test set (NWIN=151), int8 group side-info to match the carve-out.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors.torch import load_file as _sf_load

import weights.evoq_moe as em

CARVE_CFG = "routed-gate/up=2 DOWN_K=3 ATTN_K=4 SHARED_K=4 DENSE_K=4 INT8_GS=True AWQ=False(a=0.5)"
KV = ("kv_a_proj_with_mqa", "kv_b_proj")
FP16, CARVE = 6.3070, 6.9616            # full-set references (151 win)


def run():
    nwin = int(os.environ.get("EVOQ_NWIN", "151"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(nwin)
    cdir = em.cache_dir(CARVE_CFG)
    print(f"M1: carve-out with kv_a/kv_b dropped to 2-bit | cache {cdir} | {nwin} win", flush=True)
    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")    # original fp16 weights
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))   # carve-out 4-bit weights
        for name, mod in layer.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if name.endswith(KV):                                  # re-quantize KV-latent to 2-bit
                W = mod.weight.detach().float().cpu().numpy()       # from the ORIGINAL weight
                wh, _bw = em.trellis_quant(W, K=2)
                mod.weight.data = torch.from_numpy(wh).to(mod.weight.device, torch.float32)
            elif name in cw:                                       # everything else: carve-out as-is
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        h = em._run_layer(layer, h, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 6 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s) hfin={bool(torch.isfinite(h).all())}",
                  flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    line = (f"M1 KV-LATENT-AT-2BIT-IN-CARVEOUT (DeepSeek, full set, {nwin} win): "
            f"carve-out={CARVE:.4f} | carve-out+kv@2bit={ppl:.4f} (delta {ppl-CARVE:+.4f}) | "
            f"fp16={FP16:.4f} -> dropping just the 2 KV-latent tensors to 2-bit costs {ppl-CARVE:+.3f} ppl")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
