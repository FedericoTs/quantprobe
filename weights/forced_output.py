"""forced_output.py -- the complement of forced_routing.py: how much of the uniform-2bit collapse is
the RESIDUAL WRITERS (o_proj, down_proj) vs the internal projections (q/kv, gate/up)?

We load the uniform-2bit DeepSeek model but keep ONLY the residual-writer linears (o_proj writes the
attention output; down_proj writes the MLP output) at fp16, leaving everything else at 2-bit. If
perplexity recovers toward fp16, the writers' direct quantization error is the dominant failure -> a
2-tensor "protect the writers" rule should suffice, and we can drop the internal projections to 2-bit
(fewer bits at the same quality). Diagnostic (writers at fp16), not a deployable config (that's 4-bit).
Runs off the uniform cache + shard reads (no re-quant)."""
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

UNIFORM_CFG = "routed-gate/up=2 DOWN_K=2 ATTN_K=2 SHARED_K=2 DENSE_K=2 INT8_GS=True AWQ=False(a=0.5)"
# which linears to keep at fp16 (the rest stay uniform-2bit). Default = residual writers; override via
# EVOQ_KEEP_FP16 to test the complement (internal projections q/kv/gate/up).
WRITERS = tuple(os.environ.get("EVOQ_KEEP_FP16", "o_proj,down_proj").split(","))
UNI = float(os.environ.get("EVOQ_UNI_PPL", "15.3798"))         # baseline ppls for recovery% (default
FP16 = float(os.environ.get("EVOQ_FP16_PPL", "5.6570"))        # = NWIN=8 subset; PASS full-set values
# IMPORTANT: recovery% = (UNI - intervention)/(UNI - FP16). For a full-set (NWIN=151) run, override with
# EVOQ_UNI_PPL=18.3145 EVOQ_FP16_PPL=6.3070, else the printed % mixes a 151-win intervention with the
# 8-win baseline and is WRONG (this caused the stale 77% kv-latent log vs the correct 87%).


def run():
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(8)
    cdir = em.cache_dir(UNIFORM_CFG)
    print(f"writer-fp16 ablation: keep {WRITERS} at fp16 in the uniform-2bit model | cache {cdir}",
          flush=True)
    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")  # all weights fp16 from shards
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))   # uniform-2bit weights
        for name, mod in layer.named_modules():
            if name in cw and not name.endswith(WRITERS):      # overwrite NON-writers with 2-bit;
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)  # writers stay fp16
        h = em._run_layer(layer, h, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 6 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s) hfin={bool(torch.isfinite(h).all())}",
                  flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    rec = 100.0 * (UNI - ppl) / (UNI - FP16)
    line = (f"WRITER-FP16 ABLATION (DeepSeek): uniform-2bit={UNI:.4f} | writers(o_proj,down_proj)@fp16 "
            f"= {ppl:.4f} | fp16={FP16:.4f} -> recovers {rec:.0f}% of the collapse "
            f"(vs forced-routing's 16%)")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
