"""m1_param_control.py -- parameter-matched control for the KV-latent thesis (red-team R4).

Concern: "the KV-latent is the bottleneck" might just be "small/low-rank tensors are fragile."
Control: in the deployed carve-out, drop a KV-PARAM-SIZED row-slice of q_proj (a full-rank attention
internal in the SAME block) to 2-bit, and compare its ppl cost to the +5.27 from dropping the actual
KV-latent (kv_a+kv_b) to 2-bit. If the q-slice costs << +5.27 from the same parameter count, the
KV-latent is special, not merely small. Reuses the carve-out cache (4-bit) for everything else.
Full WikiText-2 test set (NWIN=151), int8 group side-info to match the carve-out.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors.torch import load_file as _sf_load

import weights.evoq_moe as em

CARVE_CFG = "routed-gate/up=2 DOWN_K=3 ATTN_K=4 SHARED_K=4 DENSE_K=4 INT8_GS=True AWQ=False(a=0.5)"
FP16, CARVE, KV_DROP = 6.3070, 6.9616, 12.2320          # full-set references (151 win)


def run():
    nwin = int(os.environ.get("EVOQ_NWIN", "151"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(nwin)
    cdir = em.cache_dir(CARVE_CFG)
    print(f"PARAM-MATCHED CONTROL: KV-sized slice of q_proj -> 2-bit | cache {cdir} | {nwin} win", flush=True)
    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))
        attn = layer.self_attn
        kv_params = attn.kv_a_proj_with_mqa.weight.numel() + attn.kv_b_proj.weight.numel()
        for name, mod in layer.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if name.endswith("q_proj"):
                W = mod.weight.detach().float().cpu().numpy()           # original q_proj [out, in]
                in_dim = W.shape[1]
                n = min(W.shape[0], max(1, round(kv_params / in_dim)))  # rows matching KV param count
                wh2, _bw = em.trellis_quant(W[:n], K=2)                 # fresh 2-bit slice
                qc = cw[name].to(torch.float32)                        # cached 4-bit q_proj (whole)
                qc[:n] = torch.from_numpy(wh2).to(qc.dtype)            # overwrite top n rows with 2-bit
                mod.weight.data = qc.to(mod.weight.device)
                if li == 0:
                    print(f"  q_proj [{W.shape[0]},{in_dim}]: first {n} rows @2bit "
                          f"(~{n*in_dim/1e6:.2f}M params = KV {kv_params/1e6:.2f}M); rest @4bit", flush=True)
            elif name in cw:
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        h = em._run_layer(layer, h, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 6 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s) hfin={bool(torch.isfinite(h).all())}", flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    line = (f"PARAM-MATCHED CONTROL (q_proj KV-sized slice @2bit, full set, {nwin} win): carve-out={CARVE:.4f} | "
            f"q-slice@2bit={ppl:.4f} (delta {ppl-CARVE:+.4f}) | KV-latent@2bit={KV_DROP:.4f} (+{KV_DROP-CARVE:.2f}) | "
            f"fp16={FP16:.4f} -> same param count, q-slice costs {ppl-CARVE:+.3f} vs KV-latent +{KV_DROP-CARVE:.2f}")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
