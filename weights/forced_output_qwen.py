"""forced_output_qwen.py -- the internal-vs-writers ablation on Qwen1.5-MoE: does the finding that the
INTERNAL projections (q/k/v, gate/up) dominate the 2-bit collapse (not the residual writers o/down)
generalize beyond DeepSeek's MLA attention? Keep EVOQ_KEEP_FP16 suffixes at fp16 in the uniform-2bit
Qwen model, measure ppl, off the Qwen uniform cache. Compare:
  EVOQ_KEEP_FP16=o_proj,down_proj                 (writers)
  EVOQ_KEEP_FP16=q_proj,k_proj,v_proj,gate_proj,up_proj   (internal)
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
from weights.evoq_moe_qwen import _eval_setup_qwen, _run_layer_qwen

QWEN_UNIFORM_CFG = "qwen-moe routed-gate/up=2 DOWN_K=2 ATTN_K=2 SHARED_K=2 INT8_GS=True"
KEEP = tuple(os.environ.get("EVOQ_KEEP_FP16", "o_proj,down_proj").split(","))
UNI, FP16 = 11.9372, 6.2492                                    # measured Qwen references (NWIN=8)


def run():
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, posemb, mask, pos = _eval_setup_qwen(8)
    cdir = em.cache_dir(QWEN_UNIFORM_CFG)
    print(f"qwen ablation: keep {KEEP} at fp16 in uniform-2bit | cache {cdir}", flush=True)
    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.cuda()
        cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))
        for name, mod in layer.named_modules():
            if name in cw and not name.endswith(KEEP):
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        h = _run_layer_qwen(layer, h, posemb, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 6 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s) hfin={bool(torch.isfinite(h).all())}",
                  flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    rec = 100.0 * (UNI - ppl) / (UNI - FP16)
    line = (f"QWEN ABLATION keep-fp16={'+'.join(KEEP)}: uniform={UNI:.4f} | this={ppl:.4f} | "
            f"fp16={FP16:.4f} -> recovers {rec:.0f}% of the collapse")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
