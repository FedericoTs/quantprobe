"""evoq builder + runtime gate for Qwen2.5-0.5B.

encode: calibrate (quant_lab), encode all 168 target linears with the champion codec
        (AWQ alpha=.5 + rotation + ECVQ lam=.008 + 0.5% outliers), save .evoq container.
run:    build the bf16 model, swap target Linears -> QuantLinear (compute bf16 to mirror the
        arena's bf16 path), load container, eval held-out ppl.
GATE:   |ppl - 4.6302| < 0.02  (the arena champion @ seed 0 measured in amax_snap).

Usage:  python -m weights.evoq_build encode|run|both
"""
from __future__ import annotations

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
from transformers import AutoTokenizer
from weights.evoq import QuantLinear, encode_tensor, load_container, save_container
from weights.quant_lab import (CFG, WPATH, _awq_scale, build_model, calibrate,
                               load_fp16, ppl, quant_keys)

CONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qwen05b.evoq")
GATE_PPL = 4.6302


def encode():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    calib = calibrate(model, tok)
    del model
    tensors = {}
    t0 = time.time()
    with safe_open(WPATH, framework="pt") as f:
        keys = sorted(quant_keys(f))
        for i, k in enumerate(keys):
            W = f.get_tensor(k).float().numpy()
            s = _awq_scale(calib, k, 0.5).astype(np.float32)
            tensors[k] = encode_tensor(W, s, self_check=(i % 40 == 0))
            if i % 40 == 0:
                print(f"  [{i+1}/{len(keys)}] {k}  {W.shape}  ({time.time()-t0:.0f}s)", flush=True)
    save_container(CONT, tensors, dict(model="Qwen2.5-0.5B", lam=0.008, seed=0))
    sz = os.path.getsize(CONT) / 1e6
    nW = sum(c["rows"] * c["cols"] for c in tensors.values())
    print(f"container: {CONT}  {sz:.0f} MB  ({sz*8e6/nW:.2f} bits/weight resident)", flush=True)


def swap_in_quantlinears(model, comps, compute_dtype):
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}.weight" if name else f"{child_name}.weight"
            if isinstance(child, nn.Linear) and full in comps:
                ql = QuantLinear(comps[full], child.bias.detach().clone() if child.bias is not None else None,
                                 compute_dtype=compute_dtype)
                setattr(mod, child_name, ql)
                n += 1
    return n


def run(compute_dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)                      # loads everything incl. non-quantized tensors + biases
    if compute_dtype == torch.float32:
        model = model.float()             # whole skeleton fp32 (mirrors the 7B GPU path)
    meta, comps = load_container(CONT)
    n = swap_in_quantlinears(model, comps, compute_dtype=compute_dtype)
    del comps                              # buffers hold their own refs; drop the dict
    print(f"swapped {n} Linears -> QuantLinear ({compute_dtype}, packed6 resident)", flush=True)
    t0 = time.time()
    p = ppl(model, tok)
    dt = time.time() - t0
    print(f"\nevoq runtime held-out ppl = {p:.4f}   (gate: arena champion {GATE_PPL} +/- 0.02; "
          f"eval {dt:.0f}s)")
    ok = abs(p - GATE_PPL) < 0.02
    print("GATE PASSED -- runtime reproduces the arena champion exactly" if ok else
          f"GATE FAILED (delta {p-GATE_PPL:+.4f}) -- debug before 7B")
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("encode", "both"):
        encode()
    if mode in ("run", "both"):
        run(torch.bfloat16)
    if mode == "runfp32":
        run(torch.float32)   # record the fp32-compute delta (GPU path uses fp32 on sm_61)
