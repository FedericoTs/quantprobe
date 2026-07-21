"""R25 scale point #3 -- Qwen2.5-3B, CPU-only fp16 eval (6.2GB fits 16GB RAM; avoids the
device_map meta-tensor offload problem). Champion PTQ codecs + naive RTN, held-out enwik8 ppl.
Extends the 0.5B->1.5B gap-halving trend to a third point. Restore between schemes by re-reading
the shards (no giant snapshot).

Run:  python -m weights.quant_scale3b
"""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from weights import codec_zoo
from weights.quant_lab import CALIB_TEXT, EVAL_TEXT, LINEAR

MDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "weights", "data", "qwen3b_base")


def shard_map():
    idx = os.path.join(MDIR, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(MDIR, v) for k, v in wm.items()}
    one = os.path.join(MDIR, "model.safetensors")
    with safe_open(one, framework="pt") as f:
        return {k: one for k in f.keys()}


def qkeys(names):
    return {k for k in names if k.endswith(LINEAR)}


def load_weights(model, smap, only=None):
    """Copy (a subset of) weights from shards into the CPU model, in place."""
    msd = model.state_dict()
    byfile = {}
    for k, fp in smap.items():
        if only is None or k in only:
            byfile.setdefault(fp, []).append(k)
    for fp, keys in byfile.items():
        with safe_open(fp, framework="pt") as f:
            for k in keys:
                if k in msd:
                    msd[k].copy_(f.get_tensor(k).to(msd[k].dtype))
    model.tie_weights()


def ppl(model, tok):
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :1024]
    with torch.no_grad():
        return float(torch.exp(model(ids, labels=ids).loss))


def calibrate(model, tok, qk):
    scales, hooks = {}, []

    def mk(key):
        def h(mod, inp):
            x = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).mean(0)
            scales[key] = x.numpy() if key not in scales else scales[key] + x.numpy()
        return h
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (name + ".weight") in qk:
            hooks.append(mod.register_forward_pre_hook(mk(name + ".weight")))
    ids = tok(CALIB_TEXT, return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return scales


def main():
    tok = AutoTokenizer.from_pretrained(MDIR)
    smap = shard_map()
    qk = qkeys(smap.keys())
    print(f"building Qwen2.5-3B on CPU (fp16, {len(qk)} target tensors) ...", flush=True)
    cfg = AutoConfig.from_pretrained(MDIR)
    torch.set_num_threads(os.cpu_count() or 8)
    model = AutoModelForCausalLM.from_config(cfg).half().eval()
    load_weights(model, smap)
    fp16 = ppl(model, tok)
    print(f"Qwen2.5-3B fp16 held-out ppl = {fp16:.3f}\n", flush=True)
    calib = calibrate(model, tok, qk)

    msd = model.state_dict()

    def quantize_all(q):
        bits_tot, elem = 0.0, 0
        for k in qk:
            a = msd[k].detach().float().numpy()
            wh, bits = q(a, k)
            msd[k].copy_(torch.from_numpy(wh).to(msd[k].dtype))
            bits_tot += bits; elem += a.size
            del a, wh
        model.tie_weights()
        return bits_tot / elem

    def rtn(a, k):
        from weights.quant_smoke import rtn_group
        return rtn_group(a, 3, 128)

    schemes = [
        ("naive RTN 3b", rtn),
        ("ECVQ.008", lambda a, k: codec_zoo.ecvq_mid(a, k, calib)),
        ("ECVQ.005", lambda a, k: codec_zoo.ecvq_005(a, k, calib)),
        ("ECVQ.003", lambda a, k: codec_zoo.ecvq_hi(a, k, calib)),
        ("entropy32", lambda a, k: codec_zoo.entropy32(a, k, calib)),
    ]
    print(f"{'scheme':<16}{'bits/wt':>9}{'ppl':>9}{'gap':>8}")
    print("-" * 42)
    rows = []
    for nm, q in schemes:
        load_weights(model, smap, only=qk)              # restore target weights from shards
        try:
            bpw = quantize_all(q)
            p = ppl(model, tok)
            rows.append((nm, bpw, p))
            print(f"{nm:<16}{bpw:>9.3f}{p:>9.3f}{p-fp16:>8.3f}", flush=True)
        except Exception as e:
            print(f"{nm:<16} FAILED {type(e).__name__}: {e}", flush=True)
        gc.collect()

    print(f"\n=== SCALE TREND (gap-to-fp16 at ~3.13 b/w ECVQ.008) ===")
    print(f"  0.5B: fp16 3.944  ECVQ.008 gap +0.539")
    print(f"  1.5B: fp16 3.128  ECVQ.008 gap +0.345")
    e8 = [r for r in rows if r[0] == "ECVQ.008"]
    if e8:
        print(f"  3.0B: fp16 {fp16:.3f}  ECVQ.008 gap +{e8[0][2]-fp16:.3f}  <== new point")


if __name__ == "__main__":
    main()
