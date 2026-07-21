"""ROUND 22 -- SCALE-VALIDATE the PTQ-ECVQ frontier at 1.5B (the quant track was never
scale-tested; the delta track was). Loads Qwen2.5-1.5B-Instruct in fp16 on the GTX 1060
(~3GB, fits 6GB for PTQ eval), reuses the proven codec_zoo ECVQ codecs, and reports held-out
perplexity + honest bits/weight. Hypothesis: the 3-bit gap to fp16 shrinks at larger scale.

Run:  python -m weights.quant_scale
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weights import codec_zoo
from weights.quant_lab import CALIB_TEXT, EVAL_TEXT, LINEAR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "weights", "data", "qwen1.5b")


def shard_map(mdir):
    """name -> safetensors file (handles single-file or sharded index)."""
    idx = os.path.join(mdir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(mdir, v) for k, v in wm.items()}
    one = os.path.join(mdir, "model.safetensors")
    with safe_open(one, framework="pt") as f:
        return {k: one for k in f.keys()}


def quant_keys(names):
    return {k for k in names if k.endswith(LINEAR)}


def build():
    cfg = AutoConfig.from_pretrained(MDIR)
    return AutoModelForCausalLM.from_config(cfg).half().to(DEV).eval()


def load_fp16(model, smap):
    msd = model.state_dict()
    files = {}
    for k, fp in smap.items():
        files.setdefault(fp, []).append(k)
    for fp, keys in files.items():
        with safe_open(fp, framework="pt") as f:
            for k in keys:
                if k in msd:
                    msd[k].copy_(f.get_tensor(k).to(msd[k].dtype).to(DEV))
    model.tie_weights()


def calibrate(model, tok, smap):
    qk = quant_keys(smap.keys())
    scales, hooks = {}, []

    def mk(key):
        def h(mod, inp):
            x = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).mean(0)
            scales[key] = x.cpu().numpy() if key not in scales else scales[key] + x.cpu().numpy()
        return h
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (name + ".weight") in qk:
            hooks.append(mod.register_forward_pre_hook(mk(name + ".weight")))
    ids = tok(CALIB_TEXT, return_tensors="pt").input_ids[:, :512].to(DEV)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return scales


def load_quant(model, smap, quantizer):
    msd = model.state_dict()
    qk = quant_keys(smap.keys())
    files = {}
    for k, fp in smap.items():
        files.setdefault(fp, []).append(k)
    bits_tot, elem = 0.0, 0
    for fp, keys in files.items():
        with safe_open(fp, framework="pt") as f:
            for k in keys:
                if k not in msd:
                    continue
                t = f.get_tensor(k)
                if k in qk:
                    a = t.float().numpy()
                    wh, bits = quantizer(a, k)
                    msd[k].copy_(torch.from_numpy(wh).to(msd[k].dtype).to(DEV))
                    bits_tot += bits; elem += a.size
                else:
                    msd[k].copy_(t.to(msd[k].dtype).to(DEV))
    model.tie_weights()
    return bits_tot / elem


def ppl(model, tok):
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :1024].to(DEV)
    with torch.no_grad():
        return float(torch.exp(model(ids, labels=ids).loss))


def main():
    tok = AutoTokenizer.from_pretrained(MDIR)
    smap = shard_map(MDIR)
    model = build()
    load_fp16(model, smap)
    fp16 = ppl(model, tok)
    print(f"Qwen2.5-1.5B  fp16 held-out ppl = {fp16:.3f}  ({len(quant_keys(smap.keys()))} quantized tensors)\n", flush=True)
    calib = calibrate(model, tok, smap)

    schemes = [
        ("naive RTN 3b", lambda a, k: _rtn(a, 3, 128)),
        ("champ (rot+aware+NF+out)", lambda a, k: codec_zoo.champ(a, k, calib)),
        ("ECVQ lam.008", lambda a, k: codec_zoo.ecvq(a, k, calib, 0.008)),
        ("ECVQ lam.005", lambda a, k: codec_zoo.ecvq(a, k, calib, 0.005)),
        ("ECVQ lam.003", lambda a, k: codec_zoo.ecvq(a, k, calib, 0.003)),
        ("entropy32 (near-lossless)", lambda a, k: codec_zoo.entropy32(a, k, calib)),
    ]
    print(f"{'scheme':<28}{'bits/wt':>9}{'ppl':>10}")
    print("-" * 47)
    rows = []
    for name, q in schemes:
        load_fp16(model, smap)
        try:
            bpw = load_quant(model, smap, q)
            p = ppl(model, tok)
            rows.append((name, bpw, p))
            print(f"{name:<28}{bpw:>9.3f}{p:>10.3f}", flush=True)
        except Exception as e:
            print(f"{name:<28} FAILED {type(e).__name__}: {e}", flush=True)
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n0.5B ref: fp16 3.944 | ECVQ.008 4.483@3.13b | ECVQ.003 4.169@3.91b")
    print(f"1.5B fp16 {fp16:.3f}")
    for n, bpw, p in sorted(rows, key=lambda r: r[2]):
        print(f"  {p:8.3f} @ {bpw:.3f}b   {n}")


def _rtn(a, bits, g):
    from weights.quant_smoke import rtn_group
    return rtn_group(a, bits, g)


if __name__ == "__main__":
    main()
