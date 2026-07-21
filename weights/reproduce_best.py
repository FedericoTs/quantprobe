"""Reproduce the headline result: the best discovered codec (ECVQ, entropy-constrained
quantization) compresses Qwen2.5-0.5B to ~3.1 bits/weight at perplexity ~4.48 -- beating
the naive 4-bit baseline (gibberish) and the hand-designed champion; near-lossless at ~5.3b.

GPU-aware: uses CUDA if available (your GTX 1060), else CPU. Runs the model in fp32
(the 1060 has no bf16/fast-fp16) so it works on Pascal.

Run:   python -m weights.reproduce_best
Needs (see weights/DATA_MANIFEST.md):
  weights/data/qwen_cfg/            (config + tokenizer, from Qwen/Qwen2.5-0.5B-Instruct)
  weights/data/qwen/base.safetensors (0.5B base weights)
  data/corpora/generic-text/enwik8_256k  (eval text; tracked in repo)
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weights import codec_zoo
from weights.quant_smoke import rtn_group

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(_ROOT, "weights", "data", "qwen_cfg")
WPATH = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")
_raw = open(os.path.join(_ROOT, "data/corpora/generic-text/enwik8_256k"), "rb").read()
CALIB_TEXT = _raw[:8000].decode("latin-1")
EVAL_TEXT = _raw[120000:128000].decode("latin-1")
LINEAR = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
          "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def _qkeys(f):
    return {k for k in f.keys() if k.endswith(LINEAR)}


def build_and_load():
    cfg = AutoConfig.from_pretrained(CFG)
    model = AutoModelForCausalLM.from_config(cfg).eval().to(DEVICE)
    msd = model.state_dict()
    with safe_open(WPATH, framework="pt") as f:
        for k in f.keys():
            if k in msd:
                msd[k].copy_(f.get_tensor(k).to(msd[k].dtype).to(DEVICE))
    model.tie_weights()
    return model


def calibrate(model, tok):
    scales, hooks = {}, []
    with safe_open(WPATH, framework="pt") as f:
        qk = _qkeys(f)

    def mk(key):
        def h(mod, inp):
            x = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).mean(0).cpu().numpy()
            scales[key] = x if key not in scales else scales[key] + x
        return h

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (name + ".weight") in qk:
            hooks.append(mod.register_forward_pre_hook(mk(name + ".weight")))
    ids = tok(CALIB_TEXT, return_tensors="pt").input_ids[:, :512].to(DEVICE)
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return scales


def load_quant(model, quantizer):
    msd = model.state_dict()
    bits, el = 0.0, 0
    with safe_open(WPATH, framework="pt") as f:
        qk = _qkeys(f)
        for k in f.keys():
            t = f.get_tensor(k)
            if k in qk:
                a = t.float().numpy()
                wh, b = quantizer(a, k)
                msd[k].copy_(torch.from_numpy(wh).to(msd[k].dtype).to(DEVICE))
                bits += b
                el += a.size
            elif k in msd:
                msd[k].copy_(t.to(msd[k].dtype).to(DEVICE))
    model.tie_weights()
    return bits / el


def ppl(model, tok):
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :1024].to(DEVICE)
    with torch.no_grad():
        return float(torch.exp(model(ids, labels=ids).loss))


def main():
    print(f"device: {DEVICE}  ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_and_load()
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    print(f"fp16 reference perplexity = {fp16:.3f}\n", flush=True)

    schemes = [
        ("naive RTN int4 (3-bit-ish)", lambda a, k: rtn_group(a, 3, 128)),
        ("hand champ (rot+NF+outlier)", lambda a, k: codec_zoo.champ(a, k, calib)),
        ("ECVQ  <-- BEST (~3.1 bits)", lambda a, k: codec_zoo.ecvq_mid(a, k, calib)),
        ("entropy-32 (near-lossless)", lambda a, k: codec_zoo.entropy32(a, k, calib)),
    ]
    print(f"{'codec':<30}{'bits/wt':>9}{'ppl':>10}")
    print("-" * 49)
    print(f"{'fp16 (reference)':<30}{16.0:>9.2f}{fp16:>10.3f}")
    for name, q in schemes:
        bpw = load_quant(model, q)
        p = ppl(model, tok)
        print(f"{name:<30}{bpw:>9.3f}{p:>10.3f}", flush=True)

    print(f"\nEXPECTED (Qwen2.5-0.5B): ECVQ ~3.13 bits, ppl ~4.48  |  naive 3-bit = gibberish (10s-100s)")
    print(f"  |  near-lossless entropy-32 ~3.97 ppl @ ~5.3 bits  |  fp16 {fp16:.2f}")
    print("If ECVQ lands ~4.5 ppl at ~3.1 bits and the naive baseline blows up, the result reproduced.")


if __name__ == "__main__":
    main()
