"""Consolidated quantization lab toward the paper "LLM-Evolved Quantization Codecs".

Adds the credibility piece the earlier scripts lacked: CALIBRATION + an ACTIVATION-AWARE
(AWQ-style) baseline -- the real SOTA-class anchor reviewers demand -- and the EVOLVED
combo (AWQ x incoherence x NF x outliers) to test whether a discovered combination
Pareto-dominates every individual hand-designed method.

Verifier (cheap, CPU): held-out perplexity (eval text disjoint from calibration text)
+ average bits/weight. Memory-safe: one reused model, weights streamed per scheme.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weights.quant_sota import hadamard_nf_outlier, nf_group
from weights.quant_smoke import rtn_group

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(_ROOT, "weights", "data", "qwen_cfg")
WPATH = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")
torch.set_num_threads(os.cpu_count() or 4)

LINEAR = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
          "gate_proj.weight", "up_proj.weight", "down_proj.weight")

_raw = open(os.path.join(_ROOT, "data/corpora/generic-text/enwik8_256k"), "rb").read()
CALIB_TEXT = _raw[:8000].decode("latin-1")          # calibration
EVAL_TEXT = _raw[120000:128000].decode("latin-1")   # held-out (disjoint)


def quant_keys(f):
    return {k for k in f.keys() if k.endswith(LINEAR)}


def build_model():
    cfg = AutoConfig.from_pretrained(CFG)
    return AutoModelForCausalLM.from_config(cfg).eval()


def load_fp16(model):
    msd = model.state_dict()
    with safe_open(WPATH, framework="pt") as f:
        for k in f.keys():
            if k in msd:
                msd[k].copy_(f.get_tensor(k).to(msd[k].dtype))
    model.tie_weights()


def calibrate(model, tok):
    scales, hooks = {}, []
    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)

    def mk(key):
        def h(mod, inp):
            x = inp[0].detach().abs().float().reshape(-1, inp[0].shape[-1]).mean(0)
            scales[key] = x if key not in scales else scales[key] + x
        return h

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and (name + ".weight") in qk:
            hooks.append(mod.register_forward_pre_hook(mk(name + ".weight")))
    ids = tok(CALIB_TEXT, return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return {k: v.numpy() for k, v in scales.items()}


def load_quant(model, quantizer):
    msd = model.state_dict()
    bits_tot, elem = 0.0, 0
    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
        for k in f.keys():
            t = f.get_tensor(k)
            if k in qk:
                a = t.float().numpy()
                wh, bits = quantizer(a, k)
                msd[k].copy_(torch.from_numpy(wh).to(msd[k].dtype))
                bits_tot += bits
                elem += a.size
            elif k in msd:
                msd[k].copy_(t.to(msd[k].dtype))
    model.tie_weights()
    return bits_tot / elem


def ppl(model, tok):
    ids = tok(EVAL_TEXT, return_tensors="pt").input_ids[:, :1024]
    with torch.no_grad():
        return float(torch.exp(model(ids, labels=ids).loss))


# ---- activation-aware (AWQ-style) scaling ----
def _awq_scale(calib, key, alpha=0.5):
    s = calib[key]
    s = s / (s.mean() + 1e-9)
    return np.clip(s, 1e-2, 1e2) ** alpha


def awq(a, key, calib, bits, g):
    s = _awq_scale(calib, key)
    wh, bq = rtn_group(a * s[None, :], bits, g)
    return (wh / s[None, :]).astype(np.float32), bq + 16 * len(s)


def awq_hadnf(a, key, calib, bits, g, p=0.005):
    s = _awq_scale(calib, key)
    wh, bq = hadamard_nf_outlier(a * s[None, :], bits, g, p)
    return (wh / s[None, :]).astype(np.float32), bq + 16 * len(s)


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    print("calibrating ...", flush=True)
    calib = calibrate(model, tok)
    fp16_ppl = ppl(model, tok)
    print(f"fp16 held-out ppl = {fp16_ppl:.3f}\n", flush=True)

    schemes = []
    for b in (4, 3):
        schemes += [
            (f"RTN-group {b}b", lambda a, k, b=b: rtn_group(a, b, 128)),
            (f"NF-group {b}b", lambda a, k, b=b: nf_group(a, b, 128)),
            (f"AWQ {b}b (calib SOTA-class)", lambda a, k, b=b: awq(a, k, calib, b, 128)),
            (f"Had+NF+outlier {b}b", lambda a, k, b=b: hadamard_nf_outlier(a, b, 128, 0.005)),
            (f"EVOLVED AWQ+Had+NF+out {b}b", lambda a, k, b=b: awq_hadnf(a, k, calib, b, 128, 0.005)),
        ]

    print(f"{'scheme':<32}{'bits/wt':>9}{'ppl':>10}")
    print("-" * 52)
    rows = []
    for name, q in schemes:
        bpw = load_quant(model, q)
        p = ppl(model, tok)
        rows.append((name, bpw, p))
        print(f"{name:<32}{bpw:>9.3f}{p:>10.3f}", flush=True)
        gc.collect()

    print(f"\nfp16 ref = {fp16_ppl:.3f}")
    for b in ("4b", "3b"):
        grp = [r for r in rows if r[0].endswith(b)]
        awq_r = [r for r in grp if r[0].startswith("AWQ")][0]
        evo_r = [r for r in grp if r[0].startswith("EVOLVED")][0]
        best = min(grp, key=lambda r: r[2])
        print(f"  {b}: AWQ(SOTA-class) {awq_r[2]:.3f}  |  EVOLVED combo {evo_r[2]:.3f}  |  "
              f"best = {best[0].split()[0]} {best[2]:.3f}")
        if evo_r[2] < awq_r[2]:
            print(f"       => EVOLVED combo beats the AWQ SOTA-class baseline at {b}")


if __name__ == "__main__":
    main()
