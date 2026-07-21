"""Campaign-3: how FEW outliers can we keep? The runtime outlier sidecar costs ~16ms/token
(random x[col] gather over 0.5%% of weights) -- the hidden tax that separates the pure-GEMV
projection (26.6 tok/s) from the true kernel ceiling (15.3). The FWHT rotation already spreads
outliers, so 0.5%% may be conservative. Sweep p in {0.5,0.25,0.1,0.05,0}%% -> held-out ppl + the
runtime gather cost (proportional to p). Pick the smallest p within the ppl noise floor (~0.03).

Run:  python -m weights.pout_sweep
"""
from __future__ import annotations

import gc
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights.quant_lab import CFG, build_model, calibrate, load_fp16, load_quant, ppl
from weights import codec_zoo

G = 128
LAM = 0.008


def ecvq_p(a, key, calib, p):
    Ws, s = codec_zoo._act(a, key, calib)
    wh, b = codec_zoo._had_ecvq(Ws, LAM, G, p)
    # honest bits: index entropy already in b; outlier side = p*(int32 pos + fp16 val)=p*48 per weight... b includes it
    return (wh / s[None, :]).astype(np.float32), b + 16 * a.shape[1]


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model(); load_fp16(model)
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    print(f"fp16 held-out ppl = {fp16:.4f}\n", flush=True)
    print(f"{'outlier p':>10}{'bits/wt':>9}{'ppl':>9}{'d_ppl':>8}{'gather x':>9}")
    print("-" * 46)
    p0 = None
    for p in (0.005, 0.0025, 0.001, 0.0005, 0.0):
        load_fp16(model)
        b = load_quant(model, lambda a, k: ecvq_p(a, k, calib, p))
        pp = ppl(model, tok)
        if p0 is None:
            p0 = pp
        rel = p / 0.005
        tag = "  <- champion" if p == 0.005 else (f"  d={pp-p0:+.3f}" if pp - p0 < 0.03 else "  +ppl")
        print(f"{p*100:>9.2f}%{b:>9.3f}{pp:>9.4f}{pp-p0:>+8.3f}{rel:>8.2f}x{tag}", flush=True)
        gc.collect()
    print(f"\nGoal: smallest p with d_ppl < ~0.03 (noise floor). 'gather x' = runtime outlier "
          f"cost multiple vs champion (0.5%%). Each halving ~ -8ms/token at 7B.")


if __name__ == "__main__":
    main()
