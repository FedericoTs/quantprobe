"""R25v2-R6-lite -- settle the lattice near-lossless question by MEASURING (not extrapolating)
champion-family points at ppl ~= 4.02 (E8 q0.08's ppl) and comparing HONEST bits directly.

Honest bits = arena bits - A1 re-coding bank (raw side 0.2876 -> real container 0.1608 = -0.1268).
E8 q0.08 honest TOTAL = 4.640 (coords real-coded + amax 0.0792; +0.011 AWQ raw if symmetric ->
4.651). R1 oracle says champion indices are iid (order-0 = true rate), so this comparison is now
fully symmetric. If a champion point at ppl <= 4.02 costs <= 4.64 b/w, lattices STAY CLOSED.

Run:  python -m weights.champ_at_402
"""
from __future__ import annotations

import gc
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights import codec_zoo
from weights.quant_lab import CFG, build_model, calibrate, load_fp16, load_quant, ppl

A1_BANK = 0.1268     # raw 0.2876 -> real container 0.1608 (round-trip verified)
E8_REF = (4.020, 4.640)


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    print(f"fp16 {fp16:.3f} | target: beat E8 q0.08 honest {E8_REF[1]:.3f} b/w @ ppl {E8_REF[0]:.3f}\n")

    schemes = [
        ("entropy-20lev", lambda a, k: codec_zoo.entropy_q(a, k, calib, 20)),
        ("entropy-24lev", lambda a, k: codec_zoo.entropy_q(a, k, calib, 24)),
        ("ECVQ lam.002",  lambda a, k: codec_zoo.ecvq(a, k, calib, 0.002)),
        ("ECVQ lam.0015", lambda a, k: codec_zoo.ecvq(a, k, calib, 0.0015)),
    ]
    print(f"{'scheme':<16}{'arena b/w':>10}{'honest b/w':>11}{'ppl':>8}   verdict")
    print("-" * 60)
    for name, q in schemes:
        load_fp16(model)
        bpw = load_quant(model, q)
        p = ppl(model, tok)
        honest = bpw - A1_BANK
        verdict = ""
        if p <= E8_REF[0] + 0.005:
            verdict = "BEATS E8 -> lattices CLOSED" if honest <= E8_REF[1] else "E8 WINS here"
        print(f"{name:<16}{bpw:>10.3f}{honest:>11.3f}{p:>8.3f}   {verdict}", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
