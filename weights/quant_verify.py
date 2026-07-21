"""Robustness check: re-evaluate the top codecs on a SECOND, disjoint held-out text
to confirm the frontier ranking (scalar ECVQ dominates lattices) is not overfit to
one eval slice. Sanity guard after catching the lattice entropy-accounting bug.
"""
from __future__ import annotations

import gc

import torch
from transformers import AutoTokenizer

from weights import codec_zoo
from weights.quant_lab import CFG, _raw, build_model, calibrate, load_fp16, load_quant

EVAL2 = _raw[200000:208000].decode("latin-1")   # different disjoint slice

CODECS = {
    "champ": codec_zoo.champ,
    "ECVQ.008": codec_zoo.ecvq_mid,
    "ECVQ.003": codec_zoo.ecvq_hi,
    "entropy16": codec_zoo.entropy16,
    "entropy32": codec_zoo.entropy32,
    "E8 q.14": codec_zoo.e8_q14,
    "E8 q.08": codec_zoo.e8_q08,
}


def ppl2(model, tok):
    ids = tok(EVAL2, return_tensors="pt").input_ids[:, :1024]
    with torch.no_grad():
        return float(torch.exp(model(ids, labels=ids).loss))


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    calib = calibrate(model, tok)
    f = ppl2(model, tok)
    print(f"fp16 (2nd held-out text) ppl = {f:.3f}\n", flush=True)
    print(f"{'codec':<12}{'bits':>8}{'ppl':>9}")
    print("-" * 30)
    rows = []
    for name, fn in CODECS.items():
        try:
            bpw = load_quant(model, lambda a, k: fn(a, k, calib))
            p = ppl2(model, tok)
            rows.append((name, bpw, p))
            print(f"{name:<12}{bpw:>8.3f}{p:>9.3f}", flush=True)
        except Exception as e:
            print(f"{name:<12} FAILED {type(e).__name__}", flush=True)
        gc.collect()

    print("\nRanking check (does scalar ECVQ still dominate E8 at equal quality?):")
    e8 = [r for r in rows if r[0] == "E8 q.14"]
    ev = [r for r in rows if r[0] == "ECVQ.003"]
    if e8 and ev:
        print(f"  ECVQ.003 {ev[0][2]:.3f} ppl @ {ev[0][1]:.2f}b  vs  E8 q.14 {e8[0][2]:.3f} ppl @ {e8[0][1]:.2f}b")
        print("  => " + ("ECVQ dominates (fewer bits, <=ppl) -- conclusion holds"
                         if ev[0][1] < e8[0][1] and ev[0][2] <= e8[0][2] + 0.05
                         else "ranking shifted -- re-examine"))


if __name__ == "__main__":
    main()
