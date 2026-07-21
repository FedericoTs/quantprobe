"""R25-A2 deliverable -- held-out ppl NOISE FLOOR. Many roadmap kill criteria use 0.02-0.03 ppl
margins; this measures whether such margins are decidable. Quantization is deterministic, so the
real verifier noise is the variance of held-out ppl across the CHOICE of held-out slice. We eval
fp16 and the ECVQ champion on K disjoint enwik8 slices and report mean/std of ppl AND of the
quant-vs-fp16 GAP (the quantity codec comparisons actually rank on).

Run:  python -m weights.noise_floor
"""
from __future__ import annotations

import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights import codec_zoo
from weights.quant_lab import (CFG, build_model, calibrate, load_fp16, load_quant)

# CPU-native (quant_lab.calibrate uses .numpy() on activations; codec_zoo is numpy)
_RAW = open("data/corpora/generic-text/enwik8_256k", "rb").read()
# K disjoint held-out windows, all past the calibration region (calib uses [:8000])
OFFSETS = [120000, 134000, 148000, 162000, 176000, 190000, 204000, 218000]
WIN = 8000


def ppl_on(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    with torch.no_grad():
        return float(torch.exp(model(ids, labels=ids).loss))


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    slices = [_RAW[o:o + WIN].decode("latin-1") for o in OFFSETS]
    calib = calibrate(model, tok)

    fp16 = np.array([ppl_on(model, tok, s) for s in slices])
    # champion ECVQ.008 quantize once, eval on all slices
    load_quant(model, lambda a, k: codec_zoo.ecvq_mid(a, k, calib))
    q = np.array([ppl_on(model, tok, s) for s in slices])
    gap = q - fp16

    print(f"K={len(slices)} disjoint held-out slices (WIN={WIN}B, 1024 tok each)\n")
    print(f"{'slice@offset':>14}{'fp16 ppl':>11}{'ECVQ ppl':>11}{'gap':>9}")
    for o, f, qq, g in zip(OFFSETS, fp16, q, gap):
        print(f"{o:>14}{f:>11.4f}{qq:>11.4f}{g:>9.4f}")
    print(f"\nfp16 ppl   : mean {fp16.mean():.4f}  std {fp16.std():.4f}  range {fp16.max()-fp16.min():.4f}")
    print(f"ECVQ ppl   : mean {q.mean():.4f}  std {q.std():.4f}  range {q.max()-q.min():.4f}")
    print(f"GAP (q-fp16): mean {gap.mean():.4f}  std {gap.std():.4f}  range {gap.max()-gap.min():.4f}")
    print(f"\n=> single-slice ppl noise ~ +/-{q.std():.3f}; the gap (what codecs rank on) noise ~ "
          f"+/-{gap.std():.3f}.")
    print(f"   Kill margins below ~{2*gap.std():.3f} ppl on ONE slice are NOT decidable -- "
          f"need multi-slice mean or paired comparison.")
    # paired SE of the mean gap if we averaged all K slices:
    se = gap.std(ddof=1) / np.sqrt(len(slices))
    print(f"   Averaging all {len(slices)} slices: gap SE ~ +/-{se:.4f} ppl (decidable margin ~{2*se:.3f}).")


if __name__ == "__main__":
    main()
