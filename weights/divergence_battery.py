"""R25-C1 -- DIVERGENCE BATTERY retro-audit. 24 rounds hill-climbed ONE enwik8 ppl slice.
This asks: does that ranking survive a multi-domain KL lens? For each frontier codec we measure
KL(fp16 || quant) of next-token distributions across diverse-domain probes (wiki, code, math,
dialogue, JSON, multilingual) + mean/worst, alongside the enwik8 held-out ppl. A codec that wins
on enwik8-ppl but has high worst-domain KL is a verifier-overfit artifact.

Anti-gaming: fp16 reference distributions cached once on FIXED probes; KL is a property of the
output distribution (not a rate the codec can fake); models loaded serially.

Run:  python -m weights.divergence_battery
"""
from __future__ import annotations

import gc
import sys

import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights import codec_zoo
from weights.quant_lab import (CFG, build_model, calibrate, load_fp16, load_quant, ppl)

# Diverse-domain probes (held-out in spirit; none used for calibration which is enwik8[:8000])
PROBES = {
    "wiki": "The Treaty of Westphalia, signed in 1648, ended the Thirty Years' War and established "
            "the principle of state sovereignty that underpins the modern international system.",
    "code": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n"
            "    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
    "math": "Theorem: for any integer n, n^2 is even if and only if n is even. Proof: if n = 2k then "
            "n^2 = 4k^2 = 2(2k^2), which is even. Conversely, if n is odd, n = 2k+1, so n^2 = 4k^2+4k+1, odd.",
    "dialogue": "User: Can you explain why the sky is blue?\nAssistant: The sky appears blue because "
                "molecules in the air scatter shorter blue wavelengths of sunlight more than longer red ones.",
    "json": '{"model": "qwen2.5", "params": 494032768, "layers": 24, "quantization": {"bits": 3.13, '
            '"method": "ecvq", "lossless": false}, "perplexity": 4.483}',
    "multiling": "La compression des poids de réseaux neuronaux permet de réduire la taille des "
                 "modèles. Die Quantisierung ist ein wichtiger Schritt. 量子化は重要な技術です。",
}


def dists(model, tok, text, dev="cpu"):
    ids = tok(text, return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        logits = model(ids).logits[0].float()       # [T, V]
    return F.log_softmax(logits, dim=-1)


def kl_from_ref(ref_logp, q_logp):
    """KL(P_fp16 || Q_quant) averaged over positions, in nats."""
    p = ref_logp.exp()
    return float((p * (ref_logp - q_logp)).sum(-1).mean())


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    calib = calibrate(model, tok)
    fp16_ppl = ppl(model, tok)
    ref = {name: dists(model, tok, t) for name, t in PROBES.items()}
    print(f"fp16 enwik8 ppl {fp16_ppl:.4f}; reference distributions cached on {len(PROBES)} probes\n", flush=True)

    codecs = [
        ("champ", lambda a, k: codec_zoo.champ(a, k, calib)),
        ("ECVQ.008", lambda a, k: codec_zoo.ecvq_mid(a, k, calib)),
        ("ECVQ.005", lambda a, k: codec_zoo.ecvq_005(a, k, calib)),
        ("ECVQ.003", lambda a, k: codec_zoo.ecvq_hi(a, k, calib)),
        ("entropy16", lambda a, k: codec_zoo.entropy16(a, k, calib)),
        ("entropy32", lambda a, k: codec_zoo.entropy32(a, k, calib)),
        ("ECVQ.020(aggr)", lambda a, k: codec_zoo.ecvq_lo(a, k, calib)),
        ("E8 q.22", lambda a, k: codec_zoo.e8_q22(a, k, calib)),
    ]
    dom = list(PROBES.keys())
    print(f"{'codec':<16}{'ppl':>8}" + "".join(f"{d[:5]:>7}" for d in dom) + f"{'meanKL':>8}{'maxKL':>8}")
    print("-" * (24 + 7 * len(dom) + 16))
    rows = []
    for name, q in codecs:
        load_fp16(model)
        try:
            bpw = load_quant(model, q)
            p = ppl(model, tok)
            kls = {d: kl_from_ref(ref[d], dists(model, tok, PROBES[d])) for d in dom}
            mean_kl = float(np.mean(list(kls.values())))
            max_kl = float(np.max(list(kls.values())))
            rows.append((name, p, bpw, kls, mean_kl, max_kl))
            print(f"{name:<16}{p:>8.3f}" + "".join(f"{kls[d]:>7.3f}" for d in dom) +
                  f"{mean_kl:>8.3f}{max_kl:>8.3f}", flush=True)
        except Exception as e:
            print(f"{name:<16} FAILED {type(e).__name__}: {e}", flush=True)
        gc.collect()

    print("\n=== ranking cross-check (does enwik8-ppl agree with divergence?) ===")
    by_ppl = sorted(rows, key=lambda r: r[1])
    by_mkl = sorted(rows, key=lambda r: r[4])
    by_xkl = sorted(rows, key=lambda r: r[5])
    print("  by enwik8 ppl :", " > ".join(r[0] for r in by_ppl))
    print("  by mean KL    :", " > ".join(r[0] for r in by_mkl))
    print("  by worst KL   :", " > ".join(r[0] for r in by_xkl))
    # Spearman-ish: rank disagreement of worst-KL vs ppl
    order_ppl = {r[0]: i for i, r in enumerate(by_ppl)}
    order_xkl = {r[0]: i for i, r in enumerate(by_xkl)}
    d2 = sum((order_ppl[n] - order_xkl[n]) ** 2 for n in order_ppl)
    nn = len(rows)
    rho = 1 - 6 * d2 / (nn * (nn * nn - 1)) if nn > 2 else float('nan')
    print(f"\n  Spearman(ppl-rank, worstKL-rank) = {rho:.3f}  "
          f"(<0.9 => worst-domain divergence carries info the ppl slice misses => battery justified)")


if __name__ == "__main__":
    main()
