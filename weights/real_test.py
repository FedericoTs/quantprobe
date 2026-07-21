"""Validate the weight-space abliteration detector on REAL HuggingFace models:
huihui-ai abliterated Qwen2.5-Instruct vs the true Qwen base. No synthetic data."""
from __future__ import annotations

import os
import sys

from weights.abliteration_detect import analyze, load_writers, verdict
from weights.abliteration_v3 import subspace_sig

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_models")

PAIRS = [
    ("Qwen2.5-0.5B-Instruct", "qwen05_base.safetensors", "qwen05_ablit.safetensors"),
    ("Qwen2.5-1.5B-Instruct", "qwen15_base.safetensors", "qwen15_ablit.safetensors"),
]


def main():
    for name, bf, af in PAIRS:
        bp, ap = os.path.join(R, bf), os.path.join(R, af)
        if not (os.path.exists(bp) and os.path.exists(ap)):
            print(f"[skip] {name}: files not ready"); continue
        try:
            base = load_writers(bp)
            ablit = load_writers(ap)
        except Exception as e:
            print(f"[skip] {name}: {e}"); continue
        a = analyze(base, ablit)
        score, flag = verdict(a)
        print(f"\n=== {name}  (huihui-ai abliterated vs Qwen base) ===")
        print(f"  writer matrices : {a['n']}")
        print(f"  rank1_energy    : {a['r1']:.3f}")
        print(f"  consistency     : {a['cons']:.3f}")
        print(f"  shared_energy   : {a['shared_energy']:.3f}")
        print(f"  VERDICT         : {flag}")
        del base, ablit


if __name__ == "__main__":
    main()
