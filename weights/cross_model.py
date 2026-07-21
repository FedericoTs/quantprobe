"""Can one foundation model derive another, losslessly? (the boundary of the technique)

Test two INDEPENDENTLY-trained, same-architecture foundation generations: Qwen2-0.5B vs
Qwen2.5-0.5B (retrained from scratch, different data/recipe). If they are independent, the
cross-generation delta gives NO advantage over storing the model standalone (both ~33%) --
in contrast to a true descendant (abliteration), where the delta (99%) >> standalone (33%).
This empirically maps where lossless cross-derivation is possible (lineage) vs impossible
(independent foundations).
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import wcodec as wc  # noqa: E402

D = os.path.join(_ROOT, "weights", "data")
QWEN25 = os.path.join(D, "qwen", "base.safetensors")          # Qwen2.5-0.5B-Instruct
QWEN2 = os.path.join(D, "qwen_gen", "qwen2-0.5b.safetensors")  # Qwen2-0.5B-Instruct (prev gen)
ABLIT = os.path.join(D, "qwen", "ablit.safetensors")          # a true descendant of Qwen2.5


def save_pct(path, ref=None):
    raw = os.path.getsize(path)
    comp = len(wc.compress_file(path, ref, level=12))
    return (1 - comp / raw) * 100


def main():
    print("Lossless cross-derivation test (Qwen 0.5B)\n")
    std = save_pct(QWEN25)
    print(f"  Qwen2.5 standalone (no reference):              {std:5.1f}% saved")
    cross = save_pct(QWEN25, QWEN2)
    print(f"  Qwen2.5 as a delta vs Qwen2 (independent gen):  {cross:5.1f}% saved")
    desc = save_pct(ABLIT, QWEN25)
    print(f"  abliteration as a delta vs Qwen2.5 (descendant):{desc:5.1f}% saved")
    print()
    benefit = cross - std
    print(f"  cross-generation delta benefit over standalone: {benefit:+.1f} pts")
    if benefit < 3:
        print("  => the shared 'base' buys ~NOTHING: independent foundations do NOT cross-derive")
        print("     losslessly (entropy/independence wall). Only LINEAGE-shared models compress.")
    else:
        print("  => unexpected: the generations share exploitable structure.")


if __name__ == "__main__":
    main()
