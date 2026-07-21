"""Ecosystem-redundancy measurement (the paper's headline systems result).

A real model FAMILY = one base + many independent derivatives (fine-tunes, variants,
re-uploads). We measure how much the family compresses when each derivative is stored as
a lossless delta vs the base, instead of in full or compressed-standalone. If a family of
N real models collapses to ~1 base + N small deltas, the public hub is storing the same
model many times over. Uses the fast multithreaded file codec.

  python -m weights.ecosystem
"""

from __future__ import annotations

import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import wcodec as wc  # noqa: E402

D = os.path.join(_ROOT, "weights", "data")
BASE = os.path.join(D, "qwen", "base.safetensors")
# auto-discover the whole family: every derivative in qwen_family/ + the abliteration
MEMBERS = {os.path.splitext(os.path.basename(p))[0]: p
           for p in sorted(glob.glob(os.path.join(D, "qwen_family", "*.safetensors")))}
MEMBERS["abliterated"] = os.path.join(D, "qwen", "ablit.safetensors")


def raw_size(path):
    _, _, _, doff = wc.parse(path)
    return os.path.getsize(path) - doff  # data section (header is ~KB)


def main():
    avail = {k: p for k, p in MEMBERS.items() if os.path.exists(p) and os.path.getsize(p) > 1e6}
    base_raw = raw_size(BASE)
    base_std = len(wc.compress_file(BASE, level=12))  # standalone (single mode)

    print(f"Ecosystem redundancy: 1 base + {len(avail)} real derivatives "
          f"(Qwen2.5-0.5B-Instruct family)\n")
    print(f"{'derivative':<20}{'raw':>9}{'standalone':>12}{'delta vs base':>15}{'edit':>8}")
    print("-" * 64)
    print(f"{'(base)':<20}{base_raw/1e6:>7.0f}M{base_std/1e6:>11.0f}M{'-':>15}{'-':>8}")

    raw_total = base_raw
    std_total = base_std
    chain_total = base_std  # base standalone; derivatives as deltas
    rows = []
    for k, p in avail.items():
        r = raw_size(p)
        std = len(wc.compress_file(p, level=12))
        delta = len(wc.compress_file(p, BASE, level=12))
        edit = "light" if delta < 0.05 * r else ("medium" if delta < 0.3 * r else "heavy")
        print(f"{k:<20}{r/1e6:>7.0f}M{std/1e6:>11.0f}M{delta/1e6:>13.1f}M{edit:>8}")
        rows.append((k, r, std, delta))
        raw_total += r
        std_total += std
        chain_total += delta

    n = len(avail) + 1
    print("-" * 64)
    print(f"\nFamily of {n} models (base + {len(avail)} derivatives):")
    print(f"  raw (store each in full):      {raw_total/1e6:>8.0f} MB")
    print(f"  each compressed standalone:    {std_total/1e6:>8.0f} MB  (save {(1-std_total/raw_total)*100:4.1f}%)")
    print(f"  base + lossless deltas:        {chain_total/1e6:>8.0f} MB  (save {(1-chain_total/raw_total)*100:4.1f}%)")
    print(f"  -> {n} models stored in {chain_total/base_raw:.2f}x one model's size "
          f"(vs {n}x); {raw_total/chain_total:.1f}x smaller than raw.")

    # light-derivative subset (the common hub case: duplicates/abliterations/preference-tuning)
    light = [r for r in rows if r[3] < 0.15 * r[1]]  # delta < 15% of raw
    if light:
        lraw = base_raw + sum(r[1] for r in light)
        lchain = base_std + sum(r[3] for r in light)
        print(f"\nLight-derivative subset ({len(light)} of {len(avail)}; delta<15%): "
              f"{len(light)+1} models -> {lraw/1e6:.0f}->{lchain/1e6:.0f} MB "
              f"(save {(1-lchain/lraw)*100:.1f}%, {lraw/lchain:.1f}x). "
              f"The common hub case (re-uploads/quants/abliterations/LoRA/preference) lands here.")


if __name__ == "__main__":
    main()
