"""A2-day-0 BINDING GATE -- exact pair-tANS achievable rate from REAL index histograms.

Pair-decoding halves the per-token symbol throughput the GPU decoder must sustain (the A1 gate
arithmetic), but a 64x64=4096-symbol alphabet quantized into an L-slot tANS table loses rate.
This computes the EXACT average rate (cross-entropy vs the slot-quantized distribution; state
effects negligible) on the real qwen05b.evoq index streams:

  single-symbol tANS  (alphabet <=64)            at L = 2^11, 2^12
  pair-symbol tANS    (alphabet <=4096)          at L = 2^12, 2^13, 2^14
  pair + pruned alphabet (top-N + ESC -> 2 raw 6-bit literals) variants

All-in accounting: + table bytes (2B/kept-entry, per tensor) + 32-bit state flush per 4096-weight
substream. GATE (binding for A2): best pair all-in <= single all-in + 0.03 b/w.

Run:  python -m weights.pair_tans_gate
"""
from __future__ import annotations

import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from weights.evoq import load_container, unpack6_t

CONT = "weights/data/qwen05b.evoq"
CHUNK_W = 4096                  # substream size (weights) per roadmap A2
STATE_BITS = 32                 # rANS state flush per substream
TBL_BYTES_PER_ENTRY = 2


def tans_rate(counts: np.ndarray, L: int):
    """Exact avg bits/symbol of a tANS with slots proportional to counts (>=1 per kept symbol)."""
    p = counts / counts.sum()
    keep = counts > 0
    slots = np.maximum(1, np.round(p[keep] * L)).astype(np.int64)
    # renormalize slots to exactly L (largest-remainder style trim/add)
    diff = L - slots.sum()
    order = np.argsort(-p[keep])
    i = 0
    while diff != 0 and len(order):
        j = order[i % len(order)]
        if diff > 0:
            slots[j] += 1; diff -= 1
        elif slots[j] > 1:
            slots[j] -= 1; diff += 1
        i += 1
    q = slots / L
    rate = float(-(p[keep] * np.log2(q)).sum())
    return rate, int(keep.sum())


def pair_pruned_rate(pc: np.ndarray, L: int, topn: int):
    """Pair tANS with pruned alphabet: top-N pairs + ESC (ESC emits 2 raw 6-bit literals)."""
    tot = pc.sum()
    order = np.argsort(-pc)
    kept = order[:topn]
    kept = kept[pc[kept] > 0]
    p_kept = pc[kept] / tot
    p_esc = 1.0 - p_kept.sum()
    counts = np.concatenate([pc[kept], [max(1, round(p_esc * tot))]])
    r, n_entries = tans_rate(counts, L)
    # split symbol-level rate: coder emits kept-pair OR ESC(+12 raw bits)
    p = counts / counts.sum()
    slots_rate_components = None
    # average bits per PAIR = sum over kept p_i*(-log2 q_i) + p_esc*((-log2 q_esc) + 12)
    # tans_rate already gives sum p*(-log2 q) over ALL incl esc; add raw-literal cost:
    bits_per_pair = r + (p[-1] * 12.0)
    return bits_per_pair / 2.0, n_entries        # bits/weight, table entries


def main():
    meta, comps = load_container(CONT)
    totW = 0
    acc = {k: 0.0 for k in ["single_L11", "single_L12", "pair_full_L12", "pair_full_L13",
                            "pair_full_L14", "pair_p512_L12", "pair_p1024_L13", "pair_p2048_L14",
                            "tbl_single", "tbl_pair_full", "tbl_p512", "tbl_p1024", "tbl_p2048"]}
    for name, c in comps.items():
        idx = unpack6_t(c["packed"], int(c["n_idx"])).numpy().astype(np.int64)
        n = int(c["rows"]) * int(c["cols"])
        idx = idx[:n]
        totW += n
        sc = np.bincount(idx, minlength=64)
        for L, key in ((2048, "single_L11"), (4096, "single_L12")):
            r, ne = tans_rate(sc, L)
            acc[key] += r * n
        acc["tbl_single"] += 64 * TBL_BYTES_PER_ENTRY * 8
        # pairs (non-overlapping, within stream; boundary effects negligible)
        m = n - (n % 2)
        pairs = idx[:m:2] * 64 + idx[1:m:2]
        pc = np.bincount(pairs, minlength=4096)
        for L, key in ((4096, "pair_full_L12"), (8192, "pair_full_L13"), (16384, "pair_full_L14")):
            r, ne = tans_rate(pc, L)
            acc[key] += (r / 2.0) * n
            if key == "pair_full_L13":
                acc["tbl_pair_full"] += ne * TBL_BYTES_PER_ENTRY * 8
        for topn, L, key, tk in ((512, 4096, "pair_p512_L12", "tbl_p512"),
                                 (1024, 8192, "pair_p1024_L13", "tbl_p1024"),
                                 (2048, 16384, "pair_p2048_L14", "tbl_p2048")):
            bpw, ne = pair_pruned_rate(pc, L, topn)
            acc[key] += bpw * n
            acc[tk] += ne * TBL_BYTES_PER_ENTRY * 8

    state_ovh = STATE_BITS / CHUNK_W                     # b/w, same for all variants
    print(f"{totW/1e6:.0f}M weights | substream {CHUNK_W}w -> state overhead {state_ovh:.4f} b/w\n")
    print(f"{'variant':<18}{'rate b/w':>9}{'tables b/w':>11}{'ALL-IN b/w':>11}")
    print("-" * 50)
    base = None
    rows = []
    for key, tk in [("single_L11", "tbl_single"), ("single_L12", "tbl_single"),
                    ("pair_full_L12", "tbl_pair_full"), ("pair_full_L13", "tbl_pair_full"),
                    ("pair_full_L14", "tbl_pair_full"),
                    ("pair_p512_L12", "tbl_p512"), ("pair_p1024_L13", "tbl_p1024"),
                    ("pair_p2048_L14", "tbl_p2048")]:
        rate = acc[key] / totW
        tbl = acc[tk] / totW
        allin = rate + tbl + state_ovh
        rows.append((key, allin))
        if key == "single_L12":
            base = allin
        print(f"{key:<18}{rate:>9.4f}{tbl:>11.4f}{allin:>11.4f}")
    best_pair = min(a for k, a in rows if k.startswith("pair"))
    print(f"\nGATE: best pair all-in {best_pair:.4f} vs single+0.03 = {base+0.03:.4f}")
    print("PAIR-tANS PASSES -- halved symbol throughput at acceptable rate cost"
          if best_pair <= base + 0.03 else
          "PAIR-tANS FAILS the binding gate -> drop pair table, keep layout (per roadmap)")


if __name__ == "__main__":
    main()
