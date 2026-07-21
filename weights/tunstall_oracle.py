"""Campaign2 NEW-HYPOTHESIS-1 oracle -- can a TUNSTALL (variable-to-fixed) code on our PROVEN-iid
ECVQ index stream achieve near-rANS rate? If yes, we get a FULLY-PARALLEL, random-access,
branch-free GPU decoder (each D-bit codeword = one independent table lookup emitting k symbols),
killing rANS's sequential state dependency -- the decode bottleneck.

Tunstall for an iid source: greedily expand the highest-prob leaf into A=|alphabet| children until
#leaves <= 2^D. Expected symbols/codeword = 1 + sum(expanded-leaf probs). Rate = D / E[symbols].
Compare to order-0 entropy H (= rANS achievable). Decode parallelism: codewords independent;
per-codeword symbol count is table-known -> warp prefix-sum gives output offsets.

Also reports the table footprint (entries x avg-string-len) and the rate at table sizes a 1060's
shared memory (48KB/SM) can hold.

Run:  python -m weights.tunstall_oracle
"""
from __future__ import annotations

import heapq
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from weights.evoq import load_container, unpack6_t

CONT = "weights/data/qwen05b.evoq"


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def tunstall_rate(p, D):
    """Exact Tunstall rate (bits/symbol) for iid dist p, D-bit codewords (<=2^D leaves).
    Returns (rate, n_leaves, avg_symbols_per_codeword, mean_string_len_for_table_mem)."""
    A = len(p)
    Lmax = 1 << D
    # max-heap of leaf probs (negate for min-heap); track expanded-prob sum
    heap = [(-float(pi), i) for i, pi in enumerate(p)]
    heapq.heapify(heap)
    n_leaves = A
    expanded_sum = 0.0
    # depth bookkeeping for table memory: track current leaves' lengths via a counter dict is heavy;
    # mean string length = E[symbols] (same quantity), table entries = n_leaves
    while n_leaves + (A - 1) <= Lmax:
        negq, _ = heapq.heappop(heap)
        q = -negq
        expanded_sum += q
        for i in range(A):
            child = q * p[i]
            if child > 0:
                heapq.heappush(heap, (-child, i))
        n_leaves += A - 1
    avg_syms = 1.0 + expanded_sum
    rate = D / avg_syms
    return rate, n_leaves, avg_syms


def main():
    meta, comps = load_container(CONT)
    # pool indices (iid => single global dictionary optimal)
    allc = np.zeros(64, np.int64)
    totW = 0
    for name, c in comps.items():
        idx = unpack6_t(c["packed"], int(c["n_idx"])).numpy().astype(np.int64)
        n = int(c["rows"]) * int(c["cols"])
        allc += np.bincount(idx[:n], minlength=64)
        totW += n
    p = allc / allc.sum()
    H = entropy(p)
    k_used = int((p > 0).sum())
    print(f"{totW/1e6:.0f}M weights | {k_used} active levels | order-0 entropy H = {H:.4f} b/w "
          f"(= rANS achievable, the SEQUENTIAL decoder)\n")
    print(f"{'codeword bits D':>16}{'#leaves':>10}{'syms/word':>11}{'rate b/w':>10}{'gap-to-H':>10}{'tbl ~KB':>9}")
    print("-" * 66)
    for D in (10, 12, 14, 16, 18, 20):
        rate, nleaf, avgs = tunstall_rate(p, D)
        # table: nleaf entries, each stores a string of avg `avgs` symbols (1 byte each) + len
        tbl_kb = nleaf * (avgs + 1) / 1024
        print(f"{D:>16}{nleaf:>10}{avgs:>11.3f}{rate:>10.4f}{rate-H:>+10.4f}{tbl_kb:>9.1f}", flush=True)

    print("\nKILL/PASS (roadmap-style, pre-registered): a Tunstall variant within +0.10 b/w of H at a "
          "table that fits ~48KB shared mem (D<=14, ~16K entries) => PARALLEL DECODE PATH OPENS "
          "(rANS sequential bottleneck removed). >+0.10 b/w => stay with rANS substreams.")
    print("NOTE: indices are iid (D_idx=0, triple-verified C1) so a SINGLE global table is optimal; "
          "decode = independent per-codeword LUT + warp prefix-sum for output offsets.")


if __name__ == "__main__":
    main()
