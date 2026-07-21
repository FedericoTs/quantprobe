"""R25-A1 step 1 -- MEASURE the champion's side-info planes (zero-risk, CPU, no ppl change).

The champion (rotate + ECVQ + entropy + 0.5% outliers) pays RAW:
  - 16 bits per (row,group) fp16 amax  -> 16/128 = 0.125 b/w
  - 32 bits per outlier (position implicit in a 32b record) -> ~0.005*32 = 0.16 b/w
This script measures what those planes ACTUALLY cost under achievable coding:
  amax plane : order-0 entropy of the fp16 bit patterns, order-0 of log2-bucketed values,
               and conditional entropy given the previous group in the same row (spatial ctx).
  outliers   : binomial bitmap rate H2(p) per position + order-0 entropy of 8-bit mu-law-ish
               bucketed magnitudes (value prior), vs the raw 32b/outlier.
Output: per-plane b/w now vs achievable, and the total bank in b/w for the champion config.

Run:  python -m weights.scale_plane
"""
from __future__ import annotations

import math

import numpy as np
import torch
from safetensors import safe_open

from weights.quant_lab import WPATH, quant_keys
from weights.quant_sota import _fwht_rows

G = 128
P_OUT = 0.005


def h0(arr):
    """Order-0 entropy (bits/symbol) of an integer array."""
    _, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def h_cond(cur, prev):
    """Conditional entropy H(cur | prev) via joint - marginal (bits/symbol)."""
    joint = (cur.astype(np.int64) << 32) ^ (prev.astype(np.int64) & 0xFFFFFFFF)
    return h0(joint) - h0(prev)


def main():
    tot_w = 0
    tot_groups = 0
    tot_out = 0
    # accumulators: amax plane
    bits_amax_raw = 0.0
    bits_amax_h0fp16 = 0.0
    bits_amax_h0log = 0.0
    bits_amax_ctx = 0.0
    # outliers
    bits_out_raw = 0.0
    bits_out_coded = 0.0

    with safe_open(WPATH, framework="pt") as f:
        qk = sorted(quant_keys(f))
        print(f"{len(qk)} tensors; champion config g={G}, outliers {P_OUT*100:.1f}%\n")
        for k in qk:
            W = f.get_tensor(k).float().numpy()
            rows, cols = W.shape
            tot_w += W.size

            # ---- outlier plane (champion keeps top-p |W| in fp16, pre-rotation) ----
            n_out = max(1, int(round(W.size * P_OUT)))
            thr = np.partition(np.abs(W).ravel(), W.size - n_out)[W.size - n_out]
            mask = np.abs(W) >= thr
            n_out = int(mask.sum())
            tot_out += n_out
            p = n_out / W.size
            # bitmap at binomial rate + 16b fp16 value with an 8-bit bucketed-magnitude prior
            hbit = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0
            vals = W[mask]
            # value prior: sign + log-magnitude bucketed to 64 bins + residual ~ entropy of buckets + ~6b residual
            lm = np.clip(np.log2(np.abs(vals) + 1e-12), -30, 10)
            buck = np.round((lm + 30) / 40 * 63).astype(np.int32)
            hval = 1.0 + h0(buck) + 6.0     # sign + bucket entropy + conservative 6b mantissa residual
            bits_out_raw += 32.0 * n_out
            bits_out_coded += hbit * W.size + hval * n_out

            # ---- amax plane (champion: per (row,group) amax of the ROTATED weights) ----
            base = W.copy()
            base[mask] = 0.0
            pad = (-cols) % G
            A = np.pad(base, ((0, 0), (0, pad))) if pad else base
            N = A.reshape(rows, -1, G).reshape(-1, G)
            signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
            R = _fwht_rows(N * signs) / np.sqrt(G)
            amax = np.abs(R).max(1)                     # one fp16 per row-group
            ng = amax.size
            tot_groups += ng
            a16 = amax.astype(np.float16).view(np.uint16).astype(np.int64)
            bits_amax_raw += 16.0 * ng
            bits_amax_h0fp16 += h0(a16) * ng
            # log-bucketed (256 bins) -- what a practical coder would code + tiny residual
            la = np.clip(np.log2(amax + 1e-12), -24, 8)
            lb = np.round((la + 24) / 32 * 255).astype(np.int64)
            bits_amax_h0log += (h0(lb) + 4.0) * ng       # +4b conservative residual to stay lossless-ish
            # spatial context: previous group in the same row (row-major group order)
            ngc = (cols + G - 1) // G
            grid = lb.reshape(rows, ngc)
            cur = grid[:, 1:].ravel()
            prev = grid[:, :-1].ravel()
            hc = h_cond(cur, prev) if cur.size else h0(lb)
            first_col = grid[:, 0]
            ctx_bits = h0(first_col) * first_col.size + hc * cur.size + 4.0 * ng
            bits_amax_ctx += ctx_bits

    print("=== amax plane ===")
    print(f"  groups: {tot_groups:,}   raw 16b/group        = {bits_amax_raw/tot_w:.4f} b/w")
    print(f"  order-0 fp16 patterns                          = {bits_amax_h0fp16/tot_w:.4f} b/w")
    print(f"  log-bucket(256)+4b residual                    = {bits_amax_h0log/tot_w:.4f} b/w")
    print(f"  + row-context (prev-group) conditional         = {bits_amax_ctx/tot_w:.4f} b/w")
    print("=== outlier plane ===")
    print(f"  outliers: {tot_out:,} ({100*tot_out/tot_w:.2f}%)  raw 32b/outlier = {bits_out_raw/tot_w:.4f} b/w")
    print(f"  bitmap H2(p) + sign/log-bucket/6b-residual     = {bits_out_coded/tot_w:.4f} b/w")
    print("=== TOTAL SIDE-INFO ===")
    now = (bits_amax_raw + bits_out_raw) / tot_w
    best = (min(bits_amax_h0fp16, bits_amax_h0log, bits_amax_ctx) + bits_out_coded) / tot_w
    print(f"  champion pays now: {now:.4f} b/w   achievable: {best:.4f} b/w")
    print(f"  BANK = {now-best:.4f} b/w at IDENTICAL ppl  (champion 3.13 -> {3.13-(now-best):.3f} b/w)")


if __name__ == "__main__":
    main()
