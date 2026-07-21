"""R25v2 A1-X4 -- LOSSY amax-snapping (A1's untested upside). A1 banked 0.127 b/w losslessly but
the amax LOW byte is incompressible noise (2.80MB, 0% gain). Snapping each per-group amax to a
small per-tensor log-spaced codebook (V entries) drops the low byte entirely -> amax side falls
from ~0.079 to ~log2(V)/128 b/w (V=32 -> 0.039), banking another ~0.04 b/w. BUT it perturbs the
dequantization (lossy) so it MUST pass the held-out ppl gate. Snap UP to the bin ceiling (never
down -> never push normalized weights outside the level grid -> no clip blowups); an ESCAPE keeps
exact fp16 amax for the rare groups beyond the top centroid.

Compares champion ECVQ.008 with raw amax vs V-snapped amax (V in {16,32,64}) at matched ppl gate.

Run:  python -m weights.amax_snap
"""
from __future__ import annotations

import gc
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights import codec_zoo
from weights.quant_lab import CFG, build_model, calibrate, load_fp16, load_quant, ppl

G = 128
ESC = 0.001   # fraction of groups kept exact (beyond top centroid)


def snap_amax_codec(a, k, calib, V):
    """champion ECVQ.008 but per-group amax SNAPPED-UP to a V-entry per-tensor log codebook."""
    Ws, sc = codec_zoo._act(a, k, calib)
    # reproduce _had_ecvq's rotation/amax, snap amax, then quantize with snapped amax
    from weights.quant_sota import _fwht_rows
    rows, cols = Ws.shape
    mask = np.abs(Ws) >= np.quantile(np.abs(Ws), 1.0 - 0.005)
    base = Ws.copy(); base[mask] = 0.0
    pad = (-cols) % G
    A = np.pad(base, ((0, 0), (0, pad))) if pad else base
    N = A.reshape(rows, -1, G).reshape(-1, G)
    signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(G)
    amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
    # log-domain codebook (per tensor), snap UP to ceiling
    la = np.log2(amax.ravel())
    centers = np.quantile(la, (np.arange(V) + 0.5) / V)
    top = centers[-1]
    esc = la > np.quantile(la, 1.0 - ESC)
    # ceil to nearest center >= la (snap up); for esc keep exact
    idx = np.searchsorted(centers, la, side="left")
    idx = np.clip(idx, 0, V - 1)
    snapped = np.where(esc, la, centers[idx])
    amax_s = (2.0 ** snapped).reshape(-1, 1)
    amax_s = np.maximum(amax_s, amax)                 # guarantee snap-UP (>= original)
    amax_s[esc.reshape(-1, 1)] = amax[esc.reshape(-1, 1)]
    Rn = R / amax_s
    lv = codec_zoo._ecvq_levels(Rn.ravel()[np.random.default_rng(1).integers(0, Rn.size, 20000)], 64, 0.008)
    idxq = codec_zoo._nearest_idx(Rn.ravel(), lv)
    Rh = lv[idxq].reshape(Rn.shape) * amax_s
    back = (_fwht_rows(Rh) / np.sqrt(G)) * signs
    wh = back.reshape(rows, -1)[:, :cols]
    wh[mask] = Ws[mask]
    ent = codec_zoo._entropy_bits(idxq, len(lv))
    # honest bits: index entropy + SNAPPED amax (log2(V)/G + esc exact fp16) + levels + outliers
    n_esc = int(esc.sum())
    amax_bits = np.log2(V) * N.shape[0] + n_esc * 16
    bits = ent * a.size + amax_bits + len(lv) * 16 + int(mask.sum()) * 32 + 16 * a.shape[1]
    return (wh / sc[None, :]).astype(np.float32), bits


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    print(f"fp16 {fp16:.4f}\n", flush=True)

    # baseline: champion ECVQ.008 (raw amax, A1-coded side ~0.079 amax b/w)
    load_fp16(model)
    b0 = load_quant(model, lambda a, k: codec_zoo.ecvq_mid(a, k, calib))
    p0 = ppl(model, tok)
    honest0 = b0 - 0.127                              # A1 lossless re-baseline
    print(f"{'scheme':<22}{'arena b/w':>10}{'honest b/w':>11}{'ppl':>9}{'d_ppl':>8}")
    print("-" * 60)
    print(f"{'champ ECVQ.008 (raw)':<22}{b0:>10.3f}{honest0:>11.3f}{p0:>9.4f}{0.0:>8.3f}", flush=True)

    for V in (64, 32, 16):
        load_fp16(model)
        b = load_quant(model, lambda a, k: snap_amax_codec(a, k, calib, V))
        p = ppl(model, tok)
        # honest: this codec already counts snapped-amax; subtract only the outlier A1-bank (~0.048)
        honest = b - 0.048
        dp = p - p0
        tag = ""
        if honest < honest0 - 0.005:
            tag = "  WIN" if dp < 0.03 else "  (cheaper but +ppl)"
        print(f"{'amax-snap V='+str(V):<22}{b:>10.3f}{honest:>11.3f}{p:>9.4f}{dp:>+8.3f}{tag}", flush=True)
        gc.collect()

    print(f"\n(champion honest {honest0:.3f} b/w @ ppl {p0:.4f}. amax-snapping wins if honest drops "
          f">~0.03 b/w at d_ppl < ~0.03, i.e. within the seed-noise floor 0.067.)")


if __name__ == "__main__":
    main()
