"""R25-A5 seed + A3 SYMMETRY counter-test. The champion's 'honest bits' for its ECVQ INDEX stream
is the order-0 entropy of the indices -- it assumes the indices are iid. But the per-group
Hadamard rotation only whitens WITHIN a group; across groups / rows / layers the index stream may
retain structure a context coder captures. This re-codes the actual ECVQ index stream with cmcore
+ zstd-19 and compares to the order-0 'honest bits'. Two payoffs:
  (1) if cmcore < order-0, the champion gets ANOTHER free win (A5) AND it is the symmetric, fair
      baseline the A3 lattice re-score must be compared against (both streams context-coded);
  (2) if cmcore == order-0, the index stream is truly iid -> the champion's accounting was right
      and the A3 lattice comparison stands as-is.

Run:  python -m weights.ecvq_cmcore [lam]   (default lam=0.008 = champion ECVQ.008)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import zstandard as zstd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors import safe_open
from weights.codec_zoo import _ecvq_levels, _nearest_idx, _entropy_bits
from weights.quant_lab import WPATH, quant_keys
from weights.quant_sota import _fwht_rows

G = 128
P_OUT = 0.005
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMCORE = os.path.join(_ROOT, "cmcore", "target", "release",
                      "cmcore.exe" if os.name == "nt" else "cmcore")


def cmcore_ratio(b, head=24_000_000):
    if not os.path.exists(CMCORE):
        return None
    chunk = b[:head]
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "i"); cm = os.path.join(td, "o"); dec = os.path.join(td, "d")
        open(src, "wb").write(chunk)
        try:
            subprocess.run([CMCORE, "c", src, cm], check=True, capture_output=True, timeout=2400)
            subprocess.run([CMCORE, "d", cm, dec], check=True, capture_output=True, timeout=2400)
        except Exception:
            return None
        if open(dec, "rb").read() != chunk:
            return None
        return os.path.getsize(cm) / len(chunk)


def main():
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.008
    idx_bytes = bytearray()
    # also a ROW-TRANSPOSED variant: indices ordered by (group, row) to expose cross-row structure
    order0_bits = 0.0
    tot = 0
    nlev_max = 0
    with safe_open(WPATH, framework="pt") as f:
        for k in sorted(quant_keys(f)):
            W = f.get_tensor(k).float().numpy()
            rows, cols = W.shape
            n_out = max(1, int(round(W.size * P_OUT)))
            thr = np.partition(np.abs(W).ravel(), W.size - n_out)[W.size - n_out]
            mask = np.abs(W) >= thr
            base = W.copy(); base[mask] = 0.0
            pad = (-cols) % G
            A = np.pad(base, ((0, 0), (0, pad))) if pad else base
            N = A.reshape(rows, -1, G).reshape(-1, G)
            signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
            R = _fwht_rows(N * signs) / np.sqrt(G)
            amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
            Rn = (R / amax)
            rng = np.random.default_rng(1)
            samp = Rn.ravel()[rng.integers(0, Rn.size, min(20000, Rn.size))]
            lv = _ecvq_levels(samp, 64, lam)
            idx = _nearest_idx(Rn.ravel(), lv).astype(np.uint8)   # nlev<=64 fits uint8
            nlev_max = max(nlev_max, len(lv))
            order0_bits += _entropy_bits(idx, len(lv)) * idx.size
            idx_bytes += idx.tobytes()
            tot += W.size

    order0 = order0_bits / tot
    raw = idx_bytes.__len__() * 8 / tot               # 8 bits/index uncoded
    z = len(zstd.ZstdCompressor(level=19).compress(bytes(idx_bytes)))
    zbpw = z * 8 / tot
    r = cmcore_ratio(bytes(idx_bytes))
    cm_bpw = (len(idx_bytes) * r) * 8 / tot if r else None
    print(f"ECVQ lam={lam} index stream ({tot:,} weights, <= {nlev_max} levels):\n")
    print(f"  order-0 entropy ('honest bits' today) = {order0:.4f} b/w")
    print(f"  zstd-19 real container                = {zbpw:.4f} b/w   ({'WIN ' if zbpw<order0-1e-3 else 'tie/loss'} {order0-zbpw:+.4f})")
    if cm_bpw:
        print(f"  cmcore (24MB-head extrap) container   = {cm_bpw:.4f} b/w   ({'WIN ' if cm_bpw<order0-1e-3 else 'tie/loss'} {order0-cm_bpw:+.4f})")
    best = min([x for x in [zbpw, cm_bpw] if x is not None])
    print(f"\n  => index-stream context-coding bank = {max(0.0, order0-best):.4f} b/w")
    if order0 - best > 0.01:
        print(f"     A5 CONFIRMED: champion indices are NOT iid -> another free win; AND the fair "
              f"A3 baseline must context-code ECVQ too (lattice edge may shrink by this much).")
    else:
        print(f"     Indices ~iid: champion order-0 accounting was right; A3 lattice comparison stands.")


if __name__ == "__main__":
    main()
