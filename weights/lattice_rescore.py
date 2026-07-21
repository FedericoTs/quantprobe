"""R25-A3 step 1 -- LATTICE RATE AMNESTY: re-score the R9-R12 E8/D4 lattice codecs under a
REAL decodable coder instead of the per-coordinate entropy formula.

History: R10 scored E8 with JOINT empirical entropy (undercounts ~2x at fine q -- caught as a
bug); R12 "fixed" it with PER-COORDINATE entropy, which OVERCOUNTS (~1 b/w, hostile review)
because lattice coordinates are correlated through the parity constraint + radial shaping.
The truth sits between, and it is constructive: serialize the actual integer lattice
coordinates and compress with cmcore (context-mixing captures intra-vector correlation
automatically) and zstd-19. Real bytes, fully decodable, symmetric with the A1 champion
accounting. The ppl values are unchanged from the arena archive -- only the bits move.

Run:  python -m weights.lattice_rescore
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import zstandard as zstd
from safetensors import safe_open

from weights.codec_zoo import _d4_nearest, _e8_nearest, _coord_entropy
from weights.quant_lab import WPATH, quant_keys, _awq_scale
from weights.quant_sota import _fwht_rows

G = 128
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMCORE = os.path.join(_ROOT, "cmcore", "target", "release",
                      "cmcore.exe" if os.name == "nt" else "cmcore")

# arena-archived held-out ppl (unchanged by re-scoring) + R12 per-coordinate bpw claims
# (E8 q.08 and q.14 already re-scored: TOTAL 4.640@4.020 and 3.836@4.151 -- skip)
ARCHIVE = {
    ("E8", 0.22): (4.5353, 3.920), ("E8", 0.28): (4.9440, 3.584),
    ("D4", 0.16): (4.5366, 3.371), ("D4", 0.20): (4.8546, 3.062),
}
CHAMP_NOTE = "champion after A1 re-baseline: 4.483 ppl @ ~2.98-3.00 b/w | ECVQ.003 4.169 @ ~3.76"


def cmcore_ratio_estimate(b, head=16_000_000):
    """cmcore on a HEAD slice only (full 357MB stream is infeasible — Hutter coder ~MB/s).
    Returns compressed_bytes/raw_bytes on the head, round-trip-checked; None if unavailable."""
    if not os.path.exists(CMCORE):
        return None
    chunk = b[:head]
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "i"); cm = os.path.join(td, "o"); dec = os.path.join(td, "d")
        open(src, "wb").write(chunk)
        try:
            subprocess.run([CMCORE, "c", src, cm], check=True, capture_output=True, timeout=1800)
            subprocess.run([CMCORE, "d", cm, dec], check=True, capture_output=True, timeout=1800)
        except Exception:
            return None
        if open(dec, "rb").read() != chunk:
            return None
        return os.path.getsize(cm) / len(chunk)


_RN_CACHE = None   # rotation Rn is q-independent -> compute the FWHT pipeline ONCE, reuse per q


def _build_rn_cache():
    global _RN_CACHE
    if _RN_CACHE is not None:
        return _RN_CACHE
    cache, tot_elem, over_groups = [], 0, 0
    with safe_open(WPATH, framework="pt") as f:
        for k in sorted(quant_keys(f)):
            W = f.get_tensor(k).float().numpy()
            rows, cols = W.shape
            tot_elem += W.size
            pad = (-cols) % G
            A = np.pad(W, ((0, 0), (0, pad))) if pad else W
            N = A.reshape(rows, -1, G).reshape(-1, G)
            signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
            R = _fwht_rows(N * signs) / np.sqrt(G)
            amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
            cache.append((R / amax).ravel().astype(np.float32))
            over_groups += N.shape[0]
    _RN_CACHE = (cache, tot_elem, over_groups)
    return _RN_CACHE


def lattice_codes(lat, q):
    """Lattice quant reusing the cached q-independent rotation. Returns int8(2*coord) stream +
    element count + per-coord entropy + group count."""
    d = 8 if lat == "E8" else 4
    nearest = _e8_nearest if lat == "E8" else _d4_nearest
    cache, tot_elem, over_groups = _build_rn_cache()
    chunks, pc_bits = [], 0.0
    for flat in cache:
        padv = (-len(flat)) % d
        f2 = np.concatenate([flat, np.zeros(padv, np.float32)]) if padv else flat
        V = f2.reshape(-1, d) / q
        P = nearest(V)
        pc_bits += _coord_entropy(P) / d * (V.size - padv)
        P2 = P * 2.0                                      # E8 half-integer coords -> 2P integer
        Pi = np.round(P2).astype(np.int8)
        if not np.array_equal(Pi.astype(np.float64), P2):
            raise RuntimeError("int8(2P) changed a lattice point -- widen dtype")
        chunks.append(Pi.tobytes())
    return b"".join(chunks), tot_elem, pc_bits, over_groups


def main():
    AMAX_SIDE = 0.0792   # A1-measured honest lossless amax rate (b/w), symmetric w/ champion
    print(f"{CHAMP_NOTE}\n")
    print(f"{'codec':<10}{'ppl':>8}{'R12claim':>10}{'zstd19':>9}{'cm~est':>9}{'side':>7}{'TOTAL':>9}")
    print("-" * 62)
    import gc
    import traceback
    for (lat, q), (ppl, claimed) in ARCHIVE.items():
        try:
            stream, n, pc_bits, ngroups = lattice_codes(lat, q)
            z = len(zstd.ZstdCompressor(level=19).compress(stream))
            zbpw = z * 8 / n
            r = cmcore_ratio_estimate(stream)
            del stream
            cm_bpw = None if r is None else r * 8  # ratio(bytes_out/bytes_in) * 8 bits/byte = b/coordbyte = b/w
            coord_bpw = min(zbpw, cm_bpw) if cm_bpw else zbpw
            total = coord_bpw + AMAX_SIDE
            cm_s = f"{cm_bpw:8.3f}" if cm_bpw else "      --"
            print(f"{lat} q{q:<5}{ppl:>8.3f}{claimed:>10.3f}{zbpw:>9.3f}{cm_s:>9}{AMAX_SIDE:>7.3f}{total:>9.3f}", flush=True)
        except BaseException as e:
            print(f"{lat} q{q:<5} FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        gc.collect()
    print("\n(claim=R12 per-coord-entropy b/w incl side [SUSPECTED OVERCOUNT]; zstd19=REAL decodable "
          "container for the coord stream; cm~est=cmcore head-ratio extrapolated; TOTAL=best coord "
          "coder + A1 honest amax side 0.079. Champion honest ~3.00@4.483; ECVQ.003 ~3.76@4.169. "
          "A lattice TOTAL < champion at <= its ppl REOPENS the branch.)")


if __name__ == "__main__":
    main()
