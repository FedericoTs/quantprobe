"""R25-A1 step 2 -- REAL side-info container (honest bytes, round-trip verified).

Serializes the champion's actual side-info planes and compresses them with REAL decodable
coders, then DECODES and verifies byte-identical reconstruction. Honest bits = container bytes.

Streams (concatenated across all 168 tensors, manifest counted):
  S1 amax-hi : high byte of each fp16 group-amax, row-major group order (sign+exp+hi-mantissa)
  S2 amax-lo : low byte of each fp16 group-amax (noisy mantissa)
  S3 out-pos : outlier linear indices per tensor, sorted, delta-gap varint
  S4 out-hi  : high byte of each fp16 outlier value
  S5 out-lo  : low byte of each fp16 outlier value
Coders: zstd-19 (baseline, instant) and optionally cmcore (context-mixing, if binary present).

Run:  python -m weights.scale_plane_codec
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Win cp1252 console safety
except Exception:
    pass

import numpy as np
import zstandard as zstd
from safetensors import safe_open

from weights.quant_lab import WPATH, quant_keys
from weights.quant_sota import _fwht_rows

G = 128
P_OUT = 0.005
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMCORE = os.path.join(_ROOT, "cmcore", "target", "release",
                      "cmcore.exe" if os.name == "nt" else "cmcore")


def planes():
    """Yield (key, amax_fp16[ng], out_idx[int64 sorted], out_val_fp16) per tensor --
    exactly the champion's side info (same seed/logic as codec_zoo/scale_plane)."""
    with safe_open(WPATH, framework="pt") as f:
        for k in sorted(quant_keys(f)):
            W = f.get_tensor(k).float().numpy()
            rows, cols = W.shape
            n_out = max(1, int(round(W.size * P_OUT)))
            flat = np.abs(W).ravel()
            thr = np.partition(flat, W.size - n_out)[W.size - n_out]
            mask = np.abs(W) >= thr
            idx = np.flatnonzero(mask.ravel()).astype(np.int64)
            vals = W.ravel()[idx].astype(np.float16)
            base = W.copy(); base[mask] = 0.0
            pad = (-cols) % G
            A = np.pad(base, ((0, 0), (0, pad))) if pad else base
            N = A.reshape(rows, -1, G).reshape(-1, G)
            signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
            R = _fwht_rows(N * signs) / np.sqrt(G)
            amax = np.abs(R).max(1).astype(np.float16)
            yield k, amax, idx, vals


def varint(arr):
    """Delta-gap varint encode a sorted int64 index array -> bytes."""
    gaps = np.diff(arr, prepend=arr[:1] - (arr[:1] - 0))  # first = abs index
    gaps = np.concatenate([arr[:1], np.diff(arr)])
    out = bytearray()
    for v in gaps:
        v = int(v)
        while True:
            b = v & 0x7F
            v >>= 7
            out.append(b | (0x80 if v else 0))
            if not v:
                break
    return bytes(out)


def varint_decode(buf, n):
    vals = np.empty(n, np.int64)
    pos = 0
    for i in range(n):
        shift = 0; v = 0
        while True:
            b = buf[pos]; pos += 1
            v |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        vals[i] = v
    return np.cumsum(vals), pos


def zstd19(b):
    return zstd.ZstdCompressor(level=19).compress(b)


def zstd_dec(b):
    return zstd.ZstdDecompressor().decompress(b)


def cmcore_pair(b):
    """Compress+decompress through the cmcore binary; returns (comp_len, roundtrip_ok)."""
    if not os.path.exists(CMCORE):
        return None, False
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.bin"); cmp_ = os.path.join(td, "out.cm"); dec = os.path.join(td, "dec.bin")
        open(src, "wb").write(b)
        try:
            subprocess.run([CMCORE, "c", src, cmp_], check=True, capture_output=True, timeout=3600)
            subprocess.run([CMCORE, "d", cmp_, dec], check=True, capture_output=True, timeout=3600)
        except Exception:
            return None, False
        ok = open(dec, "rb").read() == b
        return os.path.getsize(cmp_), ok


def main():
    s_ahi = bytearray(); s_alo = bytearray(); s_pos = bytearray()
    s_vhi = bytearray(); s_vlo = bytearray()
    manifest = bytearray()
    tot_w = 0
    orig = {}
    for k, amax, idx, vals in planes():
        a = amax.view(np.uint16)
        s_ahi += (a >> 8).astype(np.uint8).tobytes()
        s_alo += (a & 0xFF).astype(np.uint8).tobytes()
        pv = varint(idx)
        s_pos += pv
        v = vals.view(np.uint16)
        s_vhi += (v >> 8).astype(np.uint8).tobytes()
        s_vlo += (v & 0xFF).astype(np.uint8).tobytes()
        manifest += f"{k}\t{amax.size}\t{idx.size}\t{len(pv)}\n".encode()
        with safe_open(WPATH, framework="pt") as f:
            pass
        orig[k] = (amax.copy(), idx.copy(), vals.copy())
        tot_w += 0
    # total weights for b/w
    with safe_open(WPATH, framework="pt") as f:
        tot_w = sum(int(np.prod(f.get_slice(k).get_shape())) for k in quant_keys(f))

    streams = {"amax-hi": bytes(s_ahi), "amax-lo": bytes(s_alo), "out-pos": bytes(s_pos),
               "out-hi": bytes(s_vhi), "out-lo": bytes(s_vlo), "manifest": bytes(manifest)}
    raw_bits = (len(s_ahi) + len(s_alo)) * 8 + 32 * (len(s_vhi))  # champion raw: 16b/amax + 32b/outlier
    print(f"total weights {tot_w:,}; champion raw side-info = {raw_bits/tot_w:.4f} b/w\n")
    print(f"{'stream':<10}{'raw MB':>9}{'zstd19 MB':>11}{'cmcore MB':>11}")
    tot_z = 0; tot_cm = 0; cm_all_ok = True
    for name, b in streams.items():
        z = zstd19(b)
        assert zstd_dec(z) == b
        cl, ok = cmcore_pair(b) if name != "manifest" else (None, False)
        cm_all_ok &= (ok or cl is None)
        tot_z += len(z)
        tot_cm += (cl if (cl and ok) else len(z))
        cm_s = f"{cl/1e6:10.3f}" if (cl and ok) else "        --"
        print(f"{name:<10}{len(b)/1e6:>9.3f}{len(z)/1e6:>11.3f}{cm_s}", flush=True)

    # round-trip verify reconstruction from decompressed streams
    ahi = np.frombuffer(zstd_dec(zstd19(streams['amax-hi'])), np.uint8)
    alo = np.frombuffer(zstd_dec(zstd19(streams['amax-lo'])), np.uint8)
    rec_amax_all = ((ahi.astype(np.uint16) << 8) | alo).view(np.float16)
    off_a = 0; off_p = 0
    pos_buf = zstd_dec(zstd19(streams['out-pos']))
    vhi = np.frombuffer(zstd_dec(zstd19(streams['out-hi'])), np.uint8)
    vlo = np.frombuffer(zstd_dec(zstd19(streams['out-lo'])), np.uint8)
    rec_vals_all = ((vhi.astype(np.uint16) << 8) | vlo).view(np.float16)
    off_v = 0
    for line in bytes(manifest).decode().strip().split("\n"):
        k, ng, nout, plen = line.split("\t")
        ng, nout, plen = int(ng), int(nout), int(plen)
        ra = rec_amax_all[off_a:off_a + ng]; off_a += ng
        ridx, _ = varint_decode(pos_buf[off_p:off_p + plen], nout); off_p += plen
        rv = rec_vals_all[off_v:off_v + nout]; off_v += nout
        oa, oi, ov = orig[k]
        assert np.array_equal(ra.view(np.uint16), oa.view(np.uint16)), f"amax mismatch {k}"
        assert np.array_equal(ridx, oi), f"idx mismatch {k}"
        assert np.array_equal(rv.view(np.uint16), ov.view(np.uint16)), f"val mismatch {k}"
    print("\nROUND-TRIP: byte-identical reconstruction of ALL planes [OK]")

    print(f"\nzstd-19 container : {tot_z/1e6:.3f} MB = {tot_z*8/tot_w:.4f} b/w  "
          f"(bank {raw_bits/tot_w - tot_z*8/tot_w:.4f} b/w)")
    if tot_cm != tot_z:
        print(f"cmcore container  : {tot_cm/1e6:.3f} MB = {tot_cm*8/tot_w:.4f} b/w  "
              f"(bank {raw_bits/tot_w - tot_cm*8/tot_w:.4f} b/w)")
    print(f"champion 3.130 b/w -> {3.130 - (raw_bits/tot_w - min(tot_z, tot_cm)*8/tot_w):.3f} b/w at IDENTICAL ppl")


if __name__ == "__main__":
    main()
