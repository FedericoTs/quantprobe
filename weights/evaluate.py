"""The evaluation harness + diagnostics.

evaluate_codec(): runs a codec over the manifest, verifies byte-exact round-trip
(SHA-256), and reports size-weighted ratio/save% + enc/dec throughput, overall and
per-dtype. Round-trip is the hard gate.

diagnose(): per-dtype, per-plane empirical entropy (order-0 and order-1) for the
exponent vs mantissa planes, and an XOR-delta variant -- this reveals the theoretical
floor and whether spatial/sequential structure exists to exploit (the creative levers).
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from .codecs import DTYPE_ELEM

HERE = os.path.dirname(os.path.abspath(__file__))


def load_manifest():
    with open(os.path.join(HERE, "data", "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)["items"]


def _read(item) -> bytes:
    with open(os.path.join(HERE, item["path"]), "rb") as fh:
        return fh.read()


def evaluate_codec(codec, items, dtypes=None) -> dict:
    items = [it for it in items if dtypes is None or it["dtype"] in dtypes]
    rows = []
    tin = tout = 0
    enc_t = dec_t = 0.0
    all_ok = True
    per_dtype: dict = {}
    for it in items:
        data = _read(it)
        if not data:
            continue
        t0 = time.perf_counter()
        blob = codec.compress(data, it["dtype"])
        t1 = time.perf_counter()
        out = codec.decompress(blob)
        t2 = time.perf_counter()
        ok = len(out) == len(data) and hashlib.sha256(out).digest() == hashlib.sha256(data).digest()
        if not ok:
            all_ok = False
        rows.append({"id": it["id"], "dtype": it["dtype"], "in": len(data),
                     "out": len(blob), "rt": ok})
        tin += len(data)
        tout += len(blob)
        enc_t += t1 - t0
        dec_t += t2 - t1
        d = per_dtype.setdefault(it["dtype"], {"in": 0, "out": 0, "et": 0.0, "dt": 0.0, "ok": True})
        d["in"] += len(data)
        d["out"] += len(blob)
        d["et"] += t1 - t0
        d["dt"] += t2 - t1
        d["ok"] = d["ok"] and ok

    def agg(tin, tout, et, dt, ok):
        return {
            "in_bytes": tin, "out_bytes": tout,
            "ratio": round(tin / tout, 4) if tout else 0.0,
            "save_pct": round((1 - tout / tin) * 100, 2) if tin else 0.0,
            "enc_MBps": round(tin / 1e6 / et, 1) if et > 0 else 0.0,
            "dec_MBps": round(tin / 1e6 / dt, 1) if dt > 0 else 0.0,
            "rt_ok": ok,
        }

    return {
        "codec": codec.name,
        "config": codec.config(),
        "overall": agg(tin, tout, enc_t, dec_t, all_ok),
        "by_dtype": {dt: agg(d["in"], d["out"], d["et"], d["dt"], d["ok"])
                     for dt, d in sorted(per_dtype.items())},
        "n_tensors": len(rows),
    }


# --------------------------------------------------------------------------
# Diagnostics: where are the bits, and is there structure to exploit?
# --------------------------------------------------------------------------
def _h0(a: np.ndarray) -> float:
    c = np.bincount(a, minlength=256).astype(np.float64)
    p = c / c.sum()
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def _h1(a: np.ndarray) -> float:
    if a.size < 2:
        return _h0(a)
    idx = a[:-1].astype(np.int64) * 256 + a[1:]
    bg = np.bincount(idx, minlength=65536).reshape(256, 256).astype(np.float64)
    prev = bg.sum(axis=1)
    tot = a.size - 1
    h = 0.0
    nzp = np.where(prev > 0)[0]
    for i in nzp:
        p = bg[i] / prev[i]
        nz = p > 0
        h += (prev[i] / tot) * float(-np.sum(p[nz] * np.log2(p[nz])))
    return h


def diagnose(items, dtypes=("bf16", "fp16", "fp32")) -> dict:
    out: dict = {}
    for dtype in dtypes:
        group = [it for it in items if it["dtype"] == dtype]
        if not group:
            continue
        e = DTYPE_ELEM[dtype]
        raw = b"".join(_read(it) for it in group)
        arr = np.frombuffer(raw[: len(raw) // e * e], dtype=np.uint8).reshape(-1, e)
        n = arr.shape[0]
        planes = {}
        for p in range(e):
            col = arr[:, p]
            planes[f"byte{p}"] = {"h0": round(_h0(col), 3), "h1": round(_h1(col), 3)}
        # XOR-delta of consecutive elements (integer bit-patterns) -> does sequential
        # correlation exist? (Gorilla-style; if yes, predictive coding helps)
        if e == 2:
            u = np.frombuffer(raw[: n * 2], dtype="<u2").copy()
        elif e == 4:
            u = np.frombuffer(raw[: n * 4], dtype="<u4").copy()
        else:
            u = None
        xor = {}
        if u is not None:
            d = u.copy()
            d[1:] = u[1:] ^ u[:-1]
            db = d.view(np.uint8).reshape(-1, e)
            for p in range(e):
                xor[f"byte{p}"] = {"h0": round(_h0(db[:, p]), 3)}
        # theoretical floor (order-1 per plane) in bytes
        floor1 = sum(planes[f"byte{p}"]["h1"] for p in range(e)) / 8.0 * n
        out[dtype] = {
            "n_elems": n,
            "raw_bytes": n * e,
            "planes": planes,
            "xor_delta_planes": xor,
            "order1_floor_bytes": int(floor1),
            "order1_floor_save_pct": round((1 - floor1 / (n * e)) * 100, 2),
        }
    return out
