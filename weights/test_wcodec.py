"""Bulletproof round-trip suite for wcodec. A lossless codec must NEVER corrupt a byte,
across all dtypes, edge shapes, metadata, every delta mode, and sharding.

  python -m weights.test_wcodec
"""

from __future__ import annotations

import json
import os
import struct
import tempfile

import numpy as np

from weights import wcodec as wc

DTYPES = {  # safetensors name -> numpy dtype
    "F32": np.float32, "F16": np.float16, "BF16": np.uint16,
    "I8": np.int8, "I64": np.int64, "BOOL": np.bool_, "U8": np.uint8,
}

_fails = []
_passed = 0


def check(cond, msg):
    global _passed
    if cond:
        _passed += 1
    else:
        _fails.append(msg)
        print(f"  FAIL: {msg}")


def make_st(tensors, metadata=None):
    header = {}
    data = bytearray()
    for name, (dt, arr) in tensors.items():
        b = len(data)
        data += arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [b, len(data)]}
    if metadata:
        header["__metadata__"] = metadata
    hb = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(hb)) + hb + bytes(data)


def rng(seed):
    return np.random.RandomState(seed)


def gen(dt, shape, seed):
    r = rng(seed)
    npd = DTYPES[dt]
    if dt in ("F32", "F16"):
        return (r.randn(*shape) * 0.05).astype(npd)
    if dt == "BF16":
        return (r.randint(0, 65536, shape)).astype(np.uint16)
    if dt == "BOOL":
        return (r.rand(*shape) > 0.5)
    if dt == "I64":
        return r.randint(-1 << 40, 1 << 40, shape, dtype=np.int64)
    if dt == "U8":
        return r.randint(0, 256, shape).astype(np.uint8)
    return r.randint(-128, 128, shape).astype(npd)  # I8


def roundtrip_single(blob, tmp):
    p = os.path.join(tmp, "m.safetensors")
    open(p, "wb").write(blob)
    comp = wc.compress_file(p)
    return wc.decompress_file(comp)


def roundtrip_delta(blob, refblob, tmp):
    p = os.path.join(tmp, "m.safetensors")
    rp = os.path.join(tmp, "ref.safetensors")
    open(p, "wb").write(blob)
    open(rp, "wb").write(refblob)
    comp = wc.compress_file(p, rp)
    return wc.decompress_file(comp, rp)


def main():
    tmp = tempfile.mkdtemp(prefix="wctest_")

    # 1. every dtype, single mode
    print("[1] all dtypes, single mode")
    for dt in DTYPES:
        blob = make_st({"w": (dt, gen(dt, (64, 33), 1)), "b": (dt, gen(dt, (17,), 2))})
        check(roundtrip_single(blob, tmp) == blob, f"single dtype {dt}")

    # 2. edge shapes
    print("[2] edge shapes")
    edge = {
        "empty": ("F32", np.zeros((0,), np.float32)),
        "scalar": ("F32", gen("F32", (1,), 3)),
        "1xN": ("BF16", gen("BF16", (1, 128), 4)),
        "odd": ("F16", gen("F16", (7, 13), 5)),
        "big": ("BF16", gen("BF16", (512, 512), 6)),
    }
    blob = make_st(edge, metadata={"format": "pt", "note": "edge"})
    check(roundtrip_single(blob, tmp) == blob, "single edge shapes + metadata")

    # 3. delta modes: identical (COPY), sparse, dense, missing, shape-mismatch
    print("[3] delta modes")
    base = make_st({
        "ident": ("BF16", gen("BF16", (128, 64), 10)),
        "sparse": ("BF16", gen("BF16", (256, 64), 11)),
        "dense": ("BF16", gen("BF16", (256, 64), 12)),
        "fp32dense": ("F32", gen("F32", (128, 64), 13)),
        "shape": ("BF16", gen("BF16", (64, 64), 14)),
    })
    # build variant from same arrays then perturb
    a_ident = gen("BF16", (128, 64), 10)
    a_sparse = gen("BF16", (256, 64), 11).copy()
    idx = rng(99).choice(a_sparse.size, size=a_sparse.size // 50, replace=False)
    a_sparse.ravel()[idx] = (a_sparse.ravel()[idx].astype(np.int32) + 3).astype(np.uint16)
    a_dense = (gen("BF16", (256, 64), 12).astype(np.int32) + rng(7).randint(-2, 3, (256, 64))).astype(np.uint16)
    a_fp32 = gen("F32", (128, 64), 13) + (gen("F32", (128, 64), 31) * 1e-4).astype(np.float32)
    var = make_st({
        "ident": ("BF16", a_ident),
        "sparse": ("BF16", a_sparse),
        "dense": ("BF16", a_dense),
        "fp32dense": ("F32", a_fp32),
        "shape": ("BF16", gen("BF16", (80, 64), 15)),  # different shape -> single fallback
        "newone": ("BF16", gen("BF16", (32, 32), 16)),  # not in base -> single fallback
    })
    check(roundtrip_delta(var, base, tmp) == var, "delta mixed modes round-trip")

    # 4. multi-shard model round-trip (single + delta)
    print("[4] multi-shard model")
    from weights.shard_model import shard
    mdir = os.path.join(tmp, "model")
    refdir = os.path.join(tmp, "ref")
    open(os.path.join(tmp, "full.safetensors"), "wb").write(var)
    open(os.path.join(tmp, "fullref.safetensors"), "wb").write(base)
    shard(os.path.join(tmp, "full.safetensors"), mdir, 3)
    shard(os.path.join(tmp, "fullref.safetensors"), refdir, 2)
    comp = wc.compress_model(mdir, level=12)
    rest = wc.decompress_model(comp)
    ok = all(open(s, "rb").read() == rest[os.path.basename(s)] for s in wc.shard_files(mdir))
    check(ok, "multi-shard single round-trip")
    compd = wc.compress_model(mdir, refdir, level=12)
    restd = wc.decompress_model(compd, refdir)
    okd = all(open(s, "rb").read() == restd[os.path.basename(s)] for s in wc.shard_files(mdir))
    check(okd, "multi-shard delta round-trip (ref sharded differently)")

    # 5. mode-selection sanity (the right modes get chosen)
    print("[5] mode selection")
    ct, cp = wc._codecs(12), wc._codecs(3)
    bident = gen("BF16", (128, 64), 10).tobytes()
    m, _ = wc._enc_tensor(bident, "BF16", bident, ct, cp, 12, 3)
    check(m == wc.M_COPY, "identical -> M_COPY")
    m, _ = wc._enc_tensor(a_sparse.tobytes(), "BF16", gen("BF16", (256, 64), 11).tobytes(), ct, cp, 12, 3)
    check(m in (wc.M_SPARSE, wc.M_ARITH), f"sparse-ish -> SPARSE/ARITH (got {m})")

    # 6. integrity: wrong reference must be DETECTED, not silently corrupt
    print("[6] integrity / wrong-reference detection")
    p = os.path.join(tmp, "iv.safetensors")
    rp = os.path.join(tmp, "ir.safetensors")
    wp = os.path.join(tmp, "iw.safetensors")
    open(p, "wb").write(var)
    open(rp, "wb").write(base)
    wrong = make_st({  # same names/shapes as base, but 'ident' differs
        "ident": ("BF16", gen("BF16", (128, 64), 777)),
        "sparse": ("BF16", gen("BF16", (256, 64), 11)),
        "dense": ("BF16", gen("BF16", (256, 64), 12)),
        "fp32dense": ("F32", gen("F32", (128, 64), 13)),
        "shape": ("BF16", gen("BF16", (64, 64), 14)),
    })
    open(wp, "wb").write(wrong)
    comp = wc.compress_file(p, rp)
    check(wc.decompress_file(comp, rp) == var, "delta with correct ref ok")
    try:
        wc.decompress_file(comp, wp)
        raised = False
    except ValueError:
        raised = True
    check(raised, "WRONG reference detected by integrity hash (no silent corruption)")

    # 7. low-rank delta mode across the LoRA rank spectrum (adaptive rank)
    print("[7] low-rank delta mode (ranks 2/8/32)")
    rs = rng(55)
    m, n = 512, 256
    for rk in (2, 8, 32):
        base_f = (rs.randn(m, n) * 0.02).astype(np.float32)
        a = (rs.randn(m, rk) * 0.12).astype(np.float32)
        bb = (rs.randn(rk, n) * 0.12).astype(np.float32)
        ft_f = base_f + a @ bb
        base_u16 = wc._to_bf16(base_f)
        ft_u16 = wc._to_bf16(ft_f)
        base_lr = make_st({"w": ("BF16", base_u16), "x": ("BF16", gen("BF16", (64, 64), 8))})
        ft_lr = make_st({"w": ("BF16", ft_u16), "x": ("BF16", gen("BF16", (64, 64), 8))})
        check(roundtrip_delta(ft_lr, base_lr, tmp) == ft_lr, f"low-rank r{rk} round-trip")
        mode, _ = wc._enc_tensor(ft_u16.tobytes(), "BF16", base_u16.tobytes(), ct, cp, 12, 3, [m, n])
        check(mode == wc.M_LOWRANK, f"rank-{rk} delta -> M_LOWRANK (got {mode})")
    # fp16 low-rank (rank-4)
    base_f = (rs.randn(m, n) * 0.02).astype(np.float32)
    ft_f = base_f + (rs.randn(m, 4) * 0.12).astype(np.float32) @ (rs.randn(4, n) * 0.12).astype(np.float32)
    b16, f16 = wc._f2u(base_f, "fp16"), wc._f2u(ft_f, "fp16")
    filler = gen("BF16", (64, 64), 8)
    base_lr = make_st({"w": ("F16", b16), "x": ("F16", filler)})
    ft_lr = make_st({"w": ("F16", f16), "x": ("F16", filler)})
    check(roundtrip_delta(ft_lr, base_lr, tmp) == ft_lr, "fp16 low-rank round-trip")
    mode, _ = wc._enc_tensor(f16.tobytes(), "F16", b16.tobytes(), ct, cp, 12, 3, [m, n])
    check(mode == wc.M_LOWRANK, f"fp16 rank-4 delta -> M_LOWRANK (got {mode})")

    print(f"\n{_passed} checks passed, {len(_fails)} failed")
    if _fails:
        print("FAILURES:")
        for f in _fails:
            print("  -", f)
        raise SystemExit(1)
    print("ALL ROUND-TRIPS BYTE-EXACT - OK")


if __name__ == "__main__":
    main()
