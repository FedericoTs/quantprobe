"""wcodec -- end-to-end lossless codec for .safetensors weight files.

Turns the validated research into a real tool. Two modes:

  single : each tensor compressed with the smart per-plane codec (compress the
           exponent plane, store the random mantissa raw). Beats ZipNN on bf16.
  delta  : each tensor compressed as a per-plane XOR-delta vs the same-named tensor
           in a REFERENCE file (base model / previous checkpoint). ~69% on real
           Pythia training deltas; auto-falls back to single for tensors with no
           match. Sparse deltas (LoRA/pruned) additionally use a bitmap+values path.

The container stores the original safetensors header verbatim, so decompression
reproduces the input file byte-for-byte (SHA-256 gated). dtypes we don't model are
stored with a plain backend (still lossless).

CLI:
  python -m weights.wcodec compress  IN.safetensors [--ref REF] -o OUT.wc
  python -m weights.wcodec decompress OUT.wc [--ref REF] -o RESTORED.safetensors
  python -m weights.wcodec bench IN.safetensors [--ref REF]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evocompress import backends  # noqa: E402
from weights import codecs as cd  # noqa: E402

MAGIC = b"WCD1"
# safetensors dtype string -> our codec dtype
ST2DT = {"F64": "fp64", "F32": "fp32", "F16": "fp16", "BF16": "bf16", "I8": "int8",
         "F8_E4M3": "f8e4m3", "F8_E5M2": "f8e5m2", "U8": "uint8",
         "I16": "int16", "I32": "int32", "I64": "int64"}
UVIEW = {"bf16": "<u2", "fp16": "<u2", "fp32": "<u4", "fp64": "<u8", "int8": "<u1",
         "f8e4m3": "<u1", "f8e5m2": "<u1", "uint8": "<u1",
         "int16": "<u2", "int32": "<u4", "int64": "<u8"}

# per-tensor storage modes
M_RAW = 0     # plain backend (dtype we don't model, or fallback)
M_SINGLE = 1  # smart per-plane single
M_DELTA = 2   # per-plane XOR delta vs ref
M_SPARSE = 3  # sparse XOR delta: bitmap of changed elems + changed values (raw)
M_COPY = 4    # delta vs ref but identical -> store nothing
M_ARITH = 5   # arithmetic float-ordered delta (zig-zag of ULP moves), 2-byte floats
M_LOWRANK = 6  # low-rank residual: store int16 rank-r factors + exact arith residual (bf16)

_ZSTD = backends.get_backend("zstd")
_SPARSE_MAX = 0.5  # use sparse path only if <50% of elements changed


def _codecs(level):
    return cd.SplitSmartCodec("zstd", level), cd.SplitPerPlaneCodec(["zstd"], level, "pp")


# stateless decoders (zstd decompress ignores level)
_SMART = cd.SplitSmartCodec("zstd", 1)
_PERPLANE = cd.SplitPerPlaneCodec(["zstd"], 1, "pp")


# ----------------------------- safetensors parsing -----------------------------

def parse(path):
    """Return (raw, header_bytes, header_dict, data_off) for a .safetensors file.

    `raw` is a read-only mmap so large models never have to be fully resident; slicing
    `raw[a:b]` copies only that tensor. data_off = 8 + header_len.
    """
    f = open(path, "rb")
    try:
        raw = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    finally:
        f.close()  # the mapping stays valid after the fd is closed
    (hlen,) = struct.unpack("<Q", raw[:8])
    header_bytes = raw[8:8 + hlen]
    header = json.loads(header_bytes)
    return raw, header_bytes, header, 8 + hlen


def _tensors_in_order(header):
    """Tensor (name, dtype, begin, end) sorted by data offset; skips __metadata__."""
    items = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        b, e = meta["data_offsets"]
        items.append((name, meta["dtype"], int(b), int(e)))
    items.sort(key=lambda t: t[2])
    return items


# ---- arithmetic float-ordered delta (2-byte floats) ----
# Map a 16-bit float bit pattern to a monotonic integer key so that adjacent
# representable values get adjacent keys; then (key_ft - key_base) is the signed
# number of ULP steps the weight moved -- small for gentle fine-tuning, and far
# lower entropy than the XOR when small moves cross an exponent boundary.

def _mono16(u16):
    sign = (u16 >> 15) & 1
    return np.where(sign == 1, (~u16) & 0xFFFF, u16 | 0x8000).astype(np.uint16)


def _inv_mono16(k16):
    top = (k16 >> 15) & 1
    return np.where(top == 1, k16 & 0x7FFF, (~k16) & 0xFFFF).astype(np.uint16)


def _arith_enc16(u, r, perplane):
    d = _mono16(u).astype(np.int64) - _mono16(r).astype(np.int64)
    zz = ((d << 1) ^ (d >> 63)).astype(np.uint32)        # zig-zag, fits u32
    return perplane.compress(zz.astype("<u4").tobytes(), "fp32")  # 4 planes, hi~0


def _arith_dec16(blob, ref_buf, perplane):
    zz = np.frombuffer(perplane.decompress(blob), "<u4")
    d = (zz >> 1).astype(np.int64) ^ -(zz & 1).astype(np.int64)
    r = np.frombuffer(ref_buf, "<u2")
    k = (_mono16(r).astype(np.int64) + d).astype(np.uint16)
    return _inv_mono16(k).tobytes()


def _mono32(u32):
    sign = (u32 >> 31) & 1
    return np.where(sign == 1, (~u32) & 0xFFFFFFFF, u32 | 0x80000000).astype(np.uint32)


def _inv_mono32(k32):
    top = (k32 >> 31) & 1
    return np.where(top == 1, k32 & 0x7FFFFFFF, (~k32) & 0xFFFFFFFF).astype(np.uint32)


def _arith_enc32(u, r, perplane):
    d = _mono32(u).astype(np.int64) - _mono32(r).astype(np.int64)
    zz = ((d << 1) ^ (d >> 63)).astype(np.uint64)       # zig-zag, fits u64
    return perplane.compress(zz.astype("<u8").tobytes(), "fp64")  # 8 planes, hi~0


def _arith_dec32(blob, ref_buf, perplane):
    zz = np.frombuffer(perplane.decompress(blob), "<u8")
    d = (zz >> 1).astype(np.int64) ^ -(zz & 1).astype(np.int64)
    r = np.frombuffer(ref_buf, "<u4")
    k = (_mono32(r).astype(np.int64) + d).astype(np.uint32)
    return _inv_mono32(k).tobytes()


def _arith_enc(u, r, perplane, dt):
    return _arith_enc16(u, r, perplane) if dt in ("bf16", "fp16") else _arith_enc32(u, r, perplane)


def _arith_dec(blob, ref_buf, perplane, dt):
    return _arith_dec16(blob, ref_buf, perplane) if dt in ("bf16", "fp16") \
        else _arith_dec32(blob, ref_buf, perplane)


# ---- low-rank residual delta (bf16 2D tensors) ----
# Fine-tuning / abliteration / LoRA-merge updates are often low-rank. We store a small
# rank-r factorisation of the float delta plus the EXACT residual vs a deterministically
# reconstructed reference. Factors are int16 and the reconstruction uses EXACT integer
# matmul, so decode is bit-identical on any machine; the residual absorbs all factor
# quantisation error, keeping it lossless regardless of factor quality.

def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _to_bf16(f32):
    u = np.ascontiguousarray(f32).view(np.uint32)
    return (((u.astype(np.uint64) + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16))


def _rsvd(M, r_max, power=2, oversample=6):
    """Randomised SVD up to rank r_max. Returns A:m×k, B:k×n, singular values S.
    Encode-only; randomness doesn't affect round-trip (factors are stored)."""
    m, n = M.shape
    rs = np.random.RandomState(0)
    rr = min(r_max + oversample, n, m)
    Y = M @ rs.standard_normal((n, rr)).astype(np.float32)
    for _ in range(power):
        Y = M @ (M.T @ Y)
    Q, _ = np.linalg.qr(Y)
    Ub, S, Vt = np.linalg.svd(Q.T @ M, full_matrices=False)
    U = Q @ Ub
    k = min(r_max, S.size)
    return (U[:, :k] * S[:k]).astype(np.float32), Vt[:k].astype(np.float32), S[:k]


def _u2f(u16, dt):  # 16-bit float bits -> float32
    return _bf16_to_f32(u16) if dt == "bf16" else u16.view(np.float16).astype(np.float32)


def _f2u(f32, dt):  # float32 -> 16-bit float bits
    return _to_bf16(f32) if dt == "bf16" else np.ascontiguousarray(f32).astype(np.float16).view(np.uint16)


def _refprime(base_f, Ai, Bi, sA, sB, dt):
    """Deterministic reference' = round(base + Ai@Bi*sA*sB). Exact integer matmul, so it
    reconstructs bit-identically on any machine; the residual absorbs the rest."""
    L = (Ai.astype(np.int64) @ Bi.astype(np.int64)).astype(np.float64) * (sA * sB)
    return _f2u((base_f + L.astype(np.float32)).ravel().copy(), dt)


_LR_MAX_ELEMS = int(os.environ.get("WCODEC_LR_MAX_ELEMS", 64_000_000))


def _lowrank_enc(ft_u16, base_u16, m, n, perplane, level, dt, r_max=64):
    if m * n > _LR_MAX_ELEMS:
        return None  # bound rsvd working memory on very large matrices (embeddings etc.)
    base_f = _u2f(base_u16, dt).reshape(m, n).astype(np.float32)
    dW = _u2f(ft_u16, dt).reshape(m, n).astype(np.float32) - base_f
    if not np.isfinite(dW).all():
        return None  # NaN/Inf weights: skip low-rank, let elementwise modes handle it
    try:
        A, B, S = _rsvd(dW, min(r_max, min(m, n) - 1))
    except np.linalg.LinAlgError:
        return None
    tot = float(np.einsum("ij,ij->", dW, dW, dtype=np.float64))  # ||dW||_F^2, no f64 copy
    if tot <= 0:
        return None
    S64 = S.astype(np.float64)
    if float((S64 ** 2).sum() / tot) < 0.5:
        return None  # not low-rank enough; let elementwise modes handle it
    # adaptive rank = NUMERICAL rank: singular values above 1% of the largest. This is
    # the actual signal rank (rank-1 abliterations .. rank-64+ LoRAs); the bf16-rounding
    # noise floor (full-rank, tiny singulars) is left to the residual, which codes it far
    # more cheaply than spending factor bytes chasing 99.9% of a noisy energy budget.
    r_eff = max(1, min(int((S64 > 0.01 * S64[0]).sum()), S.size))
    A, B = A[:, :r_eff], B[:r_eff]
    # scales MUST be float32-exact: they are stored as float32 and the decoder
    # recomputes the reference with them, so encode must use the same rounded value
    sA = float(np.float32((float(np.abs(A).max()) or 1.0) / 32767.0))
    sB = float(np.float32((float(np.abs(B).max()) or 1.0) / 32767.0))
    Ai = np.round(A / sA).astype(np.int16)
    Bi = np.round(B / sB).astype(np.int16)
    refp = _refprime(base_f, Ai, Bi, sA, sB, dt)
    res = _arith_enc16(ft_u16, refp, perplane)
    cA = _ZSTD.compress(Ai.tobytes(), level)
    cB = _ZSTD.compress(Bi.tobytes(), level)
    return (struct.pack("<IIH", m, n, Ai.shape[1]) + struct.pack("<ff", sA, sB)
            + struct.pack("<II", len(cA), len(cB)) + cA + cB + res)


def _lowrank_dec(blob, base_u16, perplane, dt):
    m, n, r = struct.unpack("<IIH", blob[:10])
    sA, sB = struct.unpack("<ff", blob[10:18])
    lenA, lenB = struct.unpack("<II", blob[18:26])
    pos = 26
    Ai = np.frombuffer(_ZSTD.decompress(blob[pos:pos + lenA]), np.int16).reshape(m, r); pos += lenA
    Bi = np.frombuffer(_ZSTD.decompress(blob[pos:pos + lenB]), np.int16).reshape(r, n); pos += lenB
    base_f = _u2f(base_u16, dt).reshape(m, n).astype(np.float32)
    refp = _refprime(base_f, Ai, Bi, sA, sB, dt)
    return _arith_dec16(blob[pos:], refp.tobytes(), perplane)


# ----------------------------- per-tensor coders -----------------------------

def _sparse_blob(u, changed, cb):
    return struct.pack("<I", u.size) + struct.pack("<I", len(cb)) + cb + u[changed].tobytes()


def _enc_tensor(buf, stdt, ref_buf, codecs_t, codecs_p, level, probe_level, shape=None):
    """Encode one tensor's bytes. Returns (mode_byte, blob).

    Delta mode tries every applicable strategy and keeps the smallest -- so the codec is
    never worse than any single method and adapts per tensor/dtype. To stay fast at high
    levels, we PROBE the elementwise candidates at a cheap level to pick the winner, then
    compress only that winner at the target level; the low-rank candidate (for 2D bf16
    tensors whose delta is low-rank) is computed in full and compared by actual size.
    """
    smart, perplane = codecs_t
    smart_p, perplane_p = codecs_p
    dt = ST2DT.get(stdt)
    if dt is None:
        return M_RAW, _ZSTD.compress(buf, level)
    if ref_buf is not None and len(ref_buf) == len(buf):
        if ref_buf == buf:
            return M_COPY, b""
        u = np.frombuffer(buf, UVIEW[dt])
        r = np.frombuffer(ref_buf, UVIEW[dt])
        changed = u != r
        frac = float(changed.mean())
        x = (u ^ r).astype(UVIEW[dt])
        # low-rank candidate (full blob) for 2D 16-bit-float tensors whose delta is low-rank
        lr_blob = None
        if dt in ("bf16", "fp16") and shape is not None and len(shape) == 2 and min(shape) >= 16:
            lr_blob = _lowrank_enc(u, r, int(shape[0]), int(shape[1]), perplane, level, dt)
        applicable = [M_DELTA]
        if dt in ("bf16", "fp16", "fp32"):
            applicable.append(M_ARITH)
        if frac < _SPARSE_MAX:
            applicable.append(M_SPARSE)
        if len(applicable) == 1:
            best = M_DELTA  # only one elementwise strategy applies; skip the probe
        else:
            probe = [(M_DELTA, len(perplane_p.compress(x.tobytes(), dt)))]
            if M_ARITH in applicable:
                probe.append((M_ARITH, len(_arith_enc(u, r, perplane_p, dt))))
            if M_SPARSE in applicable:
                cb_p = _ZSTD.compress(np.packbits(changed).tobytes(), probe_level)
                probe.append((M_SPARSE, len(cb_p) + int(changed.sum()) * u.itemsize + 8))
            best = min(probe, key=lambda c: c[1])[0]
        if best == M_DELTA:
            mode, blob = M_DELTA, perplane.compress(x.tobytes(), dt)
        elif best == M_ARITH:
            mode, blob = M_ARITH, _arith_enc(u, r, perplane, dt)
        else:
            cb = _ZSTD.compress(np.packbits(changed).tobytes(), level)
            mode, blob = M_SPARSE, _sparse_blob(u, changed, cb)
        if lr_blob is not None and len(lr_blob) < len(blob):
            return M_LOWRANK, lr_blob
        return mode, blob
    return M_SINGLE, smart.compress(buf, dt)


def _dec_tensor(mode, blob, stdt, nbytes, ref_buf):
    if mode == M_COPY:
        return ref_buf
    if mode == M_RAW:
        return _ZSTD.decompress(blob)
    if mode == M_SINGLE:
        return _SMART.decompress(blob)
    dt = ST2DT[stdt]
    if mode == M_DELTA:
        x = np.frombuffer(_PERPLANE.decompress(blob), UVIEW[dt])
        r = np.frombuffer(ref_buf, UVIEW[dt])
        return (x ^ r).tobytes()
    if mode == M_ARITH:
        return _arith_dec(blob, ref_buf, _PERPLANE, dt)
    if mode == M_LOWRANK:
        return _lowrank_dec(blob, np.frombuffer(ref_buf, "<u2"), _PERPLANE, dt)
    if mode == M_SPARSE:
        n = struct.unpack("<I", blob[:4])[0]
        cblen = struct.unpack("<I", blob[4:8])[0]
        cb = blob[8:8 + cblen]
        vals = blob[8 + cblen:]
        changed = np.unpackbits(np.frombuffer(_ZSTD.decompress(cb), np.uint8))[:n].astype(bool)
        r = np.frombuffer(ref_buf, UVIEW[dt]).copy()
        r[changed] = np.frombuffer(vals, UVIEW[dt])
        return r.tobytes()
    raise ValueError(f"bad mode {mode}")


# ----------------------------- file-level API -----------------------------

_EQ_CHUNK = 1 << 24  # 16 MB: compare big tensors in blocks, never fully materialized


def _ref_lookup_from_file(ref_path):
    rraw, _, rheader, rdoff = parse(ref_path)  # mmap; slice tensors on demand
    idx = {name: (rdoff + b, rdoff + e) for name, dt, b, e in _tensors_in_order(rheader)}
    rmv = memoryview(rraw)

    def get(name):
        ent = idx.get(name)
        return rraw[ent[0]:ent[1]] if ent is not None else None

    def eq(name, src_mv, ab, ae):
        ent = idx.get(name)
        if ent is None or (ent[1] - ent[0]) != (ae - ab):
            return False
        rb = ent[0]
        for o in range(0, ae - ab, _EQ_CHUNK):
            end = min(o + _EQ_CHUNK, ae - ab)
            if src_mv[ab + o:ab + end] != rmv[rb + o:rb + end]:
                return False
        return True

    return get, eq


def _workers(n=None):
    if n is None:
        n = int(os.environ.get("WCODEC_WORKERS", 0)) or max(1, (os.cpu_count() or 2) - 1)
    return min(n, 16)


def compress_file(path, ref_path=None, level=19, ref_lookup=None, workers=None, ref_eq=None):
    raw, header_bytes, header, data_off = parse(path)
    tensors = _tensors_in_order(header)
    probe_level = min(level, 3)
    codecs_t = _codecs(level)
    codecs_p = _codecs(probe_level)
    if ref_lookup is None and ref_path:
        ref_lookup, ref_eq = _ref_lookup_from_file(ref_path)
    mode = 1 if ref_lookup is not None else 0
    raw_mv = memoryview(raw)

    def enc(t):
        name, stdt, b, e = t
        ab, ae = data_off + b, data_off + e
        # chunked copy check FIRST: detect unchanged tensors without materializing them
        if ref_eq is not None and ref_eq(name, raw_mv, ab, ae):
            return M_COPY, b""
        buf = raw[ab:ae]
        ref_buf = ref_lookup(name) if ref_lookup is not None else None
        shape = header[name]["shape"]
        return _enc_tensor(buf, stdt, ref_buf, codecs_t, codecs_p, level, probe_level, shape)

    nw = _workers(workers)
    # per-tensor encode parallelises well: numpy + zstd release the GIL
    if nw > 1 and len(tensors) > 1:
        with ThreadPoolExecutor(max_workers=nw) as ex:
            results = list(ex.map(enc, tensors))
    else:
        results = [enc(t) for t in tensors]

    out = bytearray()
    out += MAGIC
    out += struct.pack("<B", mode)
    out += hashlib.sha256(raw).digest()  # integrity hash of the ORIGINAL file
    out += struct.pack("<I", len(header_bytes)) + header_bytes
    out += struct.pack("<I", len(tensors))
    for m, blob in results:
        out += struct.pack("<B", m) + struct.pack("<Q", len(blob)) + blob
    return bytes(out)


def decompress_file(blob, ref_path=None, ref_lookup=None, out_path=None):
    assert blob[:4] == MAGIC, "bad magic"
    pos = 4
    (mode,) = struct.unpack("<B", blob[pos:pos + 1]); pos += 1
    digest = blob[pos:pos + 32]; pos += 32
    (hlen,) = struct.unpack("<I", blob[pos:pos + 4]); pos += 4
    header_bytes = blob[pos:pos + hlen]; pos += hlen
    header = json.loads(header_bytes)
    (nt,) = struct.unpack("<I", blob[pos:pos + 4]); pos += 4
    tensors = _tensors_in_order(header)
    assert len(tensors) == nt, "tensor count mismatch"

    if mode == 1:
        if ref_lookup is None and ref_path:
            ref_lookup, _ = _ref_lookup_from_file(ref_path)
        if ref_lookup is None:
            raise ValueError("delta container needs --ref")
    else:
        ref_lookup = None

    # parse per-tensor entries serially (cheap slicing), then decode in parallel
    entries = []
    for name, stdt, b, e in tensors:
        (m,) = struct.unpack("<B", blob[pos:pos + 1]); pos += 1
        (blen,) = struct.unpack("<Q", blob[pos:pos + 8]); pos += 8
        tblob = blob[pos:pos + blen]; pos += blen
        entries.append((name, stdt, b, e, m, tblob))

    head = struct.pack("<Q", hlen) + header_bytes
    data_len = max((e for _, _, _, e in tensors), default=0)

    def dec(entry):
        name, stdt, b, e, m, tblob = entry
        rb = ref_lookup(name) if ref_lookup is not None else None
        buf = _dec_tensor(m, tblob, stdt, e - b, rb)
        assert len(buf) == e - b, f"{name}: {len(buf)} != {e-b}"
        return b, e, buf

    nw = _workers()
    _BAD = ("integrity check failed: wrong reference or corrupt archive "
            "(decompressed bytes do not match the original SHA-256)")

    if out_path is not None:
        # stream tensors to disk in offset order: bounded memory + incremental SHA
        h = hashlib.sha256(); h.update(head)
        expect = 0
        ex = ThreadPoolExecutor(max_workers=nw) if (nw > 1 and len(entries) > 1) else None
        gen = ex.map(dec, entries) if ex else (dec(e) for e in entries)
        try:
            with open(out_path, "wb") as fo:
                fo.write(head)
                for b, e, buf in gen:
                    if b != expect:
                        raise ValueError("non-contiguous data section (use in-memory decode)")
                    fo.write(buf); h.update(buf); expect = e
        finally:
            if ex is not None:
                ex.shutdown()
        if h.digest() != digest:
            try:
                os.remove(out_path)
            except OSError:
                pass
            raise ValueError(_BAD)
        return None

    data = bytearray(data_len)
    if nw > 1 and len(entries) > 1:
        with ThreadPoolExecutor(max_workers=nw) as ex:
            for b, e, buf in ex.map(dec, entries):
                data[b:e] = buf
    else:
        for entry in entries:
            b, e, buf = dec(entry)
            data[b:e] = buf
    result = head + bytes(data)
    if hashlib.sha256(result).digest() != digest:
        raise ValueError(_BAD)
    return result


# ----------------------------- multi-shard model API -----------------------------
# Real LLMs ship sharded: model-00001-of-000NN.safetensors + model.safetensors.index.json.
# We compress each shard with the single-file codec; the delta reference is resolved by
# TENSOR NAME across all reference shards (read on demand, so memory stays bounded).

MAGIC_M = b"WCDM"


def parse_header(path):
    """Read only the safetensors header (not the data section)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


def shard_files(path):
    """Resolve a model path (file | dir | index.json) to an ordered list of shards."""
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return [path]
    if path.endswith(".index.json") and os.path.isfile(path):
        d = os.path.dirname(path)
        with open(path) as f:
            idx = json.load(f)
        names = sorted(set(idx["weight_map"].values()))
        return [os.path.join(d, n) for n in names]
    if os.path.isdir(path):
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(idx):
            return shard_files(idx)
        single = os.path.join(path, "model.safetensors")
        if os.path.isfile(single):
            return [single]
        import glob
        return sorted(glob.glob(os.path.join(path, "*.safetensors")))
    raise FileNotFoundError(f"no safetensors found at {path}")


class RefIndex:
    """Tensor-name -> (file, abs_begin, abs_end) over all reference shards; reads bytes
    on demand so a multi-GB reference never has to be fully resident."""

    def __init__(self, model_path):
        self.map = {}
        for f in shard_files(model_path):
            header, doff = parse_header(f)
            for name, dt, b, e in _tensors_in_order(header):
                self.map[name] = (f, doff + b, doff + e)

    def get(self, name):
        ent = self.map.get(name)
        if ent is None:
            return None
        path, b, e = ent
        with open(path, "rb") as f:
            f.seek(b)
            return f.read(e - b)

    def eq(self, name, src_mv, ab, ae):
        ent = self.map.get(name)
        if ent is None or (ent[2] - ent[1]) != (ae - ab):
            return False
        path, b, _ = ent
        with open(path, "rb") as f:
            f.seek(b)
            o = 0
            while o < ae - ab:
                end = min(o + _EQ_CHUNK, ae - ab)
                if src_mv[ab + o:ab + end] != f.read(end - o):
                    return False
                o = end
        return True


def compress_model(src_path, ref_path=None, level=19):
    """Compress a (possibly sharded) model into one archive. Returns bytes."""
    shards = shard_files(src_path)
    ref = RefIndex(ref_path) if ref_path else None
    out = bytearray()
    out += MAGIC_M
    out += struct.pack("<B", 1 if ref else 0)
    out += struct.pack("<I", len(shards))
    for sh in shards:
        rel = os.path.basename(sh).encode("utf-8")
        cont = compress_file(sh, level=level, ref_lookup=(ref.get if ref else None),
                             ref_eq=(ref.eq if ref else None))
        out += struct.pack("<H", len(rel)) + rel
        out += struct.pack("<Q", len(cont)) + cont
    return bytes(out)


def decompress_model(blob, ref_path=None, out_dir=None):
    """Reconstruct each shard file byte-exact into out_dir. Returns {relname: bytes}."""
    assert blob[:4] == MAGIC_M, "bad model magic"
    pos = 4
    (mode,) = struct.unpack("<B", blob[pos:pos + 1]); pos += 1
    (nsh,) = struct.unpack("<I", blob[pos:pos + 4]); pos += 4
    ref = RefIndex(ref_path) if (mode == 1 and ref_path) else None
    if mode == 1 and ref is None:
        raise ValueError("delta archive needs --ref")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result = {}
    for _ in range(nsh):
        (nl,) = struct.unpack("<H", blob[pos:pos + 2]); pos += 2
        rel = blob[pos:pos + nl].decode("utf-8"); pos += nl
        (cl,) = struct.unpack("<Q", blob[pos:pos + 8]); pos += 8
        cont = blob[pos:pos + cl]; pos += cl
        lk = ref.get if ref else None
        if out_dir:
            # stream straight to disk (bounded memory; integrity-checked inside)
            decompress_file(cont, ref_lookup=lk, out_path=os.path.join(out_dir, rel))
            result[rel] = None
        else:
            result[rel] = decompress_file(cont, ref_lookup=lk)
    return result


# ----------------------------- CLI -----------------------------

def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _is_model(path):
    """A multi-file model (directory or index.json) rather than a single .safetensors."""
    return os.path.isdir(path) or path.endswith(".index.json")


def cmd_bench(args):
    import shutil
    import tempfile
    mode = "delta" if args.ref else "single"
    is_model = _is_model(args.input)
    if is_model:
        shards = shard_files(args.input)
        raw = sum(os.path.getsize(s) for s in shards)
        label = f"model:  {args.input}  ({len(shards)} shards, {raw/1e6:.1f} MB, mode={mode}, zstd-{args.level})"
    else:
        raw = os.path.getsize(args.input)
        label = f"file:   {os.path.basename(args.input)}  ({raw/1e6:.1f} MB, mode={mode}, zstd-{args.level})"
    mb = raw / 1e6
    t0 = time.perf_counter()
    comp = compress_model(args.input, args.ref, args.level) if is_model else \
        compress_file(args.input, args.ref, args.level)
    enc_t = time.perf_counter() - t0
    # round-trip via streaming to a temp dir: bounded memory, and a successful decode
    # (no integrity error) PROVES byte-exactness against the original's stored SHA-256
    tmp = tempfile.mkdtemp(prefix="wcbench_")
    ok = True
    t1 = time.perf_counter()
    try:
        if is_model:
            decompress_model(comp, args.ref, out_dir=tmp)
        else:
            decompress_file(comp, ref_path=args.ref, out_path=os.path.join(tmp, "out.safetensors"))
    except ValueError as exc:
        ok = False
        print(f"decode error: {exc}")
    dec_t = time.perf_counter() - t1
    shutil.rmtree(tmp, ignore_errors=True)
    print(label)
    if args.ref:
        print(f"ref:    {args.ref}")
    print(f"out:    {len(comp)/1e6:.1f} MB   save {(1-len(comp)/raw)*100:.1f}%   ratio {raw/len(comp):.2f}x")
    print(f"speed:  enc {mb/enc_t:.0f} MB/s   dec {mb/dec_t:.0f} MB/s")
    print(f"round-trip: {'OK (byte-exact)' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_compress(args):
    if _is_model(args.input):
        comp = compress_model(args.input, args.ref, args.level)
        raw = sum(os.path.getsize(s) for s in shard_files(args.input))
    else:
        comp = compress_file(args.input, args.ref, args.level)
        raw = os.path.getsize(args.input)
    with open(args.output, "wb") as f:
        f.write(comp)
    print(f"{args.input} -> {args.output}  "
          f"{raw/1e6:.1f} -> {len(comp)/1e6:.1f} MB ({(1-len(comp)/raw)*100:.1f}% saved)")


def cmd_decompress(args):
    blob = open(args.input, "rb").read()
    if blob[:4] == MAGIC_M:
        decompress_model(blob, args.ref, out_dir=args.output)
        print(f"{args.input} -> {args.output}/  (model restored)")
    else:
        decompress_file(blob, ref_path=args.ref, out_path=args.output)  # streams, integrity-checked
        print(f"{args.input} -> {args.output}  ({os.path.getsize(args.output)/1e6:.1f} MB restored)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="wcodec")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("compress", "decompress", "bench"):
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("--ref", default=None)
        sp.add_argument("--level", type=int, default=19, help="zstd level (1-22)")
        if name != "bench":
            sp.add_argument("-o", "--output", required=True)
    args = p.parse_args(argv)
    return {"compress": cmd_compress, "decompress": cmd_decompress, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
