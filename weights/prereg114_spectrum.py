"""Pre-registration #114, stage A - does a low-rank correction buy more quality per byte than
precision? Measured in llama.cpp's own error metric, before any adapter file is built.

    python weights/prereg114_spectrum.py validate --src S.gguf --quant Q.gguf [--imatrix I] [--names ...]
    python weights/prereg114_spectrum.py measure [--out weights/data/prereg114.json]

The metric is the one llama-quantize minimises: per-weight squared error weighted by the
importance-matrix value of the weight's input column, d_j = in_sum2_j / count (the mean squared
activation of channel j). A rank-r correction B.A fitted to that metric removes exactly the top r
squared singular values of E.D^(1/2) (Eckart-Young), so the spectrum of the weighted error IS the
adapter's best case at every rank - no adapter has to be trained or written to know it.

The precision points come from llama.cpp's own quantizer (ggml_quantize_chunk in ggml-base.dll,
called in-process) from the same build that wrote #113's arms. `validate` proves that byte for byte
against a file llama-quantize wrote; `measure` refuses to run until that check has passed for both
formats it is anchored on.

Reads only: the Q8_0 source, the importance matrix, and (validate) one quantized file. Writes one
JSON. CPU only.
"""

import argparse
import ctypes
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gguf
from gguf import quants

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = "C:/Users/Federico/Documents/evo-compress/tools/llamacpp-b10098"  # #113's quantizer build
SRC = "D:/evo-compress-data/gguf/Qwen3.8-27B-Q8_0.gguf"
IMAT = "D:/evo-compress-data/prereg113/qwen38-27b-wikitrain-128.imatrix.gguf"
ARM_Q2 = "D:/evo-compress-data/prereg113/Qwen3.8-27B-SPREAD-I.gguf"  # blk.5 FFN is Q2_K here
ARM_Q4 = "D:/evo-compress-data/prereg113/Qwen3.8-27B-OURS-I.gguf"  # blk.51 FFN is Q4_K here
OUT = os.path.join(HERE, "data", "prereg114.json")

T = gguf.GGMLQuantizationType
FORMATS = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]  # the precision ladder, all with the imatrix
SPECTRA = ["Q2_K", "Q2_K-noimat", "Q3_K"]  # whose weighted error gets an eigen-spectrum
BLOCKS_BAND = [51, 54, 57, 60, 63]
BLOCKS_OUT = [5, 15, 25, 35, 45]
KINDS = ["ffn_gate", "ffn_up", "ffn_down"]
RANKS = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512]
THREADS = max(1, (os.cpu_count() or 4) - 1)


# ---------------------------------------------------------------------------- llama.cpp's quantizer
def load_ggml(llama_dir=LLAMA):
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.abspath(llama_dir))
    lib = ctypes.CDLL(os.path.join(llama_dir, "ggml-base.dll" if os.name == "nt" else "libggml-base.so"))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_row_size.restype = ctypes.c_size_t
    lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
    lib.ggml_quantize_init.restype = None
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    return lib


def quantize(lib, w, qtype, imatrix):
    """llama-quantize's own bytes for float32 matrix w (rows x n_per_row). Rows are independent,
    so splitting them across threads cannot change a byte."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    nrows, npr = w.shape
    rs = lib.ggml_row_size(int(qtype), npr)
    out = np.empty((nrows, rs), dtype=np.uint8)
    imp = None
    if imatrix is not None:
        imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
        assert imatrix.shape == (npr,)
        imp = imatrix.ctypes.data
    lib.ggml_quantize_init(int(qtype))
    step = max(1, -(-nrows // (THREADS * 4)))
    spans = [(r, min(nrows, r + step)) for r in range(0, nrows, step)]

    def work(span):
        r0, r1 = span
        n = lib.ggml_quantize_chunk(int(qtype), w.ctypes.data, out.ctypes.data, r0 * npr, r1 - r0, npr, imp)
        assert n == (r1 - r0) * rs, f"quantize_chunk wrote {n}, expected {(r1 - r0) * rs}"

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(work, spans))
    return out


# ---------------------------------------------------------------------------- files
def tensor_map(path):
    r = gguf.GGUFReader(path)
    return r, {t.name: t for t in r.tensors}


def as_f32(t):
    """A GGUF tensor as float32 (rows x n_per_row), dequantized exactly as llama.cpp does."""
    if t.tensor_type == T.F32:
        return np.asarray(t.data, dtype=np.float32).reshape(int(t.shape[1]), int(t.shape[0]))
    if t.tensor_type == T.F16:
        return np.asarray(t.data).view(np.float16).astype(np.float32).reshape(int(t.shape[1]), int(t.shape[0]))
    return quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32, copy=False)


def imatrix_values(itens, name):
    """llama-quantize's per-column weights for tensor `name`: in_sum2 / count, 1 where count is 0."""
    s, c = itens.get(name + ".in_sum2"), itens.get(name + ".counts")
    if s is None or c is None:
        return None
    sums = np.asarray(s.data, dtype=np.float32).ravel()
    cnt = float(np.asarray(c.data, dtype=np.float32).ravel()[0])
    return sums / cnt if cnt > 0 else np.ones_like(sums)


# ---------------------------------------------------------------------------- measurement
def bpw(lib, qtype, npr):
    return lib.ggml_row_size(int(qtype), npr) * 8.0 / npr


def adapter_bpw(r, m, n):
    """Bits per base weight of a rank-r LoRA on an m x n (rows x n_per_row) matrix, stored as
    llama.cpp allows: Q8_0 (8.5 bits) for a row length that is a multiple of 32, F16 otherwise.
    lora_a is r x n (rows of n), lora_b is m x r (rows of r)."""
    a_bits = 8.5 if n % 32 == 0 else 16.0
    b_bits = 8.5 if r % 32 == 0 else 16.0
    return (r * n * a_bits + m * r * b_bits) / (m * n)


def wsum_sq(x, d=None, chunk=512):
    """sum_ij d_j x_ij^2 in float64, a few rows at a time: a float64 copy of a whole 89M-weight
    matrix would be 700 MB, and a float32 sum over 89M terms loses digits the trace check needs."""
    d64 = None if d is None else np.asarray(d, dtype=np.float64)
    tot = 0.0
    for i in range(0, x.shape[0], chunk):
        xc = x[i:i + chunk].astype(np.float64)
        xc *= xc
        tot += float(xc.sum()) if d64 is None else float((xc @ d64).sum())
    return tot


def spectrum(e, d):
    """Eigenvalues (descending) of the Gram matrix of E.D^(1/2): the squared singular values, so
    their running sum is the error a rank-r weighted correction removes. Also returns the
    directly summed energy of E.D^(1/2), for the trace check."""
    et = e * np.sqrt(d).astype(np.float32)[None, :]
    g = et @ et.T if et.shape[0] <= et.shape[1] else et.T @ et
    lam = np.linalg.eigvalsh(g.astype(np.float64))[::-1]
    return np.clip(lam, 0.0, None), wsum_sq(et)


def one_tensor(lib, src, itens, name):
    t0 = time.time()
    w = as_f32(src[name])
    m, n = w.shape
    d = imatrix_values(itens, name)
    if d is None:
        raise SystemExit(f"VOID: the importance matrix has no entry for {name}")
    d = d.astype(np.float32)
    energy = wsum_sq(w, d)
    rec = {"name": name, "rows": m, "n_per_row": n, "weighted_energy": energy, "formats": {}, "spectra": {}}
    for f in FORMATS + ["Q2_K-noimat"]:
        qt = getattr(T, f.split("-")[0])
        q = quantize(lib, w, qt, None if f.endswith("noimat") else d)
        e = w - quants.dequantize(q, qt).astype(np.float32, copy=False)
        del q
        eps = wsum_sq(e, d)
        rec["formats"][f] = {"bpw": bpw(lib, qt, n), "eps": eps, "rel_db": float(10 * np.log10(eps / energy))}
        if f in SPECTRA:  # now, while this error exists: holding three would cost a gigabyte
            lam, tot = spectrum(e, d)
            if not np.isfinite(lam).all() or abs(lam.sum() - eps) > 1e-3 * eps or abs(tot - eps) > 1e-3 * eps:
                raise SystemExit(f"VOID: {name} {f}: Gram trace {lam.sum():.6g}, direct {tot:.6g}, "
                                 f"eps {eps:.6g} disagree")
            cum = np.cumsum(lam)
            rec["spectra"][f] = {"top": [float(cum[r - 1]) for r in RANKS if r <= len(cum)],
                                 "ranks": [r for r in RANKS if r <= len(cum)],
                                 "min_dim": int(min(m, n))}
        del e
    rec["adapter_bpw"] = {str(r): adapter_bpw(r, m, n) for r in RANKS}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def validate(a):
    lib = load_ggml(a.llama_dir)
    _, src = tensor_map(a.src)
    _, qf = tensor_map(a.quant)
    itens = tensor_map(a.imatrix)[1] if a.imatrix else {}
    names = a.names or [n for n in qf if ".ffn_" in n and qf[n].tensor_type in (T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K)][:a.n]
    ok = True
    for name in names:
        t = qf[name]
        d = imatrix_values(itens, name) if a.imatrix else None
        if a.imatrix and d is None:
            print(f"  {name}: no imatrix entry - cannot reproduce an imatrix build")
            ok = False
            continue
        mine = quantize(lib, as_f32(src[name]), t.tensor_type, d)
        theirs = np.asarray(t.data, dtype=np.uint8).reshape(mine.shape)
        same = np.array_equal(mine, theirs)
        ok &= same
        diff = "" if same else f" ({int((mine != theirs).any(axis=1).sum())}/{mine.shape[0]} rows differ)"
        print(f"  {name}: {t.tensor_type.name} {'IDENTICAL' if same else 'DIFFERENT'}{diff}")
    print("validation:", "PASS - in-process bytes equal llama-quantize's" if ok else "FAIL")
    return 0 if ok else 1


def measure(a):
    lib = load_ggml(a.llama_dir)
    for arm, name in ((ARM_Q2, "blk.5.ffn_down.weight"), (ARM_Q4, "blk.51.ffn_down.weight")):
        ns = argparse.Namespace(llama_dir=a.llama_dir, src=SRC, quant=arm, imatrix=IMAT, names=[name], n=1)
        if validate(ns) != 0:
            raise SystemExit(f"VOID: in-process quantization does not reproduce {os.path.basename(arm)}")
    _, src = tensor_map(SRC)
    _, itens = tensor_map(IMAT)
    res = json.load(open(a.out, encoding="utf-8")) if os.path.isfile(a.out) else {"tensors": {}}
    res["protocol"] = {"src": SRC, "imatrix": IMAT, "llama": os.path.basename(a.llama_dir), "formats": FORMATS,
                       "spectra": SPECTRA, "ranks": RANKS, "blocks_band": BLOCKS_BAND, "blocks_out": BLOCKS_OUT,
                       "validated": True}
    for blk in BLOCKS_BAND + BLOCKS_OUT:
        for kind in KINDS:
            name = f"blk.{blk}.{kind}.weight"
            if name in res["tensors"]:
                continue
            rec = one_tensor(lib, src, itens, name)
            rec["block"], rec["kind"], rec["band"] = blk, kind, blk in BLOCKS_BAND
            res["tensors"][name] = rec
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(res, f, indent=1)
                f.write("\n")
            q2 = rec["formats"]["Q2_K"]
            print(f"  {name:24s} Q2_K {q2['rel_db']:6.2f} dB  ({rec['seconds']}s)", flush=True)
    res["status"] = "complete"
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    print(f"done -> python weights/prereg114_score.py --results {a.out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--src", required=True)
    v.add_argument("--quant", required=True)
    v.add_argument("--imatrix")
    v.add_argument("--names", nargs="*")
    v.add_argument("--n", type=int, default=4)
    v.add_argument("--llama-dir", default=LLAMA)
    m = sub.add_parser("measure")
    m.add_argument("--out", default=OUT)
    m.add_argument("--llama-dir", default=LLAMA)
    a = ap.parse_args()
    return validate(a) if a.cmd == "validate" else measure(a)


if __name__ == "__main__":
    sys.exit(main())
