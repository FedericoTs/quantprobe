"""Acquire a representative dataset of LLM weight tensors for evaluation.

We compress the *raw bytes* of tensors, so we parse safetensors directly (8-byte
header length + JSON header + data section) and keep raw bytes + dtype label -- no
need to interpret the floats. Real trained models give real exponent/mantissa
distributions; we also derive a realistic bf16 variant from fp32/fp16 reals (modern
LLMs ship bf16), and add controlled synthetic tensors.

Writes weights/data/<id>.bin files and weights/data/manifest.json.
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Small, public, safetensors-format trained models (varied architectures/dtypes).
REAL_MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "model.safetensors"),  # fp32 encoder ~90MB
    ("EleutherAI/pythia-70m", "model.safetensors"),                   # small decoder LM
    ("prajjwal1/bert-tiny", "model.safetensors"),                     # tiny BERT
]

# safetensors dtype string -> (our label, element bytes)
ST_FLOAT = {"F32": ("fp32", 4), "F16": ("fp16", 2), "BF16": ("bf16", 2)}

PER_MODEL_BUDGET = 6_000_000  # bytes of selected fp32-equivalent tensors per model


def parse_safetensors(path: str):
    """Yield (name, dtype_str, shape, raw_bytes) for each tensor."""
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hlen))
        data = fh.read()
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        yield name, meta["dtype"], meta["shape"], data[a:b]


def to_bf16(raw: bytes, dtype_str: str) -> bytes:
    """Convert real fp32/fp16 bytes to bf16 (round-to-nearest-even)."""
    if dtype_str == "F32":
        u32 = np.frombuffer(raw, dtype=np.uint32)
    else:  # F16
        f32 = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        u32 = f32.view(np.uint32)
    bf16 = ((u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
    return bf16.tobytes()


def _write(items, idx, source, kind, dtype, shape, raw):
    path = os.path.join(DATA, f"{idx:04d}.bin")
    with open(path, "wb") as fh:
        fh.write(raw)
    items.append({
        "id": f"{idx:04d}", "source": source, "kind": kind, "dtype": dtype,
        "shape": list(shape), "nbytes": len(raw), "path": os.path.relpath(path, HERE),
    })


def fetch_real(items):
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        print("  huggingface_hub unavailable; skipping real models")
        return
    for repo, fname in REAL_MODELS:
        try:
            print(f"  downloading {repo}/{fname} ...")
            path = hf_hub_download(repo_id=repo, filename=fname)
        except Exception as exc:
            print(f"    skip {repo}: {type(exc).__name__}: {exc}")
            continue
        # select largest float tensors up to the per-model budget
        tensors = [(n, dt, sh, raw) for n, dt, sh, raw in parse_safetensors(path)
                   if dt in ST_FLOAT]
        tensors.sort(key=lambda t: len(t[3]), reverse=True)
        budget = 0
        for name, dt, shape, raw in tensors:
            if len(raw) < 4096:
                continue
            label, _ = ST_FLOAT[dt]
            scale = 4 / ST_FLOAT[dt][1]  # normalize to fp32-equivalent for budgeting
            if budget + len(raw) * scale > PER_MODEL_BUDGET:
                continue
            budget += len(raw) * scale
            _write(items, len(items), repo, "real", label, shape, raw)
            # realistic bf16 variant from fp32/fp16 reals
            if dt in ("F32", "F16"):
                _write(items, len(items), repo, "real-bf16", "bf16", shape, to_bf16(raw, dt))
        print(f"    {repo}: selected ~{budget/1e6:.1f} MB (fp32-equiv)")


def gen_synthetic(items, seed=0):
    rs = np.random.RandomState(seed)
    specs = [
        ("gauss-small", -2.2, -0.8, 0.005),   # tight weights
        ("gauss-wide", -1.5, 0.0, 0.01),      # wider magnitudes
        ("outliery", -2.0, -0.5, 0.05),       # heavy-tailed (5% outliers x4-15)
    ]
    for nm, lo, hi, outfrac in specs:
        per = 800_000
        std = float(10 ** rs.uniform(lo, hi))
        w = rs.standard_normal(per).astype(np.float32) * std
        nout = int(per * outfrac)
        oi = rs.randint(0, per, nout)
        w[oi] *= rs.uniform(4, 15, nout).astype(np.float32)
        u32 = w.view(np.uint32)
        bf16 = ((u32.astype(np.uint64) + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
        _write(items, len(items), f"synthetic:{nm}", "synthetic", "bf16", [per], bf16.tobytes())
        _write(items, len(items), f"synthetic:{nm}", "synthetic", "fp32", [per], w.tobytes())


def main():
    os.makedirs(DATA, exist_ok=True)
    items: list = []
    print("== fetching real models ==")
    fetch_real(items)
    print("== generating synthetic ==")
    gen_synthetic(items)
    manifest = os.path.join(DATA, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({"items": items}, fh, indent=2)
    by_dtype: dict = {}
    for it in items:
        d = by_dtype.setdefault(it["dtype"], [0, 0])
        d[0] += 1
        d[1] += it["nbytes"]
    print(f"\nmanifest: {len(items)} tensors, {sum(i['nbytes'] for i in items)/1e6:.1f} MB total")
    for dt, (cnt, nb) in sorted(by_dtype.items()):
        print(f"  {dt:<6} {cnt:>3} tensors  {nb/1e6:>7.1f} MB")
    print(f"  -> {manifest}")


if __name__ == "__main__":
    main()
