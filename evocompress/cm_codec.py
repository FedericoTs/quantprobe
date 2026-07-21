"""Python wrapper around the evolvable Rust context-mixing codec (`cmcore`).

Builds the crate on demand, runs compress/decompress as a subprocess, and verifies
byte-exact round-trip.  This is the bridge between the Rust evolvable artifact and
the Python evolution/evaluation harness.

Prize-size accounting (the Hutter metric):
    total = |compressed stream| + |decompressor executable|
The decompressor is the self-contained `cmcore` binary (online learning => no model
weights shipped).  On a small dev slice the binary dominates and the metric is
dominated by fixed overhead; at enwik9 scale the binary is negligible.  We report
both the compressed-only bpc and the prize-size view.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CM_DIR = os.path.join(ROOT, "cmcore")
_EXE_NAME = "cmcore.exe" if os.name == "nt" else "cmcore"
EXE = os.path.join(CM_DIR, "target", "release", _EXE_NAME)


def cargo_path() -> str | None:
    found = shutil.which("cargo")
    if found:
        return found
    cand = os.path.join(os.path.expanduser("~"), ".cargo", "bin",
                        "cargo.exe" if os.name == "nt" else "cargo")
    return cand if os.path.exists(cand) else None


def build(release: bool = True) -> str:
    cargo = cargo_path()
    if not cargo:
        raise RuntimeError("cargo not found; install Rust (rustup) to build cmcore")
    args = [cargo, "build"] + (["--release"] if release else [])
    subprocess.run(args, cwd=CM_DIR, check=True)
    return EXE


def ensure_built(rebuild: bool = False) -> str:
    if rebuild or not os.path.exists(EXE):
        build()
    return EXE


def decompressor_size() -> int:
    return os.path.getsize(ensure_built())


def _run(args: list[str]) -> None:
    subprocess.run([ensure_built(), *args], check=True)


def compress_file(src: str, dst: str) -> None:
    _run(["c", src, dst])


def decompress_file(src: str, dst: str) -> None:
    _run(["d", src, dst])


def evaluate_path(path: str) -> dict:
    """Compress + decompress a file with cmcore; verify round-trip; measure."""
    ensure_built()
    n = os.path.getsize(path)
    with tempfile.TemporaryDirectory() as td:
        cmp = os.path.join(td, "c")
        dec = os.path.join(td, "d")
        t0 = time.perf_counter()
        compress_file(path, cmp)
        t1 = time.perf_counter()
        decompress_file(cmp, dec)
        t2 = time.perf_counter()
        c = os.path.getsize(cmp)
        with open(path, "rb") as fh:
            h_src = hashlib.sha256(fh.read()).hexdigest()
        with open(dec, "rb") as fh:
            h_dec = hashlib.sha256(fh.read()).hexdigest()
    ok = h_src == h_dec
    dec_size = decompressor_size()
    enc_s = max(t1 - t0, 1e-9)
    dec_s = max(t2 - t1, 1e-9)
    return {
        "name": "cmcore",
        "in_bytes": n,
        "compressed": c,
        "decompressor": dec_size,
        "prize_total": c + dec_size,
        "bpc": 8.0 * c / n if n else 0.0,
        "bpc_with_decompressor": 8.0 * (c + dec_size) / n if n else 0.0,
        "roundtrip_ok": ok,
        "encode_MBps": (n / 1e6) / enc_s,
        "decode_MBps": (n / 1e6) / dec_s,
    }


def compress_bytes(data: bytes) -> bytes:
    ensure_built()
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in")
        dst = os.path.join(td, "out")
        with open(src, "wb") as fh:
            fh.write(data)
        compress_file(src, dst)
        with open(dst, "rb") as fh:
            return fh.read()


def decompress_bytes(blob: bytes) -> bytes:
    ensure_built()
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in")
        dst = os.path.join(td, "out")
        with open(src, "wb") as fh:
            fh.write(blob)
        decompress_file(src, dst)
        with open(dst, "rb") as fh:
            return fh.read()
