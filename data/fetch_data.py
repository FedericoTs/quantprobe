"""Fetch / generate corpora and provide a TRAIN/TEST split helper.

Design goals:
  * RUNS OFFLINE for the default domain.  The synthetic generators are fully
    deterministic (seeded numpy) so corpora are reproducible byte-for-byte.
  * Real datasets (enwik8 text, Silesia binary) download on demand via ``--download``;
    failures degrade gracefully to a synthetic fallback so nothing blocks.

Domains:
  time-series    correlated float32 telemetry channels (DEFAULT)
  server-logs    synthetic Apache-combined access logs (text)
  genomic-fastq  synthetic FASTQ reads (ACGT + quality)
  ml-weights     float32 tensors with low-rank structure
  generic-text   enwik8 (download) or a Zipf-text fallback
  generic-binary Silesia corpus (download) or a structured-binary fallback

CLI:
  python data/fetch_data.py --domain time-series
  python data/fetch_data.py --domain generic-text --download --max-bytes 5000000
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = os.path.join(HERE, "corpora")

DOMAINS = [
    "time-series",
    "server-logs",
    "genomic-fastq",
    "ml-weights",
    "generic-text",
    "generic-binary",
]


def domain_dir(domain: str) -> str:
    return os.path.join(CORPORA, domain)


def _ensure(domain: str) -> str:
    d = domain_dir(domain)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Synthetic generators (deterministic)
# ---------------------------------------------------------------------------
def generate_time_series(n_files: int = 18, rows: int = 4096, channels: int = 5, seed: int = 0) -> str:
    """Each file = one device's telemetry: ``channels`` correlated float32 columns
    with trend + seasonality + noise, stored as interleaved (row-major) float32."""
    d = _ensure("time-series")
    for fi in range(n_files):
        rs = np.random.RandomState(seed * 1000 + fi)
        t = np.arange(rows)
        # a few shared latent factors drive cross-channel correlation
        n_factors = 2
        factors = np.zeros((n_factors, rows))
        for k in range(n_factors):
            period = rs.uniform(200, 900)
            factors[k] = np.sin(2 * np.pi * t / period + rs.uniform(0, 6.28))
        cols = []
        for c in range(channels):
            trend = rs.uniform(-1e-3, 1e-3) * t
            season = 5.0 * np.sin(2 * np.pi * t / rs.uniform(80, 400) + rs.uniform(0, 6.28))
            mix = factors.T @ rs.uniform(-3, 3, size=n_factors)
            noise = rs.normal(0, rs.uniform(0.05, 0.6), size=rows)
            level = rs.uniform(-50, 50)
            col = level + trend + season + mix + noise
            if c == channels - 1:
                # one near-constant channel (highly compressible after delta)
                col = level + np.round(noise * 0.1, 2)
            cols.append(col.astype(np.float32))
        arr = np.stack(cols, axis=1)  # (rows, channels)
        with open(os.path.join(d, f"device_{fi:02d}.bin"), "wb") as fh:
            fh.write(arr.tobytes())
    return d


def generate_server_logs(n_files: int = 12, lines: int = 4000, seed: int = 0) -> str:
    d = _ensure("server-logs")
    paths = ["/", "/index.html", "/api/v1/users", "/api/v1/orders", "/static/app.js",
             "/static/style.css", "/images/logo.png", "/health", "/login", "/favicon.ico"]
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        "curl/8.4.0",
        "python-requests/2.31.0",
    ]
    methods = ["GET", "GET", "GET", "POST", "PUT"]
    statuses = [200, 200, 200, 200, 304, 404, 500]
    for fi in range(n_files):
        rs = np.random.RandomState(seed * 2000 + fi)
        out = []
        for ln in range(lines):
            ip = f"{rs.randint(10,200)}.{rs.randint(0,255)}.{rs.randint(0,255)}.{rs.randint(1,254)}"
            ts = f"[{10+ln%20:02d}/Jun/2026:{rs.randint(0,24):02d}:{rs.randint(0,60):02d}:{rs.randint(0,60):02d} +0000]"
            method = methods[rs.randint(len(methods))]
            path = paths[rs.randint(len(paths))]
            status = statuses[rs.randint(len(statuses))]
            size = rs.randint(120, 80000)
            agent = agents[rs.randint(len(agents))]
            out.append(
                f'{ip} - - {ts} "{method} {path} HTTP/1.1" {status} {size} '
                f'"https://example.com{paths[rs.randint(len(paths))]}" "{agent}"'
            )
        with open(os.path.join(d, f"access_{fi:02d}.log"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(out) + "\n")
    return d


def generate_fastq(n_files: int = 10, reads: int = 2500, read_len: int = 150, seed: int = 0) -> str:
    d = _ensure("genomic-fastq")
    bases = np.array(list(b"ACGT"), dtype=np.uint8)
    for fi in range(n_files):
        rs = np.random.RandomState(seed * 3000 + fi)
        chunks = []
        for r in range(reads):
            seq = bases[rs.randint(0, 4, size=read_len)].tobytes().decode("ascii")
            # quality skewed high (Phred), correlated along the read
            base_q = rs.randint(30, 40)
            q = np.clip(base_q + (rs.randn(read_len) * 3).astype(int), 2, 40) + 33
            qual = bytes(q.astype(np.uint8)).decode("latin-1")
            chunks.append(f"@SRR{fi:03d}.{r} length={read_len}\n{seq}\n+\n{qual}\n")
        with open(os.path.join(d, f"reads_{fi:02d}.fastq"), "w", encoding="latin-1", newline="\n") as fh:
            fh.write("".join(chunks))
    return d


def generate_ml_weights(n_files: int = 10, size: int = 16384, seed: int = 0) -> str:
    """Low-rank-ish float32 weight blobs (structured, not pure noise)."""
    d = _ensure("ml-weights")
    for fi in range(n_files):
        rs = np.random.RandomState(seed * 4000 + fi)
        dim = int(np.sqrt(size))
        rank = max(2, dim // 8)
        a = rs.randn(dim, rank).astype(np.float32)
        b = rs.randn(rank, dim).astype(np.float32)
        w = (a @ b) * 0.1 + rs.randn(dim, dim).astype(np.float32) * 0.01
        with open(os.path.join(d, f"weights_{fi:02d}.bin"), "wb") as fh:
            fh.write(w.astype(np.float32).tobytes())
    return d


def _zipf_text(rs, n_words: int) -> str:
    vocab = [f"w{i:04d}" for i in range(2000)]
    ranks = np.arange(1, len(vocab) + 1)
    probs = 1.0 / ranks
    probs /= probs.sum()
    idx = rs.choice(len(vocab), size=n_words, p=probs)
    words = []
    for j, i in enumerate(idx):
        words.append(vocab[i])
        if j % 12 == 11:
            words.append(".\n")
    return " ".join(words)


def generate_generic_text_fallback(n_files: int = 6, words: int = 80000, seed: int = 0) -> str:
    d = _ensure("generic-text")
    for fi in range(n_files):
        rs = np.random.RandomState(seed * 5000 + fi)
        with open(os.path.join(d, f"text_{fi:02d}.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(_zipf_text(rs, words))
    return d


def generate_generic_binary_fallback(n_files: int = 6, size: int = 200000, seed: int = 0) -> str:
    d = _ensure("generic-binary")
    for fi in range(n_files):
        rs = np.random.RandomState(seed * 6000 + fi)
        # mixed regions: runs, structured ramps, and noise -> varied compressibility
        ramp = (np.arange(size // 3) % 251).astype(np.uint8)
        runs = np.repeat(rs.randint(0, 256, size // 30).astype(np.uint8), 10)[: size // 3]
        noise = rs.randint(0, 256, size - len(ramp) - len(runs)).astype(np.uint8)
        blob = np.concatenate([ramp, runs, noise]).astype(np.uint8)
        with open(os.path.join(d, f"binary_{fi:02d}.bin"), "wb") as fh:
            fh.write(blob.tobytes())
    return d


# ---------------------------------------------------------------------------
# Real downloads (graceful)
# ---------------------------------------------------------------------------
def _download(url: str, dest: str, timeout: int = 60) -> bool:
    try:
        print(f"  downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "evo-compress/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        return True
    except Exception as exc:  # network unavailable / blocked
        print(f"  download failed ({type(exc).__name__}: {exc})")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_enwik8(max_bytes: int = 0) -> bool:
    d = _ensure("generic-text")
    zip_path = os.path.join(d, "enwik8.zip")
    urls = ["https://mattmahoney.net/dc/enwik8.zip", "http://prize.hutter1.net/enwik8.zip"]
    ok = False
    for url in urls:
        if _download(url, zip_path):
            ok = True
            break
    if not ok:
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("enwik8", d)
        raw = os.path.join(d, "enwik8")
        if max_bytes:
            with open(raw, "rb") as fh:
                data = fh.read(max_bytes)
            # write a few shards so we get a real train/test split
            shard = max_bytes // 6
            for i in range(6):
                with open(os.path.join(d, f"enwik8_{i:02d}.txt"), "wb") as fh:
                    fh.write(data[i * shard : (i + 1) * shard])
            os.remove(raw)
        return True
    except Exception as exc:
        print(f"  extract failed ({exc})")
        return False
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def download_silesia() -> bool:
    d = _ensure("generic-binary")
    zip_path = os.path.join(d, "silesia.zip")
    urls = [
        "https://sun.aei.polsl.pl//~sdeor/corpus/silesia.zip",
        "http://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip",
    ]
    ok = any(_download(url, zip_path) for url in urls)
    if not ok:
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(d)
        return True
    except Exception as exc:
        print(f"  extract failed ({exc})")
        return False
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


# ---------------------------------------------------------------------------
# Public API used by experiments
# ---------------------------------------------------------------------------
def ensure_domain(domain: str, download: bool = False, max_bytes: int = 0, seed: int = 0) -> str:
    """Make sure a corpus exists for ``domain``; generate or download as needed.
    Returns the domain directory."""
    d = domain_dir(domain)
    if os.path.isdir(d) and any(os.scandir(d)):
        return d  # already populated

    if domain == "time-series":
        return generate_time_series(seed=seed)
    if domain == "server-logs":
        return generate_server_logs(seed=seed)
    if domain == "genomic-fastq":
        return generate_fastq(seed=seed)
    if domain == "ml-weights":
        return generate_ml_weights(seed=seed)
    if domain == "generic-text":
        if download and download_enwik8(max_bytes=max_bytes or 6_000_000):
            return domain_dir(domain)
        print("  using synthetic Zipf-text fallback (no enwik8)")
        return generate_generic_text_fallback(seed=seed)
    if domain == "generic-binary":
        if download and download_silesia():
            return domain_dir(domain)
        print("  using synthetic structured-binary fallback (no Silesia)")
        return generate_generic_binary_fallback(seed=seed)
    raise ValueError(f"unknown domain {domain!r}; choose from {DOMAINS}")


def list_files(domain: str) -> list[str]:
    d = domain_dir(domain)
    files = []
    for entry in sorted(os.scandir(d), key=lambda e: e.name):
        if entry.is_file() and not entry.name.endswith((".zip",)):
            files.append(entry.path)
    return files


def split_train_test(paths: list[str], train_frac: float = 0.7, seed: int = 0):
    """Deterministic disjoint split.  Sort, seeded-shuffle, slice."""
    paths = sorted(paths)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(paths))
    n_train = max(1, int(round(len(paths) * train_frac)))
    n_train = min(n_train, len(paths) - 1) if len(paths) > 1 else len(paths)
    train_idx = set(order[:n_train].tolist())
    train = [paths[i] for i in range(len(paths)) if i in train_idx]
    test = [paths[i] for i in range(len(paths)) if i not in train_idx]
    return train, test


def load_files(paths: list[str], max_bytes: int = 0) -> list[bytes]:
    out = []
    for p in paths:
        with open(p, "rb") as fh:
            data = fh.read()
        if max_bytes:
            data = data[:max_bytes]
        out.append(data)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fetch/generate corpora")
    ap.add_argument("--domain", default="time-series", choices=DOMAINS)
    ap.add_argument("--download", action="store_true", help="attempt real downloads")
    ap.add_argument("--max-bytes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    d = ensure_domain(args.domain, download=args.download, max_bytes=args.max_bytes, seed=args.seed)
    files = list_files(args.domain)
    total = sum(os.path.getsize(f) for f in files)
    print(f"domain={args.domain}  dir={d}")
    print(f"  {len(files)} files, {total:,} bytes total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
