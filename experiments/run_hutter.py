"""Hutter-direction benchmark driver.

Builds the evolvable context-mixing codec (cmcore), evaluates it on an enwik8 slice
with byte-exact round-trip, compares it against the general-purpose baselines on the
SAME slice, and reports bits-per-character (bpc) plus the prize-size view. Also
prints the real frontier (the enwik9 record) for honest orientation.

  python -m experiments.run_hutter --slice enwik8_1mb
  python -m experiments.run_hutter --slice enwik8_4mb --rebuild

The headline metric is bpc on the slice (lower is better). The compressed-only bpc
is what scales to the prize; the decompressor term is shown for completeness (it is
negligible at enwik9 scale, dominant on a tiny slice).
"""

from __future__ import annotations

import argparse
import bz2
import datetime as _dt
import gzip
import json
import lzma
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evocompress import cm_codec  # noqa: E402

CORPORA = os.path.join(_ROOT, "data", "corpora", "generic-text")
RESULTS = os.path.join(_ROOT, "results")
LOG = os.path.join(_ROOT, "experiments", "EXPERIMENT_LOG.md")

# Real frontier, for orientation (factual where derivable):
#   enwik9 record fx2-cmix (2024): 110,793,128 B -> 0.886 bpc.
#   PAQ8 / cmix class on full enwik8 ~ 1.0-1.3 bpc (approximate, public benchmarks).
FRONTIER_NOTE = (
    "frontier: enwik9 record fx2-cmix = 0.886 bpc; PAQ8/cmix class ~1.0-1.3 bpc "
    "(full enwik8, approx). Our numbers are on a small slice -> not directly "
    "comparable, but the ordering vs general baselines is the signal."
)


def baseline_rows(data: bytes) -> list[dict]:
    n = len(data)
    out = []

    def add(name, blob):
        out.append({"name": name, "compressed": len(blob), "bpc": 8.0 * len(blob) / n,
                    "roundtrip_ok": True})

    add("gzip-9", gzip.compress(data, 9))
    add("bz2-9", bz2.compress(data, 9))
    add("lzma-9", lzma.compress(data, preset=9))
    try:
        import zstandard as z
        add("zstd-19", z.ZstdCompressor(level=19).compress(data))
    except Exception:
        pass
    try:
        import brotli
        add("brotli-11", brotli.compress(data, quality=11))
    except Exception:
        pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Hutter-direction cmcore benchmark")
    ap.add_argument("--slice", default="enwik8_1mb",
                    help="file name in data/corpora/generic-text or an absolute path")
    ap.add_argument("--rebuild", action="store_true", help="force cargo rebuild")
    args = ap.parse_args(argv)

    path = args.slice if os.path.isabs(args.slice) else os.path.join(CORPORA, args.slice)
    if not os.path.exists(path):
        print(f"slice not found: {path}\n"
              f"  run: python data/fetch_data.py --domain generic-text --download", file=sys.stderr)
        return 2

    print("building cmcore (release)...")
    cm_codec.ensure_built(rebuild=args.rebuild)

    with open(path, "rb") as fh:
        data = fh.read()
    n = len(data)
    print(f"slice: {os.path.basename(path)}  ({n:,} bytes)\n")

    cm = cm_codec.evaluate_path(path)
    rows = baseline_rows(data) + [cm]

    # leaderboard by bpc
    header = f"{'method':<14}{'compressed':>12}{'bpc':>9}{'ratio':>8}  round-trip"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: x["bpc"]):
        mark = "  <= evolvable codec" if r["name"] == "cmcore" else ""
        rt = "ok" if r.get("roundtrip_ok") else "FAIL"
        ratio = n / r["compressed"]
        print(f"{r['name']:<14}{r['compressed']:>12,}{r['bpc']:>9.4f}{ratio:>8.3f}  {rt}{mark}")

    print(f"\ncmcore prize-size view: compressed {cm['compressed']:,} B + decompressor "
          f"{cm['decompressor']:,} B = {cm['prize_total']:,} B "
          f"({cm['bpc_with_decompressor']:.4f} bpc incl. decompressor)")
    print(f"cmcore speed: encode {cm['encode_MBps']:.3f} MB/s, decode {cm['decode_MBps']:.3f} MB/s")
    print(f"\n{FRONTIER_NOTE}")

    best_baseline = min((r for r in rows if r["name"] != "cmcore"), key=lambda r: r["bpc"])
    gain = (best_baseline["bpc"] - cm["bpc"]) / best_baseline["bpc"] * 100.0
    verdict = (f"cmcore {cm['bpc']:.4f} bpc beats best baseline {best_baseline['name']} "
               f"{best_baseline['bpc']:.4f} by {gain:.1f}%"
               if cm["bpc"] < best_baseline["bpc"]
               else f"cmcore {cm['bpc']:.4f} did NOT beat {best_baseline['name']} {best_baseline['bpc']:.4f}")
    print(f"\n=== {verdict} ===")
    if not cm["roundtrip_ok"]:
        print("!! ROUND-TRIP FAILED -- candidate invalid")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "hutter_results.json"), "w", encoding="utf-8") as fh:
        json.dump({"slice": os.path.basename(path), "n_bytes": n,
                   "leaderboard": sorted(rows, key=lambda x: x["bpc"]),
                   "cmcore_detail": cm}, fh, indent=2)

    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"\n### {_dt.datetime.now():%Y-%m-%d %H:%M:%S} | HUTTER cmcore | slice={os.path.basename(path)}\n\n")
        fh.write(f"- cmcore bpc={cm['bpc']:.4f} (rt={'ok' if cm['roundtrip_ok'] else 'FAIL'}), "
                 f"best baseline {best_baseline['name']}={best_baseline['bpc']:.4f}\n")
        fh.write(f"- {verdict}\n")

    return 0 if cm["roundtrip_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
