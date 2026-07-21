"""CLI to evaluate weight codecs on the dataset and record results.

  python -m weights.run_eval --all            # evaluate every baseline + record
  python -m weights.run_eval --diagnose       # entropy diagnostics (where the bits are)
  python -m weights.run_eval --codec NAME      # evaluate one registered codec

Headline KPI = size-weighted save% with byte-exact round-trip; bf16 reported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import codecs as codecs_mod  # noqa: E402
from weights import evaluate as ev  # noqa: E402
from weights import track  # noqa: E402


def print_result(r: dict) -> None:
    ov = r["overall"]
    print(f"\n{r['codec']:<22} overall: save {ov['save_pct']:>6.2f}%  ratio {ov['ratio']:>6.3f}  "
          f"enc {ov['enc_MBps']:>6.1f}  dec {ov['dec_MBps']:>7.1f} MB/s  "
          f"rt={'OK' if ov['rt_ok'] else 'FAIL'}")
    for dt, m in r["by_dtype"].items():
        print(f"    {dt:<6} save {m['save_pct']:>6.2f}%  ratio {m['ratio']:>6.3f}  "
              f"dec {m['dec_MBps']:>7.1f} MB/s  rt={'OK' if m['rt_ok'] else 'FAIL'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="evaluate weight codecs")
    ap.add_argument("--all", action="store_true", help="evaluate all baseline codecs")
    ap.add_argument("--codec", help="evaluate one named baseline codec")
    ap.add_argument("--diagnose", action="store_true", help="entropy diagnostics")
    ap.add_argument("--dtypes", default=None, help="comma list, e.g. bf16,fp32")
    ap.add_argument("--record", action="store_true", help="record to leaderboard + log")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args(argv)

    items = ev.load_manifest()
    dtypes = args.dtypes.split(",") if args.dtypes else None
    tot = sum(it["nbytes"] for it in items)
    print(f"dataset: {len(items)} tensors, {tot/1e6:.1f} MB"
          + (f" (dtypes={dtypes})" if dtypes else ""))

    if args.diagnose:
        diag = ev.diagnose(items)
        print("\n== diagnostics: per-plane entropy (bits/byte) & structure ==")
        for dt, d in diag.items():
            print(f"\n{dt}: {d['n_elems']:,} elems, {d['raw_bytes']/1e6:.1f} MB raw")
            for plane, m in d["planes"].items():
                xo = d["xor_delta_planes"].get(plane, {})
                xnote = f"   xor-delta h0={xo.get('h0','-')}" if xo else ""
                print(f"    {plane}: h0={m['h0']:>5.2f}  h1={m['h1']:>5.2f} bits/byte{xnote}")
            print(f"    -> order-1 floor: save {d['order1_floor_save_pct']}% "
                  f"({d['order1_floor_bytes']:,} bytes)")
        return 0

    factories = {c.name: c for c in codecs_mod.baseline_codecs()}
    if args.all:
        chosen = list(factories.values())
    elif args.codec:
        if args.codec not in factories:
            print(f"unknown codec {args.codec}; known: {sorted(factories)}", file=sys.stderr)
            return 2
        chosen = [factories[args.codec]]
    else:
        print("specify --all, --codec NAME, or --diagnose", file=sys.stderr)
        return 2

    results = []
    for c in chosen:
        r = ev.evaluate_codec(c, items, dtypes=dtypes)
        print_result(r)
        results.append(r)
        if args.record and not args.no_record:
            track.record(r, note="baseline" if c.name in factories else "")

    # leaderboard snapshot for this run
    results.sort(key=lambda r: r["overall"]["save_pct"], reverse=True)
    print("\n== ranking (by overall save%) ==")
    for r in results:
        bf = r["by_dtype"].get("bf16", {})
        print(f"  {r['codec']:<22} {r['overall']['save_pct']:>6.2f}%  "
              f"(bf16 {bf.get('save_pct','-')}%)  dec {r['overall']['dec_MBps']} MB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
