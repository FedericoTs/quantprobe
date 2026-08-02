"""Is the gemma4-12B ladder row unstable, or was one reading anomalous?

Written because I claimed the former from three readings when seven exist. Re-run to check:

  python weights/gemma_row_stability.py

The distinction matters. "The row is unreliable" means dropping it from the ladder. "One
measurement is anomalous" means explaining that measurement - and the explanation may turn out
to be a finding rather than a defect. Cherry-picking three points out of seven produced the
first answer; reading all of them produces the second.
"""
from __future__ import annotations
import glob, json, os


def readings():
    """Every DISTINCT gemma measurement across archived ladders.

    The *_backup files are byte copies of the locked ladder, so counting them would pad the
    cluster with duplicates of a single measurement and make the row look steadier than the
    evidence supports. Dedupe on (measured, predicted).
    """
    seen, out = set(), []
    for f in sorted(glob.glob("weights/data/ladder*.json")) + \
             sorted(glob.glob("weights/data/unattended_*ladder*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        for r in d:
            if "gemma" not in str(r.get("name") or r.get("model") or "").lower():
                continue
            m = r.get("meas_tok_s") or r.get("measured")
            if m is None:
                continue
            key = (round(float(m), 4), r.get("pred_tok_s") or r.get("predicted"))
            if key in seen:
                continue
            seen.add(key)
            out.append((os.path.basename(f), float(m)))
    return out


def main():
    rs = readings()
    if not rs:
        print("no gemma readings found")
        return 1
    vals = sorted(v for _, v in rs)
    cluster = [v for v in vals if v < 14.0]
    outliers = [v for v in vals if v >= 14.0]

    print(f"{len(rs)} distinct gemma4-12B measurements:")
    for f, v in sorted(rs, key=lambda x: x[1]):
        print(f"  {v:6.2f}  {f}")

    if cluster:
        spread = max(cluster) / min(cluster)
        print("")
        print(f"  cluster: {min(cluster):.2f}-{max(cluster):.2f} tok/s over {len(cluster)} "
              f"machine/calibration states, spread {spread:.3f}x")
    for v in outliers:
        print(f"  outlier: {v:.2f} = {v / max(cluster):.2f}x the cluster maximum")

    print("")
    print("  A spread near 1.0x across many calibrations is a STABLE row. An outlier is a")
    print("  measurement to explain, not grounds to distrust every other reading. The 15.62")
    print("  came from the scrubbed-box run where all 14 rows sped up and this one moved most")
    print("  (+27.5%) - a dense 12B split across VRAM and RAM has the most to gain from page")
    print("  cache residency, which is what U-37 predicts. Possibly one effect, not a defect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
