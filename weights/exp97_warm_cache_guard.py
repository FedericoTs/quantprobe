"""Prereg #97 - make the warm-cache trap a GUARD instead of a sentence in a docstring.

THE DEFECT, demonstrated live on 2026-08-01. measure_disk()'s own docstring says: "treat a
single number above ~1.5 GB/s on a SATA-class device as evidence of a warm cache rather than
a fast disk (C-17, still open)." It asks the USER to perform the check the CODE should do.
Hours later verify.py caught exactly that: reads [0.413, 3.171, 0.415] GB/s on one file,
because exp96b had streamed 15 GB of it minutes earlier. The middle sample is RAM. A single
draw has a real chance of BEING that sample and shipping it as disk bandwidth - which is how
C-17's 6.8x error happened in the first place, through a different door.

The realistic user path is worse than our accident: `quantprobe calibrate --model X` right
after `fetch` downloaded X measures the page cache by construction.

PROPOSED CHANGE: draw N disjoint random regions, return the MINIMUM, and warn when
max/min exceeds a threshold. C-17 established the principle - the COLD number is the disk;
fast reads are cache. The minimum over disjoint regions is the best available estimate.

WHY MIN AND NOT MEDIAN, stated before measuring: a warm region inflates; nothing deflates a
read below true disk speed except a transient stall. If that assumption is wrong the min will
sit materially below a single cold sample, and P1's kill rule catches it.

EXPLICITLY NOT DOING: nudging this number toward 0.25 GB/s to make C-21's under-prediction go
away. C-23 measured llama.cpp streaming at 0.2505 GB/s against 0.452-0.459 for raw reads -
that gap is a RUNTIME inefficiency and belongs in the law, not in a probe that is supposed to
measure the DEVICE. Letting a mis-measured probe cancel an unmodelled runtime cost is exactly
the mutually-consistent-presets trap C-17 exists to warn about. Two errors that cancel are
still two errors.

STAKED BEFORE ANY CODE CHANGES (2026-08-01):
  P1  COLD FILE, NO COST: on a file with no warmed region, min-of-N is within 20% of a single
      sample. KILL RULE: if min-of-N sits >20% BELOW the single sample, the min is harvesting
      transient stalls rather than truth - say so and switch to median.
  P2  CONSTRUCTED FAILING INPUT: with a region deliberately warmed, the CURRENT single-sample
      probe must return > 1.5 GB/s on at least one draw out of 8 (proving the defect is real
      and not hypothetical), while min-of-N stays below 1.0 GB/s.
  P3  THE WARNING IS SPECIFIC: max/min > 2.0 fires on the warmed file and does NOT fire on the
      cold one. A warning that always fires is noise, not a guard.

  Run BEFORE the fix: P2 must FAIL against current code, or there is no defect to fix and the
  whole exercise is theatre.

  python weights/exp97_warm_cache_guard.py [--after]
"""
from __future__ import annotations
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantprobe import detect

# --- REVISION 2, after run 1 failed on MY design rather than on the world -------------------
# Run 1 scored P1 FAIL / P2 UNINFORMATIVE / P3 FAIL. Three flaws, all in the harness:
#   (1) P1 compared min to the MEAN of the draws - but the mean is inflated by exactly the
#       outlier under study (cold mean 0.8023 because one draw hit 2.8304). Against the MEDIAN
#       the min is 0.2% off and passes. A mis-specified metric, not a biased estimator.
#   (2) 3 GB warmed on a 37 GB file: a random 512 MB draw lands in it 8% of the time, 0.65
#       expected hits over 8 draws. The failing input was never actually constructed.
#   (3) P3 inherited (2) - the "warm" arm was barely warm, so it could not fire.
# FIXED HERE: median-based P1, and a 13.7 GB file with 10 GB warmed (73%) so the lever moves.
# THRESHOLDS ARE UNCHANGED - 20%, 1.5 GB/s, 1.0 GB/s, 2.0x all stand exactly as staked. Fixing
# a broken lever is legitimate (see #96 -> #96b); moving a goalpost after seeing a number is not.
BIG = "D:/evo-compress-data/gguf/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf"   # 13.7 GB
COLD_EVICTOR = "D:/evo-compress-data/gguf/Qwen3.5-35B-A3B-Q8_0.gguf"
OUT = "weights/data/exp97_warm_cache_guard.json"
WARM_SPAN = 10 << 30         # 10 GB of 13.7 = 73% of the file deliberately warmed
DRAWS = 8


def stream(path, nbytes, off=0):
    with open(path, "rb", buffering=0) as f:
        f.seek(off)
        left = nbytes
        while left > 0:
            b = f.read(1 << 24)
            if not b:
                break
            left -= len(b)


def single_draws(path, n=DRAWS):
    """What the CURRENT shipped probe does, n independent times."""
    return [round(detect.measure_disk(path), 4) for _ in range(n)]


def main():
    res = {"utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "after_fix": "--after" in sys.argv, "file": BIG}

    print("=== arm COLD: evict the test file, then draw ===")
    stream(COLD_EVICTOR, 16 << 30)          # push the test file out of page cache
    cold = single_draws(BIG)
    res["cold_draws"] = cold
    res["cold_min"], res["cold_max"] = min(cold), max(cold)
    res["cold_spread_x"] = round(max(cold) / min(cold), 3) if min(cold) else None
    print(f"  draws {cold}")
    print(f"  min {min(cold):.4f}  max {max(cold):.4f}  spread {res['cold_spread_x']}x")

    print("\n=== arm WARM: deliberately warm 3 GB of the test file, then draw ===")
    stream(BIG, WARM_SPAN, off=0)
    warm = single_draws(BIG)
    res["warm_draws"] = warm
    res["warm_min"], res["warm_max"] = min(warm), max(warm)
    res["warm_spread_x"] = round(max(warm) / min(warm), 3) if min(warm) else None
    print(f"  draws {warm}")
    print(f"  min {min(warm):.4f}  max {max(warm):.4f}  spread {res['warm_spread_x']}x")

    # score. P1 is measured against the MEDIAN: the mean is inflated by the very outlier the
    # experiment exists to study, which is what broke run 1.
    import statistics as st
    med = st.median(cold)
    p1 = abs(min(cold) - med) / med
    res["cold_median"] = round(med, 4)
    res["P1_min_vs_median_cold"] = round(p1, 4)
    res["P1"] = "PASS" if p1 <= 0.20 else f"FAIL (min sits {p1*100:.0f}% below the cold median)"

    hot = [x for x in warm if x > 1.5]
    res["P2_warm_draws_over_1p5"] = len(hot)
    if not hot:
        res["P2"] = ("UNINFORMATIVE - no draw exceeded 1.5 GB/s even with 3 GB warmed, so the "
                     "failing input was not constructed and nothing here tests the guard.")
    elif min(warm) < 1.0:
        res["P2"] = (f"PASS - {len(hot)}/{DRAWS} single draws returned >1.5 GB/s (max "
                     f"{max(warm):.3f}, i.e. RAM shipped as disk) while min-of-N held at "
                     f"{min(warm):.3f}.")
    else:
        res["P2"] = f"FAIL - even the minimum is warm ({min(warm):.3f} >= 1.0)."

    fires_warm = (res["warm_spread_x"] or 0) > 2.0
    fires_cold = (res["cold_spread_x"] or 0) > 2.0
    res["P3"] = ("PASS" if fires_warm and not fires_cold else
                 f"FAIL - warm fires {fires_warm}, cold fires {fires_cold}; a guard must "
                 f"discriminate, not always fire")

    res["verdict"] = ("DEFECT CONFIRMED, guard justified" if res["P2"].startswith("PASS")
                      and res["P3"] == "PASS" else "see individual predictions")
    print(f"\n  P1 {res['P1']}\n  P2 {res['P2']}\n  P3 {res['P3']}")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
