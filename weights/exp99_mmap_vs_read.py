"""Prereg #99 - is C-23's 1.82x streaming gap MMAP PAGE-FAULT OVERHEAD, or intrinsic?

WHERE THIS COMES FROM. C-23 measured llama.cpp streaming a 37 GB GGUF at 0.2505 GB/s effective
while raw reads on the same drive measure 0.452-0.459 (D-28). The mechanism was left
undetermined ON PURPOSE, because the experiment that produced it could not separate a per-fault
cost from plain lower throughput - fault counts were derived from byte counts, so the two models
are the same equation in different units.

The obvious next step was airllm (E-11): a second runtime that loads with explicit read() rather
than mmap. It needs a multi-GB safetensors download and a quiet box. THIS IS CHEAPER AND MORE
DIRECT: llama.cpp mmaps the GGUF and touches pages as tensors are needed. If mmap page faults
are the cost, a bare mmap page-touch loop in Python will show the same gap against read() on the
SAME FILE, SAME DRIVE, with no model, no runtime, and no inference in the way.

2x2, because mechanism and pattern must not be confounded - D-28 already showed pattern is free
for read() at 512 MB granularity, but says nothing about 4 KB granularity:

                    SEQUENTIAL              SCATTERED
    read()          A                       C
    mmap+touch      B                       D

STAKED BEFORE RUNNING (2026-08-02):
  P1 MMAP IS THE MECHANISM: B/A >= 1.30 (sequential mmap materially slower than sequential
     read). Then llama.cpp's deficit is a property of HOW it reads, a runtime fix exists, and
     Law 4 should keep pricing the DEVICE bandwidth it already measures.
  P2 MMAP IS NOT THE MECHANISM: B/A <= 1.10. Then mmap is free and C-23's gap belongs to
     something llama.cpp does around the read - dequantisation interleaved with I/O, thread
     stalls, or per-tensor overhead - and Law 4 needs a streaming-efficiency term it does not
     have. This kills the hypothesis I currently favour.
  P3 otherwise INCONCLUSIVE, declared in advance.

  P4 (secondary, free): does 4 KB granularity cost anything read() does not pay at 512 MB?
     C/A >= 1.30 would mean D-28's "pattern is free" result does NOT extend downward, and the
     scattered-4K arm is the one that matters for expert access.

  P5 (the number to hit): if any arm lands near 0.25 GB/s it reproduces C-23's effective rate
     from first principles, with no model in the loop at all.

  KR-A CANNOT-VARY / OVERHEAD BOUND: a WARM mmap touch (same region twice) must exceed
     2.0 GB/s. This does double duty - it proves the harness distinguishes RAM from disk, AND
     it bounds the Python per-page interpreter overhead, which is the one way this experiment
     could manufacture a fake mmap penalty. If warm is slow, the loop itself is the bottleneck
     and NO arm is quotable.
  KR-B disjoint cold regions per arm, fixed offsets so a re-run reads the same bytes.
  KR-C every arm reads the SAME number of bytes. A ratio between arms that moved different
     volumes is not a bandwidth comparison.

  CONTENTION: an active agent adds disk traffic, which slows every arm roughly equally and
  biases ratios toward 1.0 - i.e. toward P2, AGAINST the hypothesis I favour. A P1 result under
  contention is therefore conservative; a P2 result would need a quiet re-run before it is
  trusted.

  python weights/exp99_mmap_vs_read.py
"""
from __future__ import annotations
import json, mmap, os, time

F = "D:/evo-compress-data/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf"
OUT = "weights/data/exp99_mmap_vs_read.json"
GB = 1 << 30
PAGE = 4096
SPAN = 2 * GB              # every arm moves exactly this many bytes (KR-C)
BLOCK = 1 << 24            # 16 MB for the read() arms


def read_seq(f, off, span):
    f.seek(off); left = span; t0 = time.perf_counter()
    while left > 0:
        b = f.read(min(BLOCK, left))
        if not b:
            break
        left -= len(b)
    return span - left, time.perf_counter() - t0


def read_scattered(f, base, span):
    """Same volume, but in PAGE-sized pread-alikes at scattered offsets inside the region."""
    n = span // PAGE
    step = 1_299_827 * PAGE            # coprime-ish stride: covers the region without repeats
    got = 0; t0 = time.perf_counter()
    for i in range(n):
        f.seek(base + (i * step) % (span - PAGE))
        got += len(f.read(PAGE))
    return got, time.perf_counter() - t0


def mmap_touch(mm, base, span, scattered):
    n = span // PAGE
    step = 1_299_827 * PAGE if scattered else PAGE
    acc = 0; t0 = time.perf_counter()
    for i in range(n):
        off = base + ((i * step) % (span - PAGE) if scattered else i * PAGE)
        acc += mm[off]                  # one byte per page = one fault when cold
    return n * PAGE, time.perf_counter() - t0, acc


def main():
    size = os.path.getsize(F)
    res = {"utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "file": os.path.basename(F), "size_gb": round(size / GB, 1),
           "span_gb": SPAN / GB, "page": PAGE, "arms": {}}
    # KR-B: disjoint fixed regions, well apart, all inside the file
    R = {"A_read_seq": 2 * GB, "B_mmap_seq": 6 * GB,
         "C_read_scattered": 10 * GB, "D_mmap_scattered": 14 * GB, "WARM": 20 * GB}
    assert max(R.values()) + SPAN < size, "regions do not fit in the file"

    with open(F, "rb", buffering=0) as f:
        gb, dt = read_seq(f, R["A_read_seq"], SPAN)
        res["arms"]["A_read_seq"] = round(gb / 1e9 / dt, 4)
        print(f"  A read()  sequential : {res['arms']['A_read_seq']:.4f} GB/s")

        gb, dt = read_scattered(f, R["C_read_scattered"], SPAN)
        res["arms"]["C_read_scattered"] = round(gb / 1e9 / dt, 4)
        print(f"  C read()  scattered  : {res['arms']['C_read_scattered']:.4f} GB/s   ({PAGE}B units)")

    with open(F, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            gb, dt, _ = mmap_touch(mm, R["B_mmap_seq"], SPAN, False)
            res["arms"]["B_mmap_seq"] = round(gb / 1e9 / dt, 4)
            print(f"  B mmap    sequential : {res['arms']['B_mmap_seq']:.4f} GB/s")

            gb, dt, _ = mmap_touch(mm, R["D_mmap_scattered"], SPAN, True)
            res["arms"]["D_mmap_scattered"] = round(gb / 1e9 / dt, 4)
            print(f"  D mmap    scattered  : {res['arms']['D_mmap_scattered']:.4f} GB/s")

            small = 256 << 20
            mmap_touch(mm, R["WARM"], small, False)                 # warm it
            gb, dt, _ = mmap_touch(mm, R["WARM"], small, False)     # measure the warm pass
            res["arms"]["WARM_mmap"] = round(gb / 1e9 / dt, 4)
            print(f"  WARM mmap re-touch   : {res['arms']['WARM_mmap']:.4f} GB/s   (KR-A, must be >2.0)")
        finally:
            mm.close()

    A, B = res["arms"]["A_read_seq"], res["arms"]["B_mmap_seq"]
    C, D = res["arms"]["C_read_scattered"], res["arms"]["D_mmap_scattered"]
    W = res["arms"]["WARM_mmap"]
    res["B_over_A"] = round(A / B, 3) if B else None      # >1 means mmap is SLOWER
    res["C_over_A"] = round(A / C, 3) if C else None
    res["D_over_A"] = round(A / D, 3) if D else None
    res["c23_target_gbs"] = 0.2505

    print(f"\n  mmap penalty  (A/B) : {res['B_over_A']}x")
    print(f"  4K-read penalty (A/C): {res['C_over_A']}x")
    print(f"  mmap+4K       (A/D) : {res['D_over_A']}x")

    if W <= 2.0:
        res["verdict"] = (f"UNINFORMATIVE - KR-A fired: warm mmap re-touch {W:.3f} GB/s <= 2.0. "
                          f"The Python per-page loop is itself the bottleneck, so any 'mmap "
                          f"penalty' here could be interpreter overhead. No arm is quotable.")
    elif res["B_over_A"] >= 1.30:
        res["verdict"] = (f"P1 SUPPORTED - sequential mmap is {res['B_over_A']}x slower than "
                          f"sequential read() on the same file and drive. Page-fault overhead is "
                          f"real, so C-23's gap is a property of HOW llama.cpp reads, a runtime "
                          f"fix is possible, and Law 4 should keep pricing device bandwidth.")
    elif res["B_over_A"] <= 1.10:
        res["verdict"] = (f"P2 SUPPORTED - mmap costs {res['B_over_A']}x, i.e. nothing. The "
                          f"hypothesis I favoured is REFUTED: C-23's 1.82x does not come from "
                          f"page faults, it comes from something llama.cpp does around the read, "
                          f"and Law 4 needs a streaming-efficiency term it does not have. "
                          f"Contention biases toward this result, so confirm on a quiet box "
                          f"before wiring anything.")
    else:
        res["verdict"] = (f"INCONCLUSIVE - mmap penalty {res['B_over_A']}x sits in the "
                          f"pre-declared dead band (1.10, 1.30). Neither reading may be claimed.")
    near = [k for k, v in res["arms"].items() if k != "WARM_mmap" and 0.20 <= v <= 0.31]
    if near:
        res["p5_note"] = (f"P5: arm(s) {near} land in 0.20-0.31 GB/s, reproducing C-23's 0.2505 "
                          f"effective rate with no model in the loop.")
        print("\n  " + res["p5_note"])
    print("\n" + res["verdict"])
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
