"""Prereg #94 - does the disk probe measure the wrong ACCESS PATTERN?

C-21 measured the disk tier 30% faster than the law predicts. Inverting the term, the I/O
component is 1.485x quicker than modelled. Exactly two things explain that, and the single
datapoint cannot separate them:

  H1 ACCESS PATTERN. Effective bandwidth under llama.cpp's streaming is ~0.698 GB/s, while
     our probe - which jitters a 512 MB window to a RANDOM offset - reports 0.47 GB/s. If
     large contiguous reads beat scattered ones by ~1.485x on this drive, H1 explains the
     whole miss and the probe is measuring the pessimistic end of the drive's envelope.
  H2 MISS FRACTION. The page-cache miss fraction is ~0.429, not the modelled 0.637, because
     expert usage is skewed and hot experts never leave RAM. Nothing to do with the drive.

This script tests H1 ONLY, and can only ever REFUTE or SUPPORT it - H2 needs task #52.

STAKED BEFORE RUNNING (2026-08-01):
  P1  Arm SEQ (one contiguous 4 GB read) / Arm RND (8 x 512 MB at random offsets) >= 1.30x
      => H1 SUPPORTED: the probe's access pattern costs us real bandwidth.
  P2  ratio <= 1.10x => H1 REFUTED. The drive does not care about the pattern at this size,
      the probe is fine, and the 30% miss belongs to H2. Say so and go measure expert skew.
  P3  1.10x < ratio < 1.30x => INCONCLUSIVE. Report the number, claim nothing, and do NOT
      pick whichever hypothesis is more convenient.

  KILL RULE / CANNOT-VARY GUARD: arm WARM re-reads a region already in page cache and MUST
  exceed 2.0 GB/s. If WARM lands near SEQ and RND, the harness is emitting a constant and no
  verdict may be issued from any arm - the same trap that made the old disk probe report RAM.

  COLDNESS: every arm reads from a 39.7 GB file on a 16 GiB box, and arms are assigned
  DISJOINT regions so no arm warms another. Region assignment is fixed below, not random,
  so a re-run measures the same bytes.

  NOT TESTED HERE: llama.cpp's real pattern is neither purely sequential nor purely random -
  it mmaps and touches expert tensors scattered across the file. SEQ and RND bracket that
  behaviour, they do not reproduce it. A ratio >= 1.30x means the pattern MATTERS, not that
  sequential is what llama.cpp does.

  python weights/exp94_access_pattern.py
"""
from __future__ import annotations
import json, os, time

F = "D:/evo-compress-data/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf"
OUT = "weights/data/exp94_access_pattern.json"
GB = 1 << 30
CHUNK = 1 << 24


def read_span(f, off, nbytes):
    f.seek(off)
    left, t0 = nbytes, time.perf_counter()
    while left > 0:
        b = f.read(min(CHUNK, left))
        if not b:
            break
        left -= len(b)
    return (nbytes - left) / 1e9, time.perf_counter() - t0


def main():
    size = os.path.getsize(F)
    print(f"file {os.path.basename(F)}  {size/GB:.1f} GiB")
    # Disjoint regions, fixed so a re-run measures the same bytes.
    #   RND  draws inside [2 GB, 14 GB)      SEQ reads [18 GB, 22 GB)      WARM re-reads [30, 30.5)
    res = {"file": F, "size_gb": round(size / GB, 2), "arms": {}}

    with open(F, "rb", buffering=0) as f:
        # RND: 8 x 512 MB at scattered offsets - what measure_disk() does today
        gb, dt = 0.0, 0.0
        for i in range(8):
            off = 2 * GB + (i * 1_500_000_009) % (12 * GB - 512 * (1 << 20))
            g, t = read_span(f, off, 512 * (1 << 20))
            gb += g; dt += t
        res["arms"]["RND_scattered_512MB_x8"] = {"gb": round(gb, 3), "s": round(dt, 2),
                                                 "gbs": round(gb / dt, 3)}
        print(f"  RND  {gb:.2f} GB in {dt:.1f}s -> {gb/dt:.3f} GB/s")

        # SEQ: one contiguous 4 GB read - the opposite end of the envelope
        g, t = read_span(f, 18 * GB, 4 * GB)
        res["arms"]["SEQ_contiguous_4GB"] = {"gb": round(g, 3), "s": round(t, 2),
                                             "gbs": round(g / t, 3)}
        print(f"  SEQ  {g:.2f} GB in {t:.1f}s -> {g/t:.3f} GB/s")

        # WARM: read a small region twice; the second pass must be RAM or the harness is broken
        read_span(f, 30 * GB, 512 * (1 << 20))
        g, t = read_span(f, 30 * GB, 512 * (1 << 20))
        res["arms"]["WARM_recache_512MB"] = {"gb": round(g, 3), "s": round(t, 2),
                                             "gbs": round(g / t, 3)}
        print(f"  WARM {g:.2f} GB in {t:.1f}s -> {g/t:.3f} GB/s")

    rnd = res["arms"]["RND_scattered_512MB_x8"]["gbs"]
    seq = res["arms"]["SEQ_contiguous_4GB"]["gbs"]
    warm = res["arms"]["WARM_recache_512MB"]["gbs"]
    ratio = seq / rnd if rnd else None
    res["ratio_seq_over_rnd"] = round(ratio, 3) if ratio else None

    print()
    if warm <= 2.0:
        res["verdict"] = (f"UNINFORMATIVE - cannot-vary guard fired: WARM {warm:.3f} GB/s <= 2.0. "
                          f"The harness has not been shown to distinguish RAM from disk, so "
                          f"neither SEQ nor RND may be quoted.")
    elif ratio >= 1.30:
        res["verdict"] = (f"H1 SUPPORTED - SEQ/RND {ratio:.3f}x >= 1.30. Access pattern costs "
                          f"real bandwidth; measure_disk()'s random-offset draw reports the "
                          f"pessimistic end. This does NOT by itself close C-21: it shows the "
                          f"pattern matters, not that llama.cpp reads sequentially.")
    elif ratio <= 1.10:
        res["verdict"] = (f"H1 REFUTED - SEQ/RND {ratio:.3f}x <= 1.10. The drive does not care "
                          f"about the pattern at this size, the probe is sound, and C-21's 30% "
                          f"miss belongs to H2 (miss fraction / expert skew, task #52).")
    else:
        res["verdict"] = (f"INCONCLUSIVE - SEQ/RND {ratio:.3f}x sits between the staked 1.10 and "
                          f"1.30. Claim nothing; neither hypothesis is picked.")
    print(res["verdict"])
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
