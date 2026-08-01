"""quantprobe hw - read the local machine's memory tiers, no questions asked.

Every printed value is tagged with its source:
  [os]      read from the operating system / driver (capacity, stick speed, GPU name)
  [table]   looked up from the device name (bandwidth spec, eta class)
  [default] a conservative fallback - override it with flags
Bandwidths are THEORETICAL peaks (the law's eta absorbs realism, same convention as the presets).
Nothing is sent anywhere; this only reads local OS interfaces.
"""
from __future__ import annotations
import os, platform, re, shutil, subprocess


def _run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


# name-fragment -> (VRAM bandwidth GB/s, geta, gl). 1060 measured on the reference box; rest spec-sheet [table].
GPU_TABLE = [
    ("5090", 1792, 0.62, 0.42), ("5080", 960, 0.62, 0.42), ("5070 ti", 896, 0.62, 0.42),
    ("5070", 672, 0.6, 0.4), ("5060 ti", 448, 0.58, 0.38), ("5060", 448, 0.58, 0.38),
    ("4090", 1008, 0.62, 0.42), ("4080", 717, 0.6, 0.4),
    ("4070", 504, 0.55, 0.35), ("4060", 272, 0.5, 0.3),
    ("3090", 936, 0.6, 0.4), ("3080", 760, 0.58, 0.38), ("3070", 448, 0.52, 0.32),
    ("3060 ti", 448, 0.5, 0.3), ("3060", 360, 0.5, 0.3), ("3050", 224, 0.45, 0.28),
    ("2080", 448, 0.45, 0.25), ("2070", 448, 0.45, 0.25), ("2060", 336, 0.42, 0.22),
    ("1080", 320, 0.38, 0.06), ("1070", 256, 0.36, 0.05), ("1060", 192, 0.35, 0.04),
    ("a100", 1935, 0.7, 0.55), ("h100", 3350, 0.75, 0.6), ("rtx 6000", 960, 0.62, 0.42),
]
MAC_BW = {"m1 ultra": 800, "m1 max": 400, "m1 pro": 200, "m1": 68,
          "m2 ultra": 800, "m2 max": 400, "m2 pro": 200, "m2": 100,
          "m3 ultra": 819, "m3 max": 400, "m3 pro": 150, "m3": 100,
          "m4 max": 546, "m4 pro": 273, "m4": 120}


def gpus():
    """[(name, vram_gb)] via nvidia-smi; empty list if none/AMD (AMD: pass flags for now)."""
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    gs = []
    for line in out.strip().splitlines():
        if "," in line:
            name, mem = line.rsplit(",", 1)
            try:
                gs.append((name.strip(), float(mem) / 1024))
            except ValueError:
                pass
    return gs


def gpu_lookup(name):
    n = name.lower()
    for frag, bw, geta, gl in GPU_TABLE:
        if frag in n:
            return bw, geta, gl, "[table]"
    return 300, 0.45, 0.27, "[default: unknown GPU, pass --vram-bw]"


def ram_windows():
    """(total_gb[os], mts[os], channels[os]) via CIM (wmic fallback)."""
    ps = _run(["powershell", "-NoProfile", "-c",
               "$m=Get-CimInstance Win32_PhysicalMemory; "
               "($m|Measure-Object Capacity -Sum).Sum; "
               "($m|Select-Object -First 1).ConfiguredClockSpeed; ($m|Measure-Object).Count"])
    vals = [v.strip() for v in ps.strip().splitlines() if v.strip()]
    if len(vals) >= 3:
        try:
            return float(vals[0]) / 2**30, float(vals[1]), int(vals[2])
        except ValueError:
            pass
    return None, None, None


WIDE_CPU = ("threadripper", "epyc", "xeon w-3", "xeon(r) w9", "xeon(r) w7", "xeon gold",
            "xeon platinum", "xeon silver")


def ram_channels(sticks, cpu_name):
    """(channels, provenance) - CHANNEL COUNT IS NOT STICK COUNT.

    The first external replication (RTX 3090 + Ryzen 8600G, 4 DIMMs on dual-channel AM5) hit
    exactly this: `min(sticks, 8)` assumed 4-channel and quoted 173 GB/s where the platform
    delivers ~86 peak - a 2x input error that a correct Law 4 turned into a 2x wrong prediction.
    Consumer desktop platforms (AM4/AM5, LGA17xx/18xx) are DUAL-channel regardless of stick
    count; only HEDT/server parts go wider, and those we recognise by CPU name. When in doubt:
    2, plus a note - `quantprobe calibrate` measures the real stream and overrides all of this.

    THIS IS A FUNCTION so the rule can be tested. Before v1.24.0 the same logic was inline in
    `detect()` and its guard test re-implemented the WIDE list in the test body, which meant the
    test stayed green with the 2x bug restored (verified by mutation). A regression test that
    cannot see the code it guards is not a regression test.
    """
    cpu = (cpu_name or "").lower()
    if any(w in cpu for w in WIDE_CPU):
        n = max(1, min(sticks or 4, 8))
        return n, f"assumes {n}-channel [HEDT/server CPU detected]"
    n = min(sticks, 2) if sticks else 2
    if sticks and sticks > 2:
        return n, (f"dual-channel [consumer platform; {sticks} sticks does NOT mean {sticks} "
                   f"channels]")
    return n, f"assumes {n}-channel"


def detect():
    """Return the machine as quantprobe hardware kwargs + a provenance report."""
    sysname = platform.system()
    hw, notes = {}, []

    if sysname == "Darwin":
        mem = _run(["sysctl", "-n", "hw.memsize"]).strip()
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).lower()
        total = float(mem) / 2**30 if mem else 16.0
        bw = next((b for frag, b in sorted(MAC_BW.items(), key=lambda x: -len(x[0])) if frag in chip), 100)
        hw = dict(vram=round(total * 0.8), vram_bw=bw, ram=8, ram_bw=bw, disk_bw=3.5,
                  geta=0.26, gl=0.24)
        notes.append(f"Apple unified memory: {total:.0f} GB [os], {bw} GB/s [table, est eta 0.26 - unvalidated: bench me]")
        return hw, notes

    # RAM
    total, mts, sticks = (ram_windows() if os.name == "nt" else (None, None, None))
    if total is None and os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            kb = int(re.search(r"MemTotal:\s*(\d+)", f.read()).group(1))
        total, mts, sticks = kb / 2**20, None, None
    if total is None:
        total = 16.0; notes.append("RAM capacity: 16 GB [default - detection failed]")
    channels, chan_src = ram_channels(sticks, platform.processor())
    if mts:
        ram_bw = round(channels * mts * 8 / 1000)   # theoretical peak, preset convention
        notes.append(f"RAM: {total:.0f} GB, {sticks} stick(s) @ {mts:.0f} MT/s [os] -> {ram_bw} GB/s peak "
                     f"({chan_src}); the DELIVERED stream is far below peak - the one box we have "
                     f"measured ran 26.1 of a 48 GB/s peak (0.544; n=1 machine, so this is a "
                     f"datapoint, not a population). Run `quantprobe calibrate` to measure yours")
    else:
        ram_bw = 48
        notes.append(f"RAM: {total:.0f} GB [os]; speed unknown -> 48 GB/s [default: DDR4-3000 dual, pass --ram-bw]")

    # GPU(s)
    gs = gpus()
    if gs:
        vram = sum(g[1] for g in gs)
        per = [gpu_lookup(g[0]) for g in gs]                        # per-card lookup (mixed pairs differ)
        bw_sum = sum(p[0] for p in per)
        vram_bw = bw_sum * (1.0 if len(gs) == 1 else 0.85)          # aggregate w/ tensor-parallel loss
        geta, gl, src = per[0][1], per[0][2], per[0][3]             # eta class from the primary card
        names = " + ".join(g[0] for g in gs)
        notes.append(f"GPU: {names}, {vram:.0f} GB total [os], {vram_bw:.0f} GB/s {src}"
                     + (f" (x{len(gs)} per-card sum, 0.85 TP efficiency [est]; slower card gates its share)" if len(gs) > 1 else ""))
        hw.update(vram=vram, vram_bw=round(vram_bw), geta=geta, gl=gl)
    else:
        hw.update(vram=0, vram_bw=0)
        notes.append("GPU: none detected (nvidia-smi absent/empty; AMD/Intel: pass --vram/--vram-bw) [os]")

    # disk: class default; a real measured number needs `quantprobe hw --measure` (reads a large file)
    hw.update(ram=round(total), ram_bw=ram_bw, disk_bw=0.5)
    notes.append("disk: 0.5 GB/s [default: SATA-class; NVMe ~3.5, Gen4 ~7 - pass --disk-bw or run hw --measure]")
    return hw, notes


def probe_offset(size, span, rnd=None):
    """Where the next disk probe reads from. Uniform over the WHOLE file, not the tail.

    C-17: the old probe read a fixed 512 MB TAIL region jittered by at most 7 MB, so ~98.6% of
    the span overlapped between calls, and `buffering=0` does NOT bypass the OS page cache.
    Measured on this box: cold 0.44 GB/s, then 2.99 / 2.99 GB/s on re-reads - the warm number is
    RAM, not disk, and it shipped as a disk-tier input 6.8x too fast.

    Split out of `measure_disk` so the property can be tested WITHOUT a multi-gigabyte file and
    without timing anything: draw N offsets and check they span the file. The previous regression
    test needed a >2 GB fixture, silently skipped without one, and asserted only that repeated
    timings AGREE - which a fully page-cached file also satisfies, so it could not tell "fixed"
    from "warm every time". This one fails on the tail-jitter code by construction.
    """
    room = max(0, size - span)
    if room <= 0:
        return 0
    # 8 bytes, not 4. `os.urandom(4)` caps the draw at 2**32-1, so on any file larger than 4 GiB
    # the probe could never read past the 4 GiB mark: on a 20 GB GGUF - exactly the size class
    # the disk tier exists to model - 80% of the file was unreachable, and the reachable prefix
    # is the part a partial download or a header read has already warmed. The C-17 fix shipped
    # half-done and this is the other half; caught by the offset test in tests/smoke.py.
    r = rnd() if rnd else int.from_bytes(os.urandom(8), "big")
    return r % (room + 1)


def _one_read(path, mb):
    """One timed read of one random region. GB/s."""
    import time
    size = os.path.getsize(path)
    span = min(mb * 1024 * 1024, size)
    off = probe_offset(size, span)
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        f.seek(off)
        left = span
        while left > 0:
            chunk = f.read(min(1 << 24, left))
            if not chunk:
                break
            left -= len(chunk)
    dt = time.perf_counter() - t0
    return (span - left) / 1e9 / dt


def measure_disk(path, mb=512, samples=5, detail=False):
    """Disk read bandwidth, GB/s: the MINIMUM over `samples` disjoint random regions.

    WHY THE MINIMUM AND NOT ONE SAMPLE (prereg #97, measured 2026-08-01). The previous version
    took a single sample and told the USER, in this docstring, to "treat a single number above
    ~1.5 GB/s as evidence of a warm cache rather than a fast disk." Asking the caller to perform
    the check the code should do is not a guard. Hours after that text was written, verify.py
    caught it live: reads of [0.413, 3.171, 0.415] GB/s on one file, because an experiment had
    streamed 15 GB of it minutes earlier. A single draw can BE that middle sample.

    Measured, 8 draws per arm on a 13.7 GB GGUF:
      file with 73% deliberately warmed : 6 of 8 draws returned >1.5 GB/s, max 2.854 - RAM
                                          reported as disk, a 6.3x error
      file after deliberate eviction    : still 1 of 8 draws at 2.092 - a 4.7x error
      minimum over the draws            : 0.4499 and 0.4537 - both correct against the
                                          independent raw-read baseline of 0.452-0.459 (D-28)

    The second line is the one that matters: on this box a cold read cannot be *guaranteed* even
    after evicting 16 GB of an unrelated file, so single-sample probing is structurally
    unreliable rather than merely unlucky-after-a-download. Nothing reads FASTER than the device
    except cache, so the minimum is the estimator; a warm region can only push a sample up.

    NOT DONE HERE, deliberately: this does not nudge the number toward the ~0.25 GB/s that
    llama.cpp actually achieves while streaming (C-23). That gap is a RUNTIME inefficiency and
    belongs in the law, not in a probe whose job is to measure the DEVICE. Letting a
    mis-measured probe cancel an unmodelled runtime cost is the mutually-consistent-presets trap
    C-17 exists to warn about - two errors that cancel are still two errors.

    Cost: `samples` x `mb` of reading (default ~2.5 GB, a few seconds) on a once-per-machine
    calibration path. `detail=True` also returns the individual draws and how many look warm.
    """
    reads = [_one_read(path, mb) for _ in range(max(1, samples))]
    lo = min(reads)
    # POST-HOC (labelled as such): the staked spread test max/min > 2.0 was REFUTED - it fired
    # on both arms, because neither arm was truly cold. The warm FRACTION does discriminate
    # (1/8 evicted vs 7/8 warmed), but it was chosen after seeing those numbers and has not
    # been confirmed on an independent run. It is reported, never used to alter the estimate.
    warm = sum(1 for r in reads if r > 2.0 * lo)
    if detail:
        return lo, {"draws": [round(r, 4) for r in reads], "disk_gbs": round(lo, 4),
                    "warm_draws": warm, "samples": len(reads)}
    return lo


def run(a):
    hw, notes = detect()
    print("quantprobe hw - this machine, as the law sees it\n")
    for n in notes:
        print("  " + n)
    if getattr(a, "measure", None):
        p = a.measure
        if os.path.isfile(p):
            bw, info = measure_disk(p, detail=True)
            hw["disk_bw"] = round(bw, 2)
            print(f"  disk MEASURED on {os.path.basename(p)}: {bw:.2f} GB/s sequential [measured]")
            print(f"    minimum of {info['samples']} probes at random offsets: {info['draws']}")
            if info["warm_draws"]:
                print(f"    {info['warm_draws']} of {info['samples']} draws returned >2x the "
                      f"minimum - page cache, not disk. The minimum is\n    used and is the "
                      f"right number. Measured on a deliberately warmed 13.7 GB file, 6 of 8 "
                      f"single\n    draws came back above 1.5 GB/s (max 2.854, a 6.3x error) "
                      f"while the minimum stayed correct;\n    even after evicting 16 GB, 1 in 8 "
                      f"draws still hit cache. One sample was never safe. (#97)")
        else:
            print(f"  --measure: file not found: {p}")
    flags = (f"--vram {hw['vram']:g} --vram-bw {hw['vram_bw']:g} --ram {hw['ram']:g} "
             f"--ram-bw {hw['ram_bw']:g} --disk-bw {hw['disk_bw']:g}")
    print(f"\n  equivalent flags (for sharing / estimating this box elsewhere):\n  {flags}")
    print("\n  every command now uses these automatically when you pass no hardware flags;")
    print("  pass --machine or explicit flags to estimate a DIFFERENT machine instead.")
    return hw
