"""quantprobe hw — read the local machine's memory tiers, no questions asked.

Every printed value is tagged with its source:
  [os]      read from the operating system / driver (capacity, stick speed, GPU name)
  [table]   looked up from the device name (bandwidth spec, eta class)
  [default] a conservative fallback — override it with flags
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
    ("5090", 1792, 0.62, 0.42), ("4090", 1008, 0.62, 0.42), ("4080", 717, 0.6, 0.4),
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
    channels = max(1, min(sticks or 2, 8)) if sticks else 2
    if mts:
        ram_bw = round(channels * mts * 8 / 1000)   # theoretical peak, preset convention
        notes.append(f"RAM: {total:.0f} GB, {sticks} stick(s) @ {mts:.0f} MT/s [os] -> {ram_bw} GB/s peak "
                     f"(assumes {channels}-channel)")
    else:
        ram_bw = 48
        notes.append(f"RAM: {total:.0f} GB [os]; speed unknown -> 48 GB/s [default: DDR4-3000 dual, pass --ram-bw]")

    # GPU(s)
    gs = gpus()
    if gs:
        vram = sum(g[1] for g in gs)
        bw0, geta, gl, src = gpu_lookup(gs[0][0])
        vram_bw = bw0 * len(gs) * (1.0 if len(gs) == 1 else 0.85)   # aggregate w/ tensor-parallel loss
        names = " + ".join(g[0] for g in gs)
        notes.append(f"GPU: {names}, {vram:.0f} GB total [os], {vram_bw:.0f} GB/s {src}"
                     + (f" (x{len(gs)} aggregate, 0.85 TP efficiency [est])" if len(gs) > 1 else ""))
        hw.update(vram=vram, vram_bw=round(vram_bw), geta=geta, gl=gl)
    else:
        hw.update(vram=0, vram_bw=0)
        notes.append("GPU: none detected (nvidia-smi absent/empty; AMD/Intel: pass --vram/--vram-bw) [os]")

    # disk: class default; a real measured number needs `quantprobe hw --measure` (reads a large file)
    hw.update(ram=round(total), ram_bw=ram_bw, disk_bw=0.5)
    notes.append("disk: 0.5 GB/s [default: SATA-class; NVMe ~3.5, Gen4 ~7 - pass --disk-bw or run hw --measure]")
    return hw, notes


def measure_disk(path, mb=512):
    """Sequential read of a real file region, uncached-ish: the streaming pattern that matters."""
    import time
    size = os.path.getsize(path)
    span = min(mb * 1024 * 1024, size)
    off = max(0, size - span - (os.urandom(1)[0] % 7) * 1024 * 1024)  # tail region, jittered
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


def run(a):
    hw, notes = detect()
    print("quantprobe hw - this machine, as the law sees it\n")
    for n in notes:
        print("  " + n)
    if getattr(a, "measure", None):
        p = a.measure
        if os.path.isfile(p):
            bw = measure_disk(p)
            hw["disk_bw"] = round(bw, 2)
            print(f"  disk MEASURED on {os.path.basename(p)}: {bw:.2f} GB/s sequential [measured]")
        else:
            print(f"  --measure: file not found: {p}")
    flags = (f"--vram {hw['vram']:g} --vram-bw {hw['vram_bw']:g} --ram {hw['ram']:g} "
             f"--ram-bw {hw['ram_bw']:g} --disk-bw {hw['disk_bw']:g}")
    print(f"\n  equivalent flags (for sharing / estimating this box elsewhere):\n  {flags}")
    print("\n  every command now uses these automatically when you pass no hardware flags;")
    print("  pass --machine or explicit flags to estimate a DIFFERENT machine instead.")
    return hw
