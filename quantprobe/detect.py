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
    # AMD / Intel, added for issue #1 (an RX 5700 XT owner got "GPU: none detected" and had to
    # hand-pass the exact 448 this table now carries). Spec-sheet peaks, same [table] convention
    # as above; the geta/gl hints are the generic-unknown values because NO eta has been
    # measured on RDNA/Arc backends here - resolve_gpu_eta's size-classed floor and the user's
    # own `calibrate` anchors do the honest work, exactly as they did for E-13's +0.1%.
    ("rx 9070 xt", 640, 0.45, 0.27), ("rx 9070", 640, 0.45, 0.27),
    ("rx 7900 xtx", 960, 0.45, 0.27), ("rx 7900 xt", 800, 0.45, 0.27),
    ("rx 7900 gre", 576, 0.45, 0.27), ("rx 7800 xt", 624, 0.45, 0.27),
    ("rx 7700 xt", 432, 0.45, 0.27), ("rx 7600 xt", 288, 0.45, 0.27),
    ("rx 7600", 288, 0.45, 0.27), ("rx 6950 xt", 576, 0.45, 0.27),
    ("rx 6900 xt", 512, 0.45, 0.27), ("rx 6800 xt", 512, 0.45, 0.27),
    ("rx 6800", 512, 0.45, 0.27), ("rx 6750 xt", 432, 0.45, 0.27),
    ("rx 6700 xt", 384, 0.45, 0.27), ("rx 6700", 320, 0.45, 0.27),
    ("rx 6650 xt", 280, 0.45, 0.27), ("rx 6600 xt", 256, 0.45, 0.27),
    ("rx 6600", 224, 0.45, 0.27), ("rx 5700 xt", 448, 0.45, 0.27),
    ("rx 5700", 448, 0.45, 0.27), ("rx 5600 xt", 288, 0.45, 0.27),
    ("radeon vii", 1024, 0.45, 0.27), ("vega 64", 484, 0.45, 0.27), ("vega 56", 410, 0.45, 0.27),
    ("arc a770", 560, 0.45, 0.27), ("arc a750", 512, 0.45, 0.27),
    ("arc b580", 456, 0.45, 0.27), ("arc b570", 380, 0.45, 0.27),
]
MAC_BW = {"m1 ultra": 800, "m1 max": 400, "m1 pro": 200, "m1": 68,
          "m2 ultra": 800, "m2 max": 400, "m2 pro": 200, "m2": 100,
          "m3 ultra": 819, "m3 max": 400, "m3 pro": 150, "m3": 100,
          "m4 max": 546, "m4 pro": 273, "m4": 120}


def _parse_rocm_text(text):
    """Parse rocm-smi combined text output into per-card dicts.

    Pure parser, testable without a GPU (same convention as _parse_win_adapters).
    Section state machine over GPU[n]-prefixed lines, keyword matching (case-insensitive
    substring; metric strings drift across rocm-smi versions). Returns [{name, vram_gb,
    sclk, mclk, temp, sclk_max}, ...], one dict per card.
    """
    cards = {}
    in_sclk_section = False
    for line in (text or "").strip().splitlines():
        m = re.match(r"GPU\[(\d+)\]\s*:\s*(.*)", line)
        if not m:
            continue
        gpu_id = int(m.group(1))
        rest = m.group(2).strip()
        if not rest:
            # Bare "GPU[n] :" — terminates any active section
            in_sclk_section = False
            continue
        cur = cards.setdefault(gpu_id, dict(name=None, vram_gb=None, sclk=None,
                                            mclk=None, temp=None, sclk_max=None))
        rest_lower = rest.lower()
        # Name: "Card series" contains the full product name (e.g. "AMD Radeon RX 9070 XT").
        if "card series" in rest_lower:
            cur["name"] = rest.split(":", 1)[1].strip()
        # VRAM total
        elif "vram total memory (b):" in rest_lower:
            try:
                cur["vram_gb"] = int(rest.split(":", 1)[1].strip()) / 2**30
            except (ValueError, IndexError):
                pass
        # VRAM used
        elif "vram total used memory (b):" in rest_lower:
            try:
                cur["vram_used_b"] = int(rest.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        # sclk clock level: "3: (1200.Mhz)" or "S: (238Mhz)" — level may be numeric or a letter.
        elif "sclk clock level:" in rest_lower:
            m = re.search(r"\((\d+)\.?Mhz\)", rest)
            if m:
                cur["sclk"] = int(m.group(1))
        # mclk clock level
        elif "mclk clock level:" in rest_lower:
            val = rest.split(":", 1)[1].strip()
            m = re.search(r"\((\d+)\.?Mhz\)", val)
            if m:
                cur["mclk"] = int(m.group(1))
        # Temperature (Sensor edge)
        elif "temperature (sensor edge)" in rest_lower:
            try:
                cur["temp"] = int(float(rest.split(":", 1)[1].strip()))
            except (ValueError, IndexError):
                pass
        # Supported sclk frequencies section header
        elif "supported sclk frequencies" in rest_lower:
            in_sclk_section = True
            continue
        # Inside sclk section: lines like "0: 400Mhz" or "12: 2400Mhz *"
        # The `*` marks the current active frequency (level may be numeric or state letter like "S:").
        if in_sclk_section:
            sm = re.match(r"^\S+\s*:\s*(\d+)\s*Mhz", rest)
            if sm:
                freq = int(sm.group(1))
                if cur["sclk_max"] is None or freq > cur["sclk_max"]:
                    cur["sclk_max"] = freq
                if "*" in rest:
                    cur["sclk"] = freq
    return [cards[k] for k in sorted(cards)]


def _rocm_state():
    """Run rocm-smi, return parsed list. [] on missing tool / no driver."""
    cmds = [
        ["rocm-smi", "--showclkfrq"],
        ["rocm-smi", "--showclocks"],
        ["rocm-smi", "--showmeminfo", "vram"],
        ["rocm-smi", "--showproductname"],
        ["rocm-smi", "--showtemp"],
    ]
    parts = []
    for cmd in cmds:
        out = _run(cmd)
        if out.strip():
            parts.append(out)
    if not parts:
        return []
    merged = "\n".join(parts)
    try:
        return _parse_rocm_text(merged)
    except (subprocess.SubprocessError, OSError):
        return []


def gpus_amd():
    """[(name, vram_gb)] via rocm-smi; empty list if no AMD GPU detected."""
    state = _rocm_state()
    return [(d["name"], d["vram_gb"]) for d in state if d["name"] and d["vram_gb"]]


def _price_gpus(gs):
    """Price a list of (name, vram_gb) tuples through gpu_lookup.

    Returns (known, unknown) where known/unknown are lists of
    (name, vram_gb, bw, geta, gl, src) — src contains 'table' for known cards.
    """
    priced = [(n, gb) + gpu_lookup(n) for n, gb in gs]
    known = [p for p in priced if "table" in p[5]]
    unknown = [p for p in priced if "table" not in p[5]]
    return known, unknown


def gpus():
    """[(name, vram_gb)] via nvidia-smi; empty list if none/AMD."""
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


def _parse_win_adapters(text):
    """'DriverDesc|bytes' lines -> [(name, vram_gb)], virtual adapters filtered, dedup by max.
    Pure parser so the smoke suite can test it without the registry."""
    VIRTUAL = ("basic display", "basic render", "remote", "virtual", "vnc", "dameware",
               "parsec", "spacedesk", "idd", "usb", "mirage", "citrix")
    out = {}
    for line in (text or "").strip().splitlines():
        if "|" not in line:
            continue
        name, _, raw = line.rpartition("|")
        name = name.strip()
        if not name or any(v in name.lower() for v in VIRTUAL):
            continue
        try:
            gb = int(raw.strip()) / 2**30
        except ValueError:
            gb = 0.0
        out[name] = max(gb, out.get(name, 0.0))
    return sorted(out.items())


def gpus_other():
    """Non-NVIDIA adapters, Windows: driver registry first - qwMemorySize is the reliable VRAM
    field; Win32_VideoController.AdapterRAM is a uint32 that CAPS AT 4 GB and under-reports
    every modern card - CIM only as fallback. Issue #1's contributor ran an RX 5700 XT and this
    tool printed 'GPU: none detected'; this function is the fix."""
    if os.name != "nt":
        return []
    ps = ("$k='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
          "{4d36e968-e325-11ce-bfc1-08002be10318}\\0*';"
          "$r=Get-ItemProperty $k -ErrorAction SilentlyContinue | "
          "Where-Object { $_.DriverDesc } | "
          "ForEach-Object { $_.DriverDesc + '|' + $_.'HardwareInformation.qwMemorySize' };"
          "if (-not $r) { $r = Get-CimInstance Win32_VideoController | "
          "ForEach-Object { $_.Name + '|' + $_.AdapterRAM } };"
          "$r")
    return _parse_win_adapters(_run(["powershell", "-NoProfile", "-c", ps]))


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


def ram_free_gb():
    """FREE physical RAM right now - not installed capacity. None if it cannot be read.

    Installed capacity is a property of the machine. Free RAM is a property of the MINUTE, and it
    is the one that decides whether a model's weights stay resident in page cache between decode
    passes. C-32: we published 14.86 +/- 0.36 tok/s for a 13.15 GiB file, and could not reproduce
    it days later on the same box with the same command, because nothing recorded that free RAM
    was 12.24 GB - less than the file. A tight error bar said nothing about it; the run was
    internally consistent and externally unrepeatable."""
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = _MS()
            m.dwLength = ctypes.sizeof(m)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return None
            return round(m.ullAvailPhys / 2**30, 2)
        # Linux: MemAvailable is the kernel's own estimate of what a new allocation can get.
        # MemFree is not the same thing and is routinely near zero on a healthy box.
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 2**20, 2)
    except Exception:
        return None
    return None


def residency(model_bytes, free_gb=None):
    """Can this model's weights stay page-cache resident? -> (fits, note).

    Law 4 prices bytes streaming from RAM. When the file is larger than free RAM, part of every
    decode pass comes off disk instead - a different resource, an order of magnitude slower, and
    one that changes minute to minute with whatever else the machine is doing. That is not a
    prediction that came out wrong; it is a regime the prediction does not cover, and the honest
    move is to say so rather than absorb it into an error bar.

    `fits` is None when free RAM could not be read - unknown is not the same as fine."""
    if not model_bytes:
        return None, None
    gb = ram_free_gb() if free_gb is None else free_gb
    if gb is None:
        return None, "free RAM: unreadable - residency unknown, so treat any tok/s as one sample"
    m = model_bytes / 2**30
    if m <= gb:
        return True, (f"free RAM {gb:.1f} GiB vs model {m:.1f} GiB - the weights fit, so this "
                      f"number should repeat")
    return False, (
        f"free RAM {gb:.1f} GiB vs model {m:.1f} GiB - THE MODEL DOES NOT FIT IN FREE RAM.\n"
        f"             Part of every pass streams from disk, so this figure describes this "
        f"minute's\n"
        f"             machine state, not the file. Expect it to move when something else is "
        f"running,\n"
        f"             and do not publish it without the free-RAM number beside it (C-32).")


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
    gs = gpus() or gpus_amd()
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
        # nvidia-smi saw nothing - check AMD via rocm-smi, then Windows driver registry for
        # Intel (or any GPU with no driver tools). Field case: issue #1, RX 5700 XT, "none detected".
        amd = gpus_amd()
        others = gpus_other() if not amd else []
        # NO `if gs:` guard around this chain. _price_gpus([]) returns ([], []), so the empty
        # case falls to the else and gets vram=0 + the "none detected" note like every other
        # dead end. Guarding it skipped BOTH on a machine with no GPU at all: hw came back
        # with no vram/vram_bw keys and no GPU line, which is how None hardware reached the
        # contribute payload once already (issue #1). CI caught it; a GPU box cannot.
        known, unknown = _price_gpus(amd or others)
        if known:
            vram = sum(p[1] for p in known)
            vram_bw = sum(p[2] for p in known) * (1.0 if len(known) == 1 else 0.85)
            names = " + ".join(p[0] for p in known)
            is_amd = bool(amd)
            hw.update(vram=round(vram, 1), vram_bw=round(vram_bw),
                      geta=known[0][3], gl=known[0][4])
            src_note = ("AMD via rocm-smi (amdgpu driver required)" if is_amd
                        else "non-NVIDIA path: VRAM from the driver registry, bandwidth from spec")
            notes.append(f"GPU: {names}, {vram:.0f} GB [os], {vram_bw:.0f} GB/s [table] - "
                         f"{src_note}; eta on this backend is UNVALIDATED here, so run "
                         f"`quantprobe calibrate` with your llama.cpp build to anchor it")
        elif unknown:
            names = ", ".join(f"{p[0]} ({p[1]:.0f} GB)" for p in unknown)
            hw.update(vram=0, vram_bw=0)
            notes.append(f"GPU: {names} detected [os] but not in the bandwidth table - pass "
                         f"--vram <GB> --vram-bw <GB/s> (spec sheet) to include the GPU tier; "
                         f"planning CPU-only until then")
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
