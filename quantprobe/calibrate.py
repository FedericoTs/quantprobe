"""quantprobe calibrate - measure THIS machine's constants instead of assuming ours.

Why this command exists: every constant this tool ships was measured on one reference box, and
pre-registration #59 proved that even on that box, fitted constants do not transfer across
models - let alone across machines. The law FORMS travel; the numbers do not. So on your
machine the honest move is to measure, not assume.

What it measures (each value tagged, nothing sent anywhere):
  [measured] RAM stream bandwidth   - a real sequential read, the pattern CPU decode is bound by.
                                      Spec-sheet MT/s overstates what decode gets by 30-60%.
  [measured] disk sequential read   - only with --model FILE (reads the file's tail region).
  [measured] GPU clock health       - idle + max clocks now; sustained-under-load if an anchor
                                      runs. Catches the STUCK BOOST STATE failure mode we
                                      diagnosed on the reference box (SM locked ~18% low at cool
                                      temps after hours of GPU churn; only a reboot clears it -
                                      pre-registrations #60/#61).
  [measured] pure-CPU anchor        - only with --model: llama-bench -ngl 0 on your file, stored
                                      as a reference point for plan's predicted-vs-measured line.

Results persist to ~/.quantprobe/calibration.json. `plan` (and everything built on it) uses the
measured RAM/disk numbers automatically whenever it would otherwise auto-detect, and labels them
[calibrated] instead of [table]/[default]. Delete the file to go back to detection.
"""
from __future__ import annotations
import json, os, platform, subprocess, threading, time

CAL_DIR = os.path.join(os.path.expanduser("~"), ".quantprobe")
CAL_PATH = os.path.join(CAL_DIR, "calibration.json")
STALE_DAYS = 60          # hardware does not drift, but drivers and RAM configs do


def _run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def measure_ram_stream(gb=1.0, passes=3):
    """Sequential read bandwidth of main memory, GB/s. Best of `passes`.

    numpy sum over a float32 array bigger than any cache: the same access pattern the CPU
    expert path is bound by. Falls back to None (with the reason) if numpy is missing or
    allocation fails - a partial calibration is still a calibration.
    """
    try:
        import numpy as np
    except ImportError:
        return None, "numpy not installed - RAM stream skipped (pip install numpy)"
    n = int(gb * 1e9 / 4)
    try:
        a = np.ones(n, dtype=np.float32)
    except MemoryError:
        return None, f"could not allocate {gb:g} GB - RAM stream skipped"
    # int64-view max, not float sum: float reductions in numpy can be COMPUTE-bound (measured
    # 9.3 GB/s on the reference box where the memory delivers 26) and would silently calibrate
    # numpy's ALU instead of the memory bus. Integer max is one SIMD pass, one read per byte -
    # it reproduces the reference box's independently measured 26.1 GB/s stream.
    v = a.view(np.int64)
    best = 0.0
    for _ in range(passes):
        t0 = time.perf_counter()
        v.max()
        dt = time.perf_counter() - t0
        best = max(best, (n * 4) / 1e9 / dt)
    del a
    return round(best, 2), None


def gpu_state():
    """Current + max clocks and temperature via nvidia-smi; None if no NVIDIA GPU."""
    out = _run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu",
                "--format=csv,noheader,nounits"])
    line = out.strip().splitlines()[0] if out.strip() else ""
    if "," not in line:
        return None
    try:
        name, sm, sm_max, mem, temp = [v.strip() for v in line.rsplit(",", 4)]
        return dict(name=name, sm=int(sm), sm_max=int(sm_max), mem=int(mem), temp=int(temp))
    except ValueError:
        return None


class ClockSampler:
    """Samples clocks.sm in a thread while a benchmark runs - the boost-state instrument."""

    def __init__(self, every=3.0):
        self.every, self.samples, self._stop = every, [], threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            st = gpu_state()
            if st:
                self.samples.append((st["sm"], st["temp"]))
            self._stop.wait(self.every)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)

    def sustained(self):
        """Median SM MHz over loaded samples (>1000 MHz), or None."""
        loaded = sorted(s for s, _ in self.samples if s > 1000)
        return loaded[len(loaded) // 2] if loaded else None


def boost_verdict(sustained, sm_max, temp):
    """The #60/#61 diagnostic as a one-line verdict."""
    if sustained is None or not sm_max:
        return None
    frac = sustained / sm_max
    if frac >= 0.87:
        return f"healthy: sustained {sustained} of {sm_max} MHz max [measured]"
    hot = temp is not None and temp >= 80
    if hot:
        return (f"THROTTLED: sustained {sustained} of {sm_max} MHz at {temp} C - thermal; "
                "check cooling")
    return (f"STUCK BOOST STATE: sustained {sustained} of {sm_max} MHz at cool temps - the "
            "failure mode measured in preregistrations #60/#61 (-25-30%% real speed). Consumer "
            "cards cannot reset it live: REBOOT, then re-run calibrate")


def cpu_anchor(model_path, llama_dir=None):
    """Pure-CPU tg32 on the user's own file: a real measured reference point.

    Placement chosen because it needs no VRAM-fit logic and no MoE byte accounting - it is the
    one arm that runs identically on every machine that can load the file at all.
    """
    from .runtime import find_llama
    try:
        binp = find_llama(llama_dir, "llama-bench")
    except SystemExit:
        return None, "llama-bench not found - anchor skipped (pass --llama-dir)"
    with ClockSampler() as clk:
        out = _run([binp, "-m", model_path, "-ngl", "0", "-n", "32", "-p", "0", "-r", "1"],
                   timeout=1800)
    for line in out.splitlines():
        if "tg32" in line and "|" in line:
            try:
                toks = float(line.split("|")[-2].strip().split()[0])
                return dict(placement="pure CPU (-ngl 0)", metric="tg32", tok_s=toks,
                            model=os.path.basename(model_path),
                            model_gb=round(os.path.getsize(model_path) / 1e9, 3),
                            sustained_sm=clk.sustained()), None
            except (ValueError, IndexError):
                break
    return None, "could not parse llama-bench output - anchor skipped"


def load():
    """Return (calibration dict, age_days) or (None, None). Used by plan."""
    try:
        with open(CAL_PATH, encoding="utf-8") as f:
            cal = json.load(f)
        age = (time.time() - cal.get("ts", 0)) / 86400
        return cal, age
    except (OSError, ValueError):
        return None, None


def run(a):
    print("quantprobe calibrate - measuring THIS machine (nothing is sent anywhere)\n")
    cal = dict(ts=time.time(), date=time.strftime("%Y-%m-%d"), host=platform.node(),
               tool_version=__import__("quantprobe").__version__)

    ram_bw, why = measure_ram_stream()
    if ram_bw:
        cal["ram_bw_measured"] = ram_bw
        print(f"  RAM stream: {ram_bw:.2f} GB/s [measured] - this replaces the spec-sheet "
              "number, which overstates what decode gets")
    else:
        print(f"  RAM stream: {why}")

    st = gpu_state()
    if st:
        cal["gpu"] = st
        print(f"  GPU: {st['name']}, idle {st['sm']} MHz (max {st['sm_max']}), {st['temp']} C [os]")
    else:
        print("  GPU: none detected (nvidia-smi absent/empty)")

    model = getattr(a, "model", None)
    if model and os.path.isfile(model):
        from .detect import measure_disk
        cal["disk_bw_measured"] = round(measure_disk(model), 2)
        print(f"  disk: {cal['disk_bw_measured']:.2f} GB/s sequential on your file [measured]")
        if not getattr(a, "skip_bench", False):
            print("  anchor: running llama-bench -ngl 0 tg32 on your file (1-10 min on big models)...")
            anchor, why = cpu_anchor(model, getattr(a, "llama_dir", None))
            if anchor:
                cal["anchors"] = [anchor]
                print(f"  anchor: {anchor['tok_s']:.2f} tok/s pure-CPU on "
                      f"{anchor['model']} ({anchor['model_gb']:g} GB) [measured]")
                v = boost_verdict(anchor.get("sustained_sm"), st and st.get("sm_max"),
                                  st and st.get("temp"))
                if v:
                    cal["boost_verdict"] = v
                    print(f"  GPU under load: {v}")
            else:
                print(f"  anchor: {why}")
    elif model:
        print(f"  --model: file not found: {model}")
    else:
        print("  (pass --model your.gguf to also measure disk + a real decode anchor)")

    os.makedirs(CAL_DIR, exist_ok=True)
    with open(CAL_PATH, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=1)
    print(f"\n  saved: {CAL_PATH}")
    print("  plan/optimize/auto now use the measured values automatically (tagged [calibrated]).")
    print("  Delete the file to return to auto-detection. Re-run after RAM/driver changes.")
    return cal
