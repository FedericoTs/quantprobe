"""quantprobe run / bench — the runtime layer.

run:   plan the best placement for your model+machine, then LAUNCH llama.cpp with those exact
       flags (chat via llama-cli, or --serve for llama-server). Colibri-style one-command UX,
       riding stock llama.cpp instead of a custom engine.
bench: measure real decode tok/s with the planned flags and print predicted vs measured —
       every user becomes a validation point for the tiered decode law.
"""
from __future__ import annotations
import os, re, shutil, subprocess, sys

from . import plan as planmod


def exe(name):
    return name + (".exe" if os.name == "nt" else "")


def find_llama(explicit, tool):
    for cand in ([explicit] if explicit else []) + [os.environ.get("QUANTPROBE_LLAMA_DIR")]:
        if cand and os.path.isfile(os.path.join(cand, exe(tool))):
            return os.path.join(cand, exe(tool))
    w = shutil.which(tool) or shutil.which(exe(tool))
    if w:
        return w
    raise SystemExit(f"{tool} not found: pass --llama-dir, set QUANTPROBE_LLAMA_DIR, or add to PATH")


def best_flags(a):
    """Run the planner, return (best_config, flags_list) for the winning placement."""
    from . import spec as specmod
    specmod.apply(a)
    if getattr(a, "bits", None) is None:
        a.bits = 2.5
    m = dict(planmod.MODELS[a.model]) if getattr(a, "model", None) in planmod.MODELS else {}
    t = getattr(a, "total", None) or m.get("t") or 13.0
    ac = getattr(a, "active", None) or m.get("a") or t
    ne = getattr(a, "always_active", None) or m.get("ne") or (ac if ac >= t * 0.9 else ac * 0.35)
    moe = m.get("moe", ac < t * 0.9)
    hw = dict(planmod.MACHINES[a.machine]) if getattr(a, "machine", None) in planmod.MACHINES else {}
    if not hw and all(getattr(a, k, None) is None for k in ("vram", "vram_bw", "ram", "ram_bw", "disk_bw")):
        from . import detect as detmod
        auto, _ = detmod.detect()
        hw = dict(vc=auto["vram"], vb=auto["vram_bw"], rc=auto["ram"], rb=auto["ram_bw"],
                  db=auto["disk_bw"], geta=auto.get("geta", 0.45), gl=auto.get("gl"))
        print("[quantprobe] hardware auto-detected (run `quantprobe hw` for details; "
              "pass --machine/flags to estimate a different box)")
    vc = a.vram if a.vram is not None else hw.get("vc", 0)
    vb = a.vram_bw if a.vram_bw is not None else hw.get("vb", 0)
    rc = a.ram if a.ram is not None else hw.get("rc", 16)
    rb = a.ram_bw if a.ram_bw is not None else hw.get("rb", 40)
    db = a.disk_bw if a.disk_bw is not None else hw.get("db", 0.5)
    geta = hw.get("geta", 0.45); gl = hw.get("gl", None)
    act_scale = 1.0
    gguf = getattr(a, "gguf", None)
    if gguf and os.path.isfile(gguf):
        ab = max(a.bits, 4.5)
        size_pred = (ne * ab / 8 + (t - ne) * a.bits / 8) * 1.08
        size_real = os.path.getsize(gguf) / 1e9
        if size_pred > 0:
            act_scale = size_real / size_pred
            print(f"[quantprobe] calibrated to file: {size_real:.2f} GB on disk "
                  f"(preset assumed {size_pred:.2f} GB, scale {act_scale:.2f})")
    ctx = getattr(a, "ctx", 0) or 0
    kvp = (a.kv_per_pos * 1024 if getattr(a, "kv_per_pos", None)
           else m.get("kvp", planmod.DEFAULT_KVP))
    _, _, cfgs = planmod.evaluate(t, ac, ne, moe, a.bits, vc, vb, rc, rb, db, geta, act_scale, gl,
                                  ctx=ctx, kvp=kvp)
    best = cfgs[0]
    return best, best[3].replace('"', "").split()


def run(a):
    best, flags = best_flags(a)
    tool = "llama-server" if a.serve else "llama-cli"
    # --dry previews the plan + command WITHOUT requiring llama.cpp installed
    binp = tool if a.dry else find_llama(a.llama_dir, tool)
    cmd = [binp, "-m", a.gguf] + flags
    if (getattr(a, "ctx", 0) or 0) > 0:
        cmd += ["-c", str(a.ctx)]                         # launch with the context you planned for
    if not a.serve:
        cmd += ["-cnv"]
    if a.extra:
        cmd += a.extra.split()
    print(f"[quantprobe] placement: {best[0]}  (predicted {best[1]:.1f} tok/s"
          + (f", {best[2]}" if best[2] else "") + ")")
    print("[quantprobe] exec:", " ".join(cmd), "\n")
    if a.dry:
        return
    sys.exit(subprocess.call(cmd))


def bench(a):
    if getattr(a, "depth", None):
        a.ctx = a.depth                                   # prediction at the benched depth
    best, flags = best_flags(a)
    binp = "llama-bench" if getattr(a, "dry", False) else find_llama(a.llama_dir, "llama-bench")
    # llama-bench uses --mmap 0 rather than --no-mmap
    bflags = ["--mmap", "0" if "--no-mmap" in flags else "1"]
    for i, f in enumerate(flags):
        if f == "-ngl":
            bflags += ["-ngl", flags[i + 1]]
        if f == "-ot":
            bflags += ["-ot", flags[i + 1]]
    if flags and flags[0].startswith("-ngl") is False and "-ngl" not in flags:
        pass
    # normalize: flags like ['-ngl','99','-ot','exps=CPU','--no-mmap']
    cmd = [binp, "-m", a.gguf, "-n", "32", "-p", "0", "-r", str(a.reps)] + bflags
    if getattr(a, "depth", None):
        cmd += ["-d", str(a.depth)]
    print(f"[quantprobe] placement: {best[0]} | predicted {best[1]:.1f} tok/s")
    print("[quantprobe] bench:", " ".join(cmd))
    if a.dry:
        return
    print("[quantprobe] benchmarking (30-90s; llama-bench runs quietly, then prints the number)...", flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    txt = out.stdout + out.stderr
    mm = re.findall(r"tg\d+(?:\s*@\s*d\d+)?\s*\|\s*([0-9.]+)\s*(?:Â?±|\+/-)\s*([0-9.]+)", txt)
    if not mm:
        mm = re.findall(r"\|\s*([0-9.]+)\s*(?:Â?±)\s*([0-9.]+)\s*\|\s*$", txt, re.M)
    if mm:
        meas, err = float(mm[-1][0]), float(mm[-1][1])
        delta = (meas / best[1] - 1) * 100 if best[1] else 0
        print(f"\n[quantprobe] measured: {meas:.2f} +/- {err:.2f} tok/s "
              f"(predicted {best[1]:.1f}, {delta:+.0f}%)")
        if getattr(a, "contribute", False):
            _emit_contribution(a, best, meas, err, delta)
        else:
            print("[quantprobe] the tiered decode law just ran on your machine.")
            print("[quantprobe] help grow the law: re-run with --contribute for a one-click, "
                  "pre-filled data point (you review it first; nothing is sent automatically).")
    else:
        print("\n[quantprobe] could not parse llama-bench output; raw tail:")
        print("\n".join(txt.strip().splitlines()[-6:]))


def tier_view(a, best):
    """Rough (capacity, used) per tier for the dashboard's placement panel."""
    hw = dict(planmod.MACHINES[a.machine]) if getattr(a, "machine", None) in planmod.MACHINES else {}
    if not hw and all(getattr(a, k, None) is None for k in ("vram", "vram_bw", "ram", "ram_bw", "disk_bw")):
        from . import detect as detmod
        auto, _ = detmod.detect()
        hw = dict(vc=auto["vram"], vb=auto["vram_bw"], rc=auto["ram"], rb=auto["ram_bw"],
                  db=auto["disk_bw"], geta=auto.get("geta", 0.45), gl=auto.get("gl"))
        print("[quantprobe] hardware auto-detected (run `quantprobe hw` for details; "
              "pass --machine/flags to estimate a different box)")
    vc = a.vram if a.vram is not None else hw.get("vc", 0)
    rc = a.ram if a.ram is not None else hw.get("rc", 16)
    size = os.path.getsize(a.gguf) / 1e9 if a.gguf and os.path.isfile(a.gguf) else 0
    name = best[0]
    if name == "all in VRAM":
        return [("VRAM", vc, size), ("RAM", rc, 1.0)]
    if name.startswith("hybrid"):
        v = min(size * 0.15 + 1.2, vc)
        return [("VRAM (attention + ctx)", vc, v), ("RAM (experts)", rc, size - size * 0.15)]
    if name.startswith("split"):
        return [("VRAM", vc, vc * 0.9), ("RAM", rc, max(0.5, size - vc * 0.9))]
    if name.startswith("pure CPU"):
        return [("VRAM (idle)", vc, 0), ("RAM", rc, size)]
    return [("RAM (cache)", rc, rc - 4), ("disk (streaming)", max(size * 1.2, 1), size)]


def _emit_contribution(a, best, meas, err, delta):
    import urllib.parse
    from . import __version__
    hw = (dict(planmod.MACHINES.get(getattr(a, "machine", "") or "", {})).get("hint")
          or f"vram={a.vram} vram_bw={a.vram_bw} ram={a.ram} ram_bw={a.ram_bw} disk_bw={a.disk_bw}")
    model = getattr(a, "model", None) or f"total={a.total} active={a.active}"
    lines = [
        f"hardware: {hw}",
        f"model: {model} @ {a.bits:g}-bit",
        f"placement: {best[0]}",
        f"predicted: {best[1]:.1f} tok/s",
        f"measured: {meas:.2f} +/- {err:.2f} tok/s ({delta:+.0f}%)",
        f"quantprobe: v{__version__}",
        "",
        "Notes (optional): ",
    ]
    body = "\n".join(lines)
    title = f"[eta] {model} {a.bits:g}-bit on {str(hw)[:40]}"
    url = ("https://github.com/FedericoTs/quantprobe/issues/new?labels=eta-datapoint"
           f"&title={urllib.parse.quote(title)}&body={urllib.parse.quote(body)}")
    print("\n[quantprobe] Contribute this data point (OPT-IN). It contains ONLY what you see below --")
    print("             no system scan, no IP, nothing auto-collected. Review, then submit:\n")
    print(body)
    print("\n  Open to submit (you can edit first):\n  " + url + "\n")
    print("  Points that land OUTSIDE the predicted bands are the most valuable -- they refine the law.")
