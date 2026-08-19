"""quantprobe run / bench - the runtime layer.

run:   plan the best placement for your model+machine, then LAUNCH llama.cpp with those exact
       flags (chat via llama-cli, or --serve for llama-server). Colibri-style one-command UX,
       riding stock llama.cpp instead of a custom engine.
bench: measure real decode tok/s with the planned flags and print predicted vs measured -
       every user becomes a validation point for the tiered decode law.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

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
    raise SystemExit(
        f"{tool} not found: pass --llama-dir, set QUANTPROBE_LLAMA_DIR, or add to PATH"
    )


LOOPBACK = "127.0.0.1"


def bind_loopback(cmd):
    """Pin a spawned llama-server to loopback unless the user explicitly chose a host.

    llama.cpp's server starts with **CORS '*' and no API key** - that is its own startup warning,
    not our characterisation - so whatever it binds to is an unauthenticated LLM endpoint that any
    origin may script. We passed --port and inherited the host, which is a DEPENDENCY DEFAULT we do
    not control and which has moved across llama.cpp versions. Inheriting it means a future release
    could silently put a user's model on every interface of their LAN, from a tool they ran to
    measure tok/s.

    So quantprobe states the bind instead of assuming it. The dashboard already binds ITS own proxy
    to 127.0.0.1 explicitly and addresses the upstream as http://127.0.0.1 - the intent was always
    loopback; only the enforcement was missing.

    Serving on the network is a legitimate thing to want, so an explicit `--host` (via `--extra`)
    still wins - this sets a safe default, it does not remove the choice."""
    if "--host" in cmd:
        return cmd
    return cmd + ["--host", LOOPBACK]


def best_flags(a):
    """Run the planner, return (best_config, flags_list) for the winning placement."""
    from . import spec as specmod

    from_file = specmod.apply(a)  # True when the spec came from the GGUF itself
    # A typo'd --machine used to fall through to auto-detect here (line ~40) and quietly predict
    # for the wrong box. run / bench / dashboard all route through best_flags, so one guard here
    # gives them the same loud refusal plan/optimize/target already have. Runs AFTER apply so a
    # --gguf has already filled the spec.
    planmod.check_presets(a)
    if getattr(a, "bits", None) is None:
        a.bits = 2.5
    m = dict(planmod.MODELS[a.model]) if getattr(a, "model", None) in planmod.MODELS else {}
    t = getattr(a, "total", None) or m.get("t") or 13.0
    ac = getattr(a, "active", None) or m.get("a") or t
    ne = getattr(a, "always_active", None) or m.get("ne") or (ac if ac >= t * 0.9 else ac * 0.35)
    moe = m.get("moe", ac < t * 0.9)
    hw = (
        dict(planmod.MACHINES[a.machine]) if getattr(a, "machine", None) in planmod.MACHINES else {}
    )
    if not hw and all(
        getattr(a, k, None) is None for k in ("vram", "vram_bw", "ram", "ram_bw", "disk_bw")
    ):
        from . import detect as detmod

        auto, _ = detmod.detect()
        hw = {
            "vc": auto["vram"],
            "vb": auto["vram_bw"],
            "rc": auto["ram"],
            "rb": auto["ram_bw"],
            "db": auto["disk_bw"],
            "geta": auto.get("geta", 0.45),
            "gl": auto.get("gl"),
        }
        print(
            "[quantprobe] hardware auto-detected (run `quantprobe hw` for details; "
            "pass --machine/flags to estimate a different box)"
        )
        # the SAME calibration+anchor path plan uses - one function, so the commands can
        # never disagree about the same input (v1.10.5 bug class; layer 3 enforces this)
        planmod.apply_calibration_overrides(hw, a)
    vc = planmod.agg_cap(a.vram) if a.vram is not None else hw.get("vc", 0)
    vb = planmod.agg_bw(a.vram_bw, 0.85) if a.vram_bw is not None else hw.get("vb", 0)
    rc = a.ram if a.ram is not None else hw.get("rc", 16)
    rb = a.ram_bw if a.ram_bw is not None else hw.get("rb", 40)
    db = planmod.agg_bw(a.disk_bw, 0.75) if a.disk_bw is not None else hw.get("db", 0.5)
    geta = hw.get("geta", 0.45)
    gl = hw.get("gl", None)
    # File-size calibration corrects a PRESET's assumed size against the real file. It must NOT
    # run when autospec already read the spec from that same file: bits are then derived from
    # the file size, so scaling by (real size / predicted size) corrects the same discrepancy
    # twice. Measured 2026-07-26 on three models: the double correction made bench 8-13% more
    # optimistic than plan on identical input, and plan was the accurate one (predicted 19.6 vs
    # measured 19.88, +1.4%; bench said 22.1, -11%).
    act_scale = 1.0
    gguf = getattr(a, "gguf", None)
    if gguf and os.path.isfile(gguf) and not from_file:
        ab = max(a.bits, 4.5)
        size_pred = (ne * ab / 8 + (t - ne) * a.bits / 8) * 1.08
        size_real = specmod.gguf_size(gguf) / 1e9
        if size_pred > 0:
            act_scale = size_real / size_pred
            print(
                f"[quantprobe] calibrated to file: {size_real:.2f} GB on disk "
                f"(preset assumed {size_pred:.2f} GB, scale {act_scale:.2f})"
            )
    ctx = getattr(a, "ctx", 0) or 0
    kvp = (
        a.kv_per_pos * 1024 if getattr(a, "kv_per_pos", None) else m.get("kvp", planmod.DEFAULT_KVP)
    )
    _true = specmod.gguf_size(gguf) / 1e9 if gguf and os.path.isfile(gguf) else None
    # The contribution payload must carry the spec THE PREDICTION USED, not the raw args - under
    # autospec-failure or preset paths a.total/a.active are None, and issue #1 (the tool's first
    # external datapoint) arrived titled "total=None active=None". Same bug class as the
    # v1.26.1 hardware fix, other operand; stashed here because this is the resolution moment.
    a._resolved_spec = (t, ac)
    # same size-classed GPU-eta dispatch as plan (ONE shared function; layer 3 enforces parity)
    vb, geta = planmod.resolve_gpu_eta(hw, a, ac, a.bits, vb, geta)
    rb = planmod.resolve_cpu_bw(hw, a, ac, a.bits, rb)
    _, _, cfgs = planmod.evaluate(
        t,
        ac,
        ne,
        moe,
        a.bits,
        vc,
        vb,
        rc,
        rb,
        db,
        geta,
        act_scale,
        gl,
        ctx=ctx,
        kvp=kvp,
        true_size_gb=_true,
        n_layer=planmod.effective_n_layer(a, m),
        codebook_share=getattr(a, "codebook_share", 0.0),
    )
    # run/bench/dashboard LAUNCH stock llama.cpp, so they may only pick placements stock
    # llama.cpp can actually execute. The three-tier expert-cache row's "flags" field is a
    # PROSE description ("+ runtime-managed expert cache"), not argv - exec'ing it hands
    # llama-cli a bare "+" and it dies. optimize/auto already filter this; run/bench must too.
    runnable = [c for c in cfgs if "expert cache" not in c[0]]
    if not runnable:
        raise SystemExit(
            "no placement on this machine is runnable by stock llama.cpp for this file.\n"
            "  The planner's best row needs an expert-caching runtime (ktransformers/colibri-class).\n"
            "  See the full picture, including that row:  quantprobe plan --gguf <file>"
        )
    if runnable[0] is not cfgs[0]:
        print(
            f"[quantprobe] note: the fastest placement ({cfgs[0][0]}, {cfgs[0][1]:.1f} tok/s) needs an "
            f"expert-caching runtime; launching the fastest STOCK-llama.cpp placement instead."
        )
    best = runnable[0]
    # same --threads AND -ub logic as plan's printout - the command a user SEES must be the
    # command run/bench EXECUTE (the audit found plan printing flags run dropped, twice)
    fl_str = best[3].replace('"', "")
    ubf = planmod.ubatch_flags(best[0], ne * max(a.bits, 4.5) / 8 * 1.08 if moe else 0.0, vc)
    if ubf and "-ub" not in fl_str:
        fl_str = f"{fl_str} {ubf}"
    fl_str, _ = planmod.append_threads_flag(fl_str, best[0])
    return best, fl_str.split()


def run(a):
    best, flags = best_flags(a)
    tool = "llama-server" if a.serve else "llama-cli"
    # --dry previews the plan + command WITHOUT requiring llama.cpp installed
    binp = tool if a.dry else find_llama(a.llama_dir, tool)
    cmd = [binp, "-m", a.gguf] + flags
    if (getattr(a, "ctx", 0) or 0) > 0:
        cmd += ["-c", str(a.ctx)]  # launch with the context you planned for
    if not a.serve:
        cmd += ["-cnv"]
    if a.extra:
        cmd += a.extra.split()
    if a.serve:
        # After --extra, so a user's explicit `--extra "--host 0.0.0.0"` is honoured rather than
        # duplicated. Loopback is the DEFAULT here, not a restriction.
        cmd = bind_loopback(cmd)
    print(
        f"[quantprobe] placement: {best[0]}  (predicted {best[1]:.1f} tok/s"
        + (f", {best[2]}" if best[2] else "")
        + ")"
    )
    print("[quantprobe] exec:", " ".join(cmd), "\n")
    if a.dry:
        return
    sys.exit(subprocess.call(cmd))


def bench(a):
    if getattr(a, "depth", None):
        a.ctx = a.depth  # prediction at the benched depth
    best, flags = best_flags(a)
    binp = "llama-bench" if getattr(a, "dry", False) else find_llama(a.llama_dir, "llama-bench")
    # Forward the planned flags into llama-bench, translating where the two CLIs differ.
    #
    # This used to hand-list -ngl and -ot and SILENTLY DROP everything else, which means any flag
    # the planner learns to emit is missing from `bench` - and predicted-vs-measured then drifts
    # by exactly the size of the new lever, while both commands still look right. Pre-registration
    # #19 measured a 1.73x prefill effect from -ub; had it shipped through the old forwarder,
    # `bench` would have quietly measured the un-flagged configuration and reported the law as
    # wrong. An allow-list that drops the unknown is the same shape as every other silent-fallback
    # defect in this project's history, so unknown flags now RAISE.
    BENCH_VALUED = {
        "-ngl",
        "-ot",
        "-ub",
        "-b",
        "-c",
        "-t",
    }  # take a value, same spelling in bench
    BENCH_TRANSLATE = {"--no-mmap": ["--mmap", "0"]}  # spelled differently in llama-bench
    BENCH_VALUED_RENAME = {"--threads": "-t"}  # take a value, renamed in bench
    bflags, i = [], 0
    while i < len(flags):
        f = flags[i]
        if f in BENCH_TRANSLATE:
            bflags += BENCH_TRANSLATE[f]
            i += 1
        elif f in BENCH_VALUED_RENAME:
            bflags += [BENCH_VALUED_RENAME[f], flags[i + 1]]
            i += 2
        elif f in BENCH_VALUED:
            bflags += [f, flags[i + 1]]
            i += 2
        else:
            raise SystemExit(
                f"[quantprobe] internal: the planner emitted '{f}', which bench does not know how "
                f"to forward. Add it to BENCH_VALUED or BENCH_TRANSLATE in runtime.py - dropping "
                f"it would make every predicted-vs-measured figure wrong by the size of that flag."
            )
    if "--mmap" not in bflags:
        bflags += ["--mmap", "1"]
    cmd = [binp, "-m", a.gguf, "-n", "32", "-p", "0", "-r", str(a.reps)] + bflags
    if getattr(a, "depth", None):
        cmd += ["-d", str(a.depth)]
    print(f"[quantprobe] placement: {best[0]} | predicted {best[1]:.1f} tok/s")
    print("[quantprobe] bench:", " ".join(cmd))
    if a.dry:
        return
    print(
        "[quantprobe] benchmarking (30-90s; llama-bench runs quietly, then prints the number)...",
        flush=True,
    )
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace", check=False)
    txt = out.stdout + out.stderr
    mm = re.findall(r"tg\d+(?:\s*@\s*d\d+)?\s*\|\s*([0-9.]+)\s*(?:Â?±|\+/-)\s*([0-9.]+)", txt)
    if not mm:
        mm = re.findall(r"\|\s*([0-9.]+)\s*(?:Â?±)\s*([0-9.]+)\s*\|\s*$", txt, re.MULTILINE)
    if mm:
        meas, err = float(mm[-1][0]), float(mm[-1][1])
        delta = (meas / best[1] - 1) * 100 if best[1] else 0
        # C-14: a predicted-vs-measured pair is only meaningful inside ONE machine state, so the
        # pair is stamped with the state that produced it. Re-running `calibrate` changes the id;
        # a number carrying a different id must not be scored against this prediction.
        from . import calibrate as _cal

        _c, _ = _cal.load()
        _cid = (_c or {}).get("cal_id")
        print(
            f"\n[quantprobe] measured: {meas:.2f} +/- {err:.2f} tok/s "
            f"(predicted {best[1]:.1f}, {delta:+.0f}%)"
            + (f"  [machine state {_cid}]" if _cid else "  [uncalibrated]")
        )
        # A run whose own error bar is huge is not a measurement - saying so beats letting
        # someone quote a cold-cache artifact. (Seen 2026-07-26: 4.01 +/- 2.16 on a first
        # read from disk, where the warm number was 18.7.)
        # A TIGHT error bar is not evidence the number will repeat. Residency is a separate
        # question from variance, so it prints for EVERY measurement and before the spread guard
        # below - a noisy run is exactly when someone needs to know the file did not fit.
        # C-32: 14.86 +/- 0.36 was 2.4% spread, sailed through that guard, and still could not be
        # reproduced days later on the same box with the same command, because the file was
        # larger than free RAM and nothing recorded it.
        try:
            from . import detect as _det

            _fits, _note = _det.residency(os.path.getsize(a.gguf) if os.path.isfile(a.gguf) else 0)
            if _note:
                print(f"[quantprobe] {'residency' if _fits else 'RESIDENCY'}: {_note}")
        except Exception:
            pass
        if err > meas * 0.15:
            print(
                f"[quantprobe] WARNING: +/-{err / meas * 100:.0f}% spread - this number is not "
                f"reliable. Usually a cold file cache on the first read.\n"
                f"             Re-run it: the second run reads from RAM and is the real number."
            )
            return
        if getattr(a, "contribute", False):
            _emit_contribution(a, best, meas, err, delta)
        else:
            print("[quantprobe] the tiered decode law just ran on your machine.")
            print(
                "[quantprobe] help grow the law: re-run with --contribute for a one-click, "
                "pre-filled data point (you review it first; nothing is sent automatically)."
            )
    else:
        print("\n[quantprobe] could not parse llama-bench output; raw tail:")
        print("\n".join(txt.strip().splitlines()[-6:]))


def tier_view(a, best):
    """Rough (capacity, used) per tier for the dashboard's placement panel."""
    hw = (
        dict(planmod.MACHINES[a.machine]) if getattr(a, "machine", None) in planmod.MACHINES else {}
    )
    if not hw and all(
        getattr(a, k, None) is None for k in ("vram", "vram_bw", "ram", "ram_bw", "disk_bw")
    ):
        from . import detect as detmod

        auto, _ = detmod.detect()
        hw = {
            "vc": auto["vram"],
            "vb": auto["vram_bw"],
            "rc": auto["ram"],
            "rb": auto["ram_bw"],
            "db": auto["disk_bw"],
            "geta": auto.get("geta", 0.45),
            "gl": auto.get("gl"),
        }
        print(
            "[quantprobe] hardware auto-detected (run `quantprobe hw` for details; "
            "pass --machine/flags to estimate a different box)"
        )
        # the SAME calibration+anchor path plan uses - one function, so the commands can
        # never disagree about the same input (v1.10.5 bug class; layer 3 enforces this)
        planmod.apply_calibration_overrides(hw, a)
    vc = planmod.agg_cap(a.vram) if a.vram is not None else hw.get("vc", 0)
    rc = a.ram if a.ram is not None else hw.get("rc", 16)
    from . import spec as specmod

    size = specmod.gguf_size(a.gguf) / 1e9 if a.gguf and os.path.isfile(a.gguf) else 0
    name = best[0]
    if name == "all in VRAM":
        return [("VRAM", vc, size), ("RAM", rc, 1.0)]
    if name.startswith("hybrid"):
        v = min(size * 0.15 + 1.2, vc)
        return [
            ("VRAM (attention + ctx)", vc, v),
            ("RAM (experts)", rc, size - size * 0.15),
        ]
    if name.startswith("split"):
        return [("VRAM", vc, vc * 0.9), ("RAM", rc, max(0.5, size - vc * 0.9))]
    if name.startswith("pure CPU"):
        return [("VRAM (idle)", vc, 0), ("RAM", rc, size)]
    return [("RAM (cache)", rc, rc - 4), ("disk (streaming)", max(size * 1.2, 1), size)]


def _emit_contribution(a, best, meas, err, delta):
    import urllib.parse

    from . import __version__

    # THE HARDWARE MUST BE THE RESOLVED HARDWARE, not the raw args. Under auto-detect (the
    # default path, i.e. nearly every contributor) the args are all None, and this function
    # shipped printing "vram=None vram_bw=None ram=None..." - a datapoint whose entire purpose
    # is the machine, arriving without one. Caught by the pre-launch gauntlet on v1.26.1;
    # re-resolve here exactly as the prediction did.
    hw = dict(planmod.MACHINES.get(getattr(a, "machine", "") or "", {})).get("hint")
    if not hw:
        try:
            vc, vb, rc, rb, db, _geta, _gl, _hw = planmod.resolve_hw(a, announce=False)
            hw = f"vram={vc:g} vram_bw={vb:g} ram={rc:g} ram_bw={rb:g} disk_bw={db:g}"
        except Exception:
            hw = f"vram={a.vram} vram_bw={a.vram_bw} ram={a.ram} ram_bw={a.ram_bw} disk_bw={a.disk_bw}"
    # Model identity, best available first: preset name > GGUF filename > resolved spec. The raw
    # a.total/a.active fallback stays last - it is what shipped None/None in issue #1.
    rs = getattr(a, "_resolved_spec", None)
    spec_s = f"total={rs[0]:g} active={rs[1]:g}" if rs else f"total={a.total} active={a.active}"
    gguf_name = os.path.basename(a.gguf) if getattr(a, "gguf", None) else None
    model = getattr(a, "model", None) or (f"{gguf_name} ({spec_s})" if gguf_name else spec_s)
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
    title = f"[eta] {str(model)[:60]} {a.bits:g}-bit on {str(hw)[:40]}"
    url = (
        "https://github.com/FedericoTs/quantprobe/issues/new?labels=eta-datapoint"
        f"&title={urllib.parse.quote(title)}&body={urllib.parse.quote(body)}"
    )
    print(
        "\n[quantprobe] Contribute this data point (OPT-IN). It contains ONLY what you see below --"
    )
    print("             no system scan, no IP, nothing auto-collected. Review, then submit:\n")
    print(body)
    print("\n  Open to submit (you can edit first):\n  " + url + "\n")
    print(
        "  Points that land OUTSIDE the predicted bands are the most valuable -- they refine the law."
    )
