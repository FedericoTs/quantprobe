"""quantprobe probe — measure a GGUF's depth-fragility curve, emit the depth-aware recipe.
Adapted from the research script (weights/quant_probe.py); logic identical, llama.cpp located via
--llama-dir, QUANTPROBE_LLAMA_DIR, or PATH.
"""
from __future__ import annotations
import os, re, shutil, subprocess


def find_llama(explicit):
    for cand in ([explicit] if explicit else []) + [os.environ.get("QUANTPROBE_LLAMA_DIR")]:
        if cand and os.path.isfile(os.path.join(cand, exe("llama-quantize"))):
            return cand
    w = shutil.which("llama-quantize") or shutil.which("llama-quantize.exe")
    if w:
        return os.path.dirname(w)
    raise SystemExit("llama.cpp binaries not found: pass --llama-dir, set QUANTPROBE_LLAMA_DIR, or add to PATH")


def exe(name):
    return name + (".exe" if os.name == "nt" else "")


# Throughput constants MEASURED on the reference box (i5-7600K, 8 threads, 16 GB DDR4-3000,
# GTX 1060 6 GB), 2026-07-25/26. They are the basis of the up-front time estimate and are
# refined from the user's OWN elapsed time after the first step - see _Progress.
QUANT_GB_PER_MIN = 2.8       # 35 GB source -> 12.6 min per pass, five passes measured
PPL_MIN_PER_GB_FITS = 0.55   # intermediate fits memory: 13 GB model -> ~7 min / 32 chunks
PPL_MIN_PER_GB_SPILLS = 2.4
IMATRIX_MIN_PER_GB = 7.7      # 35 GB source, 100 chunks -> 270 min measured  # intermediate exceeds RAM: 23 GB -> ~55 min (page thrashing)


def fmt_dur(minutes):
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes/60:.1f} h"


def estimate_probe_minutes(src_gb, bands, ram_gb=16.0, vram_gb=0.0):
    """Honest up-front cost of a probe. Returns (minutes, spills) - `spills` flags the case
    where the Q6_K intermediate exceeds memory, which is what turns an hour into most of a day."""
    passes = bands + 1                        # one reference + one per band
    inter_gb = src_gb * 0.66                  # Q6_K intermediate of the source
    spills = inter_gb > max(ram_gb - 4, 1) + vram_gb * 0.9
    rate = PPL_MIN_PER_GB_SPILLS if spills else PPL_MIN_PER_GB_FITS
    return passes * (src_gb / QUANT_GB_PER_MIN + inter_gb * rate), spills


class _Progress:
    """Step counter + ETA refined from the user's own measured pace, not our constants."""

    def __init__(self, total_steps, est_minutes):
        import time
        self.t0 = time.time(); self.total = total_steps; self.done = 0; self.est = est_minutes

    def step(self, label):
        import time
        self.done += 1
        el = (time.time() - self.t0) / 60
        if self.done > 1:                     # refine from real pace once we have one data point
            remain = el / (self.done - 1) * (self.total - self.done + 1)
            eta = f", ~{fmt_dur(remain)} left (measured pace)"
        else:
            eta = f", ~{fmt_dur(max(self.est - el, 0))} left (estimated)"
        print(f"\n[quantprobe] step {self.done}/{self.total}: {label}  "
              f"[elapsed {fmt_dur(el)}{eta}]", flush=True)


def n_layers(gguf_path):
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)
    for field in r.fields.values():
        if field.name.endswith(".block_count"):
            return int(field.parts[field.data[0]][0])
    raise RuntimeError("no .block_count key in GGUF metadata")


def band_regex(lo, hi):
    return "blk\\.(" + "|".join(str(i) for i in range(lo, hi + 1)) + ")\\.ffn_.*"


def sh(cmd, dry, capture=False):
    print("  $", " ".join(cmd), flush=True)
    if dry:
        return ""
    if capture:
        return subprocess.run(cmd, capture_output=True, text=True).stdout + \
               subprocess.run(cmd, capture_output=True, text=True).stderr if False else \
               subprocess.run(cmd, capture_output=True, text=True, errors="replace").stdout
    subprocess.run(cmd, check=False)
    return ""


def _ppl_once(perp, gguf, eval_file, chunks, ngl):
    p = subprocess.run([perp, "-m", gguf, "-f", eval_file, "--chunks", str(chunks), "-ngl", str(ngl)],
                       capture_output=True, text=True, errors="replace")
    out = p.stdout + p.stderr
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out)
    return (float(m.group(1)) if m else None), out


def ppl(perp, gguf, eval_file, chunks, ngl, dry):
    print(f"  measuring perplexity on {chunks} chunks (this can take 1-3 min; llama.cpp is quiet while it works)...", flush=True)
    if dry:
        return None
    val, out = _ppl_once(perp, gguf, eval_file, chunks, ngl)
    if val is None and ngl != 0 and ("out of memory" in out.lower() or "failed to" in out.lower()):
        # a probe intermediate (Q6_K reference, or a band left at Q6_K) can be far bigger
        # than the source's final compressed size — retry CPU-only before giving up.
        print("  GPU offload failed to fit (likely VRAM); retrying at -ngl 0 (CPU, slower)...", flush=True)
        val, out = _ppl_once(perp, gguf, eval_file, chunks, 0)
    if val is None:
        tail = "\n".join(out.strip().splitlines()[-12:])
        print(f"  perplexity produced no parseable result. Last output lines:\n{tail}", flush=True)
    return val


def run(a):
    llama = find_llama(a.llama_dir)
    quant = os.path.join(llama, exe("llama-quantize"))
    perp = os.path.join(llama, exe("llama-perplexity"))
    wd = a.workdir or os.path.dirname(os.path.abspath(a.gguf))
    L = n_layers(a.gguf)
    step = (L + a.bands - 1) // a.bands
    bands = [(i, min(i + step - 1, L - 1)) for i in range(0, L, step)]
    print(f"quant-probe: {os.path.basename(a.gguf)} | {L} layers -> {len(bands)} bands {bands}", flush=True)

    # Up-front honesty about cost. A 35 GB source took 5h40m on the reference box - telling
    # people "30-60 min" (as this tool used to) is wrong by an order of magnitude on big models.
    src_gb = os.path.getsize(a.gguf) / 1e9
    ram_gb, vram_gb = 16.0, 0.0
    try:
        from . import detect as detmod
        d, _ = detmod.detect()
        ram_gb, vram_gb = float(d.get("ram", 16)), float(d.get("vram", 0))
    except Exception:
        pass
    est, spills = estimate_probe_minutes(src_gb, len(bands), ram_gb, vram_gb)
    print(f"\n  ESTIMATED TIME: ~{fmt_dur(est)}  ({len(bands)+1} quantize passes + "
          f"{len(bands)+1} perplexity runs on a {src_gb:.0f} GB source)", flush=True)
    if spills:
        print(f"  WARNING: the {src_gb*0.66:.0f} GB working file exceeds your memory, so every\n"
              f"  perplexity run pages from disk - that is what makes this slow. A smaller source\n"
              f"  (or more RAM) changes this from hours to minutes.", flush=True)
    print("  (measured on the reference box; refined from YOUR pace after step 1. "
          "Ctrl-C is safe between steps.)", flush=True)
    if est > 120 and not getattr(a, "yes", False) and not a.dry_run:
        try:
            if input(f"\n  This will take about {fmt_dur(est)}. Continue? [y/N]: ").strip().lower() != "y":
                raise SystemExit("aborted - nothing was built. Re-run with --yes to skip this prompt.")
        except EOFError:
            raise SystemExit(f"this probe needs ~{fmt_dur(est)} and there is no terminal to confirm.\n"
                             "  Re-run with --yes if you intend to commit that time.")
    prog = _Progress(2 * (len(bands) + 1), est)
    print("", flush=True)

    ref = os.path.join(wd, "_probe_ref_q6k.gguf")
    prog.step("building the Q6_K reference")
    sh([quant, "--allow-requantize", a.gguf, ref, "Q6_K", "8"], a.dry_run)
    prog.step("scoring the reference")
    p_ref = ppl(perp, ref, a.eval, a.chunks, a.ngl, a.dry_run)
    print(f"  ref PPL = {p_ref}\n", flush=True)

    print("[2/3] band probe (one band's FFNs -> Q2_K at a time)", flush=True)
    deltas = []
    for lo, hi in bands:
        out = os.path.join(wd, f"_probe_b{lo}_{hi}.gguf")
        prog.step(f"building band {lo}-{hi}")
        sh([quant, "--allow-requantize", "--tensor-type", f"{band_regex(lo, hi)}=q2_k", a.gguf, out, "Q6_K", "8"], a.dry_run)
        prog.step(f"scoring band {lo}-{hi}")
        p = ppl(perp, out, a.eval, a.chunks, a.ngl, a.dry_run)
        d = None if (p is None or p_ref is None) else p - p_ref
        deltas.append(d)
        print(f"  layers {lo}-{hi}: PPL {p}  (delta {d})", flush=True)
        if not a.dry_run and os.path.exists(out):
            os.remove(out)
    if not a.dry_run and os.path.exists(ref):
        os.remove(ref)

    print("\n[3/3] recipe", flush=True)
    if a.dry_run or any(d is None for d in deltas):
        print("  (dry-run / incomplete: curve unavailable)", flush=True)
        return
    worst = max(range(len(bands)), key=lambda i: deltas[i])
    lo, hi = bands[worst]
    print(f"  fragile band: layers {lo}-{hi} (delta +{deltas[worst]:.2f} vs "
          f"median {sorted(deltas)[len(deltas)//2]:.2f}) -> protect at Q4_K:\n", flush=True)
    # ONE source of truth for the recipe: print exactly what --apply would run. Hand-writing this
    # string separately let it silently go stale (it predated the SSM and shared-expert
    # protections, so anyone copying it built the OLD, worse recipe) - found in the 2026-07-26 audit.
    build_depthaware(a.llama_dir, a.gguf, "out-depthaware.gguf", lo, hi, bands[-1][1] + 1, dry=True)
    if getattr(a, "apply", False):
        out = a.out or os.path.splitext(a.gguf)[0] + "-depthaware.gguf"
        imat = getattr(a, "imatrix", None)
        if imat == "auto":
            imat = make_imatrix(a.llama_dir, a.gguf, a.eval, chunks=getattr(a, "imatrix_chunks", 100),
                                ngl=a.ngl, dry=a.dry_run)
        print("\n[quantprobe] --apply: building the recommended GGUF now...", flush=True)
        build_depthaware(a.llama_dir, a.gguf, out, lo, hi, bands[-1][1] + 1, dry=a.dry_run, imatrix=imat)
    else:
        print("\n  (re-run with  --apply --out model-2bit.gguf  to BUILD this GGUF automatically)", flush=True)


def _band_re(lo, hi):
    return "blk\\.(" + "|".join(str(i) for i in range(lo, hi + 1)) + ")\\.ffn_.*"


def build_depthaware(llama_dir, src, out, protect_lo, protect_hi, n_lay,
                     base="Q2_K", protect="q4_k", dry=False, imatrix=None):
    """Actually PRODUCE the compressed GGUF: base bits everywhere, fragile band + attention +
    embed + always-active tensors protected; optional importance-matrix calibration."""
    # --dry previews the exact command WITHOUT requiring llama.cpp installed
    q = exe("llama-quantize") if dry else os.path.join(find_llama(llama_dir), exe("llama-quantize"))
    cmd = [q, "--allow-requantize"]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    # ALWAYS-ACTIVE tensors first: llama.cpp resolves --tensor-type first-match-wins (verified
    # 2026-07-25), so this must precede the band rules or it silently does nothing. The shared
    # expert fires on EVERY token (routed experts fire ~8/256) and is heavy-tailed: measured
    # -3.2% ppl when protected at q8_0, for ~0.65% more bytes (pre-registration #12).
    cmd += ["--tensor-type", "ffn_.*_shexp.*=q8_0"]
    if protect_lo > 0:
        cmd += ["--tensor-type", f"{_band_re(0, protect_lo - 1)}=q2_k"]
    if protect_hi < n_lay - 1:
        cmd += ["--tensor-type", f"{_band_re(protect_hi + 1, n_lay - 1)}=q2_k"]
    cmd += ["--tensor-type", f"{_band_re(protect_lo, protect_hi)}={protect}",
            "--tensor-type", "attn_.*=q4_k", "--tensor-type", "ssm_.*=q4_k",
            "--token-embedding-type", "q4_k", src, out, base, "8"]
    print(f"[quantprobe] building depth-aware GGUF: protect layers {protect_lo}-{protect_hi} @ {protect}")
    print("  $ " + " ".join(cmd))
    if dry:
        return out
    rc = subprocess.call(cmd)
    if rc == 0 and os.path.exists(out):
        print(f"[quantprobe] done -> {out} ({os.path.getsize(out)/1e9:.2f} GB). "
              f"Run it:  quantprobe run --gguf {out} --model <preset> --machine <preset>")
    else:
        print(f"[quantprobe] quantize failed (exit {rc}).")
    return out


def make_imatrix(llama_dir, src, eval_file, out=None, chunks=100, ngl=99, dry=False):
    """Generate an importance matrix: measured -8.5% ppl at ~3 bits, at zero size and speed
    cost (pre-registration #12). The single largest quality lever in the recipe.

    The calibration corpus should NOT be your evaluation text - calibrating on the same data
    you score against inflates the result. Wikitext TRAIN is used against a wikitext TEST eval.
    """
    out = out or os.path.splitext(src)[0] + ".imatrix.gguf"
    if os.path.isfile(out):
        print(f"[quantprobe] reusing existing imatrix: {out}")
        return out
    im = exe("llama-imatrix") if dry else os.path.join(find_llama(llama_dir), exe("llama-imatrix"))
    cmd = [im, "-m", src, "-f", eval_file, "-o", out, "--chunks", str(chunks), "-ngl", str(ngl), "-c", "512"]
    src_gb = os.path.getsize(src) / 1e9 if os.path.isfile(src) else 0
    im_est = src_gb * IMATRIX_MIN_PER_GB * (chunks / 100.0)
    print(f"[quantprobe] building importance matrix over {chunks} chunks on a {src_gb:.0f} GB source")
    print(f"  ESTIMATED TIME: ~{fmt_dur(im_est)} (measured 4.5 h for a 35 GB source at 100 chunks).")
    print(f"  Worth ~8% quality at zero size/speed cost - but it is a real time commitment.")
    print(f"  Skip it with --no-imatrix; reduce it with --imatrix-chunks (quality scales down too).")
    print("  $ " + " ".join(cmd))
    if dry:
        return out
    rc = subprocess.call(cmd)
    if rc != 0 or not os.path.isfile(out):
        print(f"[quantprobe] imatrix generation failed (exit {rc}); continuing WITHOUT calibration.")
        return None
    print(f"[quantprobe] imatrix ready -> {out} ({os.path.getsize(out)/1e6:.0f} MB)")
    return out


def quantize(a):
    """Standalone compress: build a depth-aware GGUF from an explicit band (no probing)."""
    if not os.path.isfile(a.gguf):
        raise SystemExit(f"GGUF not found: {a.gguf}  (point --gguf at a real high-precision GGUF: f16/bf16/Q8)")
    n_lay = n_layers(a.gguf)
    rec_key = getattr(a, "recipe", None)
    if a.protect:
        lo, hi = (int(x) for x in a.protect.split("-"))
    elif rec_key:
        from . import recipes as recmod
        r = recmod.find(key=rec_key)
        if not r:
            raise SystemExit(f"no recipe '{rec_key}'. See what has been measured: quantprobe recipes")
        if r["model"]["n_layer"] != n_lay:
            raise SystemExit(
                f"recipe '{rec_key}' is for a {r['model']['n_layer']}-layer model; this file has "
                f"{n_lay}.\n  A fragile band is only meaningful for the model it was measured on.")
        lo, hi = r["probe"]["fragile_band"]
        print(f"[quantprobe] using measured recipe '{rec_key}': protect layers {lo}-{hi} "
              f"({r['probe']['shape']}-fragile, {r['probe']['fragility_ratio']}x median)")
        print(f"  measured {r['provenance']['measured']} on {r['provenance']['hardware']}; "
              f"evidence: {r['provenance']['raw_log']}")
    else:
        lo, hi = n_lay - a.protect_late, n_lay - 1
        # If someone has already measured THIS model, say so - the default is a guess, and a
        # measured band is strictly better information (Law 3: fragility is not predictable).
        try:
            from . import recipes as recmod
            from .spec import from_gguf
            arch = from_gguf(a.gguf).get("arch")
            r = recmod.find(arch=arch, n_layer=n_lay)
            if r:
                rlo, rhi = r["probe"]["fragile_band"]
                if (rlo, rhi) != (lo, hi):
                    print(f"[quantprobe] a MEASURED recipe exists for this model class "
                          f"({r['model']['name']}): fragile band {rlo}-{rhi}, not the default "
                          f"{lo}-{hi}.\n  Use it:  --recipe {r['model']['key']}")
        except Exception:
            pass
    out = a.out or os.path.splitext(a.gguf)[0] + "-depthaware.gguf"
    imat = getattr(a, "imatrix", None)
    if imat and not os.path.isfile(imat) and not getattr(a, "dry", False):
        raise SystemExit(f"--imatrix file not found: {imat}\n"
                         "  generate one first, or drop the flag to build without calibration.")
    build_depthaware(a.llama_dir, a.gguf, out, lo, hi, n_lay,
                     dry=getattr(a, "dry", False), imatrix=imat)
