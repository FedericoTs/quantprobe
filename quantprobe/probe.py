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
    print(f"quant-probe: {os.path.basename(a.gguf)} | {L} layers -> {len(bands)} bands {bands}\n", flush=True)

    ref = os.path.join(wd, "_probe_ref_q6k.gguf")
    print("[1/3] reference Q6_K", flush=True)
    sh([quant, "--allow-requantize", a.gguf, ref, "Q6_K", "8"], a.dry_run)
    p_ref = ppl(perp, ref, a.eval, a.chunks, a.ngl, a.dry_run)
    print(f"  ref PPL = {p_ref}\n", flush=True)

    print("[2/3] band probe (one band's FFNs -> Q2_K at a time)", flush=True)
    deltas = []
    for lo, hi in bands:
        out = os.path.join(wd, f"_probe_b{lo}_{hi}.gguf")
        sh([quant, "--allow-requantize", "--tensor-type", f"{band_regex(lo, hi)}=q2_k", a.gguf, out, "Q6_K", "8"], a.dry_run)
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
    print(f"[quantprobe] building importance matrix over {chunks} chunks "
          f"(one pass; slow on big models, but worth ~8% quality for free)")
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
    if a.protect:
        lo, hi = (int(x) for x in a.protect.split("-"))
    else:
        lo, hi = n_lay - a.protect_late, n_lay - 1
    out = a.out or os.path.splitext(a.gguf)[0] + "-depthaware.gguf"
    imat = getattr(a, "imatrix", None)
    if imat and not os.path.isfile(imat) and not getattr(a, "dry", False):
        raise SystemExit(f"--imatrix file not found: {imat}\n"
                         "  generate one first, or drop the flag to build without calibration.")
    build_depthaware(a.llama_dir, a.gguf, out, lo, hi, n_lay,
                     dry=getattr(a, "dry", False), imatrix=imat)
