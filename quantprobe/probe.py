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


def ppl(perp, gguf, eval_file, chunks, ngl, dry):
    print("  $", perp, "-m", gguf, "-f", eval_file, "--chunks", str(chunks), "-ngl", str(ngl), flush=True)
    if dry:
        return None
    p = subprocess.run([perp, "-m", gguf, "-f", eval_file, "--chunks", str(chunks), "-ngl", str(ngl)],
                       capture_output=True, text=True, errors="replace")
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", p.stdout + p.stderr)
    return float(m.group(1)) if m else None


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
    others = [f"{band_regex(b[0], b[1])}=q2_k" for i, b in enumerate(bands) if i != worst]
    print(f"  fragile band: layers {lo}-{hi} (delta +{deltas[worst]:.2f} vs "
          f"median {sorted(deltas)[len(deltas)//2]:.2f}) -> protect at Q4_K:\n", flush=True)
    flags = " ".join(f'--tensor-type "{o}"' for o in others)
    print(f'  llama-quantize {flags} --tensor-type "{band_regex(lo, hi)}=q4_k" '
          f'--tensor-type "attn_.*=q4_k" --token-embedding-type q4_k \\\n'
          f'    {os.path.basename(a.gguf)} out-depthaware.gguf Q2_K 8', flush=True)
    if getattr(a, "apply", False):
        out = a.out or os.path.splitext(a.gguf)[0] + "-depthaware.gguf"
        print("\n[quantprobe] --apply: building the recommended GGUF now...", flush=True)
        build_depthaware(a.llama_dir, a.gguf, out, lo, hi, bands[-1][1] + 1, dry=a.dry_run)
    else:
        print("\n  (re-run with  --apply --out model-2bit.gguf  to BUILD this GGUF automatically)", flush=True)


def _band_re(lo, hi):
    return "blk\\.(" + "|".join(str(i) for i in range(lo, hi + 1)) + ")\\.ffn_.*"


def build_depthaware(llama_dir, src, out, protect_lo, protect_hi, n_lay,
                     base="Q2_K", protect="q4_k", dry=False):
    """Actually PRODUCE the compressed GGUF: base bits everywhere, fragile band + attention + embed protected."""
    q = os.path.join(find_llama(llama_dir), exe("llama-quantize"))
    cmd = [q, "--allow-requantize"]
    if protect_lo > 0:
        cmd += ["--tensor-type", f"{_band_re(0, protect_lo - 1)}=q2_k"]
    if protect_hi < n_lay - 1:
        cmd += ["--tensor-type", f"{_band_re(protect_hi + 1, n_lay - 1)}=q2_k"]
    cmd += ["--tensor-type", f"{_band_re(protect_lo, protect_hi)}={protect}",
            "--tensor-type", "attn_.*=q4_k", "--token-embedding-type", "q4_k",
            src, out, base, "8"]
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


def quantize(a):
    """Standalone compress: build a depth-aware GGUF from an explicit band (no probing)."""
    n_lay = n_layers(a.gguf)
    if a.protect:
        lo, hi = (int(x) for x in a.protect.split("-"))
    else:
        lo, hi = n_lay - a.protect_late, n_lay - 1
    out = a.out or os.path.splitext(a.gguf)[0] + "-depthaware.gguf"
    build_depthaware(a.llama_dir, a.gguf, out, lo, hi, n_lay, dry=getattr(a, "dry", False))
