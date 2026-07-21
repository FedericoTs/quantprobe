"""quant_probe.py -- probe-then-quantize for llama.cpp: measure a model's depth-fragility curve, then
emit the optimal --tensor-type recipe. The direction is MODEL-SPECIFIC (Gemma late 4x, Qwen late 2-3x,
Mistral EARLY 25x -- no config flag or weight statistic predicts it), so probe first, then place bits.

Usage:
  python -m weights.quant_probe --gguf path/to/model-f16.gguf [--bands 4] [--chunks 32]
                                [--eval weights/data/wikitext2_test.raw] [--ngl 99] [--dry-run]

Pipeline (data-free, ~30-60 min): Q6_K reference -> quantize ONE band's FFNs to Q2_K at a time (rest
Q6_K) -> llama-perplexity on identical chunks -> print the curve + the recommended recipe (protect the
spiking band at Q4_K, rest Q2_K).
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = os.path.join(os.path.dirname(HERE), "tools", "llamacpp")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="f16/bf16 (or high-precision) source GGUF")
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--eval", default=os.path.join(HERE, "data", "wikitext2_test.raw"))
    ap.add_argument("--ngl", type=int, default=99, help="GPU layers for perplexity (lower if it OOMs)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    quant = os.path.join(LLAMA, "llama-quantize.exe")
    perp = os.path.join(LLAMA, "llama-perplexity.exe")
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


if __name__ == "__main__":
    main()
