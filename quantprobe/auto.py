"""quantprobe auto — one command: model in, running setup out.

    quantprobe auto qwen3-30b --tps 15          # preset
    quantprobe auto unsloth/Qwen3-30B-A3B-GGUF  # any HF GGUF repo (params read from filenames' size)

The two-speed design:
  FAST PATH (this command): detect the machine -> ask the optimizer for the best effective-bits ->
  scan the HF repo's file list and pick the closest-matching quant BY SIZE (bits = size*8/params —
  no fragile name parsing) -> fetch it -> print the prediction and the run command (--run launches).
  CUSTOM PATH (the actual product, printed at the end): probe YOUR model's fragile band and build a
  depth-aware GGUF at the same bits — better quality at the same bytes. `probe --apply` does it.
"""
from __future__ import annotations
import json, urllib.request

from . import plan as planmod
from . import optimize as optmod

# preset -> (repo, total params B, active B, always-active B, moe)
MODEL_REPOS = {
    "qwen3-30b":  ("unsloth/Qwen3-30B-A3B-GGUF", 30.5, 3.3, 1.2, True),
    "qwen3-coder": ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", 30.5, 3.3, 1.2, True),
    "glm-air":    ("unsloth/GLM-4.5-Air-GGUF", 110, 12, 2.7, True),
    "laguna-s":   ("unsloth/Laguna-S-2.1-GGUF", 117.6, 8, 2.5, True),
    "gemma-12b":  ("unsloth/gemma-4-12b-it-GGUF", 11.9, 11.9, 11.9, False),
    "mistral-7b": ("unsloth/Mistral-7B-Instruct-v0.3-GGUF", 7.2, 7.2, 7.2, False),
}


def list_ggufs(repo):
    """[(path, size_bytes)] for a HF repo, via the public tree API."""
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/main",
                                 headers={"User-Agent": "quantprobe-auto"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return [(f["path"], f.get("size", 0)) for f in json.load(r)
                if f["path"].endswith(".gguf") and f.get("size", 0) > 1e8
                and "mmproj" not in f["path"].lower() and "draft" not in f["path"].lower()]


WIKI_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"


def ensure_eval(dest_dir):
    """The held-out corpus for probing, fetched once (1.3 MB)."""
    import os, zipfile, io
    path = os.path.join(dest_dir, "wiki.test.raw")
    if os.path.isfile(path):
        return path
    print("[quantprobe auto] fetching the WikiText-2 eval corpus (1.3 MB, one time)...")
    with urllib.request.urlopen(WIKI_URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    with z.open("wikitext-2-raw/wiki.test.raw") as f, open(path, "wb") as out:
        out.write(f.read())
    return path


def pick_source(files, t):
    """High-precision source for the custom build: prefer Q8-class (smallest requantizable), else max bits."""
    cands = []
    for path, size in files:
        if "-of-" in path.lower():
            continue                                # split multi-part files: out of scope for v1
        bits = size * 8 / (t * 1e9)
        cands.append((bits, size, path))
    if not cands:
        return None
    hi = sorted([c for c in cands if c[0] >= 7.5], key=lambda c: c[1])
    return hi[0] if hi else max(cands)


def _wizard(a):
    """No model given: detect, ask two questions, go. The full pipeline with zero flags."""
    print("quantprobe auto - interactive (no flags needed; `quantprobe auto --help` for the full list)")
    try:
        from . import detect as detmod
        d, _ = detmod.detect()
        print(f"\n  this machine [auto-detected]: {d['vram']:g} GB VRAM @ {d['vram_bw']:g} GB/s | "
              f"{d['ram']:g} GB RAM @ {d['ram_bw']:g} GB/s | disk {d['disk_bw']:g} GB/s")
    except Exception:
        pass
    try:
        m = input(f"\n  model - preset ({', '.join(MODEL_REPOS)}) or HF GGUF repo id [qwen3-30b]: ").strip() or "qwen3-30b"
        print("\n  [1] best standard quant for this machine, ready to run  (skips quantization)")
        print("  [2] full custom: probe YOUR model's fragile layers (~30-60 min), build its personalized GGUF")
        print("  [3] hit a speed target (asks for tok/s)")
        c = input("  choice [1]: ").strip() or "1"
        if c == "2":
            a.custom = True
        elif c == "3":
            a.tps = float(input("  target tok/s: ").strip())
        r = input("  launch chat when ready? [Y/n]: ").strip().lower()
        a.run = r != "n"
    except EOFError:
        raise SystemExit("no model given and no terminal to ask. Pass one: quantprobe auto qwen3-30b [--custom] [--run]")
    a.target = m


def run(a):
    if a.target is None:
        _wizard(a)
    target = a.target
    if target in MODEL_REPOS:
        repo, t, ac, ne, moe = MODEL_REPOS[target]
    else:
        repo = target
        if not getattr(a, "total", None):
            raise SystemExit(f"'{target}' is not a preset ({', '.join(MODEL_REPOS)}) - for a raw HF "
                             "repo also pass --total (B params) and, for MoE, --active/--always-active")
        t, ac = a.total, a.active or a.total
        ne = a.always_active or (ac if ac >= t * 0.9 else ac * 0.35)
        moe = ac < t * 0.9
    a.total, a.active, a.always_active, a.model = t, ac, ne, None

    # 1. what does the optimizer want on THIS machine?
    a.gguf = None
    ranked = optmod.run(a)
    want_bits = ranked[0]["bits"]
    _, _, _, _, _, (vc, vb, rc, rb, db, geta, gl), ctx, kvp = optmod.resolve(a)

    # 2. pick the closest file BY SIZE (bits = size*8/params; honest, format-agnostic)
    try:
        files = list_ggufs(repo)
    except Exception as e:
        raise SystemExit(f"could not list {repo}: {e}")
    if not files:
        raise SystemExit(f"no GGUF files found in {repo}")
    scored = []
    for path, size in files:
        bits = size * 8 / (t * 1e9)
        if bits < 1.0 or bits > 9:
            continue
        _, _, cfgs = planmod.evaluate(t, ac, ne, moe, bits, vc, vb, rc, rb, db, geta, 1.0, gl,
                                      ctx=ctx, kvp=kvp)
        cfgs = [c for c in cfgs if "expert cache" not in c[0]]
        if not cfgs:
            continue
        scored.append((abs(bits - want_bits), -cfgs[0][1], path, size, bits, cfgs[0]))
    if not scored:
        raise SystemExit("no usable quant in that repo for this machine")
    scored.sort()
    _, _, path, size, bits, best = scored[0]

    if getattr(a, "custom", False) and want_bits >= 3.5 and not getattr(a, "force_custom", False):
        print(f"\n[quantprobe auto --custom] this machine doesn't need the surgery: the optimizer")
        print(f"  wants ~{want_bits:g}-bit here, and at >=3.5 bits standard community quants match the")
        print(f"  depth-aware recipe on quality - the fragile-band fix only pays below ~3 bits (Laws 1-2).")
        print(f"  Fetching the optimal standard quant instead. Build anyway: --force-custom.")
        a.custom = False
    if getattr(a, "custom", False):
        src = pick_source(files, t)
        if not src:
            raise SystemExit("no usable high-precision source in that repo (split files unsupported)")
        sbits, ssize, spath = src
        import os
        dest = getattr(a, "dir", None) or "./models"
        os.makedirs(dest, exist_ok=True)
        print("\n[quantprobe auto --custom] source: " + spath + f" ({ssize/1e9:.1f} GB, {sbits:.1f}-bit)")
        print("  pipeline: fetch source -> probe the fragile band (~30-60 min) -> build the")
        print("  depth-aware GGUF. Interrupt anytime; the fetch resumes.")
        if getattr(a, "dry", False):
            print("  (--dry: nothing downloaded)")
            return spath
        from . import fetch as fetchmod
        if not fetchmod.fetch(repo, dest, spath, fetchmod.token()):
            raise SystemExit("source download failed (re-run: it resumes)")
        srcfull = os.path.join(dest, os.path.basename(spath))
        evalf = ensure_eval(dest)
        from . import probe as probemod
        import argparse
        out = os.path.join(dest, os.path.basename(spath).rsplit(".gguf", 1)[0] + "-depthaware.gguf")
        pa = argparse.Namespace(gguf=srcfull, bands=4, chunks=32, eval=evalf, ngl=99,
                                workdir=dest, llama_dir=getattr(a, "llama_dir", None),
                                apply=True, out=out, dry_run=False)
        probemod.run(pa)
        print("\n[quantprobe auto --custom] your personalized model:")
        print("  quantprobe run --gguf " + out)
        if getattr(a, "run", False):
            from . import runtime
            a.gguf = out; a.bits = None
            runtime.run(a)
        return out
    print(f"\n[quantprobe auto] optimizer wants ~{want_bits:g}-bit; closest file in {repo}:")
    print(f"  {path}  ({size/1e9:.1f} GB, {bits:.2f} effective bits)")
    print(f"  predicted on this machine: {best[1]:.1f} tok/s  ({best[0]})")
    if getattr(a, "dry", False):
        print("  (--dry: nothing downloaded)")
        return path
    # 3. fetch it (resumable), then hand off
    from . import fetch as fetchmod
    dest = getattr(a, "dir", None) or "./models"
    import os
    os.makedirs(dest, exist_ok=True)
    ok = fetchmod.fetch(repo, dest, path, fetchmod.token())
    if not ok:
        raise SystemExit("download failed (it resumes: re-run the same command)")
    full = os.path.join(dest, os.path.basename(path))
    print(f"\n[quantprobe auto] ready. Run it:")
    print(f"  quantprobe run --gguf {full}")
    print("\n  Better quality at the SAME size: rerun with --custom - it probes YOUR model's")
    print("  fragile band (~30-60 min) and builds a depth-aware GGUF personalized to it.")
    if getattr(a, "run", False):
        from . import runtime
        a.gguf = full
        a.bits = None                              # let autospec read the real file
        runtime.run(a)
    return full
