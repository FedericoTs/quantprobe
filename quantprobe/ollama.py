"""quantprobe audit-ollama - price the models you already have, on the machine you already have.

    quantprobe audit-ollama              # read the store, predict, print the flags
    quantprobe audit-ollama --measure    # also RUN ollama and compare against the prediction

WHY THIS IS DIFFERENT FROM `plan`. `plan` needs you to know which file you want. This needs
nothing: ollama's blobs ARE GGUFs, so the real header is already on your disk. Params come from
the file, not from parsing a name - which matters because ollama tags say "7b" and the header
here says 7.615616512B total with 4.92 effective bits.

WHAT IT WILL NOT DO. It will not tell you "you are leaving X% on the table" unless you pass
--measure, because without measuring ollama we would be comparing our prediction against our
own assumption about a runtime we did not run. That is the shape of every defect this project
has published: a confident number with nothing behind it. --measure runs both sides and reports
what actually happened, including when the prediction is the thing that was wrong.
"""
from __future__ import annotations
import json, os, glob, shutil, subprocess, time

MODEL_MEDIA = "application/vnd.ollama.image.model"


def store_root(explicit=None):
    """Where ollama keeps its blobs. OLLAMA_MODELS wins, then the per-OS default."""
    if explicit:
        return explicit
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    return os.path.expanduser(os.path.join("~", ".ollama", "models"))


def installed(root=None):
    """[(name, blob_path, size_bytes)] for every model in the store.

    Walks manifests/ rather than calling `ollama list`, so it works when the daemon is not
    running and it gives the blob path directly. The manifest name is its path under
    manifests/<registry>/<namespace>/<model>/<tag>, which is how ollama itself addresses it.
    """
    root = store_root(root)
    mdir = os.path.join(root, "manifests")
    out = []
    if not os.path.isdir(mdir):
        return out
    for f in glob.glob(os.path.join(mdir, "**"), recursive=True):
        if not os.path.isfile(f):
            continue
        try:
            m = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue                       # not a manifest; skip rather than guess
        layers = [l for l in m.get("layers", []) if l.get("mediaType") == MODEL_MEDIA]
        if not layers:
            continue
        rel = os.path.relpath(f, mdir).replace(os.sep, "/").split("/")
        name = f"{rel[-2]}:{rel[-1]}" if len(rel) >= 2 else "/".join(rel)
        digest = layers[0]["digest"].split(":")[-1]
        blob = os.path.join(root, "blobs", f"sha256-{digest}")
        if os.path.isfile(blob):
            out.append((name, blob, os.path.getsize(blob)))
    return sorted(set(out))


def ollama_bin():
    return shutil.which("ollama") or shutil.which("ollama.exe")


def loaded_placement(name):
    """What ollama ACTUALLY chose: (gpu_percent, ctx) from `ollama ps`, or (None, None).

    This is the difference between an honest audit and a misleading one. quantprobe prices
    all-in-VRAM; if ollama split the model instead, the two numbers describe DIFFERENT
    placements and comparing them would look like a speed claim while actually being a
    category error. Measured here on a 6 GB card: ollama loaded qwen2.5:7b as 16%/84% CPU/GPU
    even though the model fits, so the gap was never evidence about anyone's prediction.
    """
    b = ollama_bin()
    if not b:
        return None, None
    try:
        out = subprocess.run([b, "ps"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None, None
    import re
    for line in out.splitlines():
        if not line.startswith(name.split(":")[0]):
            continue
        m = re.search(r"(\d+)%/(\d+)%\s*CPU/GPU", line)
        gpu = int(m.group(2)) if m else (100 if "100% GPU" in line else None)
        c = re.search(r"\b(\d{3,6})\b\s*$", line.replace("from now", "").strip())
        ctxm = re.search(r"\s(\d{3,6})\s+\d+\s*(?:minutes?|seconds?|hours?)", line)
        return gpu, int(ctxm.group(1)) if ctxm else None
    return None, None


def bench_blob(blob, ngl, ctx, bench_bin, n=64, reps=3, timeout=900):
    """Time ONE placement of the ollama blob with llama-bench. Returns tok/s or None.

    Both sides of the comparison go through this, at the SAME context depth, on the SAME
    file - so the number that comes out is a placement difference and nothing else.
    """
    cmd = [bench_bin, "-m", blob, "-ngl", str(ngl), "-n", str(n), "-p", "0", "-r", str(reps),
           "-o", "json"]
    if ctx:
        cmd += ["-d", str(ctx)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    out = p.stdout + p.stderr
    try:
        return json.loads(out[out.index("["):out.rindex("]") + 1])[0].get("avg_ts")
    except Exception:
        return None


def find_llamabench():
    for name in ("llama-bench", "llama-bench.exe"):
        w = shutil.which(name)
        if w:
            return w
    env = os.environ.get("QP_LLAMACPP")
    if env and os.path.isfile(os.path.join(env, "llama-bench.exe")):
        return os.path.join(env, "llama-bench.exe")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _ in range(8):
        c = os.path.join(here, "tools", "llamacpp-b10098", "llama-bench.exe")
        if os.path.isfile(c):
            return c
        nh = os.path.dirname(here)
        if nh == here:
            break
        here = nh
    return None


def measure(name, prompt="Count from 1 to 40, one number per line.", timeout=300):
    """Actually run ollama and read ITS OWN reported eval rate.

    Parsed from `--verbose` rather than timed by us: ollama reports eval_count and
    eval_duration, which excludes prompt processing and model load. Timing the wall clock
    ourselves would fold load time into a decode number and flatter or damage it depending on
    whether the model happened to be resident.
    """
    b = ollama_bin()
    if not b:
        return None, "ollama not on PATH"
    try:
        p = subprocess.run([b, "run", name, "--verbose", prompt],
                           capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    r = _parse_rate(p.stdout + p.stderr)
    if r is None:
        return None, "could not parse a generation eval rate from ollama --verbose"
    return r, None


def _parse_rate(out):
    """ollama's GENERATION rate from --verbose output, or None. Split out to be testable.

    MUST NOT match "prompt eval rate:", which ollama prints FIRST. A bare re.search for
    "eval rate" takes that one, and prompt throughput is a different quantity an order of
    magnitude larger - measured on real output, 186.59 against a true generation rate of
    19.92. That misread is not obviously wrong on sight: it looks like a plausible tok/s, and
    it made audit-ollama report ollama as already faster than anything worth recommending.
    Anchoring at line start is the fix.
    """
    import re
    m = re.search(r"^\s*eval rate:\s*([0-9.]+)\s*tokens/s", out, re.M)
    if m:
        return float(m.group(1))
    ec = re.search(r"^\s*eval count:\s*(\d+)", out, re.M)
    ed = re.search(r"^\s*eval duration:\s*([0-9.]+)\s*(ms|s|m)\b", out, re.M)
    if ec and ed:
        v = float(ed.group(1)) * {"ms": 1e-3, "s": 1.0, "m": 60.0}[ed.group(2)]
        return (int(ec.group(1)) / v) if v else None
    return None


def unload(name, need_free_mib=4500, tries=20):
    """Make ollama release the GPU before we time anything else on it.

    Without this the audit contaminates its own comparison. ollama keeps a model resident for
    ~5 minutes after a run: measured here, 5209 MiB of a 6144 MiB card still held. llama-bench
    then cannot fit all layers, silently spills to CPU, and the tool reports the recommended
    config as SLOWER than the one it is meant to beat - confidently, and backwards. Measured
    contaminated: -ngl 99 scored 4.56 tok/s; clean, the same command scores 18.83.

    Returns True only when the GPU is actually free. The caller must refuse to compare
    otherwise - a contaminated comparison that produces advice is worse than no comparison.
    """
    b = ollama_bin()
    if b:
        try:
            subprocess.run([b, "stop", name], capture_output=True, text=True, timeout=60)
        except Exception:
            pass
    for _ in range(tries):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
            used, total = [int(x) for x in r.splitlines()[0].split(",")]
            if (total - used) >= need_free_mib:
                return True, total - used
        except Exception:
            return None, None          # no nvidia-smi: cannot verify, so do not claim clean
        time.sleep(3)
    return False, None


def run(a):
    from . import plan as planmod, spec as specmod
    root = store_root(getattr(a, "store", None))
    models = installed(root)
    print(f"quantprobe audit-ollama - store: {root}")
    if not models:
        print("\n  no ollama models found. Pull one first:  ollama pull qwen2.5:7b")
        print("  (or point at a different store with --store, or set OLLAMA_MODELS)")
        return
    vc, vb, rc, rb, db, geta, gl, hw = planmod.resolve_hw(a, announce=False)
    print(f"  this machine: vram {vc:g} GB @ {vb:g} | ram {rc:g} GB @ {rb:g} | disk {db:g} GB/s"
          + (f"  [{hw['hint']}]" if hw.get("hint") else ""))
    print(f"  {len(models)} model(s) on disk\n")

    for name, blob, size in models:
        print(f"  {name}   ({size/1e9:.2f} GB)")
        try:
            s = specmod.from_gguf(blob)
        except Exception as e:
            print(f"     header unreadable ({type(e).__name__}) - skipped, not guessed\n")
            continue
        print(f"     from the file's own header: {s['t']:.2f}B total, {s['a']:.2f}B active, "
              f"{s['bits']:.2f} effective bits, {s['n_layer']} layers"
              + (", MoE" if s["moe"] else ", dense"))
        # Same evaluate() the `plan` command calls, with hardware from the same resolver, so
        # the two commands cannot quote different numbers for one file on one box.
        _, _, allrows = planmod.evaluate(
            s["t"], s["a"], s["ne"], s["moe"], s["bits"], vc, vb, rc, rb, db, geta,
            1.0, gl, ctx=getattr(a, "ctx", 0) or 0, kvp=s["kvp"], n_layer=s["n_layer"],
            true_size_gb=size / 1e9, codebook_share=s.get("codebook_share", 0.0))
        # only rows a user can actually execute: the expert-cache and layer-streaming rows
        # need runtimes ollama does not ship, so recommending them here would be a non-command
        rows = [r for r in allrows if getattr(r, "runnable", True)]
        if rows:
            best = rows[0]
            print(f"     quantprobe predicts: {best[1]:.1f} tok/s  ({best[0]})")
            if best[3]:
                print(f"     flags:  {best[3]}")
        if getattr(a, "measure", False):
            print("     running ollama to see what it actually does...", flush=True)
            got, err = measure(name)
            gpu_pct, octx = loaded_placement(name)
            if err:
                print(f"     ollama measurement FAILED: {err} - no comparison made")
                print("")
                continue
            print(f"     ollama MEASURED: {got:.1f} tok/s (its own eval rate)")
            if gpu_pct is not None:
                print(f"     ollama's PLACEMENT: {100-gpu_pct}%/{gpu_pct}% CPU/GPU"
                      + (f" at ctx {octx}" if octx else ""))
            # The comparison that means something: BOTH placements timed by the same tool, on
            # the SAME blob, at the SAME depth. Prediction-vs-ollama would compare different
            # placements and read as a speed claim while being a category error.
            bb = find_llamabench()
            if bb and gpu_pct is not None and gpu_pct < 100:
                nl = s["n_layer"] or 32
                theirs_ngl = max(0, round(nl * gpu_pct / 100))
                print("     unloading ollama so it stops holding the GPU...", flush=True)
                clean, freed = unload(name)
                if clean is None:
                    print("     cannot read GPU memory (no nvidia-smi), so a clean comparison")
                    print("     cannot be VERIFIED. Refusing to compare rather than guess.")
                    print("")
                    continue
                if not clean:
                    print("     ollama is still holding the GPU. REFUSING to compare - a")
                    print("     contaminated bench reports the right config as slower and would")
                    print("     hand you exactly the wrong advice. Retry in a minute.")
                    print("")
                    continue
                print(f"     GPU free: {freed} MiB. Timing BOTH placements at ctx {octx or 0}...",
                      flush=True)
                mine = bench_blob(blob, 99, octx or 0, bb)
                thrs = bench_blob(blob, theirs_ngl, octx or 0, bb)
                if mine and thrs:
                    print(f"       ollama's split  (-ngl {theirs_ngl}): {thrs:.2f} tok/s")
                    print(f"       all layers      (-ngl 99):        {mine:.2f} tok/s")
                    if mine > thrs:
                        print(f"     >> {mine/thrs:.2f}x AVAILABLE by forcing all layers onto the "
                              f"GPU. It fits - no OOM at ctx {octx or 0}.")
                        print(f"        In ollama:  PARAMETER num_gpu {nl}   (Modelfile), or set "
                              f"OLLAMA_NUM_GPU={nl}")
                    else:
                        print(f"     >> ollama's split is FASTER here ({thrs/mine:.2f}x). Its "
                              f"conservatism is correct on this box; no change recommended.")
                    print(f"        Both timed by llama-bench on the same blob at the same depth, "
                          f"so this is a\n        placement difference and nothing else. ollama's "
                          f"own {got:.1f} sits below both - the\n        remainder is its server "
                          f"and sampling overhead, which this does not measure.")
                else:
                    print("     could not time both placements - no comparison claimed")
            elif gpu_pct == 100:
                print("     ollama already has every layer on the GPU - nothing to recommend")
            elif not bb:
                print("     llama-bench not found, so no measured comparison was made. The")
                print("     prediction above is NOT a substitute for one.")
        print("")

    if not getattr(a, "measure", False):
        print("  No speed comparison was made. Numbers above are PREDICTIONS for this machine;")
        print("  `--measure` runs ollama and reports what it actually does, including when the")
        print("  prediction is what turns out to be wrong.")
