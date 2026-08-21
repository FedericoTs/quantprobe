"""quantprobe auto - one command: model in, running setup out.

    quantprobe auto qwen3-30b --tps 15          # preset
    quantprobe auto unsloth/Qwen3-30B-A3B-GGUF  # any HF GGUF repo (params read from filenames' size)

The two-speed design:
  FAST PATH (this command): detect the machine -> ask the optimizer for the best effective-bits ->
  scan the HF repo's file list and pick the closest-matching quant BY SIZE (bits = size*8/params -
  no fragile name parsing) -> fetch it -> print the prediction and the run command (--run launches).
  CUSTOM PATH (the actual product, printed at the end): probe YOUR model's fragile band and build a
  depth-aware GGUF at the same bits - better quality at the same bytes. `probe --apply` does it.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections import namedtuple

from . import optimize as optmod
from . import plan as planmod

# preset -> (repo, total params B, active B, always-active B, moe)
MODEL_REPOS = {
    "qwen3-30b": ("unsloth/Qwen3-30B-A3B-GGUF", 30.5, 3.3, 1.2, True),
    "qwen3-coder": ("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", 30.5, 3.3, 1.2, True),
    "glm-air": ("unsloth/GLM-4.5-Air-GGUF", 110, 12, 2.7, True),
    "laguna-s": ("unsloth/Laguna-S-2.1-GGUF", 117.6, 8, 2.5, True),
    "gemma-12b": ("unsloth/gemma-4-12b-it-GGUF", 11.9, 11.9, 11.9, False),
    "mistral-7b": ("unsloth/Mistral-7B-Instruct-v0.3-GGUF", 7.2, 7.2, 7.2, False),
    "qwen3-235b": (
        "unsloth/Qwen3-235B-A22B-GGUF",
        235.1,
        22,
        7.5,
        True,
    ),  # total exact (HF safetensors)
    "glm-4.7": ("unsloth/GLM-4.7-GGUF", 358.3, 32, 8, True),  # total exact; a/ne [est]
    "glm-744b": (
        "unsloth/GLM-5.2-GGUF",
        753.3,
        32,
        8,
        True,
    ),  # GLM-5.2, total exact 753.3B; a/ne [est]
    "kimi-k2.6": (
        "unsloth/Kimi-K2.6-GGUF",
        1058.6,
        32,
        6,
        True,
    ),  # total exact; a/ne [est]
    "gpt-oss-120b": (
        "unsloth/gpt-oss-120b-GGUF",
        120.4,
        5.1,
        1.8,
        True,
    ),  # total exact, A5.1B official
    "llama-70b": ("unsloth/Llama-3.3-70B-Instruct-GGUF", 70.6, 70.6, 70.6, False),
    "deepseek-16b": (
        "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        15.7,
        2.4,
        1.3,
        True,
    ),
}


def list_ggufs(repo):
    """[(path, size_bytes)] for a HF repo, via the public tree API (recursive: big repos
    keep quants in subfolders). Split multi-part files (-00001-of-000NN) are grouped into
    ONE logical entry: path = first part, size = sum of parts - so effective-bits stays honest.

    Sends the same HF token `fetch` sends: this listing used to be anonymous-only, so a GATED
    repo 401'd `auto` even for a user whose token would have downloaded the files fine one step
    later (caught live when unsloth gated their Mistral repo: `auto mistral-7b` died on the
    listing while `fetch` would have succeeded)."""
    import re

    from . import fetch as fetchmod

    headers = {"User-Agent": "quantprobe-auto"}
    tok = fetchmod.token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310  fixed https://huggingface.co endpoint
        raw = [
            (f["path"], f.get("size", 0))
            for f in json.load(r)
            if f["path"].endswith(".gguf")
            and f.get("size", 0) > 1e8
            and "mmproj" not in f["path"].lower()
            and "draft" not in f["path"].lower()
        ]
    groups, singles = {}, []
    for path, size in raw:
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path)
        if m:
            g = groups.setdefault(m.group(1), {"total": 0, "first": None, "n": int(m.group(3))})
            g["total"] += size
            if m.group(2) == "00001":
                g["first"] = path
        else:
            singles.append((path, size))
    for g in groups.values():
        if g["first"]:
            singles.append((g["first"], g["total"]))
    return singles


def gated_hint(e):
    """Actionable suffix for a 401/403 from the HF tree API - a bare 'Unauthorized' on a repo the
    user can see in a browser reads like a quantprobe bug, when it is a gated repo waiting for a
    token (or for the access request the repo page gates downloads behind)."""
    if any(code in str(e) for code in ("401", "403")):
        return (
            "\n  That repo is gated or private on Hugging Face. Request access on its page if "
            "you have not,\n  then authenticate: set HF_TOKEN, or `huggingface-cli login` "
            "(quantprobe reads the same\n  cached token). Or pick another preset/repo."
        )
    return ""


def split_parts(first_part):
    """All part paths of a split file, derived from the -00001-of-000NN first part."""
    import re

    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", first_part)
    if not m:
        return [first_part]
    base, n = m.group(1), int(m.group(3))
    return [f"{base}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


WIKI_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"


def ensure_eval(dest_dir):
    """The held-out corpus for probing, fetched once (1.3 MB)."""
    import io
    import zipfile

    path = os.path.join(dest_dir, "wiki.test.raw")
    if os.path.isfile(path):
        return path
    print("[quantprobe auto] fetching the WikiText-2 eval corpus (1.3 MB, one time)...")
    with urllib.request.urlopen(WIKI_URL, timeout=60) as r:  # nosec B310  fixed https://huggingface.co URL
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    with z.open("wikitext-2-raw/wiki.test.raw") as f, open(path, "wb") as out:
        out.write(f.read())
    return path


def pick_source(files, t):
    """High-precision source for the custom build: prefer Q8-class (smallest requantizable), else max bits."""
    cands = []
    for path, size in files:
        if "-of-" in path.lower():
            continue  # split multi-part files: out of scope for v1
        bits = size * 8 / (t * 1e9)
        cands.append((bits, size, path))
    if not cands:
        return None
    hi = sorted([c for c in cands if c[0] >= 7.5], key=lambda c: c[1])
    return hi[0] if hi else max(cands)


def _wizard(a):
    """No model given: detect, ask two questions, go. The full pipeline with zero flags."""
    print(
        "quantprobe auto - interactive (no flags needed; `quantprobe auto --help` for the full list)"
    )
    try:
        from . import detect as detmod

        d, _ = detmod.detect()
        print(
            f"\n  this machine [auto-detected]: {d['vram']:g} GB VRAM @ {d['vram_bw']:g} GB/s | "
            f"{d['ram']:g} GB RAM @ {d['ram_bw']:g} GB/s | disk {d['disk_bw']:g} GB/s"
        )
    except Exception:
        pass
    try:
        m = (
            input(
                f"\n  model - preset ({', '.join(MODEL_REPOS)}) or HF GGUF repo id [qwen3-30b]: "
            ).strip()
            or "qwen3-30b"
        )
        print("\n  [1] best standard quant for this machine, ready to run  (skips quantization)")
        print(
            "  [2] full custom: probe YOUR model's fragile layers (~30-60 min), build its personalized GGUF"
        )
        print("  [3] hit a speed target (asks for tok/s)")
        c = input("  choice [1]: ").strip() or "1"
        if c == "2":
            a.custom = True
        elif c == "3":
            a.tps = float(input("  target tok/s: ").strip())
        r = input("  launch chat when ready? [Y/n]: ").strip().lower()
        a.run = r != "n"
    except EOFError:
        raise SystemExit(
            "no model given and no terminal to ask. Pass one: quantprobe auto qwen3-30b [--custom] [--run]"
        )
    a.target = m


#: Everything the law needs to know about the model. auto hands the law EXPLICIT parameters
#: rather than a preset name (it must - a raw HF repo has no preset), so every fact the preset
#: carried has to be transferred deliberately. Spread across bare assignments, one was forgotten
#: - the layer count - and the flagship path silently lost its best placement for every preset
#: MoE (v1.11.1). Naming the record makes the set of facts reviewable, and `apply_to` makes the
#: transfer one auditable step instead of N chances to forget.
ModelSpec = namedtuple("ModelSpec", "repo total active always_active moe n_layer")


def atlas_lines(rec):
    """The two facts a measured recipe carries: where the fragile band is, and what proved it.

    `auto`'s preset table and the measured recipe atlas grew as separate lists, so the command
    that spends the hours never asked whether the answer was already recorded. Both callers below
    print these; keeping the wording in one place stops the two paths from drifting apart."""
    lo, hi = rec["probe"]["fragile_band"]
    return [
        (
            f"  fragile band: layers {lo}-{hi} ({rec['probe']['shape']}-fragile, "
            f"{rec['probe']['fragility_ratio']}x the median band)"
        ),
        f"  evidence:     {rec['provenance']['raw_log']}",
    ]


def resolve_model(a, target):
    """Resolve the model once: preset if we know it, explicit parameters otherwise."""
    if target in MODEL_REPOS:
        repo, t, ac, ne, moe = MODEL_REPOS[target]
    else:
        repo = target
        if not getattr(a, "total", None):
            # Before refusing: is this a model we have MEASURED? The fragility atlas and the
            # `auto` preset table are separate lists, so `auto qwen3.6-35b` used to dead-end on
            # the one model we published a build of - the exact name someone arrives with after
            # reading its HuggingFace card. A recipe cannot drive `auto` (it carries no fetchable
            # high-precision source repo), but answering "not a preset" while holding a measured
            # band and a finished GGUF is the tool withholding what it knows.
            from . import recipes as recmod

            rec = recmod.find(key=target)
            if rec:
                msg = [
                    f"'{target}' is not an `auto` preset - but quantprobe HAS MEASURED it.",
                    "",
                    *atlas_lines(rec),
                ]
                built = recmod.prebuilt_notice(rec)
                if built:
                    msg += ["", built.replace("[quantprobe] ", "  ")]
                msg += [
                    "",
                    "  build it yourself from a high-precision source (f16/bf16/Q8_0):",
                    f"    quantprobe quantize --gguf <source.gguf> --recipe {target}",
                    "",
                    "  `auto` needs a fetchable source repo, which this recipe does not carry yet.",
                ]
                raise SystemExit("\n".join(msg))
            raise SystemExit(
                f"'{target}' is not a preset ({', '.join(MODEL_REPOS)}) - for a raw HF "
                "repo also pass --total (B params) and, for MoE, --active/--always-active"
            )
        t, ac = a.total, a.active or a.total
        ne = a.always_active or (ac if ac >= t * 0.9 else ac * 0.35)
        moe = ac < t * 0.9
    return ModelSpec(
        repo=repo,
        total=t,
        active=ac,
        always_active=ne,
        moe=moe,
        n_layer=planmod.effective_n_layer(a, target),
    )


def apply_to(spec, a):
    """Transfer EVERY field of the record onto the args downstream commands read.

    Add a field to ModelSpec and forget it here and t_auto_transfers_every_model_field fails.
    That test exists because forgetting one silently degraded a recommendation by 20% rather
    than raising anything.
    """
    a.total, a.active, a.always_active = spec.total, spec.active, spec.always_active
    a.n_layer = spec.n_layer
    a.model = None  # deliberate: explicit parameters, not a preset lookup
    return spec


def local_spec_or_none(local, expected_size, n_parts):
    """Read a downloaded GGUF's header - but ONLY if the file is actually complete.

    Returns (spec_or_None, note_or_None). Split out of `run` so the decision is testable
    against a real truncated file rather than asserted on by reading source text - the first
    guard written for this was a source-inspection test, and disabling the size check left it
    green, which is the decorative-test pattern this suite exists to catch.

    "Already on disk" is not "complete". An interrupted `fetch` leaves a partial GGUF at exactly
    the path `auto` probes, and there are two failure modes:

      LOUD  from_gguf RAISES. Reproduced: a 200 KB slice of a 531 MB GGUF gives ValueError out
            of the tensor-shape reader. Unguarded, that killed the whole `auto` command with a
            traceback instead of falling back to the estimate it had already computed.
      QUIET from_gguf SUCCEEDS on a truncated file and returns a plausible-but-wrong spec, so
            `auto` prints a confident number for a model that is not there. This is the worse
            one, and only the size check catches it. A crash is recoverable; a confident wrong
            answer is the defect class this project exists to remove.
    """
    if not os.path.isfile(local):
        return None, None
    have = os.path.getsize(local)
    # multi-part repos: `expected_size` is the total while `local` is one shard, so the
    # comparison does not apply and is skipped rather than made to look meaningful.
    if n_parts == 1 and expected_size and have != expected_size:
        return None, (
            f"{os.path.basename(local)} is on disk but INCOMPLETE - {have / 1e9:.2f} GB "
            f"of {expected_size / 1e9:.2f} GB ({have / expected_size * 100:.1f}%). Its header "
            f"is not trusted; using the pre-download estimate instead. Finish it with "
            f"`quantprobe fetch --force` before quoting a number from this file."
        )
    from . import spec as specmod

    try:
        return specmod.from_gguf(local), None
    except Exception as e:
        return None, (
            f"{os.path.basename(local)} is on disk but its header could not be read "
            f"({type(e).__name__}). Falling back to the pre-download estimate. If the "
            f"download was interrupted, `quantprobe fetch --force` will replace it."
        )


def run(a):
    if a.target is None:
        _wizard(a)
    target = a.target
    spec = apply_to(resolve_model(a, target), a)
    repo, t, ac, ne, moe = (
        spec.repo,
        spec.total,
        spec.active,
        spec.always_active,
        spec.moe,
    )

    # 1. what does the optimizer want on THIS machine?
    a.gguf = None
    ranked = optmod.run(a)
    want_bits = ranked[0]["bits"]
    _, _, _, _, _, (vc, vb, rc, rb, db, geta, gl), ctx, kvp = optmod.resolve(a)

    # 2. pick the closest file BY SIZE (bits = size*8/params; honest, format-agnostic)
    try:
        files = list_ggufs(repo)
    except Exception as e:
        raise SystemExit(f"could not list {repo}: {e}{gated_hint(e)}")
    if not files:
        raise SystemExit(f"no GGUF files found in {repo}")
    scored = []
    for path, size in files:
        bits = size * 8 / (t * 1e9)
        if bits < 1.0 or bits > 9:
            continue
        _, _, cfgs = planmod.evaluate(
            t,
            ac,
            ne,
            moe,
            bits,
            vc,
            vb,
            rc,
            rb,
            db,
            geta,
            1.0,
            gl,
            ctx=ctx,
            kvp=kvp,
            n_layer=planmod.effective_n_layer(a, target),
        )
        cfgs = [c for c in cfgs if "expert cache" not in c[0]]
        if not cfgs:
            continue
        scored.append((abs(bits - want_bits), -cfgs[0][1], path, size, bits, cfgs[0]))
    if not scored:
        avail = sorted({round(s * 8 / (t * 1e9), 1) for _, s in files})
        raise SystemExit(
            f"no ready-to-run quant in {repo} for this machine yet (available bit-levels: {avail}).\n"
            "  Big new models often publish low-bit quants days after release. Meanwhile:\n"
            "  - predictions for any bits now:  quantprobe plan --model <preset> --bits 2.5\n"
            + (
                "  - or build your own from the high-precision source: re-run with --custom\n"
                if pick_source(files, t)
                else ""
            )
        )
    scored.sort()
    _, _, path, size, bits, best = scored[0]
    if abs(bits - want_bits) > 2:
        print(
            f"\n[quantprobe auto] note: no ~{want_bits:g}-bit-class file in {repo} yet - closest "
            f"available is {bits:.1f}-bit. Low-bit quants for new models often land days after "
            f"release; plan/optimize predict any bits meanwhile."
        )

    # Does this machine NEED to go below ~3 bits? That is the real question, and it is not the
    # same as which bit-width is fastest. This gate used to read `want_bits >= 3.5` - the speed
    # winner - which only ever fired because the (now-refuted, pre-registration #16) sub-4-bit
    # decode collapse made low bits look slow. With that bug removed the speed winner is always
    # the lowest bits the quality ceiling allows, and the gate became unreachable. The stated
    # rationale was always about viability, so ask that directly: if a >=3.5-bit build runs on
    # this machine without falling back to disk streaming, the surgery is not needed.
    # "Viable" means on the hardware the user already owns: >=3.5 bits, no upgrade to buy
    # (euro == 0), and not falling back to disk streaming. On the reference box a 30B at 4.5 bits
    # needs +16 GB RAM (euro 40) or disk, so the surgery still runs; on a 24 GB rig it is all in
    # VRAM for free, so it is declined.
    roomy = next(
        (
            r
            for r in ranked
            if r["bits"] >= 3.5 and r["euro"] == 0 and "disk" not in r["desc"].lower()
        ),
        None,
    )
    if getattr(a, "custom", False) and roomy and not getattr(a, "force_custom", False):
        print("\n[quantprobe auto --custom] this machine doesn't need the surgery: it runs")
        print(
            f"  ~{roomy['bits']:g}-bit at {roomy['tps']:.0f} tok/s, and at >=3.5 bits standard community quants"
        )
        print("  match the depth-aware recipe on quality - the fragile-band fix only pays below")
        print("  ~3 bits (Laws 1-2). Fetching the optimal standard quant instead.")
        print("  Build anyway: --force-custom.")
        a.custom = False
    if getattr(a, "custom", False):
        src = pick_source(files, t)
        if not src:
            raise SystemExit(
                "no usable high-precision source in that repo (split files unsupported)"
            )
        sbits, ssize, spath = src
        dest = getattr(a, "dir", None) or "./models"
        os.makedirs(dest, exist_ok=True)
        print(
            "\n[quantprobe auto --custom] source: "
            + spath
            + f" ({ssize / 1e9:.1f} GB, {sbits:.1f}-bit)"
        )
        cal = (
            "skipped (--no-imatrix)"
            if getattr(a, "no_imatrix", False)
            else "calibrate (importance matrix)"
        )
        # Say this BEFORE the hours, not after them. A preset can also be a model we already
        # probed, and re-deriving a band that is sitting in our own repo is the most expensive
        # thing this command can silently do.
        from . import recipes as recmod

        known = recmod.find(key=target)
        if known:
            print(f"\n[quantprobe auto --custom] the atlas ALREADY has a band for '{target}':")
            for ln in atlas_lines(known):
                print(ln)
            print("  skip the probe below and build straight from it (minutes, not hours):")
            # The path the fetch below will actually produce, not the bare repo filename - a
            # command the user cannot paste yet is worse than no command.
            print(
                f"    quantprobe quantize --gguf "
                f"{os.path.join(dest, os.path.basename(spath))} --recipe {target}"
            )
            print("  Continuing re-measures it on YOUR file, which is the right call if your")
            print("  source differs from the one above: the band is a property of the weights,")
            print("  not of the name. If it is the same source, you are paying twice.\n")
        print(f"  pipeline: fetch source -> probe the fragile band -> {cal} -> build the")
        print("  depth-aware GGUF. Each stage prints its own time estimate before it starts;")
        print("  on a large source this is HOURS, not minutes. Interrupt anytime; fetch resumes.")
        if getattr(a, "dry", False):
            print("  (--dry: nothing downloaded)")
            return spath
        from . import fetch as fetchmod

        if not fetchmod.fetch(repo, dest, spath, fetchmod.token()):
            raise SystemExit("source download failed (re-run: it resumes)")
        srcfull = os.path.join(dest, os.path.basename(spath))
        evalf = ensure_eval(dest)
        import argparse

        from . import probe as probemod

        out = os.path.join(dest, os.path.basename(spath).rsplit(".gguf", 1)[0] + "-depthaware.gguf")
        pa = argparse.Namespace(
            gguf=srcfull,
            bands=4,
            chunks=32,
            eval=evalf,
            ngl=99,
            workdir=dest,
            llama_dir=getattr(a, "llama_dir", None),
            apply=True,
            out=out,
            dry_run=False,
            imatrix=None if getattr(a, "no_imatrix", False) else "auto",
            imatrix_chunks=100,
        )
        probemod.run(pa)
        print("\n[quantprobe auto --custom] your personalized model:")
        print("  quantprobe run --gguf " + out)
        if getattr(a, "run", False):
            from . import runtime

            a.gguf = out
            a.bits = None
            runtime.run(a)
        return out
    parts = split_parts(path)
    print(f"\n[quantprobe auto] optimizer wants ~{want_bits:g}-bit; closest file in {repo}:")
    print(
        f"  {path}  ({size / 1e9:.1f} GB, {bits:.2f} effective bits"
        + (f", {len(parts)} parts" if len(parts) > 1 else "")
        + ")"
    )
    # C-15: this number is computed from PRESET params + a size-derived bit-width, because the
    # file is not downloaded yet - so it misses everything `plan --gguf` reads from the header:
    # the gather-only embedding (#76), the format mix (#70/#77), the true always-active split.
    # Measured divergence on the flagship: auto 26.2 vs plan 19.5 for the SAME file. If the file
    # is already on disk, use the real thing; otherwise say plainly that it is a pre-download
    # estimate and which command is authoritative.
    # "already on disk" is not the same as "complete". An interrupted `fetch` leaves a partial
    # file at exactly this path, and there are TWO failure modes, the quieter one being worse:
    #   (a) from_gguf RAISES - reproduced on a 200 KB slice of a 531 MB GGUF, ValueError out of
    #       the tensor-shape reader. Unguarded, that killed the whole `auto` command with a
    #       traceback instead of falling back to the estimate it had already computed.
    #   (b) from_gguf SUCCEEDS on a truncated file and returns a plausible-but-wrong spec, so
    #       `auto` confidently prints a number derived from a model that is not there. A crash
    #       is recoverable; a confident wrong answer is the defect class this project exists to
    #       remove. The size check below is what catches (b), and it is the more important half.
    _local = os.path.join(getattr(a, "dir", None) or "./models", os.path.basename(path))
    _s, _why = local_spec_or_none(_local, size, len(parts))
    if _why:
        print("  NOTE: " + _why)
    if _s:
        _, _, _c = planmod.evaluate(
            _s["t"],
            _s["a"],
            _s["ne"],
            _s["moe"],
            _s["bits"],
            vc,
            vb,
            rc,
            rb,
            db,
            geta,
            1.0,
            gl,
            ctx=ctx,
            kvp=_s["kvp"],
            n_layer=_s["n_layer"],
            true_size_gb=size / 1e9,
            codebook_share=_s.get("codebook_share", 0.0),
        )
        _c = [x for x in _c if "expert cache" not in x[0]]
        if _c:
            _b = max(_c, key=lambda x: x[1])
            print(
                f"  predicted on this machine: {_b[1]:.1f} tok/s  ({_b[0]})  [from the file's "
                f"own header - already on disk]"
            )
            best = _b
    else:
        print(f"  predicted on this machine: ~{best[1]:.1f} tok/s  ({best[0]})")
        print(
            "  NOTE: pre-download estimate from preset params - it cannot see this file's "
            "format mix or exact byte layout. After the download, `quantprobe plan --gguf "
            "<file>` reads the header and is the authoritative number (they have differed by "
            "34% on a MoE flagship)."
        )
    if getattr(a, "dry", False):
        print("  (--dry: nothing downloaded)")
        return path
    # 3. fetch it (resumable; all parts when split), then hand off
    from . import fetch as fetchmod

    dest = getattr(a, "dir", None) or "./models"
    os.makedirs(dest, exist_ok=True)
    for i, part in enumerate(parts):
        if len(parts) > 1:
            print(f"[quantprobe auto] part {i + 1}/{len(parts)}: {os.path.basename(part)}")
        if not fetchmod.fetch(repo, dest, part, fetchmod.token()):
            raise SystemExit("download failed (it resumes: re-run the same command)")
    full = os.path.join(dest, os.path.basename(path))
    print("\n[quantprobe auto] ready. Run it:")
    print(f"  quantprobe run --gguf {full}")
    print("\n  Better quality at the SAME size: rerun with --custom - it probes YOUR model's")
    print("  fragile band (~30-60 min) and builds a depth-aware GGUF personalized to it.")
    if getattr(a, "run", False):
        from . import runtime

        a.gguf = full
        a.bits = None  # let autospec read the real file
        runtime.run(a)
    return full
