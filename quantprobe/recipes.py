"""quantprobe recipes - the community fragility atlas.

Law 3 says where a model breaks under low-bit quantization is model-specific and must be
MEASURED. Measuring it is the expensive part of the pipeline (hours on a large model).

But the result is a property of the MODEL, not of your machine: Qwen3-30B's fragile band is
layers 36-47 whether you measured it on a GTX 1060 or an H100. So it needs measuring once,
globally, ever - and everyone after that skips straight to the build.

A recipe is a few numbers plus its evidence. Every entry carries the raw log it came from, the
eval corpus, and the hardware, because a recipe you cannot check is a recipe you should not use.
"""

from __future__ import annotations

import glob
import json
import os
import re


def _dir():
    """Recipes ship with the package; a repo checkout falls back to the source tree."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipes")
    if os.path.isdir(here):
        return here
    up = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes")
    return up if os.path.isdir(up) else here


def load_all():
    out = []
    for f in sorted(glob.glob(os.path.join(_dir(), "*.json"))):
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except Exception:
            continue  # a malformed contribution must not break the tool
    return out


def find(key=None, arch=None, n_layer=None):
    """Explicit key wins. Otherwise match on (arch, n_layer) - the two things that make a
    recipe applicable. Layer count alone is not enough: different architectures share it."""
    all_r = load_all()
    if key:
        for r in all_r:
            if r["model"]["key"] == key:
                return r
        return None
    if arch and n_layer:
        for r in all_r:
            if r["model"]["arch"] == arch and r["model"]["n_layer"] == n_layer:
                return r
    return None


def _repo_from_url(url):
    """`https://huggingface.co/Owner/Name` -> `Owner/Name`, or None if it isn't an HF model URL.

    A published artifact URL is the only place a recipe records where its build lives, so this
    is what makes `fetch <recipe-key>` able to download it rather than merely name it."""
    if not url:
        return None
    m = re.match(r"https?://(?:www\.)?huggingface\.co/([^/\s]+/[^/\s?#]+)", url)
    return m.group(1) if m else None


REPO_URL = "https://github.com/FedericoTs/quantprobe/blob/master/"


def params_from_gguf(path):
    """Measure the `params` block for a recipe from a real file. Never type these by hand.

    Parameter counts are what `plan --model <key>` needs to answer "will this run on my machine"
    BEFORE the user downloads. They are safe to store because they are a property of the
    architecture, not of the quantization: measured across 12 comparisons (4 quants of Qwen3.5-35B
    from 8.52 to 2.63 bit, 2 of Qwen2.5-7B, 2 of DeepSeek-V2-Lite) the spread was 0.000%.

    The one apparent violation was not one, and it is why `measured_from` is recorded alongside:
    Unsloth's Qwen3.6-35B UD-Q2_K_XL build reports 41 blocks and 35.51B where every other build of
    "the same" model reports 40 and 34.66B. It carries `nextn_predict_layers = 1` - an MTP head
    the other conversions strip. Two files under one name, differing by a whole block. Storing
    which file a number came from is the difference between a measurement and a rumour."""
    from . import spec as specmod

    s = specmod.from_gguf(path)
    return {
        "total_b": round(s["t"], 4),
        "active_b": round(s["a"], 4),
        "always_active_b": round(s["ne"], 4),
        "moe": bool(s["moe"]),
        "kv_per_pos": int(s["kvp"]),  # context pricing; without it `--ctx` falls back to a guess
        "n_layer": s["n_layer"],
        "measured_from": os.path.basename(path),
    }


def params(r):
    """The stored parameter block, if this recipe carries one. None is a normal answer."""
    p = (r or {}).get("params")
    if not p or not all(p.get(k) for k in ("total_b", "active_b", "always_active_b")):
        return None
    return p


def evidence_url(r):
    """The raw log that proves this recipe, as something the reader can actually open.

    The module docstring says a recipe you cannot check is a recipe you should not use - but the
    stored path is repo-relative, so for anyone who arrived through `pip install` the citation
    named a file they do not have. Every recipe's log is committed on master (checked before this
    was added); a path that resolves nowhere would be worse than no citation at all."""
    p = (r.get("provenance") or {}).get("raw_log")
    if not p or "://" in p:
        return p
    return REPO_URL + p.replace("\\", "/").lstrip("./")


def artifact(r):
    """The prebuilt file for this recipe, if someone already built and published it.

    Normalized to a dict so both call sites parse it once. A contribution that puts a bare
    prose string here still renders - the URL is pulled out of it - because a recipe that is
    slightly off-schema should degrade to less information, not to a stack trace."""
    a = (r.get("provenance") or {}).get("published_artifact")
    if not a:
        return None
    if isinstance(a, str):
        m = re.search(r"https?://\S+", a)
        if not m:
            return None
        url = m.group(0).rstrip(".,)")
        return {
            "url": url,
            "repo": _repo_from_url(url),
            "file": None,
            "bytes": None,
            "note": a,
        }
    return {
        "url": a.get("url"),
        "repo": a.get("repo") or _repo_from_url(a.get("url")),
        "file": a.get("file"),
        "bytes": a.get("bytes"),
        "note": a.get("note", ""),
    }


def prebuilt_notice(r):
    """What to print when a recipe already has a published build.

    The whole point of the atlas is skipping work someone else already did. Skipping the PROBE
    saves hours; skipping the BUILD saves hours more, plus the high-precision source download
    that dwarfs the output. If the file exists, say so before the user starts quantizing."""
    a = artifact(r)
    if not a or not a["url"]:
        return None
    size = f", {a['bytes'] / 2**30:.1f} GiB" if a.get("bytes") else ""
    lines = [
        f"[quantprobe] this recipe has ALREADY been built and published{size}:",
        f"  {a['url']}",
    ]
    if a.get("file"):
        lines.append(f"  file: {a['file']}")
    lines.append(
        "  Downloading it is the same bytes as building it, minus the source model and the hours."
    )
    return "\n".join(lines)


def describe(r):
    p, m, pr = r["probe"], r["model"], r["provenance"]
    lo, hi = p["fragile_band"]
    out = (
        f"{m['name']} ({m['n_layer']} layers, {'MoE' if m['moe'] else 'dense'}) - "
        f"fragile band layers {lo}-{hi} ({p['shape']}-fragile, {p['fragility_ratio']}x "
        f"the median band)\n"
        f"    measured {pr['measured']} on {pr['hardware']}, eval {pr['eval']}\n"
        f"    evidence: {evidence_url(r)}"
    )
    a = artifact(r)
    if a and a["url"]:
        size = f" ({a['bytes'] / 2**30:.1f} GiB)" if a.get("bytes") else ""
        out += f"\n    PREBUILT{size}: {a['url']}"
    return out


def run(a):
    """`quantprobe recipes` - what has already been measured, so you don't have to."""
    all_r = load_all()
    if not all_r:
        print("no recipes found (expected in quantprobe/recipes/*.json)")
        return
    print(f"\nquantprobe recipes - {len(all_r)} measured fragility bands\n")
    print("  A recipe is someone's probe result. The fragile band is a property of the MODEL,")
    print("  not your hardware, so it transfers: skip the probe, build in minutes.\n")
    for r in all_r:
        print("  " + describe(r).replace("\n", "\n  "))
        print(f"    use it:  quantprobe quantize --gguf <your-file> --recipe {r['model']['key']}\n")
    print("  Measured one yourself? Open a PR with the JSON and your raw probe log -")
    print("  every entry here carries its evidence, and contributions are held to the same bar.")
