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


def artifact(r):
    """The prebuilt file for this recipe, if someone already built and published it.

    Normalized to a dict so both call sites parse it once. A contribution that puts a bare
    prose string here still renders - the URL is pulled out of it - because a recipe that is
    slightly off-schema should degrade to less information, not to a stack trace."""
    a = (r.get("provenance") or {}).get("published_artifact")
    if not a:
        return None
    if isinstance(a, str):
        import re

        m = re.search(r"https?://\S+", a)
        return (
            {
                "url": m.group(0).rstrip(".,)") if m else None,
                "file": None,
                "bytes": None,
                "note": a,
            }
            if m
            else None
        )
    return {
        "url": a.get("url"),
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
        f"    evidence: {pr['raw_log']}"
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
