"""Record which flags a real llama.cpp build accepts, tool by tool, for the flag-drift test.

    python weights/llamacpp_flag_snapshot.py --llama-dir <dir with llama-*.exe> \
        --out tests/llamacpp_flags.json

Every flag-drift defect this project shipped (--no-mmap at b10875, -cnv at b10584) passed a green
suite, because the tests knew only the flags the tests' author remembered. This asks the binaries.
Only the option-name column of each --help line is read, so a flag that a description merely
mentions ("replaces --foo") is not counted as accepted. The GPU is hidden while asking: --help
needs no device, and a measurement may be running on it.
"""

import argparse
import json
import os
import re
import subprocess
import sys

TOOLS = ("llama-cli", "llama-server", "llama-bench", "llama-quantize", "llama-perplexity",
         "llama-imatrix", "llama-completion")
FLAG = re.compile(r"(?:^|[\s,\[])(-{1,2}[A-Za-z][\w-]*)")


def accepted(help_text):
    """Flags named in the option column of a llama.cpp --help text."""
    out = set()
    for line in help_text.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if not stripped.startswith("-") or indent > 8:  # continuation lines are deeply indented
            continue
        # The column before the description. Aliases are separated by a comma and padding
        # ("-cnv,  --conversation", "-t,    --threads N"), so only a gap that does not follow a
        # comma ends it - and the lookbehind also refuses whitespace, or the match would start in
        # the middle of that padding.
        names = re.split(r"(?<![,\s])\s{2,}", stripped, maxsplit=1)[0]
        out.update(m.group(1) for m in FLAG.finditer(" " + names))
    return out


def usage_flags(help_text):
    """llama-quantize prints its flags in a bracketed usage block, not an option column."""
    block = help_text.split("\n\n", 1)[0]
    return {m.group(1) for m in FLAG.finditer(block) if "usage" in block}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    snap, version = {}, ""
    for tool in TOOLS:
        exe = os.path.join(a.llama_dir, tool + (".exe" if os.name == "nt" else ""))
        if not os.path.isfile(exe):
            print(f"  {tool}: not in this build, skipped")
            continue
        r = subprocess.run([exe, "--help"], capture_output=True, text=True, errors="replace",
                           timeout=60, env=env, check=False)
        text = r.stdout + r.stderr
        flags = accepted(text) | usage_flags(text)
        snap[tool] = sorted(flags)
        print(f"  {tool}: {len(flags)} flags")
        if not version:
            v = subprocess.run([exe, "--version"], capture_output=True, text=True, errors="replace",
                               timeout=60, env=env, check=False)
            version = (v.stdout + v.stderr).strip().splitlines()
    m = re.search(r"build (\d+), commit (\w+)", " ".join(version))
    if not m:
        sys.exit(f"could not read the build number from --version: {version}")
    doc = {"build": int(m.group(1)), "commit": m.group(2), "tools": snap}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {a.out}: build {doc['build']} ({doc['commit']}), {len(snap)} tools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
