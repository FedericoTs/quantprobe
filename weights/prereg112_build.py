"""Build the three arms of pre-registration #112 (dense 27B, depth-aware vs spread).

Every arm gets IDENTICAL treatment except WHERE the protected blocks sit - that is the whole
experiment. The flag list mirrors quantprobe/probe.py:build_depthaware exactly (base Q2_K, band
Q4_K, attn/ssm Q4_K, shexp Q8_0, nextn Q8_0, token embedding Q4_K), so these arms are comparable
with prereg #104's.

    python weights/prereg112_build.py --arm OURS
    python weights/prereg112_build.py --plan          # print all three commands, build nothing

Disk is tight on this box by design (the user chose to run without deleting models), so each
build refuses to start unless the drive has room, and verifies the output afterwards.
"""

import argparse
import os
import subprocess
import sys

LLAMA = os.environ.get(
    "QUANTPROBE_LLAMA_DIR", "C:/Users/Federico/Documents/evo-compress/tools/llamacpp-b10098"
)
SRC = "D:/evo-compress-data/gguf/Qwen3.8-27B-Q8_0.gguf"
OUT_DIR = os.environ.get("PREREG112_OUT", "C:/qp112")
N_BLOCK = 65  # the FILE's blocks: 0..63 decode + blk.64 MTP (L-33)
N_DECODE = 64
BAND = (51, 63)  # measured under prereg #101, trimmed to the executed stack
MIN_FREE_GB = 11.0  # refuse to start a build that could truncate

# 13 protected blocks either way. SPREAD places the SAME budget evenly across depth instead of
# on the measured band - so the only difference is placement, not how many bytes are spent.
N_PROTECT = BAND[1] - BAND[0] + 1
SPREAD = sorted({int(i * N_DECODE / N_PROTECT) for i in range(N_PROTECT)})


def band_re(blocks):
    return "blk\\.(" + "|".join(str(i) for i in blocks) + ")\\.ffn_.*"


def arm_blocks(arm):
    if arm == "OURS":
        return list(range(BAND[0], BAND[1] + 1))
    if arm == "MTP":
        return list(range(BAND[0], BAND[1] + 2))  # includes blk.64, the never-executed head
    if arm == "SPREAD":
        return list(SPREAD)
    raise SystemExit(f"unknown arm {arm}")


def cmd_for(arm, out):
    protected = arm_blocks(arm)
    rest = [i for i in range(N_BLOCK) if i not in protected]
    q = os.path.join(LLAMA, "llama-quantize.exe")
    cmd = [q, "--allow-requantize"]
    # always-active first: llama.cpp resolves --tensor-type first-match-wins
    cmd += ["--tensor-type", "ffn_.*_shexp.*=q8_0", "--tensor-type", "nextn.*=q8_0"]
    cmd += ["--tensor-type", f"{band_re(rest)}=q2_k"]
    cmd += ["--tensor-type", f"{band_re(protected)}=q4_k"]
    cmd += ["--tensor-type", "attn_.*=q4_k", "--tensor-type", "ssm_.*=q4_k"]
    cmd += ["--token-embedding-type", "q4_k", SRC, out, "Q2_K", "8"]
    return cmd


def free_gb(path):
    import shutil

    return shutil.disk_usage(os.path.dirname(path) or ".").free / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["OURS", "SPREAD", "MTP"])
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"band {BAND[0]}-{BAND[1]} ({N_PROTECT} blocks) | SPREAD at {SPREAD}")
    if a.plan:
        for arm in ("OURS", "SPREAD", "MTP"):
            out = os.path.join(OUT_DIR, f"Qwen3.8-27B-{arm}.gguf")
            print(f"\n--- {arm} -> {out}\n{' '.join(cmd_for(arm, out))}")
        return 0
    if not a.arm:
        sys.exit("pass --arm OURS|SPREAD|MTP, or --plan")
    if not os.path.isfile(SRC):
        sys.exit(f"source missing: {SRC}")

    out = os.path.join(OUT_DIR, f"Qwen3.8-27B-{a.arm}.gguf")
    if os.path.isfile(out):
        print(f"exists already ({os.path.getsize(out):,} B): {out}")
        return 0
    have = free_gb(out)
    if have < MIN_FREE_GB:
        sys.exit(f"ABORT: only {have:.1f} GB free on the output drive, need {MIN_FREE_GB} GB. "
                 "A truncated arm is not a measurement.")
    print(f"[{a.arm}] {have:.1f} GB free, building -> {out}")
    r = subprocess.run(cmd_for(a.arm, out))
    if r.returncode != 0 or not os.path.isfile(out):
        sys.exit(f"[{a.arm}] BUILD FAILED rc={r.returncode}")
    print(f"[{a.arm}] OK {os.path.getsize(out):,} bytes ({free_gb(out):.1f} GB now free)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
