"""Prereg #95 harness: stage 1 Morris screening + stage 2 Sobol survivors, serial, resumable.

  python weights/doe_morris.py --dry-run            # print the full design, touch nothing
  python weights/doe_morris.py                      # run the night (operator starts this)
  python weights/doe_morris.py --deadline-hours 9.5

  python weights/doe_morris.py --stage2 --dry-run   # stage-2 Saltelli plan, touch nothing
  python weights/doe_morris.py --stage2 --model 7B  # night 1: the 7B block (192 runs)
  python weights/doe_morris.py --stage2 --model 30B # night 2: the 30B block (280 runs)
  python weights/doe_morris.py --stage2 --model 30B --n-blocks 56   # extend after UNDECIDED

This file EXECUTES docs/DESIGN_DOE_MORRIS.md and decides nothing: factors, levels, seeds,
run counts, timeouts, the CSV schema and its pinned header hash all come from that design,
which was frozen before any screening data existed. The scorer (weights/prereg95_score.py)
carries its OWN copy of the design generator - deliberately, because the scorer's copy is
the frozen precommit expectation and must not follow an edit made here. At score time it
imports build_design() from this file and REFUSES to score if the two copies no longer
regenerate the same 150 run_ids - so drift is caught actively, and the frozen copy wins.
Do not edit the factor tables here without expecting that refusal.

--stage2 EXECUTES docs/DESIGN_DOE_SOBOL.md section 4 the same way, with the same physics:
the Morris survivors stay variable at their UNCHANGED stage-1 levels, the non-survivors
sit FIXED at the stage-1 best-observed levels, and the seeded Saltelli A/B/AB_i blocks
land in a separate CSV (weights/data/doe_sobol_stage2.csv) under its own pinned header.
Stage-2 mode never opens the stage-1 CSV; run_ids carry an "s2" infix so the two stages
cannot cross-resume. The stage-2 scorer (weights/prereg95_sobol_score.py, in the repo
before the first stage-2 row per house rule) imports build_stage2_design() and refuses
on drift, mirroring what the stage-1 scorer does with build_design().

Discipline, all inherited rather than reinvented:
  - runner.owns_the_box(".doe_lock", ...): one measurement owns the box (C-14), and the
    guarded set lives in runner.LOCK_NAMES where the smoke suite can see it;
  - orphan kill before the first run - a squatting llama process corrupts the next number;
  - clocks/temp/VRAM recorded before and after every run, because a benchmark without
    machine-state control measures the state, not the config (preregs #60/#61: a stuck
    boost state cost 28% and looked like nothing);
  - a timeout produces a recorded DNF row, never a retry and never a hole - a config that
    hangs must not eat the night (the unattended_serial lesson: a DNF is a RESULT);
  - the CSV is append-only with per-row fsync, and completed run_ids are skipped on
    relaunch, so a second night finishes exactly the runs the first night did not.

One interpretation the design forced: section 3 lists thermal settle as step 8 (after the
row is appended) yet wants the settle seconds recorded in the row - impossible once the
row is on disk. Settle therefore runs BEFORE each measurement and is recorded in the row
it protects, which is also the only ordering where temp_pre is guaranteed post-settle.

Exit codes: 0 = complete or clean deadline stop; 2 = startup assertion refused;
3 = box busy; 4 = pre-flight corner failed (night aborted).

The harness writes measurements and logs ONLY. It never touches the prereg, the
amendment, or any verdict - the operator appends the amendment before the first run.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import time

import runner

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CSV_PATH = os.path.join(DATA, "doe_morris_stage1.csv")

# The pinned build IS the experiment (design box-state section). Absolute because the
# toolchain lives in the main checkout, not in worktrees.
EXE = "C:/Users/Federico/Documents/evo-compress/tools/llamacpp-b10098/llama-bench.exe"

# Exact files, byte sizes and block counts verified on disk at design time. Wrong bytes =
# wrong file = a different experiment; wrong block_count silently changes what every -ngl
# level MEANS, so both are asserted at startup and the harness aborts rather than measures.
MODELS = {
    "7B": {
        "path": "D:/evo-compress-data/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "bytes": 4683074240,
        "block_count": 28,
        "timeout_s": 240,     # ~2x the slowest estimated corner (design sec 3.5.4)
    },
    "30B": {
        "path": "D:/evo-compress-data/gguf/Qwen3-30B-A3B-Q2_K.gguf",
        "bytes": 11258610240,
        "block_count": 48,
        "timeout_s": 360,
    },
}

# Factor order IS the draw order of the seeded RNG and matches the CSV columns
# ngl,ub,t,ctk,mmp,fa,moe_cpu_frac. The design's regeneration paragraph names the draws
# but not the factor sequence, so this tuple is the pin - the scorer imports it rather
# than restating it. 2-level lists split the p=4 grid at 0.5 (see level_at); the mmp list
# is (1, 0) so grid 0 is the build default (mmap ON), same translation as flags_from_emit.
FACTORS = {
    "7B": (
        ("ngl", (0, 9, 19, 99)),
        ("ub", (128, 512, 1024, 2048)),
        ("t", (1, 2, 3, 4)),
        ("ctk", ("f16", "q8_0")),
        ("mmp", (1, 0)),
        ("fa", (0, 1)),
    ),
    "30B": (
        ("ngl", (0, 16, 32, 99)),
        ("ub", (128, 512, 1024, 2048)),
        ("t", (1, 2, 3, 4)),
        ("ctk", ("f16", "q8_0")),
        ("mmp", (1, 0)),
        ("fa", (0, 1)),
        ("moe_cpu_frac", (0.75, 0.833, 0.917, 1.0)),
    ),
}

R_TRAJ = 10        # Morris R: trajectories per model (design sec 2)
DELTA_IDX = 2      # delta = 2/3 on the p=4 grid {0,1/3,2/3,1}, held as integer thirds so
                   # float arithmetic can never move a grid point off its level
RUN_COUNTS = {"7B": 70, "30B": 80}     # R*(k+1): 10*(6+1) and 10*(7+1)

N_GEN, REPS = 128, 3       # tg128 r=3: the staked response
PRE_N, PRE_REPS = 8, 1     # pre-flight corners prove feasibility, they do not measure

MOE_LAYERS = 48            # Qwen3-30B-A3B stack depth; asserted from the GGUF header

SETTLE_TEMP_C = 52         # design sec 2: poll every 5 s until <= 52 C,
SETTLE_MIN_S = 30          # minimum 30 s,
SETTLE_CAP_S = 180         # cap 180 s

# Column order is load-bearing: the scorer refuses any CSV whose header line does not
# hash to the design pin, so an edit here makes the harness refuse to start (see
# _check_header_pin) instead of quietly writing a file the scorer will reject.
COLUMNS = (
    "run_id,model,traj,pos,changed_factor,ngl,ub,t,ctk,mmp,fa,moe_cpu_frac,"
    "status,tok_s,stddev_ts,reps_tok_s,wall_s,settle_s,"
    "free_ram_gb_pre,temp_pre,sm_mhz_pre,mem_mhz_pre,vram_mib_pre,power_w_pre,"
    "temp_post,sm_mhz_post,mem_mhz_post,vram_mib_post,power_w_post,ts_utc,cmd"
).split(",")
HEADER = ",".join(COLUMNS)
HEADER_SHA256 = "47ed63b0f4ecfa3c3a7e6a140a79eb38b0713841ee5109c8f49a3c8e27d0c624"

GPU_QUERY = ["nvidia-smi",
             "--query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,"
             "memory.used,power.draw",
             "--format=csv,noheader"]

# ----------------------------------------------------------------- stage 2 (Sobol)
# Everything below EXECUTES docs/DESIGN_DOE_SOBOL.md and decides nothing, exactly as
# the constants above execute DESIGN_DOE_MORRIS.md. Box, build, models, timeouts and
# settle are stage 1's, unchanged - stage 2 adds a design, never a second physics.

STAGE2_CSV_PATH = os.path.join(DATA, "doe_sobol_stage2.csv")

# Survivors (design sec 2): the smallest mu_star-descending prefix reaching >= 90% of
# the stage-1 total, UNION the model's P-3 mapping-set factors - a design that froze
# the mapping factors out could never PASS the stake it exists to score. Tuple order is
# the stage-1 FACTORS order restricted to survivors, NOT mu_star order, so the seeded
# stream layout cannot drift if a mu_star tie is ever re-ranked. Levels are the stage-1
# lists UNCHANGED: narrowing a range after seeing where the variance lives would
# quietly change what "top factor" means mid-stake.
STAGE2_SURVIVORS = {
    "7B": ("ngl", "ub", "t", "ctk"),                    # 95.7% of stage-1 mu_star
    "30B": ("ngl", "t", "ctk", "mmp", "moe_cpu_frac"),  # 96.2%
}

# Non-survivors FIXED at the stage-1 best-observed row's levels (design sec 2: 7B best
# ok row 22.486 tok/s, 30B best ok row 20.497 tok/s). A fixed factor cannot become the
# ST argmax, and every fixed factor sits outside both P-3 mapping sets, so fixing can
# only remove FAIL modes - it cannot manufacture a PASS.
STAGE2_FIXED = {
    "7B": {"mmp": 0, "fa": 0},
    "30B": {"ub": 2048, "fa": 1},
}

# Saltelli N*(k+2) run budget (design sec 3): the 7B argmax race is a landslide and
# gets N=32; the close 30B race (t vs mmp) gets N=40. Extension toward N_cap=64 rides
# the SAME seeded stream in +16-block steps on an UNDECIDED gate - never a redesign.
N_START = {"7B": 32, "30B": 40}
N_CAP = 64
STAGE2_RUN_COUNTS = {"7B": 192, "30B": 280}     # N_start*(k+2): 32*6 and 40*7

# Stage-2 schema: the stage-1 columns with traj,pos,changed_factor replaced by
# block,matrix (30 columns). Own pinned hash, same refusal discipline: the stage-2
# scorer hash-pins this exact line, and the harness refuses to start (and to resume)
# on drift rather than write a file that scorer would reject.
STAGE2_COLUMNS = (
    "run_id,model,block,matrix,ngl,ub,t,ctk,mmp,fa,moe_cpu_frac,"
    "status,tok_s,stddev_ts,reps_tok_s,wall_s,settle_s,"
    "free_ram_gb_pre,temp_pre,sm_mhz_pre,mem_mhz_pre,vram_mib_pre,power_w_pre,"
    "temp_post,sm_mhz_post,mem_mhz_post,vram_mib_post,power_w_post,ts_utc,cmd"
).split(",")
STAGE2_HEADER = ",".join(STAGE2_COLUMNS)
STAGE2_HEADER_SHA256 = "95fe4b76345921ea95d6a86cc83b30caad741f799515716d0d07f687c2d89ade"


class DeadlineReached(Exception):
    """Past --deadline-hours. A clean stop is a designed outcome, not a failure: every
    finished row is on disk and the next launch resumes exactly the remainder."""


class CornerFailed(Exception):
    """A pre-flight corner OOMed, errored or hung: structural infeasibility. Ten minutes
    of corners exist precisely so this cannot eat eight hours as serial DNFs."""


# --------------------------------------------------------------------------- design

def level_at(levels, idx):
    """Grid index 0..3 -> concrete level. 4-level factors map 1:1; 2-level factors split
    at the grid midpoint (design sec 2: first level if grid < 0.5 else second), so a
    delta = 2/3 step always flips them - they are never stranded mid-trajectory."""
    if len(levels) == 4:
        return levels[idx]
    return levels[0] if idx <= 1 else levels[1]


def config_at(tag, idx_vec):
    return {name: level_at(levels, i)
            for (name, levels), i in zip(FACTORS[tag], idx_vec)}


def _run_row(tag, traj, pos, changed, cfg):
    # run_id hashes the DESIGN coordinates plus the canonical config, never a clock -
    # that stability across relaunches is what makes resume skipping sound. desc and
    # cells carry the stage's own design coordinates into the shared measure_one, so
    # the per-run sequence exists once and both stages flow through it.
    cjson = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    rid = hashlib.sha256(f"{tag}|{traj}|{pos}|{cjson}".encode("utf-8")).hexdigest()[:16]
    return {"model": tag, "traj": traj, "pos": pos, "changed_factor": changed,
            "cfg": cfg, "run_id": rid,
            "desc": f"traj {traj} pos {pos} changed={changed}",
            "cells": (traj, pos, changed)}


def build_design(tag):
    """The seeded Morris design, regenerable forever: seed "prereg95:{tag}:20260807";
    per trajectory, draw all starts (grid {0, 1/3}), then all directions ({+1, -1}), then
    one shuffle for the step order - strictly that order, from that one stream (design
    sec 2). A factor with direction -1 starts at s + delta and steps down, so every step
    stays on the grid. The scorer imports this function; the design exists once."""
    factors = FACTORS[tag]
    k = len(factors)
    rng = random.Random("prereg95:" + tag + ":20260807")
    runs = []
    for traj in range(R_TRAJ):
        starts = [rng.choice((0, 1)) for _ in range(k)]     # {0, 1/3} in integer thirds
        dirs = [rng.choice((1, -1)) for _ in range(k)]
        order = list(range(k))
        rng.shuffle(order)
        idx = [s if d == 1 else s + DELTA_IDX for s, d in zip(starts, dirs)]
        runs.append(_run_row(tag, traj, 0, "base", config_at(tag, idx)))
        for pos, fi in enumerate(order, 1):
            idx = list(idx)
            idx[fi] += DELTA_IDX * dirs[fi]
            runs.append(_run_row(tag, traj, pos, factors[fi][0], config_at(tag, idx)))
    if len(runs) != RUN_COUNTS[tag]:
        raise RuntimeError(f"{tag}: generated {len(runs)} runs, design says {RUN_COUNTS[tag]}")
    if len({r["run_id"] for r in runs}) != len(runs):
        raise RuntimeError(f"{tag}: run_id collision - resume skipping would drop a run")
    return runs


def ot_pattern(frac):
    """f = fraction of the 48 expert layers whose experts live on CPU, exercised in the
    SHIPPED direction: early blocks' experts stay on GPU, the last N go to CPU (design
    sec 1.5). f = 1.0 is the whole-family pattern the ladder emit uses."""
    if frac >= 1.0:
        return "exps=CPU"
    n_cpu = round(frac * MOE_LAYERS)
    first = MOE_LAYERS - n_cpu
    alts = "|".join(str(i) for i in range(first, MOE_LAYERS))
    return "blk\\.(" + alts + ")\\.ffn_.*_exps\\.=CPU"


def bench_cmd(tag, cfg, n_gen, reps):
    """Design sec 3.7 templates, argument order preserved. -b 2048 and -ctv f16 are
    pinned for every run (-ctv q8_0 needs -fa on: a hard confound, so V stays f16);
    warmup stays at the build default (ON)."""
    cmd = [EXE, "-m", MODELS[tag]["path"],
           "-ngl", str(cfg["ngl"]), "-ub", str(cfg["ub"]), "-t", str(cfg["t"]),
           "-ctk", cfg["ctk"], "-ctv", "f16", "-b", "2048",
           "-mmp", str(cfg["mmp"]), "-fa", str(cfg["fa"])]
    if "moe_cpu_frac" in cfg:
        cmd += ["-ot", ot_pattern(cfg["moe_cpu_frac"])]
    cmd += ["-n", str(n_gen), "-p", "0", "-r", str(reps), "-o", "json"]
    return cmd


def cmd_string(cmd):
    """One reproducible line for the CSV and the dry-run: quote only what a shell would
    mangle (the -ot alternation), matching the design template's -ot "..." rendering."""
    return " ".join('"' + a + '"' if ("|" in a or " " in a) else a for a in cmd)


def corner_specs(tag):
    """The 4 extreme corners (design sec 3.4). Factors a corner does not name sit at
    grid 0; 'mmp 0' is a LEVEL, which lives at grid 1 because the mmp level list is
    (1, 0). moe_cpu_frac's grid 0 already IS min f, so max-vram needs no override."""
    names = [n for n, _ in FACTORS[tag]]

    def at(overrides):
        idx = {n: 0 for n in names}
        idx.update(overrides)
        return config_at(tag, [idx[n] for n in names])

    return (
        ("all-grid-0", at({})),
        ("all-grid-1", at({n: 3 for n in names})),
        ("max-vram", at({"ngl": 3, "ub": 3})),   # max ngl + max ub + min f
        ("max-cpu", at({"mmp": 3})),             # ngl 0 + t 1 + mmp 0
    )


# ------------------------------------------------------------------ stage 2 design

def _stage2_run_row(tag, block, matrix, cfg):
    # Same canonical-json hashing as stage 1; the "s2" infix keeps stage-2 run_ids
    # disjoint from stage 1 by construction, so the two CSVs can never cross-resume.
    cjson = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    rid = hashlib.sha256(
        f"{tag}|s2|{block}|{matrix}|{cjson}".encode("utf-8")).hexdigest()[:16]
    return {"model": tag, "block": block, "matrix": matrix,
            "cfg": cfg, "run_id": rid,
            "desc": f"block {block} matrix {matrix}",
            "cells": (block, matrix)}


def build_stage2_design(tag, n_blocks):
    """The seeded Saltelli design, regenerable forever from design sec 4.2 alone: seed
    "prereg95:stage2:{tag}:20260816" (the date THIS design was frozen - stage 1's
    20260807 is not reused, a seed must not claim a date the design did not exist on).
    Per block b, strictly in survivor order from that one stream: k uniforms U[0,1) for
    A_b, then k for B_b. Level map: a factor with L levels takes index
    min(L-1, floor(u*L)). Non-survivors are the FIXED constants in every config. Runs
    per block: A_b, B_b, then AB_i_b (= A_b with factor i's column from B_b) per
    survivor. Blocks are drawn sequentially, so any prefix of blocks is itself a valid
    smaller-N design - that is what makes --n-blocks extension resume-safe. No range
    check here on purpose: the CLI refuses outside [N_start, N_cap], while the stage-2
    scorer must be free to regenerate any complete prefix. It imports this function
    and refuses to score if its frozen copy no longer regenerates the same run_ids."""
    surv = STAGE2_SURVIVORS[tag]
    if surv != tuple(n for n, _ in FACTORS[tag] if n in set(surv)):
        raise RuntimeError(f"{tag}: STAGE2_SURVIVORS drifted out of FACTORS order - "
                           f"the seeded stream layout is pinned to that order")
    levels = dict(FACTORS[tag])
    rng = random.Random("prereg95:stage2:" + tag + ":20260816")

    def draw():
        cfg = dict(STAGE2_FIXED[tag])
        for name in surv:                       # k uniforms, survivor order, one stream
            lv = levels[name]
            u = rng.random()
            cfg[name] = lv[min(len(lv) - 1, int(u * len(lv)))]
        return cfg

    runs = []
    for block in range(n_blocks):
        a = draw()                              # k uniforms for A_b ...
        b = draw()                              # ... then k for B_b (design draw order)
        rows = [("A", a), ("B", b)]
        for name in surv:
            ab = dict(a)
            ab[name] = b[name]                  # A_b with factor i's level from B_b
            rows.append(("AB_" + name, ab))
        for matrix, cfg in rows:
            runs.append(_stage2_run_row(tag, block, matrix, cfg))
    if n_blocks == N_START[tag] and len(runs) != STAGE2_RUN_COUNTS[tag]:
        raise RuntimeError(f"{tag}: generated {len(runs)} runs at N_start, "
                           f"design pins {STAGE2_RUN_COUNTS[tag]}")
    if len({r["run_id"] for r in runs}) != len(runs):
        raise RuntimeError(f"{tag}: run_id collision - resume skipping would drop a run")
    return runs


def stage2_corner_specs(tag):
    """The same 4 extreme corners as stage 1 with the stage-2 FIXED levels substituted
    in (design sec 4.5): feasibility is probed exactly as stage 2 will run it - e.g.
    every 30B corner now carries the fixed ub=2048, which stage 1's grid-0 corners
    never did. Log-only, same as stage 1."""
    return [(name, {**cfg, **STAGE2_FIXED[tag]}) for name, cfg in corner_specs(tag)]


# ------------------------------------------------------------------ machine telemetry

def gpu_state():
    """The exact query the design pins (sec 3.5.3), sample:
    '2026/08/16 15:34:15.706, 36, 1506 MHz, 4006 MHz, 400 MiB, 29.35 W'. Units are
    stripped so the CSV holds bare numbers. Unreadable -> empty fields: telemetry is a
    witness here, not a gate - the settle loop is the gate, and IT fails closed."""
    try:
        r = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=20)
        p = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"temp": p[1], "sm": p[2].split()[0], "mem": p[3].split()[0],
                "vram": p[4].split()[0], "power": p[5].split()[0]}
    except Exception:
        return {"temp": "", "sm": "", "mem": "", "vram": "", "power": ""}


def gpu_temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def free_ram_gb():
    """Free physical RAM in GB via the same counter unattended_serial gates on. None if
    unreadable; the row then records an empty field rather than a guess."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                           capture_output=True, text=True, timeout=25)
        return round(float(r.stdout.strip()) * 1024 / 1e9, 2)
    except Exception:
        return None


def settle():
    """Thermal settle between runs: poll every 5 s until temperature.gpu <= 52 C, minimum
    30 s, cap 180 s (design sec 2 - the #60/#61 stuck-boost scar). Runs BEFORE each
    measurement so temp_pre is guaranteed post-settle and the recorded settle_s sits in
    the row it protects. An unreadable temperature counts as not-settled (fail closed);
    the loop then waits out the cap rather than trusting a blind box."""
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed >= SETTLE_CAP_S:
            break
        t = gpu_temp()
        if elapsed >= SETTLE_MIN_S and t is not None and t <= SETTLE_TEMP_C:
            break
        time.sleep(5)
    return round(time.time() - t0, 1)


# ------------------------------------------------------------------------ gguf header

def gguf_block_count(path):
    """Read <arch>.block_count straight from the GGUF metadata. The -ngl level lists
    assume a specific stack depth; a different count silently changes what every level
    means, so the caller aborts on mismatch instead of measuring the wrong thing."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        f.seek(4 + 8, 1)                                   # version, tensor_count
        (n_kv,) = struct.unpack("<Q", f.read(8))

        def skip(t):
            if t == 8:                                     # string
                (n,) = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
            elif t == 9:                                   # array
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if et in sizes:
                    f.seek(n * sizes[et], 1)               # fixed-size: one seek
                else:
                    for _ in range(n):                     # strings/nested: walk
                        skip(et)
            else:
                f.seek(sizes[t], 1)

        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", f.read(8))
            key = f.read(klen).decode("utf-8", "replace")
            (t,) = struct.unpack("<I", f.read(4))
            if key.endswith(".block_count") and t in fmts:
                return struct.unpack(fmts[t], f.read(sizes[t]))[0]
            skip(t)
    return None


# -------------------------------------------------------------------------------- csv

def _check_header_pin(header=HEADER, pinned=HEADER_SHA256):
    """The header string here and the hash in the design/scorer must be the same bytes
    forever. Recomputing both stages' pins at every start (dry-run included) makes a
    silent header edit impossible: the harness refuses to run rather than write a
    schema the precommitted scorer would refuse to score."""
    h = hashlib.sha256(header.encode("utf-8")).hexdigest()
    if h != pinned:
        raise RuntimeError(f"COLUMNS drifted: header sha256 {h} != pinned {pinned}")


def existing_run_ids(csv_path=CSV_PATH):
    """The resume set. A declared DNF row counts as done - a DNF is a result about that
    config, and retrying it would silently un-declare it."""
    if not os.path.isfile(csv_path):
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row.get("run_id")}


def open_csv(csv_path=CSV_PATH, columns=COLUMNS):
    """Append-only, header written once. An empty file (a crash before the header made
    it to disk) is treated as new, not as drift."""
    fresh = not (os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0)
    f = open(csv_path, "a", newline="", encoding="utf-8")
    wr = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    if fresh:
        wr.writerow(columns)
        f.flush()
        os.fsync(f.fileno())
    return f, wr


# ------------------------------------------------------------------------ measurement

def run_bench(cmd, cap):
    """One llama-bench under a hard cap. utf-8 + replace because a child emitting bytes
    the parent cannot decode once killed a reader thread mid-stream (ev1_run scar)."""
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=cap)
        return {"timed_out": False, "returncode": r.returncode,
                "stdout": r.stdout, "stderr": r.stderr,
                "wall": time.perf_counter() - t0}
    except subprocess.TimeoutExpired:
        # The cap is the worst single loss the design tolerates. Kill by image name and
        # give VRAM a beat to actually release before anyone reads post-state.
        subprocess.run(["taskkill", "/F", "/IM", "llama-bench.exe"], capture_output=True)
        time.sleep(2)
        return {"timed_out": True, "returncode": None, "stdout": "", "stderr": "",
                "wall": time.perf_counter() - t0}


def classify(res, log):
    """Finished bench -> (status, tok_s, stddev_ts, reps_json). DNFs are rows, never
    retries and never holes: the scorer treats them as poisoning exactly the two
    adjacent elementary effects, nothing more."""
    if res["timed_out"]:
        return "dnf_timeout", "", "", ""
    if res["returncode"] != 0:
        if "out of memory" in (res["stderr"] or "").lower():
            return "dnf_oom", "", "", ""
        log(f"  stderr tail: {(res['stderr'] or '').strip()[-400:]}")
        return "fail_parse", "", "", ""
    try:
        out = res["stdout"]
        row = json.loads(out[out.index("["):out.rindex("]") + 1])[0]
        tok_s = row.get("avg_ts")
        if tok_s is None:
            raise ValueError("no avg_ts in llama-bench JSON")
        sd = row.get("stddev_ts")
        ns = row.get("samples_ns") or []
        ng = row.get("n_gen") or 0
        # Per-rep tok/s READ from samples_ns, not inferred from a printed sd - the
        # unattended_serial lesson, and exactly what the design pins for reps_tok_s.
        reps = json.dumps([round(ng / (x / 1e9), 4) for x in ns]) if ns and ng else ""
        return "ok", tok_s, sd if sd is not None else "", reps
    except Exception as e:
        log(f"  parse failed: {e}")
        return "fail_parse", "", "", ""


def parse_avg_ts(out):
    try:
        return json.loads(out[out.index("["):out.rindex("]") + 1])[0].get("avg_ts")
    except Exception:
        return None


def measure_one(run, wr, csvf, log):
    """Design sec 3.5, one designed run end to end: settle, pre-state, launch under cap,
    classify, post-state, append + fsync. Serves both stages: the run dict carries its
    own design coordinates (desc for the log, cells for the CSV), so this sequence
    exists once and cannot fork."""
    tag = run["model"]
    cfg = run["cfg"]
    settle_s = settle()
    free_gb = free_ram_gb()
    pre = gpu_state()
    cmd = bench_cmd(tag, cfg, N_GEN, REPS)
    cstr = cmd_string(cmd)
    log(f"[{tag}] {run['run_id']} {run['desc']}: {cstr}")
    res = run_bench(cmd, MODELS[tag]["timeout_s"])
    status, tok_s, sd, reps = classify(res, log)
    post = gpu_state()
    wr.writerow([
        run["run_id"], tag, *run["cells"],
        cfg["ngl"], cfg["ub"], cfg["t"], cfg["ctk"], cfg["mmp"], cfg["fa"],
        cfg.get("moe_cpu_frac", ""),
        status, tok_s, sd, reps,
        round(res["wall"], 1), settle_s,
        free_gb if free_gb is not None else "",
        pre["temp"], pre["sm"], pre["mem"], pre["vram"], pre["power"],
        post["temp"], post["sm"], post["mem"], post["vram"], post["power"],
        time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), cstr,
    ])
    # fsync per row: a crash or power cut forfeits at most the run in flight, and the
    # resume path trusts every byte it reads back (design sec 3.5.7).
    csvf.flush()
    os.fsync(csvf.fileno())
    log(f"  {status}  tok/s={tok_s if tok_s != '' else '-'}  wall={res['wall']:.1f}s  "
        f"settle={settle_s}s  temp {pre['temp']}->{post['temp']}")


def preflight(tag, log, deadline_at, corners_fn=corner_specs):
    """Four extreme corners at -n 8 -r 1 before any designed run: ten minutes spent so a
    structural infeasibility cannot eat eight hours as serial DNFs (design sec 3.4).
    Results go to the log, never the CSV. Stage 2 passes stage2_corner_specs - same
    corners, fixed levels substituted in."""
    cap = MODELS[tag]["timeout_s"]
    log(f"[{tag}] pre-flight: 4 corner probes at -n {PRE_N} -r {PRE_REPS} (log-only)")
    for name, cfg in corners_fn(tag):
        check_deadline(deadline_at)
        cmd = bench_cmd(tag, cfg, PRE_N, PRE_REPS)
        log(f"[{tag}] preflight {name}: {cmd_string(cmd)}")
        res = run_bench(cmd, cap)
        if res["timed_out"]:
            raise CornerFailed(f"[{tag}] pre-flight corner '{name}' hung past {cap}s")
        if res["returncode"] != 0:
            oom = "out of memory" in (res["stderr"] or "").lower()
            reason = "OOM" if oom else f"exit {res['returncode']}"
            log(f"  stderr tail: {(res['stderr'] or '').strip()[-400:]}")
            raise CornerFailed(f"[{tag}] pre-flight corner '{name}' failed ({reason})")
        log(f"  ok in {res['wall']:.1f}s  avg_ts={parse_avg_ts(res['stdout'])}")


def check_deadline(deadline_at):
    # Checked before every launch, designed run or pre-flight probe: a breach stops the
    # night cleanly, the lock releases on exit, and relaunch finishes the remainder.
    if time.time() >= deadline_at:
        raise DeadlineReached()


# -------------------------------------------------------------------------- top level

def startup_assertions(log, csv_path=CSV_PATH, header_sha=HEADER_SHA256):
    """Refuse loudly before spending a single bench-second (design sec 3.3). Stage 2
    passes its own CSV path and pin; the exe/model assertions are shared, and stage-2
    mode never opens the stage-1 CSV (design sec 4.4)."""
    if not os.path.isfile(EXE):
        log(f"ABORT: {EXE} not found - the pinned build is part of the experiment")
        return 2
    for tag in ("7B", "30B"):
        m = MODELS[tag]
        if not os.path.isfile(m["path"]):
            log(f"ABORT: {tag} model missing: {m['path']}")
            return 2
        size = os.path.getsize(m["path"])
        if size != m["bytes"]:
            log(f"ABORT: {tag} is {size} bytes, design pins {m['bytes']} - "
                f"wrong file, wrong experiment")
            return 2
        bc = gguf_block_count(m["path"])
        if bc != m["block_count"]:
            log(f"ABORT: {tag} block_count={bc}, design pins {m['block_count']} - "
                f"the -ngl level meanings would drift")
            return 2
        log(f"{tag}: {m['path']}  {size} bytes  block_count={bc}  ok")
    if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "rb") as f:
            head = f.readline().rstrip(b"\r\n")
        h = hashlib.sha256(head).hexdigest()
        if h != header_sha:
            log(f"REFUSED: CSV header hash {h} != design hash {header_sha} - "
                f"refusing to resume into a drifted schema")
            return 2
        log("CSV present, header hash verified - resume is safe")
    return 0


def run_night(designs, deadline_at, log, csv_path=CSV_PATH, columns=COLUMNS,
              corners_fn=corner_specs, stage="stage 1",
              scorer="weights/prereg95_score.py"):
    # Orphan kill covers BOTH images: a leftover server squats VRAM, a leftover bench is
    # a second measurement. The designs dict fixes the block order: 7B then 30B (design
    # sec 2 - shorter all-in-VRAM block first, RAM-hungry split block second), unless
    # stage 2's --model gates the night to a single block.
    runner.kill_orphans("llama-server.exe", "llama-bench.exe")
    done = existing_run_ids(csv_path)
    log(f"resume state: {len(done)} designed rows already in the CSV")
    csvf, wr = open_csv(csv_path, columns)
    try:
        for tag in designs:
            runs = designs[tag]
            pending = [r for r in runs if r["run_id"] not in done]
            if not pending:
                log(f"[{tag}] block complete ({len(runs)} rows) - "
                    f"pre-flight and runs skipped")
                continue
            log(f"[{tag}] {len(pending)} of {len(runs)} designed runs pending")
            preflight(tag, log, deadline_at, corners_fn)
            for run in runs:
                if run["run_id"] in done:
                    continue
                check_deadline(deadline_at)
                measure_one(run, wr, csvf, log)
        # Completion is counted against THIS launch's designed scope, so a stage-2
        # single-model night reports its own block, not the other night's rows.
        final = existing_run_ids(csv_path)
        n = sum(1 for runs in designs.values() for r in runs if r["run_id"] in final)
        total = sum(len(runs) for runs in designs.values())
        log(f"{stage} complete: {n} of {total} designed rows on disk - "
            f"score with {scorer}")
        return 0
    except DeadlineReached:
        log("DEADLINE reached - resume tomorrow night")
        return 0
    except CornerFailed as e:
        log(f"ABORT: {e}")
        return 4
    finally:
        csvf.close()


def _dry_desc_stage1(run):
    return f"traj={run['traj']} pos={run['pos']} changed={run['changed_factor']:13s}"


def _dry_desc_stage2(run):
    return f"block={run['block']:2d} matrix={run['matrix']:15s}"


def dry_run(designs, corners_fn=corner_specs, describe=_dry_desc_stage1):
    """Print the whole night and touch nothing: no lock, no log file, no CSV, no
    subprocess, no model read. The printed count is what the operator checks against
    the design before committing the amendment. Serves both stages: the designs dict
    scopes the models, corners_fn and describe carry the only per-stage differences."""
    total = 0
    for tag in designs:
        print(f"[{tag}] pre-flight corner probes (log-only, never in the CSV):")
        for name, cfg in corners_fn(tag):
            print(f"  preflight {name:11s} :: "
                  f"{cmd_string(bench_cmd(tag, cfg, PRE_N, PRE_REPS))}")
        runs = designs[tag]
        print(f"[{tag}] {len(runs)} designed runs:")
        for run in runs:
            print(f"  {run['run_id']} {describe(run)} :: "
                  f"{cmd_string(bench_cmd(tag, run['cfg'], N_GEN, REPS))}")
            total += 1
    per_model = " + ".join(f"{len(designs[tag])} {tag}" for tag in designs)
    print(f"DESIGNED RUN COUNT: {total} ({per_model}); "
          f"plus {4 * len(designs)} pre-flight corner probes not in the CSV.")
    print("dry-run touched nothing: no lock, no log, no CSV, no process launched.")
    return 0


def main_stage2(args):
    """The stage-2 launch path (docs/DESIGN_DOE_SOBOL.md sec 4): the stage-1 skeleton
    with the Saltelli design, the stage-2 CSV/pin and the fixed-level corners swapped
    in - lock, orphan kill, settle, DNF, timeout, resume and deadline are the same
    functions stage 1 runs, not copies. --model gates a single-model night (the
    declared two-night split); --n-blocks extends the same seeded stream after an
    UNDECIDED gate and is refused outside [N_start, N_cap], because below N_start is
    not the staked design and past N_cap the stake is UNDECIDABLE by rule."""
    tags = ("7B", "30B") if args.model is None else (args.model,)
    n_blocks = {}
    for tag in tags:
        nb = N_START[tag] if args.n_blocks is None else args.n_blocks
        if not N_START[tag] <= nb <= N_CAP:
            print(f"REFUSED: --n-blocks {nb} outside [{N_START[tag]}, {N_CAP}] "
                  f"for {tag} - the design extends in-stream up to N_cap only "
                  f"(design sec 3)")
            return 2
        n_blocks[tag] = nb
    designs = {tag: build_stage2_design(tag, n_blocks[tag]) for tag in tags}
    if args.dry_run:
        return dry_run(designs, corners_fn=stage2_corner_specs,
                       describe=_dry_desc_stage2)

    deadline_at = time.time() + args.deadline_hours * 3600.0
    os.makedirs(DATA, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log, close_log = runner.make_log(os.path.join(DATA, f"doe_sobol_{stamp}.log"))
    try:
        log("=" * 70)
        log(f"prereg #95 stage 2 Sobol (Saltelli survivors)  stamp={stamp}  "
            f"pid={os.getpid()}")
        log(f"design docs/DESIGN_DOE_SOBOL.md  models {'+'.join(tags)}  "
            f"blocks {' '.join(f'{t}:{n_blocks[t]}' for t in tags)}  "
            f"runs {'+'.join(str(len(designs[t])) for t in tags)}  "
            f"deadline {args.deadline_hours}h  exe {EXE}")
        try:
            with runner.owns_the_box(".doe_lock", DATA):
                rc = startup_assertions(log, STAGE2_CSV_PATH, STAGE2_HEADER_SHA256)
                if rc:
                    return rc
                return run_night(designs, deadline_at, log,
                                 csv_path=STAGE2_CSV_PATH, columns=STAGE2_COLUMNS,
                                 corners_fn=stage2_corner_specs, stage="stage 2",
                                 scorer="weights/prereg95_sobol_score.py")
        except runner.BoxBusy as e:
            log(str(e))
            return 3
    finally:
        close_log()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Prereg #95 stage 1 Morris screening (docs/DESIGN_DOE_MORRIS.md); "
                    "--stage2 runs the Sobol survivors design (docs/DESIGN_DOE_SOBOL.md)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every planned command and the total count; touch nothing")
    ap.add_argument("--deadline-hours", type=float, default=9.5,
                    help="stop launching new runs this many hours after start "
                         "(design sec 2 deadline guard)")
    ap.add_argument("--stage2", action="store_true",
                    help="run the stage-2 Saltelli/Sobol design on the Morris "
                         "survivors (docs/DESIGN_DOE_SOBOL.md sec 4)")
    ap.add_argument("--model", choices=("7B", "30B"), default=None,
                    help="stage-2 only: run one model's block "
                         "(the declared two-night split)")
    ap.add_argument("--n-blocks", type=int, default=None,
                    help="stage-2 only: Saltelli base blocks N for the active "
                         "model(s); default N_start (7B 32, 30B 40), refused "
                         "outside [N_start, 64]")
    args = ap.parse_args(argv)
    if not args.stage2 and (args.model is not None or args.n_blocks is not None):
        ap.error("--model and --n-blocks are stage-2 flags; add --stage2")

    _check_header_pin()
    _check_header_pin(STAGE2_HEADER, STAGE2_HEADER_SHA256)
    if args.stage2:
        return main_stage2(args)

    designs = {tag: build_design(tag) for tag in ("7B", "30B")}
    if args.dry_run:
        return dry_run(designs)

    deadline_at = time.time() + args.deadline_hours * 3600.0
    os.makedirs(DATA, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log, close_log = runner.make_log(os.path.join(DATA, f"doe_morris_{stamp}.log"))
    try:
        log("=" * 70)
        log(f"prereg #95 stage 1 Morris screening  stamp={stamp}  pid={os.getpid()}")
        log(f"design docs/DESIGN_DOE_MORRIS.md  runs {RUN_COUNTS['7B']}+{RUN_COUNTS['30B']}  "
            f"deadline {args.deadline_hours}h  exe {EXE}")
        try:
            with runner.owns_the_box(".doe_lock", DATA):
                rc = startup_assertions(log)
                if rc:
                    return rc
                return run_night(designs, deadline_at, log)
        except runner.BoxBusy as e:
            log(str(e))
            return 3
    finally:
        close_log()


if __name__ == "__main__":
    sys.exit(main())
