#!/usr/bin/env python
"""Supervised runner for the prereg #86-#90 experiment batch (tasks #51-#55).

WHAT THIS IS
    A chain driver.  It runs the five adversarially-reviewed experiment scripts in
    dependency order, verifies a checkpoint after every step, and ABORTS the chain
    early rather than continuing into a wrong state.

WHAT THIS IS NOT
    It does not commit.  It does not touch findings/REGISTER.json.  It does not run
    findings.py or verify.py.  SCORING STAYS A HUMAN-SUPERVISED STEP: this runner
    produces logs and JSON, a human reads them and decides what enters the register.
    The runner asserts REGISTER.json is byte-identical before and after (see
    register_fingerprint) and fails loudly if anything moved it.

THE THREE THINGS THAT MAKE THIS NON-TRIVIAL
-------------------------------------------
1.  EXIT CODES ARE NOT UNIFORM ACROSS THE FIVE SCRIPTS.  A runner that maps
    "exit 0 -> PASS" would silently misreport two of the five:

      exp51   0=PASS 1=FAIL 2=abort
      exp53   0=PASS 1=FAIL 2=abort            (also carries `publishable`)
      exp54   0=PASS 1=FAIL 3=INCOMPLETE 2=die
      exp52   0 for BOTH PASS *and* FAIL, 2=VOID/MARGINAL/SPLIT, 3=Refuse
      exp55   0 for ANY terminal verdict INCLUDING REFUTED-LEVER, 2=VOID/REFUSED

    So exp52's FAIL and exp55's REFUTED-LEVER both exit 0.  The JSON verdict is the
    source of truth here; the exit code is only ever a CROSS-CHECK.  When the two
    disagree the run is marked INCONSISTENT and the chain aborts -- a disagreement
    means one of them is lying and neither may be published unexamined.

    exp55 has no PASS in its verdict vocabulary AT ALL, by design (the reviewers
    removed the word: its all-clear branch is "REFUTED-AS-UNNECESSARY", a
    refutation, and labelling a refutation "PASS" is the reporting asymmetry the
    protocol forbids).  VERDICTS[55] therefore has an empty pass-set.  That is not
    a bug in this table.

2.  THE GPU-IDLE GATE CANNOT USE --query-compute-apps PRESENCE.  Measured on this
    box (GTX 1060, Windows 10 WDDM):

        > nvidia-smi --query-compute-apps=pid,used_memory,process_name ...
        1560, [N/A], [Insufficient Permissions]
        5312, [N/A], C:\\Windows\\explorer.exe
        10352,[N/A], ...\\SearchApp.exe
        8156, [N/A], ...\\NVIDIA Overlay.exe

    Under WDDM the desktop compositor, shell and overlay are always listed, and
    per-process used_memory is always [N/A].  A gate of "refuse if any compute app
    is present" would fire on every run forever -- an unfireable-in-reverse check
    that looks like safety and is noise.  So the gate is:
        (a) total memory.used vs --max-gpu-mib   (measured idle baseline: ~806 MiB)
        (b) name-matched LLM/CUDA processes only (LLAMA_PROC_NAMES), never the shell.

3.  A SCIENTIFIC FAIL IS NOT A CHAIN FAILURE.  "A refutation is a successful
    experiment."  A staked kill rule firing (verdict FAIL / REFUTED-*) means the
    experiment RAN CORRECTLY and told us something; the chain continues, and only
    the DEPENDENTS of that experiment are skipped.  What aborts the chain is
    INFRASTRUCTURE failure: a crash, a timeout, a refusal, a missing input, a stale
    artifact, or an exit/JSON disagreement.  Conflating the two would let a genuine
    refutation cancel three unrelated experiments.

DEPENDENCY ORDER
    #51, #53, #54   independent (pure arithmetic/retrodiction, no GPU, no llama.cpp)
    #52             independent, but the long one (llama-imatrix, 8 runs/arm)
    #55             BLOCKED ON #52 -- prereg #90's lever is gated on #87's result.
                    Runs only if #52's raw verdict is exactly PASS.  FAIL, SPLIT,
                    MARGINAL, VOID and any SWEEP(...) all skip it: SPLIT in
                    particular is a human decision about which arm survives (KR-6
                    forbids letting one arm carry the other), and this runner is
                    not allowed to make it.

USAGE
    python weights/run_queue.py --dry-run          # print the plan, touch nothing
    python weights/run_queue.py                    # run the chain
    python weights/run_queue.py --only 54          # one experiment (deps still checked)
    python weights/run_queue.py --only 55 --force-unblock   # override the #52 gate

Logs land in weights/data/run_queue_<UTC timestamp>/ -- a fresh directory per
invocation, so no run can overwrite another's transcript.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# --------------------------------------------------------------------------- paths

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
REGISTER = os.path.join(REPO, "findings", "REGISTER.json")

# The llama.cpp build the GPU-adjacent experiments stake.  NOTE it lives in the MAIN
# checkout, not in this worktree -- both exp52 and exp55 hard-code this path and
# (post-review) REFUSE to fall back to PATH, so the runner checks the same path they do.
STAKED_LLAMA_DIR = r"<repo>\tools\llamacpp-b10098"

# Process names we are willing to kill as orphans.  Deliberately narrow: only
# llama.cpp workers and known local-inference servers.  Never the shell, never
# python (this runner is python, and so is every experiment it drives).
LLAMA_PROC_NAMES = (
    "llama-imatrix.exe",
    "llama-perplexity.exe",
    "llama-cli.exe",
    "llama-bench.exe",
    "llama-server.exe",
    "llama-quantize.exe",
    "main.exe",
    "server.exe",
    "koboldcpp.exe",
    "ollama.exe",
)

# Measured idle baseline on this box is ~806 MiB (desktop + overlay).  1500 leaves
# headroom for a browser tab without letting a 4 GB model hide under the gate.
DEFAULT_MAX_GPU_MIB = 1500


# ------------------------------------------------------------------ verdict tables

# For each experiment: where the verdict lives in the JSON, which values count as a
# staked PASS, which as a staked FAIL (kill rule fired -- a real result), and which
# exit codes are legitimate for each class.  Anything outside these sets is
# INCONCLUSIVE (VOID / MARGINAL / REFUSED / INCOMPLETE) and blocks dependents.
@dataclass(frozen=True)
class VerdictSpec:
    json_path: tuple           # key path into the JSON payload
    pass_set: frozenset
    fail_set: frozenset
    ok_exits: frozenset        # exit codes that mean "the script completed and scored"
    note: str = ""


VERDICTS = {
    51: VerdictSpec(("comparison", "verdict"), frozenset({"PASS"}), frozenset({"FAIL"}),
                    frozenset({0, 1})),
    52: VerdictSpec(("overall",), frozenset({"PASS"}),
                    frozenset({"FAIL"}), frozenset({0, 2}),
                    note="exit 0 covers BOTH PASS and FAIL; SPLIT/MARGINAL/VOID exit 2"),
    53: VerdictSpec(("verdict",), frozenset({"PASS"}),
                    frozenset({"FAIL"}), frozenset({0, 1}),
                    note="also requires payload.publishable; --offline stamps "
                         "FAIL-UNVERIFIED-SOURCE and forces a non-zero exit"),
    54: VerdictSpec(("verdict",), frozenset({"PASS"}), frozenset({"FAIL"}),
                    frozenset({0, 1, 3}),
                    note="exit 3 = INCOMPLETE (vacuous K-3 baseline), never a PASS"),
    55: VerdictSpec(("overall",), frozenset(),                     # <- intentionally empty
                    frozenset({"REFUTED-LEVER", "REFUTED-AS-UNNECESSARY"}),
                    frozenset({0, 2}),
                    note="'PASS' was REMOVED from this script's vocabulary by the "
                         "adversarial review; every terminal verdict is a refutation "
                         "or a MARGINAL, and all of them exit 0"),
}


# ------------------------------------------------------------------- experiment set

@dataclass
class Experiment:
    task: int
    prereg: int
    name: str
    script: str
    argv: list
    kill_rule: str             # the STAKED kill rule, for the summary table
    timeout_s: int
    json_out: str
    gpu: bool                  # spawns llama.cpp / may touch CUDA
    needs_net: bool
    depends_on: tuple = ()
    extra_inputs: tuple = ()   # files that must exist before we start

    @property
    def label(self) -> str:
        return f"#{self.task}"


def build_experiments() -> list:
    d = lambda *p: os.path.join(DATA, *p)
    s = lambda n: os.path.join(HERE, n)
    return [
        Experiment(
            task=51, prereg=86, name="external retrodiction (C-06 / U-33)",
            script=s("exp51_external_retrodiction.py"), argv=[],
            kill_rule="P-1 & P-2 & P-3: |err| < 15% vs their live cells "
                      "(joint band 7.650-10.000 tok/s)",
            timeout_s=900, json_out=d("exp51_external_retrodiction.json"),
            gpu=False, needs_net=True,
            extra_inputs=(os.path.join(REPO, "preregistrations",
                                       "2026-07-30-external-retrodiction.md"),),
        ),
        Experiment(
            task=53, prereg=88, name="two-resource disk tier",
            script=s("exp53_two_resource_disk_tier.py"), argv=[],
            kill_rule="K-1..K-4 on M2 (LOMO median <15%, max <35%; beat 1-resource "
                      "null; best at cost; >=4/5 k-pairs) + P-0 byte-model veto",
            timeout_s=900, json_out=d("exp53_two_resource_disk_tier.json"),
            gpu=False, needs_net=True,
            extra_inputs=(os.path.join(REPO, "preregistrations",
                                       "2026-07-30-two-resource-disk-tier.md"),),
        ),
        Experiment(
            task=54, prereg=89, name="binding constraint classifier",
            script=s("exp54_binding_constraint.py"), argv=[],
            kill_rule="K-1..K-5 (K-3 no-regression over 340 cells vs pre-change "
                      "plan.py; K-4 caveat on all all-in-VRAM winners)",
            timeout_s=1800, json_out=d("exp54_binding_constraint.json"),
            gpu=False, needs_net=False,
            extra_inputs=(os.path.join(REPO, "preregistrations",
                                       "2026-07-30-binding-constraint.md"),
                          os.path.join(REPO, "quantprobe", "plan.py")),
        ),
        Experiment(
            task=52, prereg=87, name="expert-usage skew (gate for #55)",
            script=s("exp52_expert_usage_skew.py"), argv=[],
            kill_rule="KR-1 held-out top-41/128 share >= 0.55; KR-2 cross-domain "
                      "retention >= 0.90; KR-3 VOID list; KR-4 staked budget",
            timeout_s=4 * 3600, json_out=d("exp52_expert_usage_skew.json"),
            gpu=True, needs_net=False,
            extra_inputs=tuple(d(f"exp52_corpus_{n}.txt")
                               for n in ("code_A", "code_B", "prose_A", "prose_B")),
        ),
        Experiment(
            task=55, prereg=90, name="cache-aware expert dropping",
            script=s("exp55_cache_aware_dropping.py"), argv=[],
            kill_rule="KR-1 min top-1 >= 0.99 AND max mean-KLD gate; KR-2 no width "
                      "clears both quality and 1.20x; KR-4 monotonicity; KR-5 "
                      "n_expert_used proof",
            timeout_s=6 * 3600, json_out=d("exp55_cache_aware_dropping.json"),
            gpu=True, needs_net=False, depends_on=(52,),
            extra_inputs=tuple(d(f"exp55_corpus_{n}.txt") for n in ("code_B", "prose_B")),
        ),
    ]


# ---------------------------------------------------------------------- run records

@dataclass
class Result:
    task: int
    status: str = "PENDING"     # PASS FAIL INCONCLUSIVE SKIPPED TIMEOUT CRASHED ABORTED
    verdict: Optional[str] = None
    exit_code: Optional[int] = None
    log_path: str = ""
    reason: str = ""
    seconds: float = 0.0
    json_out: str = ""

    @property
    def infra_ok(self) -> bool:
        """True when the script RAN and SCORED. A fired kill rule is infra-OK."""
        return self.status in ("PASS", "FAIL", "INCONCLUSIVE")


# ------------------------------------------------------------------------ utilities

class Abort(Exception):
    """Chain-level failure: stop, do not continue into a wrong state."""


def now_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dig(payload, path: tuple):
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


class Tee:
    """Runner transcript -> stdout AND the run-level log file."""

    def __init__(self, path: str, echo: bool = True):
        self.handle = open(path, "a", encoding="utf-8")
        self.echo = echo

    def __call__(self, msg: str = "") -> None:
        self.handle.write(msg + "\n")
        self.handle.flush()
        if self.echo:
            print(msg, flush=True)

    def close(self) -> None:
        self.handle.close()


# ----------------------------------------------------------------- GPU / orphan gate

def nvidia_smi(query: str, extra: str = "") -> Optional[list]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    cmd = [exe, f"--query-{query}", "--format=csv,noheader,nounits"]
    if extra:
        cmd.insert(1, extra)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def gpu_memory() -> Optional[tuple]:
    """(used_mib, total_mib, util_pct) or None when there is no NVIDIA GPU."""
    rows = nvidia_smi("gpu=memory.used,memory.total,utilization.gpu")
    if not rows:
        return None
    try:
        used, total, util = (int(float(x)) for x in rows[0].split(","))
    except ValueError:
        return None
    return used, total, util


def gpu_compute_apps() -> list:
    """Name-matched LLM/CUDA processes ONLY.

    Under WDDM this query also returns explorer.exe, SearchApp.exe, the NVIDIA
    Overlay and '[Insufficient Permissions]', all with used_memory=[N/A].  Those are
    the desktop, not a compute tenant; treating them as one would make the idle gate
    fire on every run forever.  We match against LLAMA_PROC_NAMES instead.
    """
    rows = nvidia_smi("compute-apps=pid,used_memory,process_name")
    if not rows:
        return []
    hits = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 3:
            continue
        pid, mem, name = parts[0], parts[1], ",".join(parts[2:])
        base = os.path.basename(name).lower()
        if base in {n.lower() for n in LLAMA_PROC_NAMES}:
            hits.append((pid, mem, name))
    return hits


def list_orphans() -> list:
    """[(pid, image)] for live llama.cpp-family processes, via tasklist."""
    exe = shutil.which("tasklist")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=60)
    except (subprocess.TimeoutExpired, OSError):
        return []
    found = []
    wanted = {n.lower() for n in LLAMA_PROC_NAMES}
    for line in out.stdout.splitlines():
        cols = re.findall(r'"([^"]*)"', line)
        if len(cols) < 2:
            continue
        image, pid = cols[0], cols[1]
        if image.lower() in wanted and pid.isdigit():
            found.append((int(pid), image))
    return found


def kill_pid_tree(pid: int) -> bool:
    exe = shutil.which("taskkill")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "/F", "/T", "/PID", str(pid)],
                             capture_output=True, text=True, timeout=60)
        return out.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def kill_orphans(log: Tee, do_kill: bool) -> None:
    orphans = list_orphans()
    if not orphans:
        log("  orphan sweep: no llama.cpp-family process alive")
        return
    for pid, image in orphans:
        if do_kill:
            ok = kill_pid_tree(pid)
            log(f"  orphan sweep: {'KILLED' if ok else 'FAILED TO KILL'} {image} pid={pid}")
            if not ok:
                raise Abort(f"could not kill orphan {image} pid={pid}; "
                            "refusing to measure alongside it")
        else:
            log(f"  orphan sweep: FOUND {image} pid={pid} (--no-kill-orphans: left alive)")
    if do_kill:
        time.sleep(3.0)          # let VRAM actually come back before we sample it
        still = list_orphans()
        if still:
            raise Abort(f"orphans survived the sweep: {still}")


def check_gpu_idle(log: Tee, max_mib: int) -> None:
    """Refuse to start a GPU experiment while another CUDA tenant holds memory."""
    apps = gpu_compute_apps()
    if apps:
        for pid, mem, name in apps:
            log(f"  GPU tenant: pid={pid} mem={mem} {name}")
        raise Abort(f"{len(apps)} LLM/CUDA process(es) still hold the GPU; "
                    "a measurement taken beside them is not comparable (C-14)")

    mem = gpu_memory()
    if mem is None:
        log("  GPU: nvidia-smi unavailable -- cannot verify the GPU is idle")
        raise Abort("GPU idle state is unverifiable (no nvidia-smi); refusing to "
                    "start a GPU experiment blind. Pass --skip-gpu-check to override "
                    "and record that override in the prereg.")
    used, total, util = mem
    log(f"  GPU: {used} / {total} MiB used, {util}% util (gate: used <= {max_mib} MiB)")
    if used > max_mib:
        raise Abort(f"GPU holds {used} MiB > {max_mib} MiB gate -- something is "
                    "resident. Close it or raise --max-gpu-mib deliberately.")


# ------------------------------------------------------------------------- preflight

def preflight(log: Tee, exps: list, args) -> None:
    log("=" * 88)
    log("PREFLIGHT")
    log("=" * 88)

    log(f"  python      {sys.executable}")
    log(f"  repo        {REPO}")
    log(f"  data        {DATA}")
    if not os.path.isdir(DATA):
        raise Abort(f"missing data directory: {DATA}")

    for exp in exps:
        if not os.path.exists(exp.script):
            raise Abort(f"{exp.label}: missing script {exp.script}")
        for path in exp.extra_inputs:
            if not os.path.exists(path):
                raise Abort(f"{exp.label}: missing required input {path}")
    log(f"  scripts     {len(exps)} present, all declared inputs present")

    needs_gpu = any(e.gpu for e in exps)
    if needs_gpu:
        for binary in ("llama-imatrix.exe", "llama-perplexity.exe"):
            path = os.path.join(STAKED_LLAMA_DIR, binary)
            if not os.path.exists(path):
                raise Abort(f"staked binary missing: {path}. exp52/exp55 refuse to "
                            "fall back to PATH, so substituting a build must be "
                            "explicit and recorded.")
        log(f"  llama.cpp   staked build present at {STAKED_LLAMA_DIR}")

    kill_orphans(log, do_kill=args.kill_orphans)

    if needs_gpu:
        if args.skip_gpu_check:
            log("  GPU: !! --skip-gpu-check -- idle gate BYPASSED, record this")
        else:
            check_gpu_idle(log, args.max_gpu_mib)
    else:
        log("  GPU: no GPU experiment in this plan; idle gate not required")

    log("")


# ------------------------------------------------------------------- the actual run

def stream_child(exp: Experiment, log_path: str, log: Tee, timeout_s: int) -> tuple:
    """Run one experiment under a watchdog. Returns (exit_code, timed_out, seconds)."""
    cmd = [sys.executable, exp.script] + list(exp.argv)
    log(f"  cmd     {subprocess.list2cmdline(cmd)}")
    log(f"  log     {log_path}")
    log(f"  timeout {timeout_s}s ({timeout_s / 3600:.2f} h)")

    started = time.time()
    with open(log_path, "w", encoding="utf-8") as sink:
        sink.write(f"# {exp.label} prereg #{exp.prereg} {exp.name}\n")
        sink.write(f"# cmd: {subprocess.list2cmdline(cmd)}\n")
        sink.write(f"# started: {_dt.datetime.now(_dt.timezone.utc).isoformat()}\n\n")
        sink.flush()
        proc = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )

        pump_done = threading.Event()

        def pump():
            try:
                for line in proc.stdout:
                    sink.write(line)
                    sink.flush()
            except Exception as exc:                       # noqa: BLE001
                sink.write(f"\n[runner] output pump died: {exc!r}\n")
            finally:
                pump_done.set()

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            sink.write(f"\n[runner] WATCHDOG: no exit after {timeout_s}s -- killing tree\n")
            sink.flush()
            kill_pid_tree(proc.pid)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        pump_done.wait(timeout=30)

    return proc.returncode, timed_out, time.time() - started


def score_run(exp: Experiment, exit_code: int, json_before: Optional[str],
              log: Tee) -> Result:
    """Verify the checkpoint: fresh JSON, parseable verdict, exit/verdict agreement."""
    res = Result(task=exp.task, exit_code=exit_code, json_out=exp.json_out)
    spec = VERDICTS[exp.task]

    if not os.path.exists(exp.json_out):
        res.status = "CRASHED"
        res.reason = f"no JSON written at {exp.json_out}"
        return res

    json_after = sha256_file(exp.json_out)
    if json_before is not None and json_after == json_before:
        # The exact "stale artifact nearly published" failure this project was burned by.
        res.status = "CRASHED"
        res.reason = ("JSON is byte-identical to the pre-run file -- this run wrote "
                      "nothing and the result on disk belongs to an earlier run")
        return res

    try:
        with open(exp.json_out, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        res.status = "CRASHED"
        res.reason = f"unreadable JSON: {exc}"
        return res

    verdict = dig(payload, spec.json_path)
    res.verdict = verdict
    if verdict is None:
        res.status = "CRASHED"
        res.reason = f"no verdict at JSON path {'.'.join(spec.json_path)}"
        return res

    # Cross-check, never a substitute. Disagreement means one of the two is lying.
    if exit_code not in spec.ok_exits:
        res.status = "CRASHED"
        res.reason = (f"exit {exit_code} outside the legitimate set "
                      f"{sorted(spec.ok_exits)} for this script (verdict {verdict!r})")
        return res

    if verdict in spec.pass_set:
        res.status = "PASS"
    elif verdict in spec.fail_set:
        res.status = "FAIL"
        res.reason = "staked kill rule fired -- a result, not a chain failure"
    else:
        res.status = "INCONCLUSIVE"
        res.reason = f"verdict {verdict!r} is neither a staked PASS nor a staked FAIL"

    # exp53 additionally carries a publishability flag (--offline / unverified source).
    if exp.task == 53 and payload.get("publishable") is False:
        res.status = "INCONCLUSIVE"
        res.reason = "payload.publishable is false (unverified source)"

    # exp52/exp55 demote to SWEEP / non-headline when the staked budget was departed from.
    if payload.get("is_staked_headline") is False:
        res.status = "INCONCLUSIVE"
        res.reason = "is_staked_headline=false -- budget deviation, not the staked run"
    if isinstance(verdict, str) and verdict.startswith("SWEEP("):
        res.status = "INCONCLUSIVE"
        res.reason = "SWEEP: departs from the staked budget (KR-4)"

    # Drift alarm. Keyed on the RAW verdict string, not on res.status: an exp55 "PASS"
    # already falls through to INCONCLUSIVE above, so a status-keyed check here would be
    # dead code that looks like a safeguard.
    if exp.task == 55 and verdict == "PASS":
        res.status = "INCONCLUSIVE"
        res.reason = ("exp55 emitted the verdict 'PASS', which the adversarial review "
                      "REMOVED from its vocabulary -- the script or the VERDICTS table "
                      "has drifted and neither may be trusted until reconciled")

    return res


def gate_reason(exp: Experiment, results: dict, force: bool) -> Optional[str]:
    """Why this experiment must be SKIPPED, or None if it may run."""
    for dep in exp.depends_on:
        got = results.get(dep)
        if got is None:
            return f"depends on #{dep}, which did not run"
        if got.status == "PASS":
            continue
        if force:
            continue
        return (f"blocked: #{dep} returned {got.status}"
                + (f" ({got.verdict})" if got.verdict else "")
                + " -- the lever's gate did not open. A SPLIT or MARGINAL is a human "
                  "decision (KR-6 forbids one arm carrying the other); rerun with "
                  "--force-unblock only after that decision is written down.")
    return None


# ---------------------------------------------------------------------- the summary

def print_summary(log: Tee, exps: list, results: dict, run_dir: str) -> None:
    log("")
    log("=" * 118)
    log("SUMMARY")
    log("=" * 118)
    head = f"{'EXP':<5} {'RESULT':<13} {'VERDICT':<24} {'STAKED KILL RULE':<44} LOG"
    log(head)
    log("-" * 118)
    for exp in exps:
        res = results.get(exp.task) or Result(task=exp.task, status="NOT RUN")
        rule = exp.kill_rule.replace("\n", " ")
        rule = rule if len(rule) <= 43 else rule[:40] + "..."
        verdict = str(res.verdict or "-")
        verdict = verdict if len(verdict) <= 23 else verdict[:20] + "..."
        log(f"{exp.label:<5} {res.status:<13} {verdict:<24} {rule:<44} "
            f"{os.path.basename(res.log_path) if res.log_path else '-'}")
    log("-" * 118)
    for exp in exps:
        res = results.get(exp.task)
        if res and res.reason:
            log(f"  {exp.label}: {res.reason}")
    log("")
    log(f"  full kill rules and transcripts: {run_dir}")
    log("")
    log("  SCORING IS NOT DONE. This runner does not commit and does not touch")
    log("  findings/REGISTER.json. A human reads these logs, decides what the result")
    log("  means, and stakes it into the register -- misses at equal prominence to hits.")
    log("=" * 118)


def write_summary_json(path: str, exps: list, results: dict, run_dir: str) -> None:
    payload = {
        "runner": "weights/run_queue.py",
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_dir": run_dir,
        "note": "advisory only -- not a register entry, not a score",
        "experiments": [
            {
                "task": e.task,
                "prereg": e.prereg,
                "name": e.name,
                "kill_rule": e.kill_rule,
                "gpu": e.gpu,
                "depends_on": list(e.depends_on),
                "status": (results.get(e.task).status if results.get(e.task) else "NOT RUN"),
                "verdict": (results.get(e.task).verdict if results.get(e.task) else None),
                "exit_code": (results.get(e.task).exit_code if results.get(e.task) else None),
                "seconds": round(results.get(e.task).seconds, 1) if results.get(e.task) else None,
                "reason": (results.get(e.task).reason if results.get(e.task) else ""),
                "log": (results.get(e.task).log_path if results.get(e.task) else ""),
                "json_out": e.json_out,
            }
            for e in exps
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)


# --------------------------------------------------------------------------- dry run

def do_dry_run(log: Tee, exps: list, args) -> int:
    log("=" * 88)
    log("DRY RUN -- nothing is executed, no file outside the log directory is written")
    log("=" * 88)
    log("")

    log("PLAN (dependency order):")
    for i, exp in enumerate(exps, 1):
        dep = (", ".join(f"#{d}" for d in exp.depends_on)) if exp.depends_on else "none"
        log("")
        log(f"  {i}. {exp.label}  prereg #{exp.prereg}  {exp.name}")
        log(f"       script    {os.path.relpath(exp.script, REPO)}")
        log(f"       cmd       {subprocess.list2cmdline([sys.executable, exp.script] + exp.argv)}")
        log(f"       depends   {dep}")
        log(f"       gpu       {'yes -- idle gate enforced' if exp.gpu else 'no'}")
        log(f"       network   {'yes (live source re-extraction)' if exp.needs_net else 'no'}")
        log(f"       watchdog  {exp.timeout_s}s ({exp.timeout_s / 3600:.2f} h)")
        log(f"       json      {os.path.relpath(exp.json_out, REPO)}")
        log(f"       verdict   {'.'.join(VERDICTS[exp.task].json_path)} "
            f"| PASS={sorted(VERDICTS[exp.task].pass_set) or 'NONE BY DESIGN'} "
            f"| FAIL={sorted(VERDICTS[exp.task].fail_set)}")
        if VERDICTS[exp.task].note:
            log(f"       caution   {VERDICTS[exp.task].note}")
        log(f"       kill rule {exp.kill_rule}")
        missing = [p for p in (exp.script,) + exp.extra_inputs if not os.path.exists(p)]
        log(f"       inputs    {'ALL PRESENT' if not missing else 'MISSING: ' + str(missing)}")

    log("")
    log("PREFLIGHT THAT WOULD RUN:")
    orphans = list_orphans()
    log(f"  orphan sweep    {'would kill ' + str(orphans) if orphans else 'nothing to kill'}"
        f"{'' if args.kill_orphans else '  (--no-kill-orphans: report only)'}")
    mem = gpu_memory()
    if mem:
        used, total, util = mem
        verdict = "PASS" if used <= args.max_gpu_mib else "WOULD REFUSE"
        log(f"  gpu idle gate   {used}/{total} MiB, {util}% util vs "
            f"{args.max_gpu_mib} MiB -> {verdict}")
    else:
        log("  gpu idle gate   nvidia-smi unavailable -> WOULD REFUSE (or --skip-gpu-check)")
    apps = gpu_compute_apps()
    log(f"  gpu tenants     {apps if apps else 'none (name-matched; desktop procs ignored)'}")
    for binary in ("llama-imatrix.exe", "llama-perplexity.exe"):
        path = os.path.join(STAKED_LLAMA_DIR, binary)
        log(f"  staked binary   {'OK  ' if os.path.exists(path) else 'MISSING '} {path}")

    log("")
    log("WHAT ABORTS THE CHAIN (vs what merely blocks a dependent):")
    log("  ABORT   crash, watchdog timeout, refusal, missing input, stale/unwritten JSON,")
    log("          exit-code/JSON-verdict disagreement, un-killable orphan, busy GPU")
    log("  CONTINUE a staked kill rule firing (FAIL / REFUTED-*). That is a RESULT.")
    log("          Only the dependents of that experiment are skipped.")
    log("")
    log("NOT DONE BY THIS RUNNER: git commit, findings.py, verify.py, any write to")
    log("findings/REGISTER.json. Scoring stays human-supervised.")
    return 0


# ------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the preflight state; execute nothing")
    ap.add_argument("--only", type=int, metavar="N",
                    help="run a single experiment by task number (51-55)")
    ap.add_argument("--force-unblock", action="store_true",
                    help="run a dependent even if its gate did not open (record why)")
    ap.add_argument("--max-gpu-mib", type=int, default=DEFAULT_MAX_GPU_MIB,
                    help=f"GPU idle gate in MiB (default {DEFAULT_MAX_GPU_MIB}; "
                         "measured desktop baseline on this box is ~806)")
    ap.add_argument("--skip-gpu-check", action="store_true",
                    help="bypass the GPU idle gate (must be recorded in the prereg)")
    ap.add_argument("--no-kill-orphans", dest="kill_orphans", action="store_false",
                    help="report llama.cpp orphans instead of killing them")
    ap.add_argument("--timeout-scale", type=float, default=1.0,
                    help="multiply every watchdog timeout (e.g. 0.05 for a smoke test)")
    ap.set_defaults(kill_orphans=True)
    args = ap.parse_args()

    exps = build_experiments()
    if args.only is not None:
        exps = [e for e in exps if e.task == args.only]
        if not exps:
            print(f"no such experiment: #{args.only} "
                  f"(known: {[e.task for e in build_experiments()]})", file=sys.stderr)
            return 2

    stamp = now_stamp()
    run_dir = os.path.join(DATA, f"run_queue_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    log = Tee(os.path.join(run_dir, "run_queue.log"))

    log(f"run_queue.py  {stamp}  (tasks: {[e.task for e in exps]})")
    log("")

    if args.dry_run:
        rc = do_dry_run(log, exps, args)
        log("")
        log(f"dry-run transcript: {os.path.join(run_dir, 'run_queue.log')}")
        log.close()
        return rc

    # The register must be untouched at the end. Fingerprint it now.
    register_before = sha256_file(REGISTER)

    results: dict = {}
    aborted_at: Optional[int] = None

    try:
        preflight(log, exps, args)

        for exp in exps:
            if aborted_at is not None:
                res = Result(task=exp.task, status="SKIPPED",
                             reason=f"chain aborted at #{aborted_at}")
                results[exp.task] = res
                continue

            skip = gate_reason(exp, results, args.force_unblock)
            if skip and args.only is None:
                results[exp.task] = Result(task=exp.task, status="SKIPPED", reason=skip)
                log(f"-- {exp.label} SKIPPED: {skip}")
                continue
            if skip and args.only is not None and not args.force_unblock:
                results[exp.task] = Result(task=exp.task, status="SKIPPED", reason=skip)
                log(f"-- {exp.label} SKIPPED: {skip}")
                continue
            if skip and args.force_unblock:
                log(f"!! {exp.label} gate overridden by --force-unblock: {skip}")

            log("=" * 88)
            log(f"{exp.label}  prereg #{exp.prereg}  {exp.name}")
            log("=" * 88)
            log(f"  kill rule  {exp.kill_rule}")

            # Re-check the GPU immediately before each GPU experiment, not just once
            # at the top: an hour of exp52 is plenty of time for something to appear.
            if exp.gpu and not args.skip_gpu_check:
                kill_orphans(log, do_kill=args.kill_orphans)
                check_gpu_idle(log, args.max_gpu_mib)

            json_before = sha256_file(exp.json_out)
            log_path = os.path.join(run_dir, f"exp{exp.task}.log")
            timeout_s = max(30, int(exp.timeout_s * args.timeout_scale))

            code, timed_out, seconds = stream_child(exp, log_path, log, timeout_s)

            if timed_out:
                res = Result(task=exp.task, status="TIMEOUT", exit_code=code,
                             reason=f"watchdog fired after {timeout_s}s; process tree killed")
            else:
                res = score_run(exp, code, json_before, log)
            res.log_path = log_path
            res.seconds = seconds
            results[exp.task] = res

            log("")
            log(f"  -> {res.status}"
                + (f"  verdict={res.verdict}" if res.verdict else "")
                + f"  exit={code}  {seconds:.1f}s")
            if res.reason:
                log(f"     {res.reason}")

            # A fired kill rule is a RESULT and the chain continues.
            # Infrastructure failure is not, and it stops everything.
            if not res.infra_ok:
                aborted_at = exp.task
                log("")
                log(f"!! CHAIN ABORTED at {exp.label} ({res.status}). Remaining "
                    "experiments are SKIPPED rather than run into a wrong state.")
                if exp.gpu:
                    kill_orphans(log, do_kill=args.kill_orphans)

    except Abort as exc:
        log("")
        log(f"!! PREFLIGHT/CHAIN ABORT: {exc}")
        aborted_at = aborted_at if aborted_at is not None else -1
        for exp in exps:
            results.setdefault(exp.task, Result(task=exp.task, status="SKIPPED",
                                                reason=f"aborted before start: {exc}"))
    except KeyboardInterrupt:
        log("")
        log("!! interrupted by operator")
        aborted_at = aborted_at if aborted_at is not None else -1
        kill_orphans(log, do_kill=args.kill_orphans)
        for exp in exps:
            results.setdefault(exp.task, Result(task=exp.task, status="SKIPPED",
                                                reason="operator interrupt"))

    print_summary(log, exps, results, run_dir)
    write_summary_json(os.path.join(run_dir, "run_queue_summary.json"),
                       exps, results, run_dir)

    register_after = sha256_file(REGISTER)
    if register_before != register_after:
        log("!! findings/REGISTER.json CHANGED during this run. This runner must never")
        log("!! write it. Investigate before trusting anything above.")
        log.close()
        return 2
    log(f"  findings/REGISTER.json unchanged ({str(register_before)[:12]}...) as required")
    log.close()

    if aborted_at is not None:
        return 2
    if any(r.status in ("INCONCLUSIVE", "SKIPPED") for r in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
