"""The measurement discipline, in one place instead of five.

Every long runner here needs the same four things: refuse to start if another measurement owns
the box, take its own lock, kill orphaned servers first, and log with timestamps to a file
nobody else is writing. That code was copy-pasted into autotune_sweep, ev1_run, phaseb_gen,
gridbench and p0_lanes - and copies drift.

They had. Surveyed 2026-08-08, mid-EV-1-night:

    ev1_run.py         checked all 5 locks
    phaseb_gen.py      checked 4  - missed .ev1_lock
    autotune_sweep.py  checked 2  - missed .grid_lock, .phaseb_lock, .ev1_lock

So with EV-1 holding the box and a 30B row in flight, autotune_sweep would have started
happily and contended for the GPU - which is precisely the overlap that voided the 2026-07-31
ladder run and forced it to be deleted rather than explained. The list has to exist once.

Adding a runner? Put its lock in LOCK_NAMES. That is the whole registration step, and
t_every_runner_guards_against_every_other_lock fails if you forget.
"""
from __future__ import annotations
import os
import subprocess
import time
from contextlib import contextmanager

# THE canonical list. Every runner refuses to start while ANY of these is held by someone else.
LOCK_NAMES = (".p0_lock", ".autotune_lock", ".grid_lock", ".phaseb_lock", ".ev1_lock")


class BoxBusy(RuntimeError):
    """Another measurement owns the machine. Never proceed - C-14 is not negotiable."""


def kill_orphans(*names):
    """Orphaned servers squat VRAM and silently corrupt the next run's numbers."""
    for n in names or ("llama-server.exe",):
        subprocess.run(["taskkill", "/F", "/IM", n], capture_output=True)
    time.sleep(2)


def make_log(path):
    """Timestamped logger writing to its own file. Returns (log, close)."""
    fh = open(path, "a", encoding="utf-8")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log, fh.close


@contextmanager
def owns_the_box(mine, data_dir, kill=True):
    """Acquire exclusive right to measure. `mine` is this runner's lock name from LOCK_NAMES.

    Refuses if any OTHER lock exists - checked against the shared list, so a runner cannot
    quietly guard a subset. Releases its own lock on the way out even if the body raises,
    because a lock left behind blocks every future run and reads as a machine that is busy.
    """
    if mine not in LOCK_NAMES:
        raise ValueError(f"{mine} is not in LOCK_NAMES - register it there, not here")
    mypath = os.path.join(data_dir, mine)
    for name in LOCK_NAMES:
        if name != mine and os.path.isdir(os.path.join(data_dir, name)):
            raise BoxBusy(f"REFUSED: {name} exists - another measurement owns the box")
    try:
        os.mkdir(mypath)
    except FileExistsError:
        raise BoxBusy(f"REFUSED: {mypath} exists - this runner is already running")
    try:
        if kill:
            kill_orphans()
        yield
    finally:
        if kill:
            kill_orphans()
        try:
            os.rmdir(mypath)
        except OSError:
            pass
