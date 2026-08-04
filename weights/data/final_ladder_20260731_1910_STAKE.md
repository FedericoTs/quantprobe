# STAKE — final-ladder validation of v1.24.0 (written BEFORE any ladder measurement)

Timestamp of stake: 2026-07-31 ~19:10 local. Nothing below may be renegotiated after a number
lands. Misses are published at the same prominence as hits.

## What is being validated

Everything on the final tree that is unmeasured: the version bump to 1.24.0, the README header
change (wordmark), and the reinstalled artifact. Runner: `weights/run_ladder_idle.sh`, which
gates GPU + CPU idleness, kills orphans, then calls `python weights/full_ladder_v124.py --bench`.

## Pre-run facts recorded before measuring (so they cannot be selected after)

- Installed artifact: quantprobe 1.24.0 at
  `C:\Users\Federico\AppData\Roaming\Python\Python312\site-packages\quantprobe`,
  byte-identical to the repo tree `quantprobe/` (recursive filecmp: zero DIFF, zero one-sided files).
- `python verify.py`: all FIVE layers PASS (128 unit tests; installed artifact 1.24.0;
  end-to-end vs llama.cpp; 4 measured anchors; findings reach the code).
- Calibration on disk at stake time: `~/.quantprobe/calibration.json`, **cal_id 2dc97d41**,
  ram_bw 24.4 GB/s, disk_bw 0.47 GB/s, boost healthy (1847/1911). The runner does NOT
  re-calibrate — contrary to the task brief, neither `run_ladder_idle.sh` nor
  `full_ladder_v124.py` invokes `quantprobe calibrate`. This run therefore reuses 2dc97d41,
  which is the SAME state as the reference ladder. Stated plainly rather than papered over.
- Reference ladder (comparison target): `weights/data/ladder_state_locked.json` at HEAD,
  cal 2dc97d41, 14/14 rows, median |err| = **8.95%**. Backed up to
  `weights/data/ladder_PRE_v124_2dc97d41_backup.json` before this run overwrites it.
- Engine delta since that reference ladder: `git diff 8cf660c^ HEAD -- quantprobe/` touches
  exactly ONE line — `__version__` in `quantprobe/__init__.py`. No plan/engine code changed.
  Working tree has zero uncommitted diff under `quantprobe/`.
- Published baseline for the pass criterion: **8.8%** (C-16/C-17). Noise floor: **+/-1 point**,
  differences under 2 points are NOT evidence (C-18).

## Staked predictions

**P-1 (primary, the task's pass criterion).** Median |err_pct| over all 14 rows lands inside
`[6.8%, 10.8%]` — i.e. within 2 points of the 8.8% baseline.
KILL RULE: median outside that closed band => FAIL.
I explicitly forfeit the right to call a low number a win: per C-18, anything in
`[6.8, 10.8]` is reported as **UNCHANGED**, never as "better". If the median lands at 7.5% I
will write "unchanged", and this sentence is on the record before the number exists.

**P-2 (completeness).** All 14 rows produce a non-null `measured` tok/s and a non-null
`err_pct`. No SKIP (missing file), no BENCH FAILED.
KILL RULE: fewer than 14 scored rows => FAIL (a median over 13 rows is a different statistic
and will not be substituted).

**P-3 (one machine state, C-14).** Every one of the 14 rows carries the identical `cal_id`, and
that id equals `2dc97d41`, the calibration present on disk at stake time.
KILL RULE: more than one distinct cal_id, or an id != 2dc97d41, or `uncalibrated` => FAIL.

**P-4 (determinism — the stake that does NOT hide behind the noise floor).** Because the only
engine delta since the reference ladder is a version string, the PREDICTION half must be
bit-identical: all 14 `predicted` tok/s, all 14 `placement` strings, and all 14 emitted
llama-server commands reproduce `ladder_PRE_v124_2dc97d41_backup.json` exactly.
KILL RULE: any single one of the 42 compared fields differs => FAIL, and the differing field is
named in the report. This is the arm that can actually catch the v1.24.0 bump changing shipped
behaviour; P-1 cannot, because a 1-2 point median wander is indistinguishable from noise.

**P-5 (predicted direction of the measured half).** The `measured` tok/s WILL wander row to row
(gemma4-12B alone moved 13.23 -> 12.25 between two runs at equivalent calibration). I predict
the GPU-only control row (Qwen2.5-0.5B Q8_0) measures within +/-3% of 151.76 tok/s, since the
card was the fixed point across all five ladders on 2026-07-31 (152.08-153.74 there, 151.76 on
the reference run).
KILL RULE: control row outside 147.2-156.3 => the box is not in the same state and P-1's verdict
is reported as UNINFORMATIVE rather than PASS/FAIL, whatever the median says.

## Non-predictions (things I am NOT claiming and will not claim afterwards)

- I am NOT claiming the ladder validates the README/wordmark change; a markdown header cannot
  reach a tok/s number. That change is covered only by verify.py layer 5, already green.
- I am NOT claiming any disk-tier coverage. Zero of the 14 rows read the disk tier (C-17's
  known coverage gap). A pass here says nothing about disk-tier predictions.
- I am NOT re-calibrating, so this run cannot detect calibration drift since 2dc97d41; the
  drift detector's own output is the only signal and it is not part of any kill rule here.

## Falsifiability check (protocol: construct the input that makes the test fail)

The scorer `weights/score_final_ladder.py` must exit non-zero on each of four constructed
failing inputs before its PASS on the real data is trusted:
 F1: 14 rows, median forced to 12.0% -> must fail P-1.
 F2: 13 scored rows (one `measured` nulled) -> must fail P-2.
 F3: two distinct cal_ids -> must fail P-3.
 F4: one `predicted` perturbed by 0.1 -> must fail P-4.
Results of F1-F4 are appended to this file BEFORE the ladder result is read.

## Falsifiability gate RESULT (run before the ladder, appended before any result was read)

`python weights/falsify_final_ladder.py`:
```
ok  F0 reference-vs-itself (must PASS, exit 0): exit 0 (wanted 0)
ok  F1 median forced to 12.0%: exit 1 (wanted 1)
ok  F2 one row unmeasured (13/14): exit 1 (wanted 1)
ok  F3 two cal_ids: exit 1 (wanted 1)
ok  F4 one predicted +0.1: exit 1 (wanted 1)

falsifiability gate: PASSED - the scorer can fail
```
The scorer exits non-zero on every constructed failing input and zero on the control, so a
PASS from it is not the "measurement that cannot vary" signature. Ladder now launched.
