# RESULT — final-ladder validation of v1.24.0

Stake: `weights/data/final_ladder_20260731_1910_STAKE.md` (written before any measurement).
Raw rows: `weights/data/final_ladder_20260731_1913_RESULT.json`
Runner log: `weights/data/final_ladder_20260731_1913_runner.log`
Reference: `weights/data/ladder_PRE_v124_2dc97d41_backup.json` (cal 2dc97d41, median 8.95%)

## THE MISS, FIRST AND AT FULL PROMINENCE

**P-1 FAILED. Median |err| = 12.3%, outside the staked band 6.8-10.8%.**
The scorer exits 1. The kill rule is honored exactly as written and is not being renegotiated.
Staked band was 8.8% +/- 2 points; 12.3% is 3.5 points above baseline.

## Scorer output (verbatim)

```
P-2 rows scored: 14/14 (rows present: 14)
P-3 cal_ids: ['2dc97d41']
P-1 median |err|: 12.3%  band 6.8-10.8
P-4 prediction-field diffs: 0/42
P-5 control Qwen2.5-0.5B Q8_0: 152.62 tok/s  band 147.21-156.31
FAIL: P-1 median 12.3% outside staked band 6.8-10.8%
VERDICT: FAIL
```

| arm | staked | measured | verdict |
|---|---|---|---|
| P-1 median in [6.8, 10.8] | median within 2pt of 8.8% | **12.3%** | **FAIL** |
| P-2 14/14 rows scored | 14 | 14 | PASS |
| P-3 one cal_id == 2dc97d41 | one | `['2dc97d41']` | PASS |
| P-4 42 prediction fields identical | 0 diffs | **0/42 diffs** | PASS |
| P-5 GPU control within +/-3% | 147.2-156.3 | 152.62 | PASS |

## Per row, this run vs the reference ladder (same cal_id, same predictions)

| row | pred | ref meas | new meas | ratio | ref s | new s |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | 124.0 | 151.76 | 152.62 | 1.01 | 3 | 3 |
| Qwen3-0.6B Q8_0 | 103.0 | 103.48 | 103.94 | 1.00 | 4 | 4 |
| Qwen3.5-4B Q4_K_M | 32.6 | 29.94 | 29.95 | 1.00 | 13 | 13 |
| Qwen2.5-7B IQ4_NL | 27.4 | 25.49 | 25.37 | 1.00 | 17 | 17 |
| Qwen2.5-7B Q4_K_M | 21.5 | 22.70 | 22.50 | 0.99 | 18 | 16 |
| gemma4-12B depth-aware | 12.4 | 12.25 | 13.86 | 1.13 | 28 | 26 |
| DS-Lite 16B IQ2_XS | 30.6 | 26.23 | 25.64 | 0.98 | 18 | 19 |
| Qwen2.5-14B Q4_K_M | 5.0 | 5.01 | 4.90 | 0.98 | 50 | 52 |
| DS-Lite 16B Q4_K_M | 20.0 | 24.04 | 23.63 | 0.98 | 30 | 31 |
| Qwen3-30B-A3B Q2_K | 19.8 | 22.02 | 19.48 | **0.88** | 35 | 37 |
| Qwen3-Coder-30B Q2_K_L | 19.8 | 21.77 | 17.36 | **0.80** | 33 | 38 |
| Qwen3.5-35B APEX-Mini | 22.0 | 23.11 | **4.25** | **0.18** | 36 | **92** |
| Qwen3.6-35B Q2_K_XL | 18.7 | 16.96 | **7.55** | **0.45** | 37 | **62** |
| Qwen3.6-35B APEX-MTP-Nano | 19.2 | 17.45 | **12.75** | **0.73** | 34 | **43** |

Rows 0-8 reproduce the reference within 0.98-1.13. Rows 9-13 collapse, monotonically, and
their wall-clock bench time balloons (36s -> 92s on APEX-Mini). A prediction error cannot
change how long llama-bench takes to run; only the machine can.

## WHY THE MEDIAN MOVED: THE RUN WAS NOT SERIAL

The predictions are bit-identical to the reference (P-4: 0 of 42 fields moved), so 100% of the
median's movement is in the MEASURED half. Direct observation of the live process table:

```
ProcessId       : 1336
ParentProcessId : 13780
CreationDate    : 31/07/2026 19:27:10
CommandLine     : ...\tools\llamacpp-b10098\llama-bench.exe
                  -m D:/evo-compress-data/gguf/Laguna-S-2.1-UD-Q2_K_XL.gguf
                  -ngl 0 -b 2048 -ub 2048 -t 4 -n 64 -p 0 -r 2 --progress
   working set: 13,663 MB     (box has 15.95 GB total; free RAM read 0.14 GB)

ProcessId    : 13780
CreationDate : 31/07/2026 19:11:00
Cmd          : C:\Python312\python.exe weights/disktier_20260731_run.py
```

A **different, unrelated experiment** — the disk-tier run, driven by `weights/disktier_20260731_run.py`,
started 19:11:00, two minutes BEFORE my ladder launched at 19:13 — was benching 117B-class
Laguna files pure-CPU (`-ngl 0 -t 4`) on all four cores throughout my ladder. It uses a
different binary (`llamacpp-b10098`, not the ladder's `llama.cpp-pristine`), so it is not an
orphan of mine.

The timeline matches the damage exactly:
- My ladder's benching ran ~19:24 to 19:32:09.
- The 13.66 GB `llama-bench` PID 1336 started **19:27:10** and was still resident at 19:33.
- Degradation begins at **row 9** and deepens through row 13 — i.e. from ~19:27 onward.
- Rows 0-8 are the all-in-VRAM and small-split rows: they need little host RAM, and they are
  clean. Rows 9-13 are the big MoE splits whose CPU tier holds many GB of experts — they
  compete directly for the RAM the foreign job took, and only they were harmed.

This violates two standing rules at once: GPU/CPU runs are SERIAL, and C-14 (never compare
across machine states — the state changed mid-run, at row 9).

## THE IDLE GATE DID NOT CATCH IT — that is a real defect, not an excuse

`run_ladder_idle.sh` passed its gate ("gates passed: GPU 750MiB, CPU idle") with three
consecutive 7% CPU samples **while a four-thread `llama-bench` was mid-sweep**. Two holes:
1. It samples `Win32_Processor.LoadPercentage`. A competing CPU-tier job that is momentarily
   blocked on disk (loading a 13 GB GGUF) reads as an idle box. The gate samples the wrong
   thing at the wrong moments.
2. **It has no free-RAM gate at all.** Free physical memory is the exact resource the RAM-tier
   ladder rows compete for, and nothing checks it. 0.14 GB free was never going to be caught.

## CORRECTION TO SOMETHING I ASSERTED EARLIER IN THIS SESSION

Mid-run I claimed the CPU-idle oscillation (44-48% then 7%) was caused by my own monitor: each
emitted event woke the agent harness, and the harness spike was what the next sample read. I
stated that as the diagnosis because the gate passed immediately after I stopped the monitor.
**That causal claim was not supported and I withdraw it.** The same pattern is equally
explained by the foreign disk-tier job alternating between disk-load phases (low CPU) and bench
phases, which I did not know about at the time. Post-hoc per-process sampling showed the claude
processes at only 6.1% and 4.1% of the box. I inferred cause from one coincidence in time;
that is the same error this protocol exists to prevent.

## WHAT COULD NOT BE MEASURED (stated plainly, no estimate substituted)

- **Whether v1.24.0 regresses the ladder median: NOT MEASURED.** This run cannot answer it.
  The clean-row subset is not a substitute and I am not offering it as one: my own stake says
  a median over fewer than 14 rows "is a different statistic and will not be substituted."
  (For the record, so it cannot be quietly deployed later as a win: rows 0-10 alone median
  8.8%. That is an 11-row number from a contaminated run. It is not evidence and must not be
  cited as a pass.)
- **Calibration freshness: NOT MEASURED.** Contrary to the task brief, neither
  `run_ladder_idle.sh` nor `full_ladder_v124.py` invokes `quantprobe calibrate`. This run
  reused the on-disk cal 2dc97d41. No fresh calibration happened; the brief's description of
  the runner is wrong.
- **Disk-tier accuracy: NOT COVERED.** Zero of the 14 rows read the disk tier (C-17's known
  coverage gap). Unchanged by this run.

## WHAT IS ESTABLISHED, AND IS NOT AFFECTED BY THE CONTAMINATION

P-4 is computed from `quantprobe plan` alone — no benching, no timing, immune to a busy box.
All 14 predicted tok/s, all 14 placement strings and all 14 emitted llama-server commands
reproduce the reference ladder **exactly (0 of 42 fields moved)**, under one cal_id, on
14/14 rows. Combined with `git diff 8cf660c^ HEAD -- quantprobe/` touching exactly one line
(`__version__`), and the installed artifact being byte-identical to the repo tree, and
`verify.py` green on all five layers: **v1.24.0 did not change any shipped prediction.**
That is a real result. It is not the staked pass criterion, and it does not substitute for one.

## ALSO: MY RUN LIKELY CONTAMINATED THEIRS

The overlap is symmetric. `weights/disktier_20260731_run.py` was benching from 19:11 while my
ladder held the GPU and ran its own benches from ~19:24. Any disk-tier number that run produced
in the 19:24-19:32 window is suspect for the same reason mine are, and whoever owns it should
be told before those numbers are scored. I did **not** kill PID 1336/13780 — it is another
session's live experiment, not an orphan, and killing it would have destroyed their data.

## REQUIRED NEXT STEP

Re-run the full 14-row ladder on a genuinely quiet box, after adding to the gate:
(a) a **free-RAM floor** (the RAM-tier rows need >13 GB free on this box),
(b) a check for **any** `llama-*` process regardless of which binary directory it came from,
(c) re-sampling CPU across the whole gate window rather than three point samples.
Until that runs, the final-ladder validation item is **unmeasured**, and it should be recorded
as unmeasured rather than as either a pass or a regression.
