# C-16 — calibration has no completeness contract, and partial calibration is worse than none

**Not a pre-registration.** This is a defect report from five ladders measured 2026-07-31
while chasing an unrelated slowdown. Recorded here because the evidence is five full
ladder runs and it belongs with them.

## How it surfaced

The 2026-07-31 ladder showed every CPU-participating row 15–30% below the a19aeee4
baseline while every GPU-only row held. Suspecting ambient CPU load, then machine session
state, we ran the ladder in five states. The GPU-only rows are the control throughout:
Qwen2.5-0.5B measured 153.39 / 153.74 / 153.00 / 152.95 / 152.08 across all five — the
card never moved.

| state | cal_id | ram | disk | 30B-A3B | median \|err\| |
|---|---|---|---|---|---|
| baseline | a19aeee4 | — | — | 21.71 | 8.8% |
| degraded, loaded | 75eb1d48 | 22.58 | 0.45 | 17.59 | 7.2% |
| degraded, idle-gated | 75eb1d48 | 22.58 | 0.45 | 17.51 | 7.9% |
| post-reboot, **stale cal** | 75eb1d48 | 22.58 | 0.45 | 21.10 | 9.9% |
| post-reboot, **no cal file** | uncalibrated | None | None | 21.72 | **27.2%** |
| post-reboot, **RAM only** | 37a91948 | 25.23 | None | 21.65 | **12.5%** |
| post-reboot, **full** | c24a253b | 24.14 | 2.99 | 21.56 | **7.9%** |

The machine slowdown was real and reboot-recoverable (measured RAM 22.58 → 25.23 GB/s,
+11.7%); post-reboot measurements reproduce a19aeee4 within ~2% on all 14 rows. That part
is closed. What it exposed is three defects in calibration, all shipping.

## The three defects

**1. Nothing ever re-measures.** Calibration is written to `~/.quantprobe/calibration.json`
and read back forever. No bench or plan path re-measures; only an explicit
`quantprobe calibrate` does. A user who calibrates once during a degraded window — on
battery, under load, mid-thermal-throttle — carries that state silently and permanently.
Measured cost here: predictions frozen at degraded values against a recovered machine,
median 8.8% → 9.9%, with every error flipping sign.

**2. A missing file degrades silently to presets.** Deleting the calibration did not
trigger re-measurement; it fell through to built-in preset defaults, reporting
`machine state: uncalibrated (ram None GB/s, disk None GB/s)` in a line nobody has to
read, and produced a **27.2% median** on a perfectly healthy machine. The tool tags
calibrated output `[calibrated]`; the uncalibrated case gets no equivalent prominence.

**3. Partial calibration is WORSE THAN NONE for the components you skipped.** This is the
non-obvious one. Calibrating RAM alone (no `--model`, so no disk and no decode anchor)
made the RAM-bound rows the most accurate ever recorded on this ladder — Qwen3-30B-A3B
−10.2% → **+1.6%**, Qwen3-Coder-30B −10.0% → **+0.9%** — while wrecking every GPU-bound
row: Qwen2.5-0.5B −18.6% → **−28.1%**, gemma4-12B 0.0% → **+12.8%**, Qwen3.6-APEX-MTP
+8.5% → **+42.0%**. Net median 12.5%, worse than the 8.8% baseline it replaced.

Calibration is a VECTOR, not a scalar. The presets are mutually consistent — each one
compensates for the others' biases. Measuring one component and leaving the rest on
presets breaks that consistency, and the damage lands on exactly the components you did
not measure. Restoring the full vector (`c24a253b`: ram 24.14, disk 2.99, decode anchor)
recovered a **7.9% median, beating the 8.8% baseline**.

## A fourth, separate defect: the disk probe was wrong by 6x

The disk-only probe read **0.46 GB/s**; the full calibration measured **2.99 GB/s** on the
same drive, same session (+550%). Every disk-tier prediction under `75eb1d48`, which
shipped 0.45, rested on it. Cause not yet diagnosed — first-touch/cold-cache is the
suspect. Filed as its own work item; do not quote any disk-tier number from a
disk-only calibration until it is.

## What worked

The drift detector fired unprompted on the state change, named both moved quantities, and
quoted C-14 back at its own authors: *"Measurements taken under the OLD state cannot be
scored against predictions from this one — that comparison moved every arm of our own
model ladder by 5-12 points."* The mechanism to catch this existed. What was missing is
that nothing forces it to run.

## The contract that is owed

1. Stamp every calibration with wall-clock time AND a boot-session id; warn loudly when
   reused across a reboot or beyond an age threshold.
2. Mark each component `measured | preset` individually, and warn on any MIXED state —
   partial calibration must be as visible as no calibration.
3. Give the uncalibrated state the same prominence the `[calibrated]` tag gets.
4. Re-measure by default in bench paths rather than on request.
5. Diagnose the disk probe before any disk-tier claim is quoted again.

Raw ladders retained: `ladder_20260731_idle_prereboot_75eb1d48.json`,
`ladder_20260731_postreboot_stalecal.json`, `ladder_20260731_uncalibrated.json`,
`ladder_20260731_ramonly_37a91948.json`, and the current locked ladder (`c24a253b`).
Degraded calibration archived as `calibration_75eb1d48_degraded.json`.

## Scoring two staked calls, at equal prominence

- *"An idle machine will restore the MoE rows to ~21.7"* — **REFUTED.** Idle-gated, the
  30B measured 17.51, and the Qwen3.6 rows got *slower* at idle than under load.
- *"A reboot will restore them"* — **CONFIRMED.** 21.10 immediately post-reboot, 21.56 on
  the final full calibration, against a 21.71 baseline.
