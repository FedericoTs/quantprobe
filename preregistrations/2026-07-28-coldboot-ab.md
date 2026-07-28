# Pre-registration #61: the cold-boot A/B — C-10's driver named, the fair comparison completed

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the reboot. **STAKED.**

## What today's diagnostics established, in order

1. #60's clean block measured the flagship arms 25-30% below yesterday's calibration.
2. A runaway `find` orphan (16,285 CPU-s, 47% idle load) was caught and excluded — but killing it
   only recovered part of the gap (15.3, not 21.6).
3. The box then COOLED for an hour (GPU 32 °C idle) and still measured 15.56 — **not thermal.**
4. Clock polling under load caught the cause: **SM sustains 1506 MHz where yesterday's 21.58 was
   measured at a recorded 1835 MHz** (mem 3802 vs 4004 spec), at 38 °C — a stuck lower boost
   state after a day of context churn, OOM events and profiler event storms.
5. `nvidia-smi -rgc` / `--lock-gpu-clocks` are unsupported on consumer Pascal. **Only a reboot
   resets it.**

## The protocol (weights/coldboot_ab.cmd — run FIRST after reboot, before anything else)

Six arms, clocks sampled before/after each, most important first: instrumented-binary tool split →
**pristine binary** (built tonight from the vendored commit f113e02 with ZERO patches) same arm →
both binaries plain `-ngl 20` → pristine pp2048 (the original 386 claim) → position-control repeat.

## Stakes

- **P-1 (the C-10 mechanism).** After reboot, sustained SM during tg ≥ **1750 MHz**.
- **P-2 (the recovery).** Instrumented arm-1 tg returns to **[19.0, 23.5]** (yesterday 21.58;
  today degraded 15.1-15.6). If P-1 holds and P-2 fails, the boost state was not the whole story.
- **P-3 (the fair-binary check).** Pristine vs instrumented agree within **±3%** on both shared
  arms — our instrumentation (all env-gated OFF) and the compiled-in E2c dispatch cost nothing.
  If this fails, every measurement this session carries a binary asterisk and gets re-based.
- **P-4 (the original claims reproduce).** Pristine pp2048 lands in **[350, 420]** (original
  386.04) and the tg parity between -ot and -ngl 20 holds cold (**ratio 0.93-1.07**) — the #60
  correction was not an artifact of the degraded state.

## KILL RULE

If **P-1 fails** (clocks stay ~1506 after reboot), the box has a persistent driver/hardware
regression, C-10 escalates from "measurement honesty" to "the reference machine changed under us",
and every constant in the tool needs a re-calibration pass before the next release.
If **P-3 fails**, the session's llama.cpp-referenced numbers are re-measured on the pristine
binary before any of them are quoted again.

## Why this completes the fair comparison the project owes

The original numbers (386.04 pp / 21.58 tg) were measured with clock state recorded (1835 MHz) on
a fresh box. Tonight's comparison runs the SAME model, SAME flags, SAME machine, cold, on TWO
binaries — the patched one used all session and a zero-patch build of the identical commit — with
clock state logged at every step. Whatever survives is the machine's honest best.

**Wired into:** pending the post-reboot log; C-10 resolves either way.
