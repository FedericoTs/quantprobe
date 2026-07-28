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

---

## Scored (2026-07-28, log: `weights/data/prereg61_coldboot.log`)

**Verdict: P-1, P-2, P-3 all HIT. P-4 half-hit with the miss explained by an arm-config error in
the stake. C-10 is RESOLVED with its mechanism confirmed, and the fair comparison is complete.**

| arm | result | clocks after |
|---|---|---|
| 1. instrumented, tool -ot split | **21.11 ± 0.95** | 1898 MHz, 4006 mem, 40 C |
| 2. PRISTINE (zero patches), same | **20.82 ± 0.43** | 1885 MHz |
| 3. instrumented, plain -ngl 20 | 20.59 ± 0.44 | 1847 MHz |
| 4. pristine, plain -ngl 20 | 20.79 ± 0.21 | 1873 MHz |
| 5. pristine, -ot split, pp2048 | **336.94 ± 1.33** | 1873 MHz |
| 6. position control (= arm 1) | **21.68 ± 0.20** | 1873 MHz |

- **P-1 HIT.** SM sustained 1847-1898 MHz everywhere (vs 1506 stuck yesterday), memory at full
  4006. **C-10's mechanism confirmed: a stuck lower boost state, cleared by reboot.** Not thermal
  (34 C at start), not OS load, not our code.
- **P-2 HIT.** 21.11 cold, 21.68 warmed - **the original 21.58 reproduced to 0.5%.** The tool's
  calibration constants were right all along; no recalibration.
- **P-3 HIT.** Pristine vs instrumented: 1.4% (-ot), 1.0% (-ngl). The session's entire
  measurement corpus stands on a clean binary footing.
- **P-4a MISS, cause identified as a staking error:** the original 386.04 pp was measured on a
  different expert split; the currently-shipped pattern (layers 11-47 to CPU, 37 CPU expert
  layers vs 32) does proportionally more CPU work per prefill token: 336.94 = -12.7%, tracking
  the +15.6% CPU-layer count. Logged as U-14: sweep resident-expert count for the pp/tg frontier
  - the current pattern may be leaving ~13% pp on the table when VRAM allows more residents.
- **P-4b HIT.** tg parity -ot vs -ngl at full clocks: 1.025 / 1.001. The #60 copy correction was
  physics, not an artifact of the degraded state.

## The fair comparison, final

```
                                     tg128           notes
llama.cpp pristine, naive -ngl 20    20.79 ± 0.21    zero patches, same commit
llama.cpp pristine, tool -ot split   20.82 ± 0.43    tg parity, as corrected in #60
our instrumented build, same arms    20.59-21.68     within 1.4% of pristine
original calibration (yesterday)     21.58           reproduced at 21.68 position-control
pp2048: tool split                   336.94          (original 386.04 was a different split - U-14)
copy-regime + ngram speculation      ~4.7x on top    (measured #28, cold-box, unchanged)
```

**The machine at its best is ~21.7 tok/s raw decode on the flagship, ~98-108 tok/s in the
copy-regime with speculation - and the deep-dive did not cost the baseline anything: the
instrumented binary that produced 61 pre-registrations measures within 1.4% of a pristine build.**

The honest bottom line for the tool: its decode advantage over informed-naive llama.cpp use
(-ngl 20) on this flagship is ~parity on tg - its value is the pp placement (2.2x, and U-14 may
add more), the speculation configuration (4.7x copy-regime), the format lever (+19% Q4_0), the
fit-finding, and now the boost-state diagnostic - the machine-state advice no other tool gives.

**Wired into:** `findings/REGISTER.json:C-10` (closed, mechanism confirmed) · `U-14` (new:
resident-expert sweep for pp) · `quantprobe/plan.py` cold-box copy refined with the mechanism.
