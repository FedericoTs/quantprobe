# Pre-registration #45: is the PCIe bus actually idle during split decode? (the dual-bus premise)

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the measurement. **Status: STAKED.**

## The idea being tested (from an adversarial red-team of our own conclusions)

Law 4 prices decode as `tok/s <= 1 / SUM_i(bytes_i / BW_i)`. **The SUM assumes the tiers are used
one at a time.** During split-placement decode this box appears to use exactly one bus at a time:

- PCIe 3.0 x16 (measured 12.2 GB/s) carries only activations — 32 host layers x ~16 KB =
  ~0.5 MB/token = ~0.04 ms of a 44.9 ms token. Nominally **>99% idle**.
- The CPU expert phase is ~19-22 ms of that token, during which the GPU has nothing to do.

If both buses ran concurrently on disjoint expert subsets, effective host bandwidth would be
23.1 + 12.2 = **35.3 GB/s (+53%)**, and the wall would move from 41.1 to ~55.6 tok/s. Prior art
exists for the mechanism (Fiddler, ICLR'25, does per-expert CPU-vs-GPU placement inside one MoE
layer), and it is favourable here precisely because our DRAM:PCIe ratio is 23.1:12.2 = 1.89 —
narrow enough that a mixed split beats both pure ones.

**But the whole idea rests on one unverified premise: that PCIe is genuinely idle and that using
it does not disturb decode.** That premise is testable with NO code change, and this
pre-registration tests only that.

## Arms — shipped split recipe, tg128, r=3, one session

| arm | what runs alongside llama-bench |
|---|---|
| A | nothing (baseline) |
| B | **PCIe saturator** — continuous pinned host->device copies from a second process |
| C | **DRAM hog** (positive control) — continuous large host-memory reads |

## Stakes

- **P-1 (the premise).** Arm B costs **< 3%** versus A. If saturating PCIe barely touches decode,
  the bus really is spare capacity and the dual-bus lever has something to work with.
- **P-2 (the control must fire).** Arm C costs **> 15%** versus A. A background DRAM reader
  competes for the exact resource the expert phase is bound by, so it MUST hurt. If C is also
  free, the harness is not creating real contention and P-1 is uninterpretable — this is the arm
  that makes P-1 meaningful rather than a null-measuring-nothing.
- **P-3 (no output effect).** Not measured here — this is a throughput test only, and no claim
  about correctness is made.

## KILL RULE

**If P-1 fails (PCIe traffic costs >3%) the dual-bus idea dies before any code is written**: the
link is not free, so moving expert bytes onto it trades one contended resource for another. If P-2
fails, the experiment is void and must be redesigned — a null in both arms means the background
load is not reaching the hardware.

## What this does and does not decide

**Decides:** whether spare PCIe capacity exists during decode. **Does not decide:** whether ggml's
scheduler can overlap a CUDA branch with a CPU branch (the red-team named this as the decisive
unknown — if `ggml-backend-sched` inserts an event sync at every split boundary the branches
serialise and the cost is the SUM, not the MAX, turning +20% into a -35% regression). That is a
source question and is out of scope here.

Honest expectation, recorded before measuring: even if this passes, the compounded prize is
~1.6x on novel content (21 -> ~35 tok/s), which does NOT reach 100. The project's "100 tok/s on
novel content is a hardware statement" survives; what would not survive is the claim that novel
decode has **no** software lever left, and the 41.1 wall underneath it.
