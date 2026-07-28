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

---

## Scored (2026-07-28, log: `weights/data/prereg45_dual_bus.log`)

**Verdict: the EXPERIMENT is void on protocol — but the IDEA is refuted by physics that the
experiment did not need to establish. Kill rule fires on the second ground, not the first.**

### The experiment failed, and the position control is what caught it

| arm | order | tg128 |
|---|---|---|
| A baseline | 1st (cold) | 14.66 ± 0.31 |
| B + PCIe saturator (7.4 GB/s H2D) | 2nd | 12.21 ± 0.17 |
| C + DRAM hog | 3rd | 16.02 ± 0.32 |
| **A2 baseline re-run** | **4th (warm)** | **18.29 ± 0.26** |

**The baseline drifted +25% across the session** (14.66 → 18.29) — larger than any effect under
test. Arms measured at different positions cannot be compared, so P-1 and P-2 are both
**unscoreable**. Reported rather than salvaged: reading B as "−16.7% vs A" would have been the
same thermal-ordering error #31 already caught once, and the position control existed precisely to
catch it a second time. A valid version interleaves arms (A,B,A,C,A,B…) to cancel drift.

### The idea is refuted anyway, and by an argument the measurement could not have improved on

The red-team's arithmetic assumes effective host bandwidth becomes `23.1 + 12.2 = 35.3 GB/s` by
running DDR4 and PCIe concurrently. **That addition is not available for host-resident weights.**

A PCIe transfer of expert bytes from host memory is a DMA read **out of host DRAM**. The bytes
cross the DRAM bus whether the CPU computes on them locally or the GPU receives them over PCIe —
the DMA engine has to fetch them first. So streaming an expert to the GPU does not spare the DRAM
bus; it adds a hop and moves the arithmetic, while the *same* bytes still occupy the *same*
bottleneck. The two "buses" are in series for this workload, not in parallel, and
`BW_dram + BW_pcie` double-counts a single read.

The only way PCIe traffic avoids DRAM is if the data already sits in VRAM — at which point it is
not host-resident and the configuration is just a different placement split, which pre-registration
#43 already swept (K = 32…48) and #21 before it.

So Law 4's `SUM` survives on this hardware for this workload — not because tiers *cannot* overlap
in general, but because **for host-resident MoE weights both paths are gated by one DRAM read.**

### What survives from the red-team's proposal

Two things worth keeping, neither of which needs the dual-bus claim:

1. **The GPU genuinely idles ~22 ms per token** while the CPU expert phase runs. That is real
   headroom for *compute* — anything the GPU could do that does not require re-reading host
   weights. Speculative verification is exactly such a workload, which is one reason speculation
   works so well here.
2. **The falsification-first instinct was correct** and is the reason this cost an hour instead of
   a week: the idea came with a zero-code test, and the test's own control killed the test.

**Wired into:** `findings/REGISTER.json:D-14` (dual-bus expert streaming, refuted on physics) ·
the protocol note on interleaving arms.
