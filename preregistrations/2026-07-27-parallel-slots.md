# Pre-registration #26: concurrency is the dimension this project has never measured

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## Why this is the largest remaining gap

Every number in this repository — every anchor, every frontier row, every η — was measured at
**one request at a time.** That is a defensible default for a single-user desktop, and it is also a
blind spot large enough to invert our advice for anyone serving more than one person.

The mechanism is already established. **L-09**: with `-ot ...=CPU` the expert tensors cross PCIe
**once per ubatch**, not once per token, because ggml offers the op to CUDA above 32 tokens. At
`-np 1` a decode step presents *one* token, so that amortisation never fires during generation —
it only helps prefill. Concurrent slots fill a decode ubatch that a single stream cannot.

So the host-resident placements should have a large, unmeasured concurrency headroom that
VRAM-resident weights do not, because a resident weight was never paying PCIe in the first place.
If that is right, **the configuration we recommend changes with the number of users**, and we have
never told anyone that.

This also matters because pre-registration #25 just collapsed the placement frontier to a single
point. If concurrency re-opens it, the tool needs a concurrency input rather than a workload-ratio
one — and if it does not, the collapse is a much stronger result than one session can otherwise
support.

## Configurations

`Qwen3-30B-A3B-Q2_K`, reference box, `llama-batched-bench`, one session, GPU state logged.
`-npl 1,2,4,8`, prompt 512, generation 128, `-c` sized to hold all slots.

| arm | placement | why |
|---|---|---|
| S | split, `-ub 1024` (the #25 winner) | what we currently recommend |
| A | all experts → CPU, `-ub 2048` | **more** host-resident, so more to amortise |

## Stakes

- **P-1 (batching pays at all).** On arm S, **aggregate** decode throughput at `npl=8` is
  **≥ 3.0×** the `npl=1` figure. Batching amortises the weight read across slots on any tier, so a
  failure here means the harness is not doing what I think it is and everything below is void.
- **P-2 (host-residency pays MORE — the actual hypothesis).** Arm A's aggregate speedup from
  `npl=1` to `npl=8` **exceeds arm S's by ≥25% relative**. A has strictly more weight sitting in
  host memory, so it has strictly more PCIe traffic to amortise, and L-09 says that traffic is
  charged per ubatch.
- **P-3 (the recommendation inverts).** At `npl=1` arm S beats arm A on decode (21.58 vs 19.79,
  measured in #25). **At `npl=8`, arm A's aggregate decode overtakes arm S's.** If this holds, the
  single-point collapse from #25 is correct *only for single-user serving*, and concurrency is a
  first-class input the tool does not have.
- **P-4 (no law changes).** A flag, not a law. All four published anchors bit-identical.

## Refuted if

**P-2 fails.** Then host-resident weights gain no more from concurrency than resident ones, the
per-ubatch PCIe amortisation does not reach decode even when the ubatch is full, and my reading of
L-09 is wrong about the generation phase specifically. That is a clean negative and it would mean
the #25 collapse holds at every concurrency — which is a *better* outcome for the tool, since it
means one recommendation covers everyone.

P-1 failing invalidates the harness rather than the hypothesis, and P-2/P-3 become uninterpretable.

## What ships

**Nothing automatic.** If P-3 holds, the honest change is to disclose the concurrency dependence
and name the alternative command — the v1.13.1 shape: state the trade, give the flags, let the user
choose. A concurrency input would only be worth adding if the crossover is large and lands at a
realistic slot count.

If P-3 fails but P-1 holds, the shipped change is a single sentence: batching is worth Nx and
quantprobe does not model it. Saying that plainly is better than leaving people to assume our
single-stream numbers describe a server.

**Explicitly NOT claimed:** anything about quality, latency SLOs, or memory pressure at high slot
counts. Aggregate throughput and per-request rate are both reported, because quoting only the
aggregate is exactly the kind of half-number this project has already had to correct twice.
