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

---

## Scored (2026-07-27, log: `weights/data/prereg26_parallel_slots.log`)

**Verdict: P-1 MISS, P-2 MISS and refuted by DIRECTION, P-3 MISS, P-4 HIT. The refutation condition
I wrote into the stake is triggered — and it is the outcome that is better for the tool.**

`llama-batched-bench`, npp 512, ntg 128, one session. Aggregate decode (`S_TG t/s`):

| `-npl` | S: split, ub 1024 | A: all experts→CPU, ub 2048 | control: dense 7B all-in-VRAM |
|---|---|---|---|
| 1 | 21.93 | 19.89 | 22.47 |
| 2 | 32.39 | 27.82 | 38.35 |
| 4 | 38.45 | 30.28 | 49.31 |
| 8 | **44.48 (2.03×)** | **37.70 (1.90×)** | **50.48 (2.25×)** |

- **P-1 (≥3.0× on arm S at npl 8): MISS.** 2.03×.
- **P-2 (arm A gains ≥25% more than arm S): MISS, and backwards.** A gained *less* (1.90× vs
  2.03×). The hypothesis was that more host-resident weight means more PCIe traffic to amortise
  across slots; the direction is wrong, so the reasoning is wrong, not merely the magnitude.
- **P-3 (A overtakes S at npl 8): MISS.** 37.70 vs 44.48. The #25 winner stays the winner.
- **P-4 (no anchor moves): HIT.**

### The control kills my explanation too

Having lost P-2 I reached for the obvious MoE story — different slots route to different experts,
so batching cannot amortise expert traffic the way dense batching would. **The dense control
refutes that as well: 2.25×, barely above the MoE split's 2.03×.** Saturation is essentially
identical across architecture (MoE vs dense), placement (split vs all-CPU vs fully resident), and
memory tier. Whatever the ceiling is, it is none of the things this project models.

What the numbers say without an explanation attached: per-step time grows **3.56×** from npl 1 to 8
while the weight bytes read per step are unchanged. So weight amortisation is working exactly as
expected; the cost that grows is per-sequence work, and this measurement does not identify it.

**I am not naming a mechanism.** In this project my first mechanism guess has now been wrong four
times — the fixed-overhead model (#15), GPU clock state (#24), bytes-per-token (#24), and
host-residency here — and each time the wrong guess survived precisely as long as nobody built a
control that could kill it. This is recorded as a measured ceiling with an open cause.

### The one place the staked mechanism IS visible

Prefill, where L-09's per-ubatch amortisation actually applies:

| `-npl` | S | A | dense |
|---|---|---|---|
| 1 | 308.17 | **200.73** | 394.07 |
| 8 | 443.82 (1.44×) | **425.31 (2.12×)** | 405.22 (1.03×) |

Arm A gains most, exactly as P-2 reasoned — because it starts furthest below the ceiling, having
the most host-resident weight to amortise. But **all three converge to ~405–445 t/s**, which looks
like a hard compute ceiling on this GPU, and convergence is why the effect cannot show up as a
placement difference at npl 8. The mechanism is real; it is simply capped before it can change any
recommendation.

### What this settles, and what ships

**The #25 single-point collapse holds at every concurrency measured.** That is what the stake said
a P-2 failure would mean, and it is the better result: one recommendation covers the single-user
desktop and the small multi-user server alike, on this hardware. No concurrency input is needed.

What ships is **one disclosed sentence**, because leaving it unsaid is the actual error: our
single-stream numbers understate aggregate server throughput by roughly **2×**, saturating by about
4 slots. Anyone reading a quantprobe figure as a server capacity number is reading it wrong in a
direction we can quantify.

**Explicitly NOT claimed:** that the ~2× ceiling generalises off a GTX 1060. A 2016 card without
tensor cores is exactly where a batched-decode ceiling would appear first, and the single most
valuable follow-up is this same sweep on any modern GPU — it costs one command and would tell us
whether this is a law or a museum piece.

**Wired into:** `findings/REGISTER.json:C-06` (the open ceiling) · `findings/REGISTER.json:U-05`
(scored) · disclosure pending in the CLI.
