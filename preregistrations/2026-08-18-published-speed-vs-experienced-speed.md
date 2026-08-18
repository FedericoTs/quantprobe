# Pre-registration #105: is our published speed the speed a user actually gets?

**Author:** Federico Sciuca · **Date staked:** 2026-08-18, **before any arm below was run.**
**VOID / UNSCORED — 2026-08-18. The premise died at the reference arm. See the verdict at the
foot of this file. Superseded by [pre-registration #106](2026-08-18-is-the-headline-reproducible.md).**

## Why

The model card for [Qwen3.6-35B-A3B-depthaware-GGUF](https://huggingface.co/FedericoSciuca/Qwen3.6-35B-A3B-depthaware-GGUF)
publishes **14.86 ± 0.36 tok/s** at `-ngl 12`. That came from `llama-bench` `tg128`, N=5.

Running the same file through `llama-cli` on the same box — the command a user would actually
type — the generation-sanity log
([`qwen36_generation_sanity.log`](../weights/data/qwen36_generation_sanity.log)) reports
**11.3, 11.4, 11.4 tok/s** across three prompts. That is **23% below the published figure**.

This is the C-31 failure mode with a different mask. C-31 was "we quoted the first line of a log
and it was the least representative request." This would be "we quoted the benchmark harness and
the harness is not the product." A headline number has to be the number the reader gets, or it
has to say out loud which mode produced it.

Both numbers are real. The question is what causes the gap, and which one belongs on the card.

## Hypothesis

`llama-bench` allocates exactly the KV cache its `-n`/`-p` need. `llama-cli` defaults to a full
context — 4096 tokens on this build — and allocates that KV up front. On a **6 GB card holding
12 of 40 layers**, VRAM is the binding resource, so a bigger KV allocation displaces weights
that would otherwise sit on the GPU. If that is the mechanism, decode rate should fall
monotonically as `-c` grows, and a small `-c` should recover the benchmark number.

If instead the rate is flat across `-c`, the gap lives somewhere else (harness overhead,
sampling stack, thread defaults) and the card's number is simply the wrong mode to quote.

## Predictions (staked before measurement)

- **P-1.** `llama-cli -c 512` reaches **≥ 13.5 tok/s** — i.e. it recovers most of the gap to
  14.86. *Refuted if it lands below 13.5.*
- **P-2.** Decode rate is **monotonically non-increasing** across `-c` ∈ {512, 2048, 4096, 8192}
  (allowing overlap within one standard deviation). *Refuted by a non-monotone ordering whose
  reversal exceeds the error bars.*
- **P-3.** `llama-cli -c 4096` (the default) reproduces the sanity log at **11.4 ± 1.0 tok/s**,
  confirming the gap is a property of the configuration and not of that one session.
  *Refuted if the default-context arm lands outside that window.*

## Amendment, pre-data (2026-08-18, before any arm ran)

Priced the KV before running, and it argues **against** my own hypothesis. `quantprobe`'s spec
reader gives this file `kv_layers = 10` (of 40 — the hybrid linear-attention split, U-51) and
`kvp = 20,480` bytes/token. So:

| context | KV cache |
|---|---|
| 512 | 10.5 MB |
| 4096 | 83.9 MB |

The whole span of the sweep moves **74 MB on a 6 GB card — 1.2% of VRAM**. That is not a
plausible cause of a 23% throughput loss. **P-1 is now expected to be refuted**, and I am
leaving it staked at 13.5 rather than softening it, because a prediction retuned once it looks
uncomfortable is not a prediction.

Adding one arm to separate the two remaining explanations, since the KV story is weak:

- **P-4.** `llama-bench -n 512` stays within **1.0 tok/s** of `llama-bench -n 128`. If it does,
  the generated-token count is not the cause and the gap is a **harness** difference —
  `llama-bench` timing the decode loop, `llama-cli` timing decode plus sampling, detokenization
  and output. *Refuted if the two llama-bench arms differ by more than 1.0 tok/s*, which would
  instead point at context depth.

If P-1 falls and P-4 holds, the kill rule's HEADLINE-CHANGES branch fires: the card leads with
what the user experiences, and the benchmark figure keeps its place as a labelled harness
number rather than the headline.

## Method

One binary (`llama-cli`, b10098), one file, one machine state, arms differing **only** in `-c`.
`-ngl 12 -n 256 -st --simple-io --seed 1234`, 2 reps per arm, reported as mean ± half-range.
C-14 holds: nothing else runs on the box. `llama-bench -n 128` is re-run in the same session so
the reference is not imported from a different machine state.

## Kill rule (committed before data exists)

Scored by [`weights/prereg105_score.py`](../weights/prereg105_score.py), written and committed
**before** the run.

- If **P-1 and P-2 both hold**: the gap is a VRAM-displacement effect. The card keeps 14.86 but
  must state the context it was measured at and carry the `-c` curve, because a user on the
  default gets 23% less and deserves to know why.
- If **P-1 is refuted**: the benchmark number is not reachable from the real command on this
  hardware. **The card's headline changes to the `llama-cli` figure**, and 14.86 is demoted to a
  labelled harness measurement. A number you cannot reach by typing the documented command is
  not a headline.
- Either way the card is edited **the same day**, and the correction is published at the same
  size as the original claim.

---

## Verdict: VOID / UNSCORED (2026-08-18)

**P-1, P-2, P-3, P-4: VOID. No treatment arm ever ran.**

The run order put the `llama-bench` reference first, so the comparison would come from one machine
state instead of being imported from another day. It came back at **11.0 tok/s**.

That is the same `llama-bench tg128` that produced the published **14.86 +/- 0.36** - byte-identical
file, same binary (b10098), same `-ngl 12`, same box, the command differing only in `-r 3` versus
`-r 5`. And it lands within 0.4 tok/s of the `llama-cli` figure this prereg was written to explain.

So the framing was wrong. There is no harness-versus-product gap: **`llama-bench` and `llama-cli`
agree at ~11 tok/s today.** The 23% is not between two tools, it is between two *sessions of the
same tool*. Every prediction here tests a distinction that does not exist, so the remaining arms
were killed rather than run - twelve minutes of exclusive box time spent answering a dead question
is twelve minutes not spent on the live one.

**What the reference arm cost, and what it bought.** It cost the prereg. It bought the actual
defect, which is worse than the one we staked: *our published headline does not reproduce on the
machine that produced it.*

Diagnostics taken immediately after, box idle:

| probe | reading | rules out |
|---|---|---|
| VRAM held by the desktop | **409 MiB** of 6144 | contention from Edge / XD / shell |
| GPU performance state | **P0**, mem clock 4004 of 4006 MHz | power or thermal throttling |
| free system RAM | **12.24 GB** of 15.95 | - |
| model file | **13.15 GiB** | - |

The last two rows are the finding. The file is *larger than free RAM*, so it cannot be held whole
in page cache, and how much of it is resident depends on whatever else the machine was doing that
minute. Law 4 prices bytes streaming from RAM. When the bytes come off disk instead, the law is
being applied outside the regime it was measured in - and nothing in our tooling said so, because
nothing in our tooling records free RAM next to a tok/s figure.

**Not amended into a new hypothesis.** One number had already been measured when the premise
broke, and rewriting predictions around data already seen is the failure this discipline exists to
prevent. #105 stays void. The successor question is staked clean in prereg #106
([`2026-08-18-is-the-headline-reproducible.md`](2026-08-18-is-the-headline-reproducible.md)), with
its scorer committed before its arms run.
