# Roadmap

## The queue, in order (updated 2026-08-04)

Close-first discipline: nothing new opens while a staked test sits unscored.

| # | item | size | track |
|---|---|---|---|
| 1 | Score **X-1** (verify-width cliff) against its staked kill rules; register | in flight | A |
| 2 | Register **U-40** (draft-driven expert prefetch) and **U-41** (expert-coherent sampling) as staked untried ideas | minutes | C |
| 3 | **A2A benchmark** — depth-aware quant vs uniform `llama.cpp` quant, *same model, same hardware, same context*: ppl + KL(+tail) + business pass-rate + tok/s, packaged reproducibly. The exact table the community asks for; the exllamav2 arm is not runnable on Pascal and goes on the `bench --contribute` ask for an Ampere volunteer | half-day | A→B |
| 4 | Gemma-4-26B **0.77×** — recheck a/ne against the GGUF header, re-score the Spark row | short | A |
| 5 | **C-23** discriminators — kimi byte counter (E-12) + airllm non-mmap (E-11) | short | A |
| 6 | **Per-shape calibration** prereg (07-31) — run or close with a written reason | short | A |
| 7 | Ladder rows for **0.6B and 7B** — the comparison surface becomes real | short | A |
| 8 | **Blind-score the 8 rubric outputs** (needs an independent judge — Federico) | short | A |
| 9 | **Batch-axis advisory** in the planner from the U-38/U-39 curves, incl. the Pascal anti-valley warning (widths 2–8 dominated; 1 or ≥9) | medium | A |
| 10 | Lane pilots **X-2/X-3** (predicate-picked best-of-16; 16-vote on t4a2) | evening | C |
| 11 | Distribution loops (leaderboard → interactive matrix → HF badge → launch post) | rolling | B |
| 12 | **S-1** specialist distillation — awaiting explicit go | days | C |

Parked with reasons: t4a2/t4c1 need a faster box or bigger budget; U-40 needs llama.cpp
surgery; older backlog (#47–51, #53, #55–57) slots behind the above.

The method does not change as the scope grows: **stake the prediction, define the kill rule,
measure, publish the misses at the same size as the hits.** Everything below is ordered so each
step either closes an open claim or builds on one already closed. Nothing here is a promise of a
result — several entries exist precisely because they might die, and saying so in advance is the
point.

## Track A — evidence debt (close what is open before opening more)

| item | state | what closes it |
|---|---|---|
| **U-38 batching crossover** | staked, sweep running | `-np` sweep N=1..16 on a 7B all-in-VRAM; K-3 lets the hypothesis die if aggregate never clears ~2× |
| **Gemma-4-26B at 0.77×** | open, violates C-02 | the one Spark row that over-predicts; either our active-parameter figure is wrong or the floor has an exception. Next: verify a/ne from the GGUF header instead of the model card |
| **C-23: the 1.82× streaming gap** | mechanism unknown | two cheap discriminators queued: airllm reads without mmap (E-11), and kimi-k3-in-c may log per-token bytes (E-12) — either separates "mmap page-fault cost" from "cost intrinsic to the access pattern" |
| **Per-shape calibration** | prereg staked 07-31 | run it or close it with a written reason; a staked prereg left dangling is worse than a miss |
| **Tier ladder, more rows** | 1 model scored | the 0.6B and 7B on the same 52 predicates — cheap, and the ladder only means something as a comparison surface |
| **Rubric blind-scoring** | 8 outputs captured, unjudged | shuffle, strip provenance, judge against the pre-written bars — the protocol exists, it has just never been executed |
| **T4 recall immunity** | caveat logged | the one T4 solve was the puzzle family models have seen; add *generated* variants (fresh constraint sets, brute-force-verified unique) so recall cannot help |

## Track B — distribution (the tool is validated; now it has to be found)

Order matters: each loop feeds the next.

1. **`bench --contribute` leaderboard + atlas.** Every contribution is a machine we did not own
   and a datapoint the law gets scored against in public. The GPU-resident eta gap and the
   Mac/50-series presets are *explicitly* waiting on this.
2. **The matrix as the shareable artifact.** [MATRIX.md](MATRIX.md) is already the thing people
   screenshot; an interactive version in the browser build (pick your hardware, see every model
   priced, binding constraint named) is the speedtest-style mechanic.
3. **HF model-card badge** — "predicted on your hardware by quantprobe" with the one-command
   reproduction line. Zero-cost distribution at the exact moment someone is choosing a quant.
4. **The launch post** is now honest to write: *we recommended a config, then made it prove
   itself — 40/40 machine-checked tasks, misses published, every number reproducible on demand.*

## Track C — model surgery, scientifically (merge · distill · prune)

The question behind this track: **can we change what a model *is*, not just where it runs, and
prove the change with the same discipline?** The instruments are already built — KL divergence
with the tail measured separately (L-27), the 52-predicate task suite, the tier ladder, and a
fragility probe that knows *which layers* of a given model break under pressure. What follows are
experiments, each with a gate it can fail.

**Hardware honesty first:** this box trains nothing beyond ~1B-parameter students. Everything
larger in this track is either inference-time composition, weight-space surgery (no training), or
waits for contributed compute from Track B.

### S-1 — the specialist student: distill ONE cluster into a 0.5B

*Claim to test:* a 0.5B tuned on a single task family (say, structured extraction) can match or
beat the 30B teacher **on that family only**, at 40× less memory and single-digit-millisecond
latency.

- Teacher: the validated 2.5-bit 30B generates a few thousand (input → output) pairs for one
  cluster, temperature 0, format-checked by the same predicates that score the suite.
- Student: Qwen3-0.6B, LoRA fine-tune (fits this box).
- **Gate, staked before training:** student must pass ≥ the teacher's rate on *held-out* tasks of
  that cluster (fresh instances, same predicates — the suite generates them mechanically), and
  must NOT regress more than a staked margin on one untouched cluster (catastrophic-forgetting
  check). Kill: if the tuned student beats its own base by <10 points, the distillation was
  noise.

### S-2 — expert pruning on the MoE (surgery without training)

The 30B routes 8 of 128 experts per token; exp95 measured usage skew on our workloads. The REAP
prereg (2026-07-24) staked expert-pruning arithmetic and was parked. Resurrect it with today's
instruments:

- Drop the coldest experts (by measured routing frequency on a fixed corpus) → smaller file →
  more experts fit in VRAM → the planner's split moves.
- **Gate:** KLD vs the unpruned model under L-27's tail metric, plus the full 40-task suite.
  Speed is predicted by the planner *before* the cut (fewer bytes/token only if routing avoids
  the dropped experts; the law prices both outcomes). Kill: any cluster drops below its
  unpruned pass rate.

### S-3 — merging two fine-tunes, with the fragility atlas as the referee

Weight-space merges (SLERP / TIES) are cheap — no training, pure arithmetic on files this box
handles. The scientific angle nobody ships: **our probe measures which layers are fragile per
model. Does merge damage concentrate in exactly those layers?**

- Merge two same-architecture fine-tunes; score the merge on the ladder vs both parents.
- Then the real experiment: a **fragility-weighted merge** (protect the layers the probe flags,
  interpolate freely elsewhere) against the naive merge. Stake the direction before running.
- Kill: if fragility-weighted ≤ naive on both KLD and the suite, the atlas does not transfer to
  merging and we say so.

### S-4 — composition without surgery (the baseline all of the above must beat)

Draft-model speculation is already "two models composing at inference time," and it is measured
here: ngram 2.10× on code, MTP +17%→−24% across placements. Any surgery result that a
speculation pair matches at zero training cost is not worth shipping. This track is the control
arm, and it is already done — that is what makes the others scoreable.

### Sequencing

S-1 first (cheapest, clearest gate, uses the suite as both teacher-filter and judge), S-2 second
(the prereg already exists), S-3 third (needs two suitable parents chosen honestly). Each gets
its own preregistration file before any weights move.

## Standing rules that bound all three tracks

- One `cal_id` per comparison; no cross-state numbers (C-14).
- Every headline number reproducible on demand, or it is not a headline.
- A staked expectation that misses is published as a miss — the T4 ladder's 0/6→1/6 already set
  that precedent.
- The kill rule is written before the experiment, and firing it is a result, not a failure.
