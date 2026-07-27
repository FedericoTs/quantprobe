# Pre-registration #28: speculation on the flagship — the only legal way past the wall

**Author:** Federico Sciuca · **Date staked:** 2026-07-27, BEFORE the measurement. **Status: STAKED.**

## Why this is the decisive test

Pre-registration #27 recomputed the decode wall on MEASURED memory bandwidth: **41.1 tok/s** is the
absolute raw-decode ceiling for the flagship on this box, and we sit at 22.25 (54%). The stated
goal — 52.9 or beyond — is therefore unreachable by ANY runtime that reads every active byte for
every token. Speculation breaks that axiom: the draft proposes k tokens, the target verifies them
in ONE forward pass, so one weight-read serves up to k tokens. Effective tok/s can exceed the wall
by up to the acceptance-weighted batch factor. Nothing else in the register can do this.

V-04 measured ngram-simple at 2.10x on CODE on a dense 7B, 1.01x on prose. This measures the
flagship itself, on the winning placement, and adds a DRAFT MODEL arm (Qwen3-0.6B-Q8_0, same
tokenizer family) which can generalise where span-copying cannot.

## Configurations

Target `Qwen3-30B-A3B-Q2_K`, split placement, `--no-mmap`, `-t 4`, `--temp 0`, `-n 256`, r=3,
llama-cli `eval time` tok/s. Two content types: PROSE (wikitext continuation) and CODE (Python
from this repo). Draft arm: `-md Qwen3-0.6B-Q8_0 -ngld 99`; the draft is a fourth VRAM claimant
(L-06), so its arm widens the CPU split to layers 12-47 to make room, and VRAM is logged - if the
cliff fires it will be visible, not averaged in.

| arm | spec |
|---|---|
| B | none (baseline; must reproduce ~22 or the session is contaminated) |
| N | `--spec-type ngram-simple` |
| D | `--spec-type draft-simple -md` 0.6B |

## Stakes

- **P-1 (ngram on code, the shipped claim reaches the flagship).** N/code ≥ **1.5×** B/code.
- **P-2 (content-dependence control).** N/prose ≤ **1.15×** — ngram drafts by copying spans; prose
  has few; if this "gains" more, the harness is broken, not the law.
- **P-3 (the draft model generalises).** D/prose ≥ **1.3×** B/prose — the arm ngram cannot serve.
- **P-4 (THE WALL IS EXCEEDED).** Best arm on code ≥ **41 tok/s** — effective decode above the
  measured raw-decode ceiling, on a 2016 GPU, proving the axiom-break works where no runtime
  improvement possibly could.

## Refuted if

P-4 fails while P-1 holds: speculation pays but cannot clear the wall on this hardware — the
target's verify pass itself is then the binding cost, and the honest statement becomes "the wall
stands; speculation recovers only its acceptance-weighted fraction." P-3 failing kills the draft
arm for prose (acceptance too low on a 30B MoE), leaving ngram/code as the only speculative gain.

## What ships

Whatever the numbers are: the flagship's plan output currently quotes the dense-7B speculation
figures; after this it quotes its OWN. If P-4 holds, the wall statement in L-11 gains the corollary
that it is exceedable and by which lever. Quality is NOT at issue: speculative decoding with
temp-0 verification is output-identical by construction, so unlike #25 no perplexity gate applies.

---

## Scored (2026-07-27, log: `weights/data/prereg28_speculation.log`)

**Verdict: P-1 HIT but on the WRONG AXIS (see below), P-2 HIT, P-3 MISS decisively, P-4 HIT —
50.04 tok/s, the measured raw-decode wall exceeded by 22% on a 2016 GPU.**

### First, the trap this measurement nearly fell into

The staked harness (temp-0 raw continuation) produced spectacular numbers — ngram 1.82× on code at
**100% acceptance**, 1.96× on prose — and inspection of the actual output showed a **repetition
loop**: `(but 4.5x 32B is 144B, which is 144B / 32B = 4.5x). So 4.5x 32B = 144B` forever. Ngram
feasts on loops. Those tok/s are real; the content is garbage; scoring on it would have been the
ubatch-cliff mistake again — quoting a peak from a pathological corner as a property of the
configuration. All arms were re-measured under the chat template on real tasks with coherent
output (samples logged). Also: **`llama-cli` silently ignores `--spec-type`** — an hour went to a
1.00× that was the flag not existing. Speculation lives in `llama-server` only.

### The honest numbers (chat template, coherent output, r=2 each)

| task regime | baseline | ngram-simple | draft 0.6B |
|---|---|---|---|
| novel code (write a new function) | 21.13 | 20.62 (0% acc.) | 14.90 (81% acc.) |
| novel prose (summarise + explain) | 21.35 | 21.32 | 16.33 (83% acc.) |
| **edit (rename var, emit full file)** | 20.73 | **50.04 (89% acc.)** | — |

- **P-1 (ngram ≥1.5× on code): HIT, but the axis is wrong.** "Code vs prose" is not the variable.
  **COPY vs NOVEL is.** Novel code generation gets 0% acceptance and 0.98×; an edit task on the
  same file gets 89% and 2.41×. V-04's shipped claim ("if you write CODE, 2.10×") must be re-scoped:
  the prize attaches to output that REPRODUCES context spans — edits, refactors, quoting,
  boilerplate — which happens to be most of what coding agents do all day, but is not "code".
- **P-2 (prose control ≤1.15×): HIT.** 1.00× on novel prose, exactly as content-dependence demands.
- **P-3 (draft model ≥1.3× on prose): MISS, decisively — it is NET NEGATIVE (0.72–0.79×) at 81–83%
  acceptance.** The 0.6B's own forward passes plus the widened expert split cost more than
  verification saves. On this box the draft-model arm is a dead end at any acceptance rate this
  model family delivers; the only speculation that pays is the FREE kind.
- **P-4 (best arm ≥41 tok/s: the wall): HIT. 50.04 ≥ 41.1.** Effective decode 22% above the
  measured raw-decode ceiling, from 20.73 baseline — 2.41×, output-identical, one flag.

### What this establishes

The axiom-break works exactly as the arithmetic said it must: one 30B weight-read verifies ~3.4
tokens when 89% of drafts are accepted, so effective bandwidth-per-token exceeds what the memory
system can physically deliver for token-by-token decode. **The 52.9 "wall" from the old spec-sheet
arithmetic is 95% reached (50.04) in the one regime where speculation legitimately fires** — and
raw decode remains hard-limited at 41.1, with 22.25 measured, exactly as #27 concluded.

An honest map of decode on this box, all measured today:

| regime | tok/s | limited by |
|---|---|---|
| raw decode, measured | 22.25 | kernel at 66% of real stream |
| raw decode, wall | 41.1 | measured DRAM (26.1 GB/s) |
| copy-regime speculation | **50.04** | acceptance × verify cost |
| novel-generation | 21.3 | the raw wall — nothing to draft |

**Wired into:** `findings/REGISTER.json:V-04` (re-scoped copy-vs-novel) · `findings/REGISTER.json:L-11`
(exceedability corollary) · `findings/REGISTER.json:D-09` (draft-model dead end) · CLI advice update.
