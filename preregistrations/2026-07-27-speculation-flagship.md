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
