# Pre-registration #67: the Big Thing — model-drafter speculation on NOVEL text, dense target

**Author:** Federico Sciuca · **Date staked:** 2026-07-28, BEFORE the runs. **STAKED.**

## The untested cell

Every drafter measurement in this project ran against the MoE FLAGSHIP on the SPLIT placement
(D-09: 0.6B draft 0.72x; L-13: acceptance collapse) — where verify rounds pay the CPU-expert
path and the economics are hostile by construction. The textbook speculative-decoding setup —
a small draft and a DENSE target, both all-in-VRAM, verify rounds entirely on GPU — has never
been run on this box. It is the one cell where the Big Thing (novel-text speculation) could
still live, and both models are on disk: Qwen2.5-0.5B-Q8_0 drafting Qwen2.5-7B-Q4_K_M
(same family, shared vocab; measured solo: 154 vs 22.6 tok/s — a 6.8x speed ratio).

## Method

`llama-speculative` (pristine build): 7B target -ngl 99, 0.5B draft -ngld 99, NOVEL prompts
(one prose, one code — no context to copy), n 256, draft lengths K in {4, 8}. Baseline: the
same binary/prompt with speculation disabled (or the measured 22.6 bench figure, both quoted).
Metrics: accept rate and end-to-end decode tok/s.

## Stakes

- **P-1 (acceptance).** Novel-CODE acceptance at K=4-8 lands in **45-75%** (L-13's short-draft
  regime measured 75%@3 collapsing to 35%@16 on the flagship; the dense-target draft is the
  same drafter, so the acceptance CURVE should transfer even though the economics differ).
- **P-2 (THE PRIZE).** End-to-end novel-code decode >= **1.25x** the no-spec baseline. With a
  6.8x speed ratio and ~60% acceptance the textbook EV is ~1.5-1.8x; 1.25x is the bar for
  "the Big Thing lives on dense targets".
- **P-3 (prose).** Novel prose >= 1.05x (prose acceptance measured lower everywhere).

## KILL RULE

**If P-2 fails (< 1.25x on code) AND prose <= 1.05x, novel-text speculation is dead on this box
in BOTH architectures' best cells** — the flagship (measured dead, D-10) and the dense AIV
(this) — and the Big Thing closes NEGATIVELY: the register records that on 6GB-class hardware
the only speculation that pays is copy-regime, full stop. That is a publishable conclusion.
If P-2 HITS, the tool's speculation advice gains a dense-model branch it has never had.

**Wired into:** pending; `speculation_advice()` and D-09/D-10/L-13 score either way.

---

## Scored (2026-07-28, log: `weights/data/prereg67_novel_spec.log`, baseline 22.63 same-state)

**Verdict: P-1 HIT (67.9% @ K=4, in the staked band). P-2 MISS — best is 1.11x at K=2, below
the 1.25x bar. P-3 MISS (prose 0.95x best). THE KILL RULE FIRES as pre-committed — and it
closes the Big Thing with a number instead of a null.**

The full K-curve on novel code (first-ever positive novel-text speculation on this box):
K=1: 85.5%/24.27 · **K=2: 79.0%/25.16 (1.11x)** · K=3: 75.5%/24.13 · K=4: 67.9%/22.92 ·
K=8: 39.1%/20.90 (0.92x) · adaptive p-min 0.75 changed nothing · prose peaked at 0.95x.

### What this settles

1. **Novel-text speculation is ALIVE but SMALL on this box: +11% on code-style output, dense
   all-in-VRAM targets, K=2.** Prose: net-negative at every K. MoE flagship: measured dead
   before (D-09/D-10). The 0.5B drafter's own cost is the binder — at a 6.8x speed ratio the
   draft+verify overhead eats most of an 79%-acceptance harvest. The arithmetic says the prize
   needs a ~20x-faster drafter (sub-100M class), which does not exist in this model family.
2. **llama.cpp's default draft length (3) is not the optimum here** — K=2 beats it; K>=4 is a
   regression on novel text. The tool's advice now says so.
3. **The "anything left on the table" question closes measured:** the last crumb on this box is
   +11% code-novel via a dense-target draft, now shipped as scoped advice. Everything larger
   requires either a drafter class that does not exist locally, batching (serving), or the
   copy-regime (measured 2.2-5x, long shipped).

**Wired into:** `speculation_advice()` (dense-target branch added) · D-09 scope amended (the
0.72x was the SPLIT flagship; the dense cell measures 1.11x) · L-13 (curve extended with the
dense-target datapoints).
