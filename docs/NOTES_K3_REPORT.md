# Notes: Kimi K3 technical report (arXiv 2607.24653) — what transfers to this project

Read 2026-08-04, full text. 2.8T MoE, **104B activated** (paper figure — replaces our "arch
est"), 16/896 routed experts, 1M context. What follows is only what changes *our* work.

## Their post-training recipe, compressed

SFT (trajectories synthesized by prior domain-specialist models + **multi-stage verification** +
human-in-loop) → RL as **nine specialists** (3 domains × {low, high, max} reasoning effort) →
**MOPD** (Multi-Teacher On-Policy Distillation) consolidates them into one model: the *student*
generates rollouts and the per-token reward is `clip(sg(log π_teacher/π_student), ±Rmax)`.
Negative result they publish: finer top-k distillation objectives — "no clear advantage."

## Transfers to S-1 (Track C)

1. **Specialize-then-consolidate is the frontier recipe; S-1 is its first half.** The sequel is
   now defined: train per-cluster specialists, then multi-teacher-distill into one student —
   feasible at 0.6B with our own specialists as teachers.
2. **S-1b (staked before running, after the vanilla arm):** an on-policy pass — student
   generates, teacher logprobs grade per token (llama.cpp serves logprobs). MOPD's negative
   result says plain logprob-ratio suffices; do not over-engineer the objective.
3. **Effort is a training axis, not an accident.** The 0.6B student is trained deliberately as
   a *low-effort* specialist: teacher targets with thinking stripped. (Matches three separate
   thinking-budget lessons this suite has already paid for.)
4. Their verification-gated everything = our predicate-filter design, at 2.8T. Keep filter
   primacy.

## Transfers to the core project

- **External convergence on depth-aware quantization:** K3 ships MXFP4 *experts* while
  attention, shared experts and routers stay higher precision, with QAT through all of
  post-training and matched train/serve quantization. "Protect the always-active, squeeze the
  routed" is our recipe's shape, deployed at 2.8T. → cite; and **task #49 (MXFP4 in
  FORMAT_EBW) rises in priority** — K3-class GGUFs will arrive as MXFP4.
- **Their serving stack is speculation** (pre-trained MTP layer fine-tuned into an EAGLE-3
  draft, 7-step unroll). The X-1 kernel rule concerns the verify pass and therefore applies to
  EAGLE-class drafts too, not just ngram — advisory can generalize.
- **MATRIX gets real K3 numbers** (2.8T / a=104B, paper): at ~4.25-bit experts that is
  ~58 GB/token — K3 is batch-and-draft territory on anything below datacenter bandwidth,
  consistent with the E-12 corrected byte model (104B activated ≈ trunk + 16 experts).
