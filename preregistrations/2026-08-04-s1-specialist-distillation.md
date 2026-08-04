# S-1 — can a 0.6B student beat a 30B teacher on ONE task cluster?

**Date staked:** 2026-08-04, before any data was generated, any weight was touched, and before
a training stack existed on this box.

**The claim under test:** a 0.6B model fine-tuned on teacher-generated data for a *single* task
family (structured extraction) matches or beats the 30B teacher **on that family only**, at ~40×
less memory and single-digit-millisecond latency — and does not collapse on an untouched family.

This is the first Track C experiment (ROADMAP): changing what a model *is*, not just where it
runs, proved with the same discipline as the placement work.

## Why extraction

The 30B scored 5/5 on it, the checks are fully executable (JSON shape + exact values, zero
judgement), and instances can be **generated mechanically** — which is what makes an honest
held-out split possible at all. Nothing here rests on a human deciding whether an answer is good.

## Design

- **Instance generator** (`weights/s1_gen.py`): templated extraction tasks with randomised
  entities, amounts, currencies, dates, IDs. Seeded. **Train draws from seeds 0–N, held-out
  draws from a disjoint seed range**, and disjointness is asserted mechanically by comparing
  prompt hashes — not assumed.
- **Teacher:** Qwen3-30B-A3B Q2_K (the validated config), temperature 0, via llama-server.
- **Filter:** every generated sample must pass the same executable predicates the suite uses.
  Rejected samples are counted and reported; only passing pairs enter the training set.
- **Student:** Qwen3-0.6B, LoRA fine-tune (the only size this box can train).
- **Untouched control family:** classification (single-label, executable) — never trained on.

## Staked gates

- **P1 (did distillation do anything):** tuned student beats its **own base** on held-out
  extraction by **≥10 points**. Below that the run is noise and S-1 reports no effect.
- **P2 (the headline):** tuned student **≥ teacher's pass rate** on held-out extraction.
- **KR-A (catastrophic forgetting, refutation):** if the tuned student drops **>15 points**
  versus its base on the untouched classification family, the specialist was bought by
  damaging the model and S-1 is a **failure even if P1 and P2 pass**. This is the gate the
  experiment is most likely to die on and it is stated first-class.
- **KR-B (data sufficiency):** ≥300 filtered-clean training pairs, else precondition-blocked —
  reported as *unable to run*, not as a null.
- **KR-C (split integrity):** zero prompt-hash overlap between train and held-out. Any overlap
  voids the run entirely.
- **KR-D (environment):** this box has **no training stack today** (torch/transformers/peft all
  absent) and a Pascal card whose support in current torch is unverified. If the stack cannot
  be made to train, S-1 is **precondition-blocked (exit 2 semantics)** — not a failed
  hypothesis. Phase 1 below runs regardless and can kill the experiment on its own.

## Phases, cheapest falsification first

1. **Teacher data generation + filter** — no training stack needed. If the teacher cannot
   produce ≥300 clean pairs (KR-B) or the split cannot be made disjoint (KR-C), S-1 stops here
   having cost one evening of GPU time and no downloads.
2. **Environment** — torch/peft on Pascal. KR-D decides.
3. **Fine-tune + score** — P1/P2/KR-A against held-out instances the student has never seen.

Raw outputs under `weights/data/s1_*`. Verdict appended to this file whichever way it goes.

---

## VERDICT (scored 2026-08-04, same day): P1 FAIL - and the failure is the finding

Phase 1: KR-B PASS (385 teacher-clean pairs of 400; teacher held-out 116/120 = 96.7%, the P2
bar). KR-C held (zero prompt-hash overlap). KR-D PASS: torch 2.5.1+cu121 runs on the Pascal
card; the 1060 trained the LoRA in ~3 minutes.

| gate | staked | measured |
|---|---|---|
| P1 gain >= +10 pts | tuned - base on held-out | **+1.7% -> FAIL** |
| P2 >= teacher 96.7% | tuned held-out | 100.0% - **pass, but VACUOUS** |
| KR-A drop > 15 pts kills | control family | **0.0% drop - holds** |

**Why P1 failed and why that is the result worth having: the BASE student scored 98.3%.** The
untuned 0.6B already beat the 30B teacher on this family before any training happened - there
was no headroom for distillation to fill, and epoch-1 mean loss of 0.0068 confirms the student
already knew the task. P2's "student >= teacher" is therefore technically true and
scientifically empty: distillation did not produce it; model generations did.

What stands: the full pipeline is proven end to end on this box (generator with KR-E, teacher
filtering, LoRA training on Pascal, one checker scoring every arm, zero forgetting on the
untouched family). What died: the assumption that "a task the 30B is good at" implies "a task
with distillation headroom".

**Design rule staked for any S-2: measure the BASE STUDENT on the candidate cluster FIRST, and
only stake a distillation claim where the base scores materially below the teacher.** The
control family here (classification, base 46.7%) is exactly such a gap - but it was this
experiment's untouched KR-A control and cannot be recycled into its own follow-up without a
fresh stake. Evidence: `weights/data/s1_train_run.log`, `s1_student_results.json`,
`s1_*_teacher.json`; adapter on local disk.
