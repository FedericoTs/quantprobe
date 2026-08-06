# PHASE B — the data engine (non-benchmark training data, screen-verified)

**Date staked:** 2026-08-05, after Phase A's verdict and before any training sample existed.
Program law (2026-08-05-benchmark-sanctity) applies to every byte this phase produces.

**What Phase B builds:** the training corpus for Phase C, from two feeds that never touch a
protected bench, plus the decontamination screen as living, tested code.

## Feeds

1. **Public corpus, re-verified here.** Candidate: `bigcode/self-oss-instruct-sc2-exec-filter-50k`
   (StarCoder2 self-generated, execution-filtered, open provenance - no closed-API-derived
   text), fallback NVIDIA OpenCodeInstruct. B1 verifies license + schema before anything else;
   a corpus whose license or provenance fails inspection is dropped, stated, and replaced.
   Every sample re-executes in OUR sandbox (provenance restored by re-verification - the
   filter-on-import rule); only passing samples proceed to the screen.
2. **Committee-generated problems.** The Coder-30B invents problem+tests from seeded diversity
   templates; 4B and 7B lanes solve (the 4B earned first chair in Phase A); plus-style
   verification filters. Fully local provenance, unlimited supply.

## The screen (weights/decon.py, unit-tested before first use)

Exact prompt-hash + sliding 8-gram token overlap (whitespace-normalized, case-folded) against
ALL protected-bench prompts AND canonical solutions (MBPP+ 378, HumanEval+ 164, pinned
hashes). Any hit -> sample EXCLUDED and counted. The screen's own tests include a
known-contaminated fixture (a verbatim bench solution and an 8-gram-sharing paraphrase) that
MUST be caught, and a clean fixture that must pass - the mutation directions are pinned.

## Staked gates

- **P-B1 (supply):** >= 3,000 screen-clean, execution-verified training samples across both
  feeds. Below that, Phase C is precondition-blocked (not enough data to train honestly), and
  the block is the published result.
- **P-B2 (committee feed quality):** >= 500 committee problems whose tests (a) pass their own
  reference solution and (b) FAIL a null candidate and a mutated reference. A test set that
  cannot reject wrong code is decoration; problems failing this are dropped and counted.
- **KR-B1 (the law):** the screen runs on every batch, counts published, an unscreened batch
  voids the run. Inherited, restated, non-negotiable.
- **KR-B2 (generator honesty):** if > 30% of committee problems fail P-B2's test-quality check,
  the generation prompt is broken - block and fix, do not cherry-pick survivors into the
  corpus without stating the drop rate.
- **KR-B3 (state):** generation is GPU work under the shared lock discipline; screen and
  imports are CPU work and may run beside nothing measuring.

## Cost and order

$0. B1: corpus license check + download + schema + re-execution sample (CPU + disk). B2:
screen implementation + tests. B3: full corpus re-execution + screening (CPU, hours). B4:
committee generation (GPU, one evening). Verdict appended here; Phase C stakes separately.

---

## VERDICT (B4 arm 1, scored 2026-08-06): both supply gates short, KR-B2 fired as forecast - remedies staked and running

| gate | staked | measured |
|---|---|---|
| P-B1 >= 3,000 verified samples | 2,343 (feed1 2,142 + feed2 201) | **SHORT - continuation slice running** |
| P-B2 >= 500 committee problems | 201 | **SHORT - behind KR-B2's fix** |
| KR-B2 drop rate <= 30% | **59.8% -> BLOCKED** | the gate did its job |
| KR-B1 screen on every batch | ran, logged, counts published | held |

**The finding worth the block: test-writer overconfidence is model-size-independent.** Among
test sets that reached the reference check, the 4B asserted wrong expected values **54.7%**
of the time (n=4,866) and the 30B - 7.5x larger, the best coder on this box - **55.7%**
(n=492). Writing correct literal expectations WITHOUT executing is a capability neither model
has; the null/mutation gates (2.4% and 1.0% leak rates) prove the tests are otherwise
substantive. Scale does not fix hallucinated arithmetic - verification does.

**Remedies, staked before launch and now running (B4b, outputs tagged _b):**
1. **P-B1:** continuation over shuffle-slice [5000:7500] of the screen-clean corpus - disjoint
   by construction, same feed1 protocol, seed unchanged. Expected +~1,070.
2. **KR-B2/P-B2:** feed2 v2 - ONE repair pass per failing test set: the model re-derives each
   expected value by stepping through its own solution. Targets the measured pathology
   directly; no second repair, no bar-lowering, repaired-count published. v2 is a new
   protocol version and its samples are tagged as such.

Screen note upheld: 12 of 500 committee problems (2.4%) echoed protected-bench idioms and
were excluded - generated text is not exempt from the law.
