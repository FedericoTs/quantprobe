# Reading notes — PrimeIntellect prime-agent (2026-08, repo study)

Source: github.com/PrimeIntellect-ai/prime-agent (MIT, harness only - no weights, no
datasets). A "self-improving RLM agent": context as variables, tools and recursive subagents
as function calls inside a persistent IPython REPL ("everything is programmatic"), plus a
"continual harness" - durable state holding prompts, memories, skills, and subagent specs,
refined through "small, evidence-backed updates." The ARC-AGI-3 95.5% claim from their
announcement is NOT in the repo - no methodology published yet, so it stays un-audited (the
E-15 restraint applies: we price claims when arithmetic exists, not tweets).

## What transfers to quantprobe now

1. **Persistent executor pool (direct P0b input).** Their persistent-REPL principle names our
   measured pain: P0's selection cost was 16 SEQUENTIAL subprocess spawns per task (~0.5-1s
   each on Windows) - most of the 3.2x wall-clock penalty that failed P2. A persistent
   sandbox worker pool + test-as-lanes-finish is the P0b engineering core, now validated by
   an independent architecture.
2. **The runner module we already wrote five times.** Their "continual harness" formalizes
   what p0_lanes/gridbench/autotune/phaseb each re-implement by hand: lock family,
   probe-first, checkpoint ledgers, GPU-state logging, cite-or-refuse outputs. One
   `weights/runner.py` refactor turns our discipline from copy-paste into policy - their
   "small, evidence-backed updates" is exactly how our protocol amendments already work,
   minus the reuse.
3. **The verifiers interface.** Their separate `verifiers` repo is becoming the de-facto
   RL-environment format. Phase C's eventual RL arm (rewards from OUR test execution) should
   adopt that interface rather than invent one - and our grid benches packaged as verifier
   environments is both interop and distribution (their users become our validators).

## The bigger thesis this supports (recorded, not yet a project)

Prime Agent optimizes a harness for CAPABILITY. A tool for professional researchers must
optimize for CREDIBILITY - and that is the product we have been hand-building all week
without naming it: **the pre-registered research harness.** Components already proven here:
- the stake as a first-class object (gates + kill rules BEFORE measurement);
- provenance enforced by machinery (locks, C-14 states, solo-provenance, screens);
- cite-or-refuse outputs (charts that cannot render unledgered numbers);
- evolution ledgers logged by default -> every experiment mints its own trend chart;
- verdicts published misses-first, with the media asset generated the day they land.
Prime Agent's architecture shows the execution substrate such a tool would sit on
(persistent programmatic env, budgets, subagents); ours shows the integrity layer no one
else ships. The combination - "a harness where an agent CANNOT fool itself or you" - is the
ultra-tool thesis, and this week's pipeline is its working prototype at n=1.

## What does not transfer

Their subscription/API-provider assumption (we are local-first), the RLM recursion depth
(our campaigns are deterministic scripts by design - auditability beats flexibility for
measurement), and any capability claims until methodology ships.
