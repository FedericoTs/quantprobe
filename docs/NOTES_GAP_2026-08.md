# Product gap notes - 2026-08

Working notes for prioritization. Every claim cites a file or an issue; where the evidence is
one datapoint, it says so. Written blunt on purpose: the science track is ahead of the product
track, and the two users who actually showed up prove it.

## 1. Who uses this today vs who could

### Today: n=2 external users, and both are off the validated path

- **pittsat (issue #1)** - Windows 11, AMD RX 5700 XT, Vulkan llama.cpp, 32 GB RAM. A
  hobbyist tinkerer ("cleaning off everything LLM to start fresh after messing with ROCM
  hacks", his comment on issue #1). He ran the full loop - hw, calibrate, plan, bench
  --contribute - and delivered the first external datapoint (+0.1% vs prediction, E-13 in
  FINDINGS.md). Getting there he hit: GPU undetected (had to pass --vram/--vram-bw by hand),
  `autospec skipped (No module named 'gguf')` silently degrading the read-your-file feature,
  and the split-GGUF spec bug that titled his contribution `total=None active=None` (fixed
  v1.26.3 per the issue #1 thread).
- **fboudra (issue #2)** - Linux, AMD RX 9070 XT. Professional-grade: when calibrate printed
  "GPU: none detected (nvidia-smi absent/empty)" he read the source, diagnosed it himself
  ("OS support is limited to Windows, mainly support for Nvidia cards"), and offered a patch
  ("I've got a patch if you're interested by a PR"). The PR is accepted in words (issue #2
  thread) and not yet landed.

The signal is loud for n=2: **100% of field contact is AMD, 50% is Linux**, and the tool's
validated core is one NVIDIA/Windows 2016 desktop (docs/HARDWARE.md). detect.py now covers
AMD via the Windows driver registry (issue #1 fix, detect.py:211) but the Linux path and
calibrate.py are still nvidia-smi-only (calibrate.py:73, detect.py:59). The people who show
up are exactly the people the code serves worst.

### Who could, in rough order of reachable size

- **Ollama installed base** - `audit-ollama` already targets them (cli.py:81, README
  "Audits a running Ollama install") and they are the largest, least expert population.
- **Mac / Apple-silicon local LLM users** - plausibly the biggest single demographic in
  local inference; README's own limits section says those presets are "extrapolated, not
  measured", and detect.py:176 tags Apple bandwidth "unvalidated: bench me".
- **Hardware buyers** - plan/target/optimize answer "what does the next euro buy"
  (docs/HARDWARE.md), a question no vendor tool answers honestly.
- **Fine-tuners with unpublished models** - the stated audience for --custom (README, Fast
  vs Custom), currently gated by probe cost (gap 7).
- **Claim-checkers** - the DGX Spark / Kimi / airllm retrodictions (README, "Check any
  speed claim") serve forum arguers and reviewers; more a distribution channel than a user
  base, but it is the repo's most shareable material.

## 2. Top 10 gaps, ranked by user-value x feasibility

Ranked product-first: what moves a real user soonest per unit of work.

1. **Linux/AMD parity in detect + calibrate.** Evidence: issue #2 verbatim; both external
   users are AMD; calibrate.py:73 and detect.py:59 are nvidia-smi-only; the Windows-registry
   AMD path (detect.py:32) does not exist on Linux. A contributor patch is already offered
   and verbally accepted (issue #2 thread). Effort: **S** to land the PR, **M** to add
   rocm-smi/sysfs/vulkaninfo detection plus a Linux CI leg so it stays fixed.
2. **No single business-readable report artifact.** Evidence: cli.py's 14 subcommands
   include no report/export; plan's output as pasted in issue #1 is ~40 dense lines with
   register IDs (#16/#52, C-14, D-10) that mean nothing outside FINDINGS.md, and the paste
   arrived line-mangled - terminal text is the only artifact users can share. A one-page
   `plan --report` (md/html: verdict, tok/s band, binding constraint, the run command, plain
   words) serves users and doubles as the marketing surface. Effort: **M**, 2-3 days.
3. **No uncertainty bands on predictions in the output.** Evidence: the honesty exists in
   prose - README states the +/-25% band, the one-sided >=0.90x floor, and "we do not have a
   point prediction for this placement... 0.32-0.56" - but plan prints a bare "36.8 tok/s"
   (issue #1 paste) with the band buried in a caveat paragraph. The repo's brand IS
   calibrated honesty; the CLI undersells it. Effort: **S** - the numbers are already
   computed and documented, print "29-46" or "floor 33" next to the point.
4. **No Windows installer / true one-liner.** Evidence: README ships its own failure note
   ("'quantprobe' is not recognized? pip put it in a folder that isn't on your PATH");
   issue #1 shows `gguf` as a soft dependency silently skipping autospec on the first
   external run; weight-touching commands require a separately obtained llama.cpp located
   via --llama-dir/env/PATH (README, Commands). Effort: **M** - make gguf a hard dep,
   pipx/uv install docs, and either fetch a llama.cpp release binary or fail loudly with
   the exact download line.
5. **bench --contribute -> atlas is a manual, lossy pipe.** Evidence: contribution is a
   hand-pasted GitHub issue (README, Contributing: "you review and submit"); issue #1's
   payload arrived broken (`total=None`) and needed two owner follow-up questions; the
   leaderboard/atlas automation is still pending (docs/ROADMAP.md Track B item 1). Every
   strategic unknown - GPU eta gap, Mac and 50-series presets, the Gemma-4-26B 0.77x row -
   is explicitly queued on this pipe (README, "When quantprobe won't help you"). Effort:
   **M** - GitHub issue-form template plus a CI parser that turns issues into
   HARDWARE_TABLE rows.
6. **Docs assume an expert reader.** Evidence: the quickstart's first output block contains
   an `-ot "blk\.(16|17|...|47)\.ffn_.*_exps\.=CPU"` regex (README); preregistration
   epistemics arrive in paragraph two, before any "type this" line; QUICKSTART.md exists
   but the front door optimizes for reviewers, not the Ollama-class user who would grow the
   atlas. Effort: **S-M** - a 90-second path (install, `quantprobe auto`, done) above the
   fold, epistemics one click deeper. No science changes.
7. **Probe cost on big models.** Evidence: README, Fast vs Custom - "~50 min for a 7B,
   ~10 h for a 35B"; the 27B probe had to run from a Q4 source because BF16 was infeasible
   on the reference box (docs/QUANT_QUALITY.md, section 4). The staked cheap replacement
   exists (U-46 effective-rank probe, docs/ROADMAP.md item 15) but is research-gated and
   may die by its own kill rule. Effort: **L** and uncertain; the shippable mitigation is
   gap 10 (grow the atlas so users skip the probe).
8. **No bring-your-own tasks.** Evidence: weights/business_tasks.py is a fixed 52-predicate
   suite; README's limits section concedes "You need task-level eval scores (MMLU,
   HellaSwag) - quantprobe measures... its own 40-task business suite". The question every
   business user actually has is "does the cheap quant survive MY workload", and there is
   no hook for it. The predicate/self-test discipline to copy already exists in
   business_tasks.py. Effort: **M-L** - a task-spec format plus runner.
9. **Recipes atlas has 5 entries.** Evidence: quantprobe/recipes/ contains mistral-7b,
   qwen2.5-7b, qwen3-30b, qwen3.5-35b, qwen3.8-27b; README's limits section says "four
   families so far". No Llama, no Gemma recipe ships despite Gemma-12B results in the
   README - so the "skip the probe" escape hatch (cli.py:98, --recipe) covers almost
   nobody's model. Effort: **M per family** locally (bounded by gap 7), or community-cheap
   once gap 5 exists.
10. **Mac / Apple silicon unmeasured.** Evidence: README, limits - "You're on a Mac or a
    50-series card. Those presets are extrapolated, not measured"; detect.py:176. Largest
    could-be audience, lowest feasibility today: needs hardware access or contributed rows,
    i.e. it is downstream of gaps 5 and 1. Effort: code **M** (Metal detection mirrors the
    AMD work), validation **blocked** on contributions.

Not on the list but noted: the linear-attention KV over-estimate on hybrids (U-51 in
FINDINGS.md) is a correctness gap in plan's memory math on the newest model family; it is
already queued as v1.28 work and is invisible to today's users at short context.

## 3. Three things this repo does better than any alternative

1. **Adversarial self-honesty as a product surface.** Every headline number is staked
   before measurement and the misses print at the same size as the hits - the -67%
   disk-tier miss sits on the README hero chart; QUANT_QUALITY.md section 2 documents the
   authors refuting their OWN favorable first reading with a pre-staked re-grade (+24.0 ->
   +23.8). No quant picker, leaderboard, or VRAM calculator does anything like this, and it
   is the reason a stranger (pittsat) ran the full loop and contributed.
2. **Capability-grade quality evidence for aggressive quants.** NAIVE vs recipe on full
   MATH-500/GSM8K/IFEval: 57.0 -> 81.0 on the 35B; a 2.6 -> 50.2 rescue on the 4B; and the
   honest size-dependence law that 2-bit on a 4B stays mediocre even rescued, "use Q4"
   (docs/QUANT_QUALITY.md sections 1-3). Community quants ship perplexity at best; nobody
   else publishes where their own method loses 30.8 points.
3. **Diagnosis, not just a number.** plan names the binding constraint and prices the
   levers - "faster RAM / XMP / more channels: NO effect (0% of the token)" in pittsat's
   own paste (issue #1); README: "3 tok/s, disk-bound means buy RAM; 3 tok/s,
   bandwidth-bound means don't bother". Plus retrodiction of other people's claims (DGX
   Spark's 5.5-7x impossible ceiling, the Kimi 200-320x unit inversion, airllm's 30x
   spread - README, "Check any speed claim"). Everything else in this space answers
   "does it fit"; nothing else answers "why, and what would fixing it buy".

## 4. The one gap to fix next

**Gap 1: Linux/AMD parity - land fboudra's PR, add rocm-smi/sysfs detection to calibrate,
stand up the Linux CI leg, and put the 9070 XT calibrate output into the atlas as the first
AMD/Linux row (exactly what the issue #2 thread already asks him for).**

Why this one and not the report artifact: it is the only gap where all three lines cross -
100% of observed demand (both issues are AMD), a volunteer holding finished work (issue #2:
"I've got a patch"), and the strategic flywheel (docs/ROADMAP.md Track B runs on contributed
rows; the eta gap and Mac presets are "explicitly waiting on this" per README - and today
the calibrate path those contributions need only works on NVIDIA/Windows). Cost is near
zero: review a PR someone else wrote. Leaving it open costs a contributor - solo projects
get very few people who read your source, diagnose your gap, and offer the fix; making that
person wait is the most expensive thing on this list.

The report artifact (gap 2) is the second fix, not the first: a beautiful one-pager about a
tool half your visitors cannot calibrate is marketing before product. Do it immediately
after - it is what turns the next pittsat's mangled paste into a link.

Blunt bottom line: the science is validated on one 2016 desktop and the field arriving at
the door is AMD, Linux, and (statistically, next) Mac. Durability for real users is platform
parity plus a shareable artifact plus bands in the output; everything else on the list can
queue behind those without losing anyone who has actually shown up.
