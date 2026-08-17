> # KILLED 2026-08-18, before a line of it was written
>
> This design does not ship. An adversarial verifier was pointed at the business thesis
> *before* implementation and refuted it with named prior art: **PIT** (2010) already ships
> our RETURN-HARDCODE, BRANCH-KILL, COMPARATOR-FLIP and THRESHOLD-PERTURB operators under
> other names; **mutatest** ships the Python branch mutation; **cosmic-ray** and **mutmut**
> both accept an arbitrary gate command. And the AI-specific pitch - "your eval would pass a
> model that always answers the same thing" - is already a shipped flag in
> **lm-evaluation-harness** (a dummy model returning a constant) and a published result
> (arXiv:2410.07137, ICLR 2025: null models score 86.5% LC on AlpacaEval 2.0).
>
> Two things survive and are worth keeping, as a feature rather than a product: **SOURCE-SEVER**
> (does a claim regenerate from the artifact it cites - the C-31 class, which no mutation
> tool packages) and the **plausible-constant** refinement (PIT returns degenerate values like
> 0 and null; a degenerate return mostly exercises your error handler, while a *plausible*
> one exercises the check).
>
> The verifier also refuted two of the case studies on their own terms: RETURN-HARDCODE could
> not have produced our 99.9 escape (that was an assignment substitution, and the function
> returns a list), and BRANCH-KILL could not have produced the unload tristate collapse (that
> was a changed return tuple, no branch involved). Both corrections are recorded in E-29.
>
> Kept in the repository unchanged below, because a design killed on evidence is worth more
> to the next person than a design quietly deleted.

# Design: `gateprobe` v0 - does your gate know how to fail?

Status: design, pre-implementation. Owner: gateprobe (new product). Target: v0.1.
Sibling product: quantprobe. Shared discipline, disjoint problem.

quantprobe answers *can this machine run this model*. gateprobe answers a question one
level up, and it is the question this repo keeps paying for: **can this test suite
actually fail?**

The thesis is stated in full in docs/article_who_checks_the_gate.md and it is not a
metaphor. In an AI system the gate - the eval suite, the verifier, the scorer, the CI
check - is the node most likely to be quietly broken, because **its failure looks exactly
like success**. A model that breaks gets loud. A gate that breaks gets green.

Everything below is grounded in four failures from this repository, all in the commit
history, all with dates.

---

## 1. The pitch, and the exact failure it detects

> **gateprobe breaks your code on purpose and reports every break your gate did not
> notice.**

The failure it detects is not "you have too few tests". It is narrower and much more
common:

> A gate that runs, exits 0, and would exit 0 even if the thing it checks were replaced
> by a constant.

That gate is not measuring anything. It is a green light with a runtime cost. The only
way to know which of your gates are in this state is to break the code underneath them
and watch.

### 1.1 The evidence table - four real cases from this repo

| # | What was mutated | What the gate said | How long it stayed green | Receipt |
|---|---|---|---|---|
| 1 | `report`'s verdict headline hardcoded to `99.9 tok/s` | **PASS** | hours, same day (2026-08-16) | tests/smoke.py:4275-4298 (docstring records the escape); commit `dc23dd9` |
| 2 | Scorer bar `CONC_SHARE = 0.70` -> `0.10` in a copy | **FAIL** - the gate held, the constant is live | n/a: this is the by-hand drill, run on every precommitted scorer since prereg #95 | weights/prereg95_score.py:122 (declared), :420 (consumed), :584 (`--self-check`); commit `31dd79c` |
| 3 | `unload()`'s AMD fallback collapsed a tristate: both failure arms `return None`, leaving the "ollama is squatting" arm unreachable | **PASS** - full suite green through merge AND release | merged `0cfdab7` 2026-08-17, shipped in v1.28.1, fixed `fdc1a4e` the same day; the sibling no-GPU fall-through reached a **tagged and uploaded release** and CI caught it 84 s after the push (`0ac2fb4`) | quantprobe/ollama.py:199-233; regression test tests/smoke.py:3859 |
| 4 | Nothing was mutated - the cited source was never read by any gate | **no gate existed** | **14 days**, published 2026-08-04 (`18fe736`), corrected 2026-08-18 (`24bc6cf`) | findings/REGISTER.json C-31 |

Read row 1 and row 4 together, because they are the same disease at two speeds.

Row 1: the parity test for `quantprobe report` asserted that the planner's tok/s appeared
*somewhere* in the report file. An adversarial pass hardcoded the headline to `99.9` and
the test passed, because a different line elsewhere in the report happened to carry the
real number. The gate was checking string presence, not numeric equality. It would have
passed a report that ignored its inputs completely. Fixed by anchoring to the two lines a
decision-maker actually reads (`PREDICTED decode speed, one user:` and the `RECOMMENDED`
placements row) and by running the same check on two machine presets whose right answers
differ, so no single memorized artifact satisfies both.

Row 4: the published headline "Qwen3-30B-A3B, 22.69 tok/s" was the **first request of a
server session** - a 16-token reply, quoted from llama.cpp's `eval time` line, where the
print is dominated by per-request overhead and is not a decode rate at all. The log it
cited carried 1,231 per-request decode rates with a median of 20.79. Nobody, including
us, had ever re-derived the number from the source it named. It shipped in the README for
two weeks. C-31's own generalisation: *any metric extracted by grepping a log for a
pattern finds the FIRST match, and the first match is systematically the least
representative request in the session.*

Row 1 was caught because someone was explicitly instructed to refute. Row 4 was caught
because someone asked a direct question and went back to the log instead of to memory.
Neither is a process. gateprobe is the process.

---

## 2. CLI surface

Small on purpose. Three verbs.

```
gateprobe run     --target <path> --gate "<command>"
                  [--ops OP[,OP...]] [--n N] [--seed S] [--timeout SEC]
                  [--sources GLOB] [--include-tests] [--workdir DIR]
                  [--json out.json] [--adjudicate FILE]

gateprobe ops     [--json]                 # list operators and what each simulates

gateprobe restore --workdir DIR            # replay a backup after a crash, verify hashes
```

`--gate` is **any command whose exit code means pass/fail**. That is the whole contract,
and it is what makes the tool portable across a pytest suite, a precommitted scorer, an
eval harness, a Makefile target, or three lines of shell:

```
--gate "pytest -q tests/"
--gate "python weights/prereg95_score.py --self-check"
--gate "python tests/smoke.py"
--gate "lm_eval --model hf --tasks gsm8k --limit 20 && python check_scores.py"
```

### 2.1 Defaults, and why

| Flag | Default | Why |
|---|---|---|
| `--ops` | all five, in table order | The five exist because each maps to a real failure class (section 3). Opting out is a choice you should have to type. |
| `--n` | `20` | Enough to be informative on a first run, small enough that the run is minutes not hours. Sites are sampled deterministically from `--seed`. |
| `--seed` | `0` | A published score that cannot be re-run is a C-31. The seed fixes the mutation set exactly. |
| `--timeout` | `max(60, 3 x baseline_wall_seconds)` | Calibrated from the baseline gateprobe just measured, and **printed**. A fixed timeout calibrated on one regime silently mis-scores another - that is commit `38d14f8` in this repo, where a 3 t/s floor tuned on quantized models failed a BF16 run that was merely slow. |
| `--sources` | files under `--target` that the baseline gate opened | SOURCE-SEVER needs a source list; inferring it is better than making the user enumerate it, and the inferred list is printed so it can be corrected. |
| `--include-tests` | off | Mutating a test and watching the suite stay green is a valid and different question. Mixing it into the headline number makes the number mean two things. |
| `--workdir` | `.gateprobe/<utc-timestamp>/` | Holds the manifest and the backups. It is the crash-recovery key (section 4). |

### 2.2 Exit codes

| Code | Meaning |
|---|---|
| `0` | Ran to completion, **every** eligible mutation killed. |
| `1` | Ran to completion, **at least one escape**. The escape list is on stdout and in `--json`. |
| `2` | **Refused to run.** Dirty git tree under `--target`, `--target` outside the repo, the baseline gate did not exit 0, no mutation sites found, or another runner holds the box. |
| `3` | **Restore verification failed.** At least one file did not hash back to its manifest entry. The tree may be dirty. Prints the workdir path and stops. |

`1` and `2` are deliberately different. CI wants escapes to be red, but a refusal must
never be readable as an escape - "we couldn't measure" and "your gate is weak" are
different sentences, and a tool that conflates them is committing the exact sin it exists
to detect (article Rule 4: give the gate a third answer).

### 2.3 What it prints

```
gateprobe v0.1  target: quantprobe/  gate: "python tests/smoke.py"
  baseline: gate exited 0 in 41.2 s  ->  timeout set to 124 s
  sites: 341 eligible, sampling 20 (seed 0, ops: all)

  [ 1/20] return-hardcode   plan.py:812   evaluate()          KILLED   (exit 1, 43.1 s)
  [ 2/20] comparator-flip   plan.py:1104  >= -> >             ESCAPED  (exit 0, 41.8 s)
  ...

  ESCAPES (3) - read these before the score
  ---------------------------------------------------------------
  #2  comparator-flip  quantprobe/plan.py:1104     mutant sha256 3f9a...
      - if free_mib >= need_free_mib:
      + if free_mib >  need_free_mib:
      note: comparator-flip has the highest equivalent-mutation rate of the five ops.

  #7  return-hardcode  quantprobe/report.py:288    mutant sha256 c410...
      ...

  mutation score: 15/18 = 0.83   (3 escaped, 1 timeout, 1 not-applied; 20 of 341 sites)
  a score is a sensitivity measure, not a correctness measure - see the escape list.
  restored: 341 files verified against manifest.
```

Two properties of that output are contractual, not cosmetic:

- **The escape list prints before the score.** There is no `--score-only` flag and there
  will not be one. A bare number that nobody re-derives from its escapes is the C-31
  failure mode, rebuilt inside the tool that exists to prevent it.
- **`restored: N files verified` is always the last line.** Its absence is a bug report.

---

## 3. Mutation operators

This is the heart of the product, and the reason it is not a repackaged mutation-testing
library. Classic mutation testing inherits its operator set from 1970s compiler research:
arithmetic operator replacement, statement deletion, constant increment. Those operators
were designed to model *programmer typos*.

Programmer typos are not the failure mode. The failure mode in an AI system is a
component that **stops computing and starts asserting** - a scorer that returns a
constant, a threshold that got refactored out of the decision path, a branch that a merge
made unreachable, a claim that no longer regenerates from its source. Every operator
below is justified by one of those, with a receipt.

### 3.1 RETURN-HARDCODE

**Mechanically:** pick a function under `--target` with at least one `return`; replace its
body with `return <plausible constant>`. The constant is inferred from the function's own
return sites and annotations - a float returner gets a float inside the range of the
literals already present, a bool gets `True`, a str gets the most frequent literal in the
file, a dict gets the same keys with plausible values. Deterministic given `(file,
qualname, seed)`.

**The single most important implementation rule in this document:** the constant must be
**plausible**. Returning `None` or `-1` usually crashes the gate, the gate exits non-zero,
and the mutation is scored as killed - which proves nothing, because it was killed by the
crash, not by any check of the value. Our escape happened precisely because `99.9` was
plausible enough to look like a real tok/s figure. An implausible constant is a mutation
that measures your traceback handler.

**What it simulates:** the eval that passes a model which always answers `0.9`. The scorer
whose headline is a property of the scorer rather than of any model. The renderer that
stopped reading its inputs.

**Which case study:** row 1, the `99.9` escape, directly. Also C-25, where three unrelated
models scored **exactly** `0.0000` on gsm8k_cot_zeroshot because the strict-match filter
demanded a sentence the prompt never asked for - 0 of 3,957 responses matched, while
flexible extraction on the identical responses gave 36.8 / 81.7 / 79.9%. A metric that is
a constant across a 0.6B and a 30B is a return-hardcode that happened by accident.

**Known limit:** a function whose return value the gate genuinely does not depend on will
survive, correctly. That is an equivalent mutation and the escape list says so.

### 3.2 THRESHOLD-PERTURB

**Mechanically:** find numeric literals used in a boolean context, and module-level
`UPPERCASE` numeric constants. Emit three mutants per site: `x * 0.1`, `x * 10`, and a
boundary nudge `x +/- eps` to probe the tie case.

**What it simulates:** a pass/fail bar that is decorative rather than live. A precommitted
threshold that a refactor moved out of the comparison. A config value that is read,
logged, and never consumed.

**Which case study:** row 2 - this is the drill this repo already runs by hand on every
precommitted scorer. `CONC_SHARE = 0.70` is declared at weights/prereg95_score.py:122 and
consumed at :420 (`"pass": top3 >= CONC_SHARE * total`); change it to `0.10` in a copy and
a synthetic dataset must flip its verdict. Precommitment (article Rule 2) is worthless if
the committed constant turns out to be ornamental.

**Known limit, and it is a sharp one:** a *killed* THRESHOLD-PERTURB proves the constant is
live. It does **not** prove the bar is meaningful. The same scorer carries an explicit
guard at :410-416 because `top3 >= 0.70 * total` passes trivially when `total == 0` -
degenerate data with no signal at all would have satisfied a fully live bar. gateprobe
measures whether the constant is in the decision path. Whether the decision is worth
making is a human's job.

### 3.3 COMPARATOR-FLIP

**Mechanically:** `>=` <-> `>`, `<=` <-> `<`, `==` <-> `!=`. Boolean-operator flips
(`and` <-> `or`) are opt-in via `--ops comparator-flip:strict` because they generate
equivalents at a much higher rate.

**What it simulates:** off-by-one at the bar; the exact tie; the boundary. This matters
disproportionately because the boundary is where the close cases live, and the close cases
are where being wrong is most expensive (article Rule 4 - the 569/1000 bootstrap refusal
exists because a binary gate would have picked the number that was 0.045 higher).

**Which case study:** it is the operator with the most live sites in our own tree.
weights/prereg95_score.py:425 decides P-4 on `sigs["ub"] > med` - strict versus non-strict
changes the verdict on an exact tie. quantprobe/ollama.py:213 gates a whole benchmark on
`(total - used) >= need_free_mib`, the difference between "enough VRAM" and "just barely
not", and a contaminated comparison there produced `-ngl 99` at 4.56 tok/s against a clean
18.83 - confidently, and backwards.

**Known limit:** highest equivalent-mutation rate of the five. gateprobe prints that prior
next to every COMPARATOR-FLIP escape rather than letting the reader discover it.

### 3.4 BRANCH-KILL

**Mechanically:** pick an `if` / `elif` / `except` / `match` arm and make it unreachable by
forcing the condition (`if cond:` -> `if False and cond:`). The body is preserved so the
file still parses and static analysis still sees it. Deletion is deliberately not used:
deleting a branch changes control flow in ways that crash, and a crash is a kill that
teaches nothing. We want the arm to go silently dead - which is what actually happens in
production.

**What it simulates:** a merge or refactor that collapses a multi-way answer into a single
one. This is the highest-value operator for agent systems specifically, because multi-way
outcomes (pass / fail / cannot-verify) are where those systems encode their honesty, and a
collapsed tristate is invisible in every green suite.

**Which case study:** row 3, exactly. `unload()` returns a tristate the caller branches on -
`True` (VRAM verified free), `False` (read the VRAM fine, ollama is still holding it),
`None` (no tool could read VRAM at all). The AMD `rocm-smi` fallback in PR #4 made both
failure paths end at `return None`, which left the `False` arm unreachable: a working
`nvidia-smi` plus a squatting ollama told the user their GPU was **unreadable**, sending
them to fix the wrong thing. The whole suite was green through merge and through the
v1.28.1 release.

Note the environment axis (article Rule 5): that collapse was structurally invisible on
this box, because reproducing it needed a specific hardware state. BRANCH-KILL does not
wait for hardware - it forces the arm. The sibling bug `0ac2fb4` (a GPU-less box falling
through every arm and returning hardware with missing fields) reached a **tagged and
uploaded release** for the same reason: the local machine has a GPU, so the broken branch
is unreachable here.

**Known limit:** a genuinely dead arm always survives. gateprobe labels that survivor
`DEAD-ARM-CANDIDATE` rather than calling it a gate weakness. Dead code found by accident
is still found.

### 3.5 SOURCE-SEVER

**Mechanically:** for each inferred source file - a data file, log, fixture, CSV, or
checkpoint the baseline gate opened - emit three mutants: **missing** (renamed aside),
**empty**, and **truncated to its first line**. Code is untouched. This is the only
operator that works on non-Python projects in v0.

**What it simulates:** the question this repo treats as its comprehension-debt alarm -
*does your claim regenerate from its cited source, or is it a memory?* If severing the
source leaves everything green, then no committed code derives that claim, and the number
is rented rather than owned.

**Which case study:** row 4, C-31, and nothing else comes close. The 22.69 tok/s headline
named `bt_server.log` as its source. Delete that log and every gate in the project stays
green, because nothing ever read it to produce the number. Fourteen days in the README. By
contrast our chart renderers refuse to render when the log they cite is missing - that
refusal is a SOURCE-SEVER kill, and it is the behaviour the operator is looking for
everywhere else.

**Known limit, stated because it is the operator's whole boundary:** SOURCE-SEVER proves a
derivation path **exists**. It cannot prove the derivation is **right**. C-26 is the
counterexample from inside this repo: the MATH-500 extractor did read its source, did
fire, and returned a plausible wrong number - it took the *last* `\boxed{}` in a response,
which was an unbalanced fragment after a degenerate repetition loop hit the token cap, so
answers that were exactly the gold value scored zero (4B AIME24 33.3% -> 50.0% after the
fix, 9 items rescued of 1,180, 0 lost). A truncation-severing operator finds nothing there.
Catching that class needs per-item invariants, and it is out of v0 scope (section 6).

### 3.6 Operator-to-case-study matrix

| Operator | Case 1 (99.9) | Case 2 (0.70 bar) | Case 3 (tristate) | Case 4 (C-31) |
|---|---|---|---|---|
| RETURN-HARDCODE | **catches** | - | - | - |
| THRESHOLD-PERTURB | - | **catches** | - | - |
| COMPARATOR-FLIP | - | partial (bar direction) | partial (the `>=` at ollama.py:213) | - |
| BRANCH-KILL | - | - | **catches** | - |
| SOURCE-SEVER | - | - | - | **catches** |

Each of the five earns its place by being the only operator in its column.

---

## 4. Safety model

Non-negotiable. gateprobe deliberately corrupts a working tree; if it ever loses one byte
of someone's code, the product is over on that day. This section is the contract.

**The guarantee, in one sentence:** gateprobe hashes every file under `--target` into a
manifest and backs up every file it will touch **before the first mutation**, applies
exactly one mutation at a time, restores from that backup and re-verifies the sha256 after
every single run, and re-verifies the entire tree against the manifest before exiting -
and on any hash mismatch it stops immediately with exit 3 rather than continuing.

This is the copy -> mutate -> run -> restore -> verify-sha256 loop we already run by hand
in this repo (the mutation passes recorded at tests/smoke.py:4295, the hash pinning at
weights/prereg95_score.py where the CSV header and the data file are both sha256'd into
the verdict JSON). gateprobe is that loop with the human removed from the restore step,
which is the step humans skip.

### 4.1 Pre-flight refusals (all exit 2, all before anything is written)

1. **Git must be clean under `--target`.** `git status --porcelain -- <target>` must be
   empty. This is not politeness. It is the thing that makes the recovery story in 4.5
   true: if `git checkout -- <target>` is a complete recovery, we can survive a kill -9,
   and it is only a complete recovery when there was no uncommitted work to lose.
2. **Never mutate outside `--target`.** Every candidate site is filtered by
   `os.path.realpath(site).startswith(os.path.realpath(target) + os.sep)` **after** symlink
   resolution. A symlink whose realpath leaves the target is skipped and listed in the
   output as skipped, never silently.
3. **`--target` may not be a repository root** (detected by a `.git` directory) unless
   `--allow-repo-root` is passed. Passing the repo root is almost always a mistake, and the
   blast radius of that mistake is the whole project.
4. **No other runner may own the box.** Inside this repo, gateprobe checks the shared lock
   set in weights/runner.py:43 (`LOCK_NAMES`) and refuses if any lock exists. It forks
   processes that can saturate a machine, and a benchmark's numbers are not recoverable.
   It then takes its own `.gateprobe_lock`, released in a `finally`, per the same module's
   discipline.
5. **The baseline gate must exit 0 on the untouched tree.** A red gate cannot kill
   anything, and a mutation score computed against one is meaningless - every mutant would
   read as "killed". The baseline run also produces the wall-clock the timeout is
   calibrated from and the source list SOURCE-SEVER uses.
6. **At least one eligible site must exist**, otherwise the run is a vacuous 0/0. A skip is
   not a pass; this suite has enforced that since v1.12 (tests/smoke.py:24-33) and
   gateprobe inherits it.

### 4.2 Manifest and backup

```
<workdir>/manifest.json     {relpath: {sha256, size, mode}} for the whole target tree
<workdir>/orig/<relpath>    byte copy of every file any planned mutation will touch
<workdir>/run.jsonl         one line per mutation, appended before the mutant is written
```

The manifest is written to disk **before the first mutation**, because it is the recovery
key and a recovery key that only exists in memory is not a recovery key. Every backup copy
is sha256-verified against the manifest entry immediately after it is written; a backup
that was never hash-checked is a hope, not a backup.

### 4.3 The per-mutation loop

```
for each mutation (strictly serial, never two at once):
    append intent to run.jsonl and fsync          # so a crash is reconstructable
    write mutant; sha256 the mutant               # the report can prove which bytes ran
    run --gate, cwd = repo root, timeout as calibrated
       capture exit code, wall seconds, last 2000 chars of combined output
    restore from <workdir>/orig/<relpath>
    re-hash the restored file; compare to manifest
       mismatch -> ABORT THE ENTIRE RUN, exit 3, print workdir path, attempt nothing else
```

Strictly serial is a **safety invariant, not a performance oversight**. It is what bounds
the crash blast radius to exactly one file (4.5). Parallelism needs a copy-per-worker tree
and is v1 work.

### 4.4 Post-run

Re-hash the entire target tree against the manifest and print
`restored: N files verified against manifest`. Any mismatch is exit 3. This line is part of
the output contract precisely so that its absence is noticeable.

### 4.5 What happens if the tool is killed

Stated exactly, because "we handle crashes" is the kind of claim this product exists to
disbelieve.

- **SIGINT / SIGTERM:** a handler restores the single in-flight mutation, re-verifies its
  hash, prints the restore line, and exits 130 / 143. Best-effort by definition - a signal
  handler can itself be interrupted.
- **kill -9, hard crash, power loss:** **at most one file is mutated**, because mutations
  are strictly serial. The workdir survives on disk with `manifest.json`, `orig/`, and a
  `run.jsonl` whose last line names the file that was in flight. Recovery is
  `gateprobe restore --workdir <dir>`, which replays every backup and verifies every hash
  against the manifest, exiting non-zero if any file cannot be restored to its recorded
  digest.
- **Workdir also destroyed:** `git checkout -- <target>` is a complete recovery. It is
  complete *because* pre-flight refused to start on a dirty tree (4.1.1). That refusal is
  load-bearing, and it is the reason it cannot be made optional.
- **Residual risk, stated rather than hidden:** a kill -9 in the window between writing the
  mutant and its bytes reaching disk can leave a partial file. gateprobe catches this on
  the *next* invocation - it refuses to start when it finds a workdir whose manifest does
  not match the current tree, and names the file. It cannot catch it during the crash,
  because nothing can.

### 4.6 The gate command

`--gate` runs through the platform shell, because "any command whose exit code means
pass/fail" requires it. Two consequences, both stated:

- gateprobe inherits the user's shell semantics, quoting and all. It is the user's command.
- **gateprobe never constructs any part of that command from file contents.** Nothing read
  out of the target tree ever reaches the command line. The mutation data flows to disk and
  to the report, never to the shell.

---

## 5. Scoring and output

### 5.1 The buckets

| Bucket | Definition | Counted in the score? |
|---|---|---|
| KILLED | Gate exited non-zero on the mutant | yes, numerator and denominator |
| ESCAPED | Gate exited 0 on the mutant | yes, denominator only |
| TIMEOUT | Gate exceeded the calibrated timeout | **no** - reported separately, always |
| NOT-APPLIED | Mutant failed `ast.parse`, or the site vanished | **no** - listed with the reason |

```
mutation score = KILLED / (KILLED + ESCAPED)
```

TIMEOUT gets its own bucket on purpose. Counting a timeout as killed inflates the score;
counting it as an escape overstates the weakness. The honest statement is that the gate
never rendered a verdict, which is a third answer, and a tool that cannot give a third
answer will manufacture confidence in exactly the close cases (article Rule 4 - the reason
the Sobol scorer printed UNDECIDED at 569/1000 instead of picking the number that was 0.045
higher). TIMEOUT is always printed, never folded away.

### 5.2 The per-mutation table

Columns: `id, op, file:line, symbol, verdict, gate_exit, wall_s, mutant_sha256`. For every
escape, additionally: a unified diff capped at 12 lines, and the operator's equivalence
prior.

`--json` carries the same rows plus the run header: `ops`, `n`, `seed`, `site_population`,
`target_tree_sha256`, `gate` string, baseline wall-clock, timeout, gateprobe version, and
the target's commit sha. That header is what makes a published score re-runnable by a
stranger, which is the whole point.

### 5.3 The caveats, which are part of the output and not an appendix

**A low score means one of two things, and they are not the same thing:**

1. **A weak gate.** The mutation changed behaviour and nothing noticed. This is the finding.
2. **An equivalent mutation.** The mutant is semantically identical to the original. No
   test can kill it and none should. This is not a finding, and reporting it as one is how
   a measurement tool becomes a nuisance.

v0 does not automate that distinction - deciding it needs to know what the code means. It
narrows it three ways:

- **The diff is printed.** Adjudication takes seconds, not a code-archaeology session.
- **Operator priors are printed.** COMPARATOR-FLIP is flagged as the highest-equivalence
  operator. A BRANCH-KILL survivor on an arm nothing else touches is labelled
  `DEAD-ARM-CANDIDATE`, which is a real finding wearing a survivor's clothes.
- **`--adjudicate escapes.json`** lets a human mark an escape `EQUIVALENT` with a one-line
  reason. The reason is stored and **reprinted in every future report**, so an
  "it's equivalent" claim goes on the record with a name attached instead of evaporating.
  Because `--seed` fixes mutation ids, adjudications stick across runs.

**Never report a score without the escape list.** This is enforced mechanically: escapes
print before the score, `--json` always carries the full escape array, and there is no
score-only mode. A headline number that nobody re-derives from its evidence is precisely
C-31 - and shipping that inside the tool built to catch C-31 would be the funniest possible
way to lose the argument.

**A score is not comparable across targets.** Different sizes, different op mixes, and
different sampling fractions produce different numbers with the same name. Every published
score carries suite path, commit sha, ops, seed, and `n of N sites`.

**The sample is a sample.** `--n 20` against 341 sites is an estimate `[est]`. v0 prints
`20 of 341 sites` and computes no confidence interval, because a fabricated interval would
be worse than an honest fraction.

---

## 6. What v0 explicitly does not do

Scope discipline, listed so that the first person to ask for each of these can be shown
this line.

1. **No LLM anywhere in the loop.** No LLM-judged mutations, no LLM-chosen sites, no
   LLM-written explanations. Operators are deterministic AST rewrites. A seed must
   reproduce a mutation set byte-for-byte, forever, on someone else's machine; a model in
   the path destroys that, and re-runnability is the product's only real defence.
2. **No semantic mutations.** Swapping an algorithm, reordering statements, changing a loop
   bound in a way that requires understanding intent - all out. Those need a spec or a
   model. The five operators are the ones that map to observed failures.
3. **No auto-fix.** gateprobe never writes a test, never proposes an assertion, never edits
   a suite. The moment a tool suggests the assertion that would kill its own mutant, teams
   start optimizing the score instead of the gate (see section 8).
4. **Python-only for the four code operators.** SOURCE-SEVER is language-agnostic because it
   touches data, and `--gate` is language-agnostic because it is a shell command, so a
   TypeScript project gets real value from v0 with one operator. Full multi-language AST
   support is v1+.
5. **No parallelism.** One mutation at a time is a safety invariant (4.3).
6. **No coverage integration.** "Only mutate covered lines" raises the score and hides
   exactly the uncovered-arm findings - the `unload()` tristate class - that we most want.
7. **Test files are not mutated by default** (`--include-tests` is opt-in).
8. **No caching, no incremental runs.** Reusing a result against a tree that may have
   changed is a correctness risk in the one place we cannot take one.
9. **Not recommended as a required CI check yet.** The exit codes are designed for it. We
   will not recommend it until we have run it against our own suite for a month and can
   publish the false-positive rate.
10. **No score badge.** Not in v0. A number on a README with no escape list attached is the
    exact artifact this tool was built to distrust.

---

## 7. The launch artifact

**The plan:** run gateprobe against the eval and test suites of N popular open-source agent
frameworks, and publish the mutation scores with every mutation.

This is a good launch because it is the only way to show what the tool sees, and a
dangerous one because the same post is one framing decision away from being a hit piece.
The fairness rules below are what make it defensible. They are written before the first
run, for the same reason our scorers are (article Rule 2: a rule written after you have
seen the results is fitted to the results).

**N and the list are precommitted.** N = 5 for the first publication. The list is chosen
from a public popularity ranking on a stated date, with star counts, and committed to the
repo - with its sha - **before the first run**. The selection is the biggest cherry-pick
surface in the whole exercise, so it gets treated like a threshold: staked in advance.

### 7.1 Fairness rules, non-negotiable

1. **We go first, and at the same size.** quantprobe's own suite (tests/smoke.py: 188
   tests, 4,552 lines) is scored and published **before any third party is contacted**,
   with the identical ops set, `--n`, and seed, at the same prominence - same post, same
   chart, same font size. If our number is the worst on the board, it is still the first
   number in the post. This repo has done this before: `3f10059` shipped a scope label on
   our own headline feature the day our own kill rule fired against it.
2. **Notify before publishing.** Every maintainer receives the complete report - escapes,
   diffs, seed, exact command, commit sha - with a **14-day window** before publication. Any
   response is published verbatim next to the score. A maintainer's correction changes the
   number, not a footnote.
3. **Publish the mutations, not just the score.** Every mutation id, every diff, the exact
   command line, the seed, and the pinned commit sha of the target. Anyone can re-run and
   get the identical mutants. A score nobody else can reproduce is a C-31 with better
   typography.
4. **No cherry-picking, mechanically prevented.** The framework list, ops, `--n`, and seed
   are committed with a hash before the first run. **Every** framework on that list is
   published - 0.10 or 0.95 - including the ones where gateprobe failed to run at all,
   which publish as `NOT-RUN` with the reason. We do not run twelve and publish five.
5. **An escape is not an accusation.** The framing is "here is what a gate can miss", not
   "this project is broken". Every escape carries the equivalence caveat. Any escape we
   could not adjudicate publishes as `UNADJUDICATED`, never as a hole. No security
   language, no CVE framing, no "vulnerable".
6. **No score without its scope.** Every published number names the suite path, commit sha,
   ops, sampling fraction (`n of N sites`), and environment. A number without those is a
   number nobody can own.
7. **We publish our own misses inside the launch.** If a framework we expected to score
   badly comes back clean, that goes in. If a maintainer demonstrates that an operator
   produces junk mutants on their codebase, that operator gets a documented limitation
   **the same day, at full prominence** - the same kill-rule discipline the binding-constraint
   line got.
8. **Standing right of re-run.** We re-run and update any published score, free, on any
   commit a maintainer names, indefinitely.

### 7.2 The honest headline

The post is not "framework X has a bad test suite". It is:

> We broke five projects' code on purpose, including our own, and counted what their gates
> did not notice. Here is every mutation, every escape, and the seed to reproduce all of it.

---

## 8. The biggest risk

**The mutation score measures the gate's sensitivity, not its correctness - and it is
trivially gameable in the direction that looks exactly like improvement.**

A team that wants 1.00 can get there by adding assertions that pin whatever the code
currently returns. Every mutant dies. The score goes green. What they have built is a
change-detector that locks in current behaviour **including its bugs**, and that is worse
than the weak gate it replaced, because it now fights every correct fix.

C-26 is the proof from inside this repo. Our MATH-500 extractor read its source, fired
correctly by every structural measure, and returned a plausible wrong number - 33.3% where
the truth was 50.0%, because it took the last `\boxed{}` rather than the last *well-formed*
one. A sensitivity-maximising suite would have pinned 33.3% as ground truth and then
failed the fix. gateprobe would have scored that gate highly, and gateprobe would have been
right about sensitivity and useless about truth.

So the property gateprobe measures - does the gate move when the code moves - is
**necessary and not sufficient**, and the launch has to say that louder than it says the
score. Mitigations, all already in the design: escapes print before the score; the
"sensitivity, not correctness" line ships in every run's output and every published table;
there is no score-only mode and no badge; and section 6 forbids the auto-fix feature that
would turn score-gaming into one keystroke.

Secondary risks, ranked: (2) equivalent-mutation false positives on unfamiliar codebases
make the launch look sloppy - mitigated by adjudication with published reasons and by
`UNADJUDICATED` as a first-class outcome; (3) reputational blowback from rule 5 being read
as an accusation anyway - mitigated by going first, at the same size, with our own number.
