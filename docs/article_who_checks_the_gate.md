# Who checks the gate?

**Graph engineering tells you to put a verifier at the end. Nobody tells you the verifier
will lie to you. Here is how to catch it — with receipts from a week where ours did.**

---

There's a good piece going around by [@0xShoopy](https://x.com/0xshoopy) on *graph
engineering*: stop giving one agent five jobs. Split the work into specialists. Wire them
with deterministic edges instead of hopeful prompts. Keep state in a file. And put a
**gate** at the end — the one node allowed to say *done* or *not done*.

I agree with all of it. We've been running that architecture on a real codebase across 105
pre-registered experiments, and I'd add nothing to the design advice.

This is about what happens next. Because we built the gates, then we started measuring
the gates, and the results were not comfortable.

The short version: **a gate you have never tried to break is not a gate. It's a green
light.**

Six receipts follow. All from one working week. Every number is in a public repo.

---

## 1. The test that passed a hardcoded answer

We shipped a feature called `report` — a one-page artifact you can forward to someone who
will never run the tool. Its headline is a predicted speed. The single load-bearing
property: **the report's number must equal the planner's number.** Two commands in one
tool disagreeing about the same file on the same machine is worse than no feature.

So we gated it. A test runs the planner, runs the report, asserts the numbers match.

Green. Green for hours.

Then an adversarial agent — one whose only instruction was *try to refute this* —
hardcoded the report's headline to `99.9 tok/s` and ran the test again.

**It passed.**

The assertion searched the whole file for the planner's number, and a different line
elsewhere in the report happened to contain it. The gate was checking that a string
existed somewhere, not that two numbers were equal. It would have passed a report that
ignored its inputs completely.

The fix took ten minutes: anchor the assertion to the two lines a human actually reads,
and run the same check on a second machine preset where the right answer is a different
number, so no single memorized artifact can satisfy both. Then mutate it four ways —
hardcode the verdict, hardcode the rows, delete a mandatory block, disable a guard. All
four now fail.

There is a postscript, and it belongs here rather than in a footnote. When I went back to
cite this story I found that the *weak* version of the assertion had been fixed in my
working tree before its first commit — so the escape existed only in a transcript. A true
story with no artifact anyone else can run is precisely the thing this article argues
against, and I had written it as receipt number one. The fix is a committed test that
reconstructs the escape in milliseconds: a document whose verdict line is wrong while
another line still carries the right number, asserting that whole-document search *passes*
it and the line-anchored form *fails* it. Run it yourself; that is the difference between a
receipt and an anecdote, and I had to be caught to produce one.

**Rule 1: mutation-test the gate.** Break the thing on purpose. If the gate still says
pass, you learned something enormous for the price of one deliberate bug. If you have
never watched your gate fail, you do not know that it can.

---

## 2. Write the gate before the data exists

Every experiment we run gets a scorer — the code that turns raw measurements into a
verdict — and that scorer is **committed to the repository before the first row of data
exists.**

This sounds like ceremony. It isn't. A gate written after you've seen the results is
fitted to the results. Not because you're dishonest; because you're human and the
threshold that "obviously" makes sense is the one that vindicates the number you're
looking at. You'll pick `>= 0.65` instead of `>= 0.70` and have a reason ready.

Precommitment makes that impossible. Our scorers hard-refuse to run when data is missing
or incomplete, print the staked thresholds verbatim from the pre-registration, and are
themselves mutation-tested: change a `0.70` to `0.10` in a copy, and a synthetic dataset
must flip its verdict, proving the constant is live in the decision path rather than
decorative.

**Rule 2: the gate is part of the experiment, not the write-up.** If it ships after the
data, it isn't a gate — it's a rationalization with a test runner.

---

## 3. The verifier's job is to refute, not to approve

In our workflows, every builder agent is followed by a verifier agent, and the verifier's
prompt does not say *check this work*. It says: **try to REFUTE the following, with
evidence.** It returns a schema-forced verdict — a boolean and a list of findings — so it
cannot hedge its way to "looks good."

The difference is not cosmetic. "Review this" produces agreement; models are agreeable.
"Refute this" produces the hardcoded-99.9 discovery above.

The strongest single upgrade: make verifiers **independently re-derive** the thing they're
checking rather than read it. On one experiment, the builder wrote a statistical estimator
and hand-derived a test fixture in exact fractions. The verifier hand-derived the same
fixture *from the specification alone* and compared. They matched — which is worth
something, because the two derivations never saw each other.

**Rule 3: adversarial by construction.** A verifier that shares the builder's assumptions
is a rubber stamp with extra latency.

---

## 4. A gate that cannot say "I don't know" will lie to you

Last night a run finished and the scorer refused to produce a verdict.

The measurement was a variance attribution over 280 benchmark runs, deciding which
configuration flag carries the most influence on speed. Two flags came back at 0.914 and
0.869 — a statistical dead heat. The precommitted rule said the top factor must hold rank
one in at least 950 of 1000 bootstrap resamples. It held in **569**.

So the scorer printed UNDECIDED, named the pre-declared remedy (extend the same seeded
design and measure more), and issued no verdict.

That refusal is the most valuable output of the night. A binary gate — pass or fail — would
have been forced to pick, and it would have picked the number that was 0.045 higher, and
we would have published a finding that a coin flip could have reversed.

**Rule 4: give the gate a third answer.** Pass, fail, and *not enough evidence*. Systems
that can only pass or fail will manufacture confidence they don't have, and they'll do it
most often in exactly the close cases where being wrong matters.

---

## 5. Your gate has an environment axis, and it is invisible from your machine

This one cost a broken release yesterday.

We merged an external contribution — a good patch, five tests included, full suite green on
my machine. I tagged it, uploaded it, announced it.

CI went red eighty-four seconds later.

The patch had refactored hardware detection and left one branch unguarded: on a machine
with **no GPU at all**, the code fell through every arm and returned hardware with missing
fields. Downstream, that became `None`, and `None` became the exact bug a user had filed
two weeks earlier.

My local suite could not have caught it. This machine has a GPU, so the broken branch is
unreachable here. CI runs on GPU-less runners, and it caught the bug on four tests — after
the tag and the upload, because I'd run them in the same sequence as the push.

Two fixes, and the second is the one that matters:

1. The regression test now stubs the hardware probes empty **in-process**, so the no-GPU
   path is exercised on GPU machines too. It was verified against the actual broken
   release file rather than a hand-written imitation of the bug: it fails there, passes
   now.
2. **CI green became a release gate, not a formality that runs after the upload.** The
   hotfix was pushed, watched to green across all five checks, and only then tagged and
   published.

**Rule 5: ask what your gate structurally cannot see.** Every gate runs in an environment,
and the environment silently defines the set of bugs it is capable of catching. Test the
empty case, the absent hardware, the missing file — in-process, so the path runs everywhere.

---

## 6. Decide in advance what result would make you change the product

The strongest gate isn't in CI. It's the one where you write down, before measuring, what
outcome would force you to change something you've already shipped — and then you honor it
when it fires.

Ours fired this week, against our own headline feature.

The tool prints a line telling you which resource limits your machine — the thing users
quote most. It's derived from a physical model. Ten days ago we staked a test: measure
which configuration flags actually drive the variance in speed, and check whether the top
one maps to the resource our model names. We wrote down the mapping in advance, and we
wrote down the consequence of failure: **if it fails, the shipped line gets a scope label
the same day, at full prominence.**

It failed. The top variance carrier was decisive — rank one in 1000 of 1000 resamples —
and it was not in the mapped set. The honest diagnosis is that we posed the test badly:
our factor ranges spanned configurations, so a configuration-level lever won by
construction.

The kill rule doesn't care that the mistake was ours. Same day, that line now prints:

> *validation: derived from the law, not confirmed by variance attribution*

— in the tool's output, in the README, and on the chart assets. It stays until a
better-posed re-derivation earns its removal.

**Rule 6: a kill rule you wrote yourself, and honored against yourself, is the only
verification anyone else has reason to trust.** Everything else is marketing with tests
attached.

---

## The two problems nobody solves with a framework

The original article names them honestly, and I want to reinforce rather than answer them.

**Comprehension debt** — the gap between what's in your repo and what you understand — grows
every time the graph ships code you didn't read. There's no tooling fix. But there is a
measurable proxy I'd offer: **can every claim in your project be regenerated from committed
data by committed code?** If a number can only be reproduced by rerunning an agent, you
don't own that number; you're renting it. Our charts refuse to render if the log they cite
is missing. That refusal is a comprehension-debt alarm.

**Cognitive surrender** — using the graph to avoid thinking rather than to think faster — has
one tell I've found reliable: *when the system produces a result you like, do you check it
as hard as one you don't?* We now mutation-test the tests that produce good news, because
those are the ones nobody re-reads.

---

## The map

Graph engineering gets you the architecture. This is the layer above it:

1. **Mutation-test the gate.** Never seen it fail = don't know that it can.
2. **Commit the gate before the data.** After the data, it's fitted.
3. **Verifiers refute, not approve.** Re-derive independently; schema-force the verdict.
4. **Give the gate a third answer.** Pass, fail, not-enough-evidence.
5. **Ask what the gate cannot see.** Environment defines its blind spots. Test empty.
6. **Write kill rules and honor them against yourself.** Publish the misses at the same
   size as the hits.

None of this makes agents smarter. It makes their output *checkable*, which is the only
property that compounds. A graph that ships fast and can't tell you when it's wrong is a
faster way to be wrong.

The gate is not the end of the graph. **The gate is the part of the graph most likely to
be quietly broken**, because it's the only node whose failure looks exactly like success.

---

*Every number here comes from a public repository with pre-registered experiments and
published misses: the hardcoded-99.9 escape, the 569/1000 refusal, the broken release, and
the kill rule that labeled our own headline feature — all in the commit history, at the
same size as the wins. Tooling: [quantprobe](https://github.com/FedericoTs/quantprobe).*
