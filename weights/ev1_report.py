"""Read the EV-1 rows off disk and decide WHICH number is publishable for each task.

This module exists because of a specific, repeated failure: three times now, a benchmark has
reported a score that was a property of its SCORER rather than of the model.

    MATH-500 (hendrycks)  extractor slices "first $ to last $" and never reads \\boxed{}.
                          0.6B scored 0.00% while 89.4% of its answers carried a boxed value.
    AIME (zero-shot)      the prompt asks for no answer format, so the 4B wrote "Answer: 49"
                          in 30 of 30 items and scored 0 - while the 0.6B boxed out of habit.
    GSM8K cot_zeroshot    strict-match requires the literal sentence "The answer is N.", which
                          the zero-shot prompt never requests. 0 of 3,957 responses across
                          three unrelated models matched it. Flexible-extract: 36.8/81.7/79.9.

The tell is the same every time and it is cheap to detect: a metric that is EXACTLY zero for
every model in the suite. Real capability differences do not line up on 0.0000 across a 0.6B
and a 30B - that is a format mismatch, not a difficulty wall. So the rule here is mechanical:
a metric that is uniformly zero across >= MIN_MODELS_FOR_ZERO_RULE models is never allowed to
become a headline number. It must be either explained (annotated as an artifact, with the
mechanism) or fixed. Silence is not an option the code offers.

This is deliberately a REFUSAL, not a correction. The guard cannot know whether a uniform zero
is an artifact or a genuine wall, and guessing would be the same error in the other direction -
so it stops and demands a human verdict recorded in ARTIFACTS.
"""
from __future__ import annotations
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

MIN_MODELS_FOR_ZERO_RULE = 3     # two models can coincide; three unrelated ones do not

# The metric that IS the score for each task, and why that one. Written down because the
# choice is a protocol decision, not a formatting detail - picking the friendlier of two
# filters after seeing the numbers would be exactly the thumb on the scale we forbid.
REPORTED = {
    "math500_boxed":      ("exact_match,none", "our extractor: last \\boxed{} + lm-eval is_equiv"),
    "aime24_boxed":       ("exact_match,none", "same extractor as MATH-500, same protocol"),
    "aime25_boxed":       ("exact_match,none", "same extractor as MATH-500, same protocol"),
    "ifeval":             ("prompt_level_strict_acc,none", "strict is the standard headline for IFEval"),
    "gpqa_main_zeroshot": ("acc,none", "multiple choice - no extraction step to get wrong"),
    "gsm8k_cot_zeroshot": ("exact_match,flexible-extract", "strict-match is a scorer artifact - see ARTIFACTS"),
}

# Uniform zeros we have DIAGNOSED. An entry here is a claim that the mechanism is understood;
# it is what lets the run proceed. Anything not listed stops the report.
ARTIFACTS = {
    ("gsm8k_cot_zeroshot", "exact_match,strict-match"):
        "Requires the literal 'The answer is N.' - a sentence gsm8k-cot-zeroshot's own "
        "doc_to_text ('Q: {question}\\nA: Let's think step by step.') never asks for. The "
        "pattern is inherited from the few-shot variant, where the exemplars demonstrated it. "
        "Measured: 0 of 3,957 responses across 0.6B/4B/7B match. Report flexible-extract.",
}


class SuspectMetric(RuntimeError):
    """A metric is uniformly zero and undiagnosed. Publishing it would be a false claim."""


BOXED_TASKS = {"math500_boxed", "aime24_boxed", "aime25_boxed"}


def rescore_boxed(model, task, root=None):
    """Re-grade a boxed row from its logged samples with the CURRENT extractor.

    Necessary because scoring happened at run time, and the extractor has since been fixed: it
    took the last \\boxed outright, so a model that looped into the token cap had its correct
    answer discarded along with the truncated fragment after it (see math500_utils). Rows
    already on disk carry the old verdict, and the row running tonight is being scored by the
    old code held in its own process - re-grading offline is the only way every row gets the
    same treatment.

    Nothing is overwritten. lm-eval's results_*.json stays the raw record; this returns the
    corrected numbers beside it so the delta is always visible and auditable.

    Returns {"exact_match": .., "emitted_boxed": .., "n": .., "rescued": ..} or None if the
    samples were not logged.
    """
    root = root or os.path.join(DATA, "ev1")
    files = sorted(glob.glob(os.path.join(root, model, task, "**", "samples_*.jsonl"),
                             recursive=True))
    if not files:
        return None
    sys.path.insert(0, os.path.join(HERE, "lm_eval_tasks"))
    try:
        import math500_utils as M
    except ImportError:
        return None
    n = ok = boxed = rescued = lost = 0
    seen = set()
    with open(files[-1], encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("doc_id") in seen:
                continue                      # one line per (doc, filter); grade each doc once
            seen.add(d.get("doc_id"))
            resp = d.get("resps") or [[""]]
            txt = resp[0][0] if isinstance(resp[0], list) else str(resp[0])
            r = M.process_results(d["doc"], [txt])
            n += 1
            ok += r["exact_match"]
            boxed += r["emitted_boxed"]
            was = int(d.get("exact_match") or 0)
            rescued += (r["exact_match"] == 1 and was == 0)
            lost += (r["exact_match"] == 0 and was == 1)
    if not n:
        return None
    return {"exact_match": ok / n, "emitted_boxed": boxed / n,
            "n": n, "rescued": rescued, "lost": lost}


def load_rows(root=None, rescore=True):
    """{(model, task): {metric: value}} for every results_*.json on disk.

    With rescore=True (the default) boxed tasks are re-graded from their logged samples with
    the current extractor, and the row carries `rescued`/`lost` counts so a reader can see
    exactly what the correction did. rescore=False returns the untouched lm-eval verdicts.
    """
    root = root or os.path.join(DATA, "ev1")
    rows = {}
    for p in glob.glob(os.path.join(root, "*", "*", "**", "results_*.json"), recursive=True):
        parts = p.replace("\\", "/").split("/")
        model, task = parts[-4], parts[-3]
        try:
            blob = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for _, metrics in blob.get("results", {}).items():
            rows[(model, task)] = {k: v for k, v in metrics.items()
                                   if isinstance(v, (int, float)) and "stderr" not in k}
    if rescore:
        for (model, task), metrics in list(rows.items()):
            if task not in BOXED_TASKS:
                continue
            fixed = rescore_boxed(model, task, root)
            if not fixed:
                metrics["_rescore_unavailable"] = 1    # samples missing: say so, do not guess
                continue
            metrics["exact_match,none"] = fixed["exact_match"]
            metrics["emitted_boxed,none"] = fixed["emitted_boxed"]
            metrics["_rescued"] = fixed["rescued"]
            metrics["_lost"] = fixed["lost"]
    return rows


SYSTEM_NEEDLE = "final answer inside"     # distinctive fragment of ev1_run.SYSTEM_INSTRUCTION


def verify_prompts(root=None):
    """Check what the models were ACTUALLY SENT, not what the command was supposed to say.

    The boxed-answer system instruction must reach every boxed row and no other row. IFEval
    grades literal instruction compliance - "respond in all lowercase", "no commas", "wrap the
    reply in quotes" - so a standing instruction to box the answer is a competing instruction
    and would depress the score for a reason that has nothing to do with the model.

    tests/smoke.py already asserts that build_cmd routes the flag correctly. That is INTENT.
    This is DELIVERY, and the two can diverge: the scoping fix landed mid-campaign, so a row
    run before it would carry the instruction no matter what the current code says. Reading it
    back out of the logged prompts is the only way to know, and it is the same discipline that
    caught the AIME budget disparity - the code as it stands now says nothing about the code a
    row was run under.

    Returns [] when every row is correct, or a list of violations.
    """
    root = root or os.path.join(DATA, "ev1")
    bad = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "**", "samples_*.jsonl"),
                                 recursive=True)):
        parts = path.replace("\\", "/").split("/")
        model, task = parts[-4], parts[-3]
        n = hit = 0
        seen = set()
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if d.get("doc_id") in seen:
                        continue
                    seen.add(d.get("doc_id"))
                    n += 1
                    if SYSTEM_NEEDLE in json.dumps(d.get("arguments", "")):
                        hit += 1
        except OSError:
            continue
        if not n:
            continue
        if task in BOXED_TASKS and hit != n:
            bad.append(f"{model}/{task}: boxed task, but only {hit} of {n} prompts asked for "
                       f"the answer format")
        if task not in BOXED_TASKS and hit:
            bad.append(f"{model}/{task}: NOT a boxed task, yet {hit} of {n} prompts carry the "
                       f"boxed-answer instruction - a competing instruction on this benchmark")
    return bad


def uniform_zeros(rows, min_models=MIN_MODELS_FOR_ZERO_RULE):
    """[(task, metric, n_models)] for metrics that are exactly 0.0 on every model measured.

    Exactly 0.0 - not "close to zero". A genuinely hard benchmark produces small nonzero
    scores and ragged ones; a format mismatch produces the same clean zero everywhere, because
    the extractor never fires at all. That sharpness is what makes the rule usable.
    """
    by_metric = {}
    for (model, task), metrics in rows.items():
        for k, v in metrics.items():
            # A leading underscore marks PROVENANCE, not a score. The rule is structural rather
            # than a list of exceptions, because a guard whose blocklist grows every time it
            # fires stops being a guard. It fired on `lost` the moment re-scoring was added -
            # correctly by its own logic, and wrongly in substance: "0 answers lost on every
            # model" is the correction working, not a scorer that never fires.
            if k in ("sample_len", "alias") or k.startswith("_"):
                continue
            by_metric.setdefault((task, k), []).append(v)
    out = []
    for (task, k), vals in sorted(by_metric.items()):
        if len(vals) >= min_models and all(v == 0.0 for v in vals):
            out.append((task, k, len(vals)))
    return out


def check_publishable(rows):
    """Raise unless every uniform zero in `rows` has a recorded diagnosis. Returns the list."""
    found = uniform_zeros(rows)
    undiagnosed = [(t, m, n) for t, m, n in found if (t, m) not in ARTIFACTS]
    if undiagnosed:
        lines = "\n".join(f"  {t}/{m}: exactly 0.0 on all {n} models" for t, m, n in undiagnosed)
        raise SuspectMetric(
            "A metric scored exactly 0.0 on every model measured. Across unrelated model sizes "
            "that is the signature of a scorer/format mismatch, not a capability wall - it has "
            "been three-for-three so far.\n" + lines +
            "\nDiagnose the mechanism and record it in ev1_report.ARTIFACTS, or fix the task. "
            "Do not publish it as a score.")
    return found


def table(rows):
    """[(model, task, metric, value, why)] using the single reported metric per task."""
    out = []
    for (model, task), metrics in sorted(rows.items()):
        pick = REPORTED.get(task)
        if not pick:
            continue
        metric, why = pick
        if metric in metrics:
            out.append((model, task, metric, metrics[metric], why))
    return out


if __name__ == "__main__":
    rows = load_rows()
    flagged = check_publishable(rows)
    for t, m, n in flagged:
        print(f"ARTIFACT (diagnosed) {t}/{m}: 0.0 on all {n} models")
    print()
    for model, task, metric, value, _ in table(rows):
        print(f"{model:>5} | {task:<22} | {metric:<32} | {100 * value:6.1f}%")
    print(f"\n{len(rows)} rows on disk")
