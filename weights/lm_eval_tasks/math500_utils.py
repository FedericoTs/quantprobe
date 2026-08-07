"""Scoring for math500_boxed - lm-eval's OWN extractor and comparator, wired correctly.

Every function that decides right-or-wrong is imported from lm_eval, unmodified. This file
contributes no equivalence logic of its own; it only reads the boxed answer out of the
response instead of slicing "first $ to last $" of the whole solution, which is the single
defect that makes the stock hendrycks_math500 report 0.00% for any model that reasons in
LaTeX. See math500_boxed.yaml for the full rationale and the measurements behind it.

`emitted_boxed` is reported alongside accuracy on purpose: it separates "the model was wrong"
from "the model never produced a gradeable answer". Those two failure modes were indistinguish-
able in the stock task, and telling them apart is what diagnosed this whole class of bug.
"""
from lm_eval.tasks.hendrycks_math.utils import (  # noqa: F401
    is_equiv,
    last_boxed_only_string,
    remove_boxed,
)


def _gold(doc):
    """AIME ships the answer under 'Answer', MATH-500 under 'answer'."""
    for k in doc:
        if k.lower() == "answer":
            return str(doc[k])
    raise KeyError("no answer field in doc")


def process_results(doc, results):
    response = results[0]
    boxed = last_boxed_only_string(response)
    candidate = None
    if boxed is not None:
        try:
            candidate = remove_boxed(boxed)
        except Exception:
            candidate = None
    if candidate is None:
        return {"exact_match": 0, "emitted_boxed": 0}
    try:
        correct = int(bool(is_equiv(candidate, _gold(doc))))
    except Exception:
        correct = 0
    return {"exact_match": correct, "emitted_boxed": 1}
