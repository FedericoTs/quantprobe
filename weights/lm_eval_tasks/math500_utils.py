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

BOXED = chr(92) + "boxed"        # built from the ordinal: a literal backslash in this project
                                 # has twice been eaten as an escape in transit


def last_well_formed_boxed(response):
    """The last \\boxed{...} whose braces BALANCE, scanning backwards from the end.

    lm-eval's last_boxed_only_string takes the last \\boxed outright and returns None if it is
    unbalanced. That is correct for a response that ends normally, and wrong for one that hits
    the token cap - which is exactly what the 4B does on AIME: it reaches the right answer, then
    repeats "\\boxed{116}" 683 times until generation is cut off mid-token. The final fragment
    is unbalanced, the extractor returns None, and a CORRECT answer scores zero.

    Measured over every banked boxed row (10 rows, 1,180 items): taking the last well-formed box
    rescues 9 answers and loses 0, and changes nothing at all on 8 of the 10 rows. A scorer
    change that only ever helps would be suspect; one that fires only where the defect occurs
    is a defect fix. The two rows it moves are the two where a model looped into the cap.

    This is C-25 pointed at our own scorer: the difference between "the model was wrong" and
    "we failed to read the answer" is worth more than either number.
    """
    i = response.rfind(BOXED)
    while i != -1:
        j = response.find("{", i)
        if j != -1:
            depth = 0
            for k in range(j, len(response)):
                if response[k] == "{":
                    depth += 1
                elif response[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return response[i:k + 1]
        i = response.rfind(BOXED, 0, i)
    return None


def _gold(doc):
    """AIME ships the answer under 'Answer', MATH-500 under 'answer'."""
    for k in doc:
        if k.lower() == "answer":
            return str(doc[k])
    raise KeyError("no answer field in doc")


def process_results(doc, results):
    response = results[0]
    boxed = last_well_formed_boxed(response)
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
