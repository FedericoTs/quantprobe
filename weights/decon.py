"""Phase B decontamination screen (prereg 2026-08-05-phase-b-data-engine.md, program law
2026-08-05-benchmark-sanctity.md).

  python weights/decon.py --selftest     # pinned mutation-direction fixtures, no network

Every training sample passes here or it does not train. Two checks, both against ALL
protected-bench prompts AND canonical solutions (MBPP+ and HumanEval+, hash-pinned):
  1. exact match: sha256 of the normalized full text;
  2. sliding 8-gram overlap on word tokens (case-folded, whitespace-collapsed) - deliberately
     strict; the failure direction is losing training data, never contaminating eval.
The screen is a kill rule: a batch whose log cannot show it ran is void.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys

NGRAM = 8


def _tokens(text):
    return re.findall(r"\w+", text.lower())


def _norm_hash(text):
    return hashlib.sha256(" ".join(_tokens(text)).encode()).hexdigest()


def _grams(toks, n=NGRAM):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def load_protected():
    """(exact_hashes, gram_set, meta) over MBPP+ and HumanEval+ prompts + canonical solutions."""
    from evalplus.data import (get_mbpp_plus, get_mbpp_plus_hash,
                               get_human_eval_plus, get_human_eval_plus_hash)
    hashes, grams = set(), set()
    n_texts = 0
    for bench in (get_mbpp_plus(), get_human_eval_plus()):
        for t in bench.values():
            for text in (t["prompt"], t["prompt"] + t["canonical_solution"],
                         t["canonical_solution"]):
                toks = _tokens(text)
                hashes.add(_norm_hash(text))
                grams |= _grams(toks)
                n_texts += 1
    meta = dict(mbpp_hash=get_mbpp_plus_hash(), humaneval_hash=get_human_eval_plus_hash(),
                n_texts=n_texts, n_grams=len(grams), ngram=NGRAM)
    return hashes, grams, meta


def screen_one(text, hashes, grams):
    """(clean, reason). Reason names the first violation for the exclusion ledger."""
    toks = _tokens(text)
    if _norm_hash(text) in hashes:
        return False, "exact-normalized-match"
    for i in range(len(toks) - NGRAM + 1):
        if tuple(toks[i:i + NGRAM]) in grams:
            return False, f"8gram@{i}:{' '.join(toks[i:i + NGRAM])[:60]}"
    return True, ""


def screen_batch(samples, hashes, grams):
    """samples: [str] -> (kept_indices, ledger). The ledger is what gets PUBLISHED."""
    kept, excluded = [], []
    for i, s in enumerate(samples):
        ok, why = screen_one(s, hashes, grams)
        (kept.append(i) if ok else excluded.append((i, why)))
    return kept, dict(total=len(samples), kept=len(kept), excluded=len(excluded),
                      reasons=excluded[:50])


def selftest():
    """Pinned mutation directions: a verbatim bench solution MUST flag, an 8-gram-sharing
    paraphrase MUST flag, a clean synthetic sample MUST pass. If any direction flips, the
    screen is broken and Phase B may not proceed (KR-B1)."""
    from evalplus.data import get_mbpp_plus
    hashes, grams, meta = load_protected()
    print(f"protected: {meta['n_texts']} texts, {meta['n_grams']:,} 8-grams "
          f"(mbpp {meta['mbpp_hash'][:8]}, he {meta['humaneval_hash'][:8]})")
    t = next(iter(get_mbpp_plus().values()))
    verbatim = t["prompt"] + t["canonical_solution"]
    ok, why = screen_one(verbatim, hashes, grams)
    assert not ok, "MUTATION FAIL: verbatim bench text passed the screen"
    toks = _tokens(t["canonical_solution"])
    if len(toks) >= NGRAM:
        stolen = " ".join(toks[:NGRAM])
        para = f"Here is a helpful training example about lists. {stolen} and then we return."
        ok2, why2 = screen_one(para, hashes, grams)
        assert not ok2, "MUTATION FAIL: 8-gram-sharing paraphrase passed the screen"
    clean = ("Write a function that converts a temperature series from celsius to kelvin "
             "while clamping sensor glitches below absolute zero and logging their indices.")
    ok3, why3 = screen_one(clean, hashes, grams)
    assert ok3, f"MUTATION FAIL: clean synthetic sample flagged: {why3}"
    kept, ledger = screen_batch([verbatim, para, clean], hashes, grams)
    assert kept == [2] and ledger["excluded"] == 2, f"batch ledger wrong: {ledger}"
    print(f"selftest OK: verbatim flagged ({why.split(':')[0]}), paraphrase flagged, "
          f"clean passed; ledger {ledger['kept']}/{ledger['total']} kept")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else 0)
