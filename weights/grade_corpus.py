"""U-43: grade the Phase B corpus for DIFFICULTY, and validate the grader before trusting it.

The screen asked "is this correct and uncontaminated". It never asked "is this worth learning
from", so a 3,626-sample corpus could be large and uniformly easy and we would not know.

CORRECTION TO THE REGISTERED PROTOCOL (2026-08-08): U-43 said difficulty could be graded from
"per-candidate outcomes already logged". It cannot - the corpus records only the SURVIVORS
(instruction/response/source/tests); no per-candidate solve rate was ever written. So
difficulty is estimated from measurable proxies instead.

What makes that honest rather than arbitrary: feed2 samples carry a DECLARED level in their
source (committee/<topic>/<level>/<n>, easy|medium|hard-but-testable). Those act as a held-out
label set - if the proxies cannot separate declared easy from declared hard, the proxies are
not measuring difficulty and no grading ships.
"""
from __future__ import annotations
import ast, json, os, re, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CORPORA = [os.path.join(HERE, "data", "phaseb_corpus.jsonl"),
           os.path.join(HERE, "data", "phaseb_corpus_b.jsonl")]
LEVELS = ("easy", "medium", "hard-but-testable")


def proxies(rec):
    """Cheap, model-free signals. Every one computable from the record alone."""
    tests = rec.get("tests") or ""
    resp = rec.get("response") or ""
    n_assert = tests.count("assert")
    try:
        tree = ast.parse(resp)
        nodes = sum(1 for _ in ast.walk(tree))
        branches = sum(1 for n in ast.walk(tree)
                       if isinstance(n, (ast.If, ast.For, ast.While, ast.Try,
                                         ast.comprehension, ast.BoolOp)))
        depth = _depth(tree)
    except SyntaxError:
        nodes = branches = depth = -1
    return dict(n_assert=n_assert, resp_lines=resp.count("\n") + 1,
                nodes=nodes, branches=branches, depth=depth,
                instr_words=len((rec.get("instruction") or "").split()))


def _depth(node, d=0):
    ch = list(ast.iter_child_nodes(node))
    return d if not ch else max(_depth(c, d + 1) for c in ch)


def declared_level(src):
    parts = (src or "").split("/")
    if len(parts) > 2 and parts[0] == "committee" and parts[2] in LEVELS:
        return parts[2]
    return None


def main():
    recs = []
    for p in CORPORA:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            r["_px"] = proxies(r)
            r["_lvl"] = declared_level(r.get("source"))
            recs.append(r)
    print(f"corpus: {len(recs)} samples")

    # --- data-quality check: the source-string parse ---
    bad = [r["source"] for r in recs
           if (r.get("source") or "").startswith("committee") and r["_lvl"] is None]
    print(f"committee rows whose level did not parse: {len(bad)}")
    for s in bad[:3]:
        print(f"    {s}")

    labelled = [r for r in recs if r["_lvl"]]
    print(f"declared-level samples (the validation set): {len(labelled)}  "
          f"{dict(Counter(r['_lvl'] for r in labelled))}")

    # --- DOES THE PROXY SEPARATE DECLARED EASY FROM DECLARED HARD? ---
    print("\n=== proxy means by DECLARED level (the validation) ===")
    keys = ("n_assert", "resp_lines", "nodes", "branches", "depth", "instr_words")
    by = defaultdict(list)
    for r in labelled:
        by[r["_lvl"]].append(r["_px"])
    print(f"{'proxy':12s}" + "".join(f"{l:>22s}" for l in LEVELS) + "   easy->hard")
    verdict = {}
    for k in keys:
        means = [sum(p[k] for p in by[l]) / max(len(by[l]), 1) if by[l] else 0 for l in LEVELS]
        ratio = means[2] / means[0] if means[0] else 0
        verdict[k] = ratio
        print(f"{k:12s}" + "".join(f"{m:22.1f}" for m in means) + f"   {ratio:.2f}x")

    best = max(verdict, key=lambda k: abs(verdict[k] - 1))
    print(f"\nstrongest separator: {best} at {verdict[best]:.2f}x easy->hard")
    if abs(verdict[best] - 1) < 0.25:
        print("VERDICT: proxies do NOT separate declared difficulty (<1.25x). "
              "No grading ships - this would be labelling noise.")
        return 1
    print("VERDICT: proxies track declared difficulty. Grading is defensible; "
          "bands below are for the UNLABELLED remainder.")

    # --- band the whole corpus on the validated signal ---
    vals = sorted(r["_px"][best] for r in recs)
    q1, q2, q3 = (vals[len(vals) // 4], vals[len(vals) // 2], vals[3 * len(vals) // 4])
    band = Counter()
    for r in recs:
        v = r["_px"][best]
        r["_band"] = ("q1" if v <= q1 else "q2" if v <= q2 else "q3" if v <= q3 else "q4")
        band[r["_band"]] += 1
    print(f"quartiles on {best}: {q1} / {q2} / {q3}   bands: {dict(band)}")

    out = os.path.join(HERE, "data", "phaseb_difficulty.json")
    json.dump({"signal": best, "quartiles": [q1, q2, q3],
               "separation_easy_to_hard": round(verdict[best], 3),
               "n": len(recs), "bands": dict(band),
               "proxy_ratios": {k: round(v, 3) for k, v in verdict.items()},
               "by_source": {r["source"]: r["_band"] for r in recs}},
              open(out, "w", encoding="utf-8"), indent=1)
    print(f"written -> {os.path.relpath(out, os.path.dirname(HERE))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
