#!/usr/bin/env python3
"""Render and VALIDATE the findings register.

Why this exists. By 2026-07-27 this project had 18 pre-registrations, four laws, six dead ends and
a growing pile of things measured once and half-remembered. Nothing held them together, so the same
class of mistake kept recurring: a result would be measured, published in LAWS.md, and go on
contradicting the shipped code for days (the sub-4-bit gate), or a figure measured at the edge of a
resource would be quoted as a property of a configuration (the ubatch cliff).

The fix is not another document to keep in sync by hand - that is the failure mode, not the cure.
`findings/REGISTER.json` is the single source of truth; FINDINGS.md is GENERATED from it; and the
checks below fail loudly when the register and the repository disagree.

    python findings.py            # validate, then regenerate FINDINGS.md
    python findings.py --check    # validate only; non-zero exit if the register has drifted
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(ROOT, "findings", "REGISTER.json")
OUT = os.path.join(ROOT, "FINDINGS.md")
PREREG = os.path.join(ROOT, "preregistrations")

SECTIONS = [
    ("laws", "Established laws", "What we believe, and the measurement that earned it."),
    ("levers", "Shipped levers", "Things the tool actually recommends, with the number attached."),
    ("dead_ends", "Measured dead ends", "Negative results. These are load-bearing: each one is a "
                                        "direction nobody has to spend a day on again."),
    ("contradictions", "Open contradictions", "Where the code, the law and the measurements do not "
                                              "agree yet. Ranked by how much damage the gap does."),
    ("untried", "Untried levers", "Staked predictions written BEFORE measuring, so a miss is "
                                  "visible. Ordered by expected value."),
    ("external", "External work to study", "Prior art that could make one of the above unnecessary."),
]

REQUIRED = ("id", "kind", "claim", "status", "confidence")


def load():
    with open(REG, encoding="utf-8") as f:
        return json.load(f)


def validate(reg):
    """Every check here failed at least once in this project's history."""
    problems, seen = [], set()

    entries = [(s, e) for s, _, _ in SECTIONS for e in reg.get(s, [])]
    for section, e in entries:
        req = ("id", "kind", "repo", "why") if e["kind"] == "external" else REQUIRED
        for k in req:
            if not e.get(k):
                problems.append(f"{e.get('id', '?')}: missing required field '{k}'")
        if e["id"] in seen:
            problems.append(f"{e['id']}: duplicate id - ids are stable and never reused")
        seen.add(e["id"])
        # An untried lever without a staked magnitude is a wish, not a hypothesis. This is the
        # whole pre-registration discipline expressed as a lint.
        if section == "untried" and not e.get("predicted_effect"):
            problems.append(f"{e['id']}: untried entry has no predicted_effect - stake it or drop it")
        # Scope is what stops a claim being applied where it was never measured. D-05 was nearly
        # generalised past the regime it was measured in.
        if e["kind"] in ("law", "lever", "dead_end") and not e.get("scope"):
            problems.append(f"{e['id']}: {e['kind']} with no scope - a claim without a scope is a guess")

    # Every scored pre-registration must appear somewhere in the register. This is the check that
    # would have caught a measured result never reaching the code.
    # NUMBERS MUST BE UNIQUE. This dict is keyed by integer, so before 2026-07-31 a second file
    # claiming a taken number silently OVERWROTE the first - and the "every staked prereg is
    # cited" check below then passed for both on one citation. Two documents both called #92
    # shipped that way (per-shape calibration and speculation x KV-quant, staked two minutes
    # apart), while a third artefact cited as "#92b" belonged to neither. A citation that
    # resolves to the wrong experiment is the same defect class as no citation at all.
    staked, dupes = {}, []
    if os.path.isdir(PREREG):
        for fn in sorted(os.listdir(PREREG)):
            if not fn.endswith(".md"):
                continue
            with open(os.path.join(PREREG, fn), encoding="utf-8") as f:
                head = f.read(4000)
            m = re.search(r"[Pp]re-registration #(\d+)", head)
            if m:
                n = int(m.group(1))
                if n in staked:
                    dupes.append(f"pre-registration #{n} is claimed by TWO documents "
                                 f"({staked[n]} and {fn}) - every citation of #{n} is ambiguous. "
                                 f"Renumber the later stake.")
                staked[n] = fn
    problems += dupes
    cited = set()
    for _, e in entries:
        for field in ("evidence", "why_it_is_promising", "what_the_data_rules_out", "magnitude"):
            cited.update(int(n) for n in re.findall(r"prereg(?:istration)? #(\d+)", str(e.get(field, ""))))
    for num, fn in sorted(staked.items()):
        if num not in cited:
            problems.append(f"pre-registration #{num} ({fn}) is not cited by any register entry")

    # `wired_into` must point at something that exists. A finding that claims to have reached the
    # code, and has not, is the exact failure layer 5 of verify.py was built for.
    for _, e in entries:
        for ref in re.findall(r"([\w/]+\.py):(\w+)", str(e.get("wired_into", ""))):
            path, sym = os.path.join(ROOT, ref[0]), ref[1]
            if not os.path.isfile(path):
                problems.append(f"{e['id']}: wired_into names {ref[0]}, which does not exist")
            elif not re.search(r"^\s*(def |class |%s\s*=)" % re.escape(sym),
                               open(path, encoding="utf-8").read(), re.M):
                problems.append(f"{e['id']}: wired_into names {ref[0]}:{sym}, which is not defined there")

    return problems


def render(reg):
    L = ["# Findings register",
         "",
         "**Generated from `findings/REGISTER.json` by `findings.py`. Do not edit this file by "
         "hand — your changes will be overwritten. Edit the register.**",
         "",
         "Everything this project has measured, refuted, or deliberately left untried, in one "
         "place. Every claim carries the scope it was measured in, because a claim applied outside "
         "its scope is a guess wearing a number.",
         "",
         f"Reference box: {reg['_schema']['reference_box']}.",
         ""]

    counts = {s: len(reg.get(s, [])) for s, _, _ in SECTIONS}
    L += ["| section | count |", "|---|---|"]
    L += [f"| {t} | {counts[s]} |" for s, t, _ in SECTIONS]
    L += [""]

    for section, title, blurb in SECTIONS:
        rows = reg.get(section, [])
        if not rows:
            continue
        L += [f"## {title}", "", blurb, ""]
        if section == "untried":
            rows = sorted(rows, key=lambda e: e.get("priority", 99))
        if section == "contradictions":
            rows = sorted(rows, key=lambda e: e.get("priority", 99))
        for e in rows:
            head = f"### {e['id']} — {e.get('repo') or e['claim']}"
            L += [head, ""]
            if e.get("repo"):
                L += [f"{e.get('why', '')}", ""]
                if e.get("question_to_answer"):
                    L += [f"**Question to answer:** {e['question_to_answer']}", ""]
            else:
                if e.get("magnitude"):
                    L += [f"**Magnitude:** {e['magnitude']}", ""]
            for label, key in (("Hypothesis", "hypothesis"),
                               ("Predicted effect (staked)", "predicted_effect"),
                               ("Why it is promising", "why_it_is_promising"),
                               ("Protocol", "protocol"),
                               ("Why this is the top item", "why_it_is_the_top_item"),
                               ("What the data rules out", "what_the_data_rules_out"),
                               ("Leading hypothesis", "leading_hypothesis"),
                               ("Also establishes", "also_establishes"),
                               ("Next action", "next_action")):
                if e.get(key):
                    L += [f"**{label}:** {e[key]}", ""]
            meta = [f"`{e['status']}`"] + ([f"`{e['confidence']}`"] if e.get("confidence") else [])
            if e.get("scope"):
                meta.append(f"scope: {e['scope']}")
            if e.get("evidence"):
                meta.append(f"evidence: {e['evidence']}")
            if e.get("cost"):
                meta.append(f"cost: {e['cost']}")
            if e.get("wired_into"):
                meta.append(f"wired into: `{e['wired_into']}`")
            L += [" · ".join(meta), ""]

    L += ["---", "",
          "## How something gets into this register", "",
          "1. **Stake it.** A pre-registration in `preregistrations/`, with numbered predictions "
          "and a kill rule, written before the measurement. An untried entry with no "
          "`predicted_effect` fails validation.",
          "2. **Measure it.** Log to `weights/data/`, `r=3`, GPU state recorded before and after.",
          "3. **Score it publicly**, hits and misses with equal prominence.",
          "4. **Wire it in** — or record explicitly that it does not ship, and why. "
          "`findings.py` checks that every `wired_into` target actually exists.",
          "5. **Check the neighbourhood.** A value that is flat next to its neighbours is a "
          "result; a value with a 45% step next to it is a coincidence. This step was added after "
          "pre-registration #23 found a shipped figure sitting one `-ub` step from a cliff.",
          ""]
    return "\n".join(L)


def main():
    reg = load()
    problems = validate(reg)
    if problems:
        print("findings register has drifted from the repository:")
        for p in problems:
            print("  -", p)
        return 1
    n = sum(len(reg.get(s, [])) for s, _, _ in SECTIONS)
    print(f"  register valid: {n} entries, all scoped, all wired_into targets exist")
    if "--check" not in sys.argv:
        with open(OUT, "w", encoding="utf-8", newline="\n") as f:
            f.write(render(reg))
        print(f"  wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
