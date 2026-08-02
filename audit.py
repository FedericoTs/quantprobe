"""quantprobe audit - does what we MEASURED actually reach what we SHIP?

This exists because of a specific failure. On 2026-07-25 we measured, and published in LAWS.md,
that the Pascal low-bit decode collapse was format-dependent rather than bit-width-dependent. The
planner went on gating decode efficiency on bit-width for another day, telling every user with a
sub-4-bit quant that their GPU was useless for it and recommending a placement 2.4x slower than
the one it rejected. Nothing was broken in the usual sense: tests were green, the release gate
passed, the finding was written down. **The finding simply never reached the code.**

No test catches that, because the code is self-consistent - it is consistent with a belief we had
already disproved. So this is a different kind of check: not "does the code do what it says", but
"does the code do what we LEARNED".

Two audits, both of which failed the night they were written:

  A. FINDINGS -> CODE.  Every scored pre-registration must declare where its result landed, in a
     line the machine can read. Naming "nothing shipped" is a valid and common answer - a refuted
     hypothesis should change no code - but it must be stated, not left implicit. An unanswered
     pre-registration is a measurement nobody acted on, which is exactly the hole above.

  B. DECISION SURFACE -> EVIDENCE.  Every placement the planner can recommend must be backed by a
     measurement, or listed as unmeasured WITH a reason. Every anchor we had was a MoE-hybrid or
     disk-stream row; not one covered "all in VRAM", the most common configuration there is - and
     that is precisely where the 9.5x error hid through a public release. A coverage hole in the
     evidence is invisible until you enumerate the surface and diff it against the anchors.

    python audit.py            # standalone, human-readable
    python verify.py           # runs this as layer 5 of the release gate
"""
from __future__ import annotations
import glob, io, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Placements the planner can emit that NOBODY has measured yet, each with the reason it is
# outstanding. Being on this list is a debt, not an exemption - it is meant to be short and to
# shrink. A placement that is neither anchored nor listed here fails the audit.
UNMEASURED_PLACEMENTS = {
    "split: N/N layers->VRAM, rest->RAM":
        "dense partial offload - never measured; needs a dense model that overflows this 6 GB card",
    "stream from disk (VRAM+RAM expert cache)":
        "needs an expert-caching runtime (ktransformers/colibri-class); stock llama.cpp cannot run it",
    "layer-by-layer streaming (one layer resident)":
        "E-11, added 2026-08-01 from the airllm placement. UNMEASURED and doubly so: no "
        "layer-streaming runtime is installed here, and the number is an explicit UPPER BOUND "
        "with two named unpriced terms - host-to-device PCIe transfer of every layer every "
        "token (no measured PCIe anchor exists and inventing one is refused), and the "
        "streaming-efficiency gap C-23 put at 1.82x on llama.cpp. It is marked runnable=False "
        "so it can never be the printed winner over a row a user can actually execute. What "
        "makes it worth shipping anyway: it is the only row that prices a placement whose VRAM "
        "cost scales with LAYER size instead of model size, and third-party airllm reports "
        "(0.5-2 tok/s RAM-resident, <0.1 disk-bound) fall inside what it predicts - "
        "encouraging, unscored, and no stake was placed before those numbers were read.",
}


def _norm(name):
    """Collapse the numeric parts of a placement label so 'split experts: 21%->VRAM' and the
    same row at 67% count as one surface, not two."""
    return re.sub(r"\d+(\.\d+)?%?", "N", name).strip()


def audit_findings_reach_code():
    """A. Every scored pre-registration declares where its finding landed, and the target exists."""
    problems, checked = [], 0
    for path in sorted(glob.glob(os.path.join(HERE, "preregistrations", "*.md"))):
        text = io.open(path, encoding="utf-8").read()
        name = os.path.basename(path)
        if not re.search(r"^#+\s*Scored\b|\*\*Status:.*SCORED", text, re.M | re.I):
            continue                      # still open: staked but not yet measured
        checked += 1
        m = re.search(r"^\*\*Wired into:\*\*\s*(.+)$", text, re.M)
        if not m:
            problems.append(f"{name}: scored but never says where the finding went. Add a "
                            f"'**Wired into:**' line naming the code path and test, or "
                            f"'**Wired into:** nothing - <why>' if it correctly changed nothing.")
            continue
        claim = m.group(1).strip()
        if re.match(r"nothing\b", claim, re.I):
            if len(claim) < len("nothing - ") + 12:
                problems.append(f"{name}: 'Wired into: nothing' needs a reason after it.")
            continue
        # verify every `path:symbol` reference actually resolves
        for ref in re.findall(r"`([\w./-]+\.(?:py|md|html|json))(?::([\w.]+))?`", claim):
            fpath, symbol = ref
            full = os.path.join(HERE, fpath)
            if not os.path.isfile(full):
                problems.append(f"{name}: claims it shipped into {fpath}, which does not exist")
                continue
            if symbol:
                body = io.open(full, encoding="utf-8", errors="replace").read()
                if symbol not in body:
                    problems.append(f"{name}: claims {fpath}:{symbol}, but '{symbol}' is not in "
                                    f"that file - the finding was reverted or renamed away")
    return checked, problems


def audit_decision_surface_is_evidenced():
    """B. Every placement the planner can recommend is anchored, or declared unmeasured."""
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, "tests"))
    from quantprobe.plan import evaluate, MACHINES
    import smoke

    # every distinct placement the planner emits across a broad sweep of the decision surface
    surface = set()
    cases = [dict(t=30.5, a=3.3, ne=1.2, moe=True), dict(t=110, a=12, ne=2.7, moe=True),
             dict(t=7.2, a=7.2, ne=7.2, moe=False), dict(t=70.6, a=70.6, ne=70.6, moe=False),
             dict(t=11.9, a=11.9, ne=11.9, moe=False)]
    for mk in ("2016-xmp", "rtx-3060", "rtx-4090", "ddr5", "mac-m3-max"):
        M = MACHINES[mk]
        hw = dict(vc=M["vc"], vb=M["vb"], rc=M["rc"], rb=M["rb"], db=M["db"],
                  geta=M["geta"], gl=M["gl"])
        for kw in cases:
            for bits in (2.0, 2.95, 4.5):
                try:
                    _, _, cfgs = evaluate(**kw, bits=bits, n_layer=48, **hw)
                except Exception:
                    continue
                surface.update(_norm(c[0]) for c in cfgs)

    covered = {_norm(a[2]) for a in smoke.MEASURED_ANCHORS}
    covered.add(_norm("all in VRAM"))                 # smoke.VRAM_GAPS, 7 measured points
    covered.add(_norm("pure CPU (GPU idle)"))         # the pure-CPU 16k pre-registration
    covered.add("split experts: N->VRAM, rest->RAM")   # pre-registration #13
    known = {_norm(k) for k in UNMEASURED_PLACEMENTS}

    # anchors are stored as row SUBSTRINGS ('hybrid', 'stream from disk (cold'), so match loosely
    def is_covered(row):
        return any(c in row or row in c for c in covered)

    problems = []
    for row in sorted(surface):
        if is_covered(row) or row in known:
            continue
        problems.append(f"placement '{row}' can be recommended to users but is neither measured "
                        f"nor listed in UNMEASURED_PLACEMENTS with a reason")
    return sorted(surface), problems


def audit_single_source_of_truth():
    """C. Is any fact expressed in two places with nothing forcing them to agree?

    This is the largest defect class in the project's history: 7 of 22 shipped defects, ~32%.
    Every one was the same disease - one fact, two expressions, no mechanism keeping them equal:
    the version in pyproject.toml vs __init__.py; the decode law in Python vs JavaScript; the
    fits-in-VRAM advisory in plan but not optimize; bench correcting file size twice; and the
    layer-count fallback hand-written at four separate call sites.

    Point-fixing each instance never worked, because the SHAPE kept reappearing. So check for the
    shape.
    """
    problems = []

    # C1. the layer-count fallback must exist in exactly one place
    reimpl = []
    for path in sorted(glob.glob(os.path.join(HERE, "quantprobe", "*.py"))):
        src = io.open(path, encoding="utf-8").read()
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if re.search(r'getattr\([^)]*["\']n_layer["\'][^)]*\)\s*or\b', line):
                reimpl.append(f"{os.path.basename(path)}:{i} re-implements the layer-count "
                              f"fallback - call plan.effective_n_layer() instead")
    problems += reimpl

    # C2. the machine/model tables exist in Python AND in the published simulator's JavaScript
    sys.path.insert(0, HERE)
    from quantprobe.plan import MACHINES
    page = os.path.join(HERE, "docs", "index.html")
    if os.path.isfile(page):
        js = io.open(page, encoding="utf-8").read()
        rows = re.findall(r'\{n:"([^"]+)",\s*vc:([\d.]+),\s*vb:([\d.]+),\s*rc:([\d.]+),'
                          r'\s*rb:([\d.]+),\s*db:([\d.]+),\s*geta:([\d.]+)(?:,\s*gl:([\d.]+))?\}', js)
        pysig = {(m["vc"], m["vb"], m["rc"], m["rb"]): (k, m) for k, m in MACHINES.items()}
        for n, vc, vb, rc, rb, db, geta, gl in rows:
            sig = (float(vc), float(vb), float(rc), float(rb))
            if sig not in pysig:
                continue                       # simulator-only entry (e.g. "Custom"); not a drift
            k, m = pysig[sig]
            for field, sval in (("db", db), ("geta", geta), ("gl", gl)):
                if not sval:
                    continue
                if abs(float(sval) - m[field]) > 1e-9:
                    problems.append(f"simulator preset '{n}' disagrees with plan.MACHINES['{k}']: "
                                    f"{field} = {sval} (JS) vs {m[field]} (Python)")

        # C3. the MODEL table is copied into the simulator too, and it HAD drifted: the page
        # carried GLM-5.2 as 744B long after v1.6.2 verified 753.3B from HF safetensors - the
        # same release whose live verification caught that GLM-4.7 is the 358B model. A stale
        # parameter count on the public calculator is a wrong prediction for everyone who uses
        # it. Matched on kvp, which is distinctive per architecture and does not drift with
        # parameter-count corrections.
        from quantprobe.plan import MODELS
        mrows = re.findall(r'\{n:"([^"]+)",\s*t:([\d.]+),\s*a:([\d.]+),\s*ne:([\d.]+),'
                           r'\s*moe:([01])(?:,\s*kvp:(\d+))?\}', js)
        bykvp = {m["kvp"]: (k, m) for k, m in MODELS.items()}
        for n, t, a, ne, moe, kvp in mrows:
            if not kvp or int(kvp) not in bykvp:
                continue                       # simulator-only entry (e.g. "Custom")
            k, m = bykvp[int(kvp)]
            for field, sval in (("t", t), ("a", a), ("ne", ne)):
                if abs(float(sval) - m[field]) > 1e-9:
                    problems.append(f"simulator model '{n}' disagrees with plan.MODELS['{k}']: "
                                    f"{field} = {sval} (JS) vs {m[field]} (Python)")
            if bool(int(moe)) != bool(m["moe"]):
                problems.append(f"simulator model '{n}' disagrees with plan.MODELS['{k}'] on moe")
    return problems


def audit_evidence_is_reachable():
    """D. Every file the register cites as EVIDENCE must actually be in the repo.

    This gate exists because it was missing and something got through it. On 2026-08-02, after
    v1.24.0 shipped, the published register cited 109 evidence paths and 38 of them were not in
    the repository - the frozen disk-tier stake, the exp94-exp98 results, the calibration
    artifacts. The register said "measured 0.476225 tok/s, evidence:
    weights/data/disktier_20260731_1857_staked.json" and a reader who went to check that got
    nothing. Layers A-C all passed: A checks `wired_into` targets, B checks placements, C checks
    duplicated constants. Nobody checked that the evidence was there.

    For a project whose entire claim is that every number was written down first and can be
    re-derived, an unverifiable citation is the worst defect class available - it looks like
    rigour and is not, the same way a test that cannot fail looks like coverage.

    Deliberately narrow: only paths INSIDE the repo are checked. External URLs, upstream
    filenames quoted from other projects, and paths under doc/ that name someone else's tree are
    skipped - we cannot vouch for those and pretending to would be the same error inverted.
    """
    reg = json.load(io.open(os.path.join(HERE, "findings", "REGISTER.json"), encoding="utf-8"))
    cited, problems = {}, []
    for key, entries in reg.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            for field in ("evidence", "wired_into"):
                text = e.get(field) or ""
                for m in re.finditer(r"[\w./-]+\.(?:json|log|md|py|sh|cmd|csv|txt)", text):
                    p = m.group(0)
                    # repo-relative only: must live under a directory we actually own
                    if p.split("/")[0] not in ("weights", "preregistrations", "findings",
                                               "tests", "quantprobe", "tools", "docs",
                                               "experiments"):
                        continue
                    # ...and must not be a path inside SOMEONE ELSE'S repo. U-34 cites
                    # "github.com/Helldez/BigMoeOnEdge docs/expert-prediction.md" - a real
                    # file, in their tree, which we cannot and should not vouch for. The
                    # first version of this gate flagged it as missing: a false positive on
                    # its very first run. A gate that cries wolf erodes exactly the trust it
                    # exists to build, so external references are skipped explicitly.
                    lead = text[max(0, m.start() - 60):m.start()]
                    if "github.com" in lead or "://" in lead or "huggingface.co" in lead:
                        continue
                    cited.setdefault(p, set()).add(e["id"])
    for p, ids in sorted(cited.items()):
        if not os.path.exists(os.path.join(HERE, p)):
            problems.append(f"{p} is cited as evidence by {', '.join(sorted(ids))} "
                            f"but is not in the repository")
    return len(cited), problems


def main():
    print("\n=== A. do measured findings reach the code? ===")
    n, pa = audit_findings_reach_code()
    print(f"  {n} scored pre-registrations checked")
    for p in pa:
        print("  MISSING  " + p)
    if not pa:
        print("  every scored finding declares where it landed, and every target resolves")

    print("\n=== B. is every recommendable placement backed by evidence? ===")
    surface, pb = audit_decision_surface_is_evidenced()
    print(f"  {len(surface)} distinct placements reachable across the decision surface")
    for row, why in UNMEASURED_PLACEMENTS.items():
        print(f"  unmeasured (declared)  {row}\n                         -> {why}")
    for p in pb:
        print("  UNEVIDENCED  " + p)
    if not pb:
        print("  no placement can be recommended without either a measurement or a stated reason")

    print("\n=== C. is any fact expressed in two places, unenforced? ===")
    pc = audit_single_source_of_truth()
    for p in pc:
        print("  DUPLICATED  " + p)
    if not pc:
        print("  the layer-count resolver is used everywhere; simulator tables agree with Python")

    print("\n=== D. can a reader actually reach the evidence we cite? ===")
    ncited, pd = audit_evidence_is_reachable()
    print(f"  {ncited} repo-relative evidence paths cited by the register")
    for p in pd:
        print("  UNREACHABLE  " + p)
    if not pd:
        print("  every cited evidence file is in the repository")

    problems = pa + pb + pc + pd
    print("\n" + "=" * 60)
    if problems:
        print(f"{len(problems)} audit finding(s). These are not test failures - they are places")
        print("where what we know and what we ship have drifted apart.")
        sys.exit(1)
    print("audit clean")


if __name__ == "__main__":
    main()
