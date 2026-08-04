"""S-1 phase 1: generate extraction instances, collect teacher answers, filter by predicate.

  python weights/s1_gen.py --selftest
  python weights/s1_gen.py --emit    train 0 400   > (writes weights/data/s1_train_prompts.json)
  python weights/s1_gen.py --teacher http://127.0.0.1:8080 --split train
  python weights/s1_gen.py --report

WHY MECHANICAL GENERATION. An honest held-out split is the whole experiment. If train and test
instances are hand-written, "held out" means "I think I didn't reuse it". Here every instance is
a template filled from a SEEDED rng, train and test draw from DISJOINT seed ranges, and the
disjointness is asserted by prompt hash before any number is reported (KR-C). The ground truth
is computed by the generator, not by the teacher - so a teacher answer can be checked exactly,
and the same checker scores the student later.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

FIRST = ["Dana", "Marta", "Tomas", "Aisha", "Piotr", "Chiara", "Rafael", "Ingrid", "Yusuf", "Lena",
         "Otto", "Sofia", "Hugo", "Nadia", "Emil", "Clara", "Viktor", "Rosa", "Bruno", "Alma"]
LAST = ["Whitfield", "Ruiz", "Novak", "Haddad", "Kowalski", "Bianchi", "Moreau", "Lindqvist",
        "Demir", "Okafor", "Vargas", "Halonen", "Sorensen", "Marchetti", "Bauer", "Costa"]
ROLE = ["VP of Partnerships", "Head of Ops", "Finance Director", "Procurement Lead",
        "Chief of Staff", "Engineering Manager", "Account Director", "Data Lead"]
COMP = ["Northwind Logistics", "Delta Freight", "Kestrel Analytics", "Bramble Foods",
        "Ardent Systems", "Solano Media", "Ferro Manufacturing", "Lyra Health"]
DOM = ["northwind-log.com", "deltafreight.eu", "kestrel.io", "bramble.co", "ardent-sys.com",
       "solano.media", "ferro-mfg.de", "lyrahealth.org"]
CUR = ["EUR", "USD", "GBP", "CHF"]
ITEM = ["Widget A", "Widget B", "Console Rail", "Filter Pack", "Drive Belt", "Sensor Kit",
        "Cable Loom", "Mount Bracket"]
MON = ["January", "March", "April", "June", "September", "November"]


def _contact(r):
    f, l = r.choice(FIRST), r.choice(LAST)
    role = r.choice(ROLE)
    i = r.randrange(len(COMP))
    email = f"{f[0].lower()}.{l.lower()}@{DOM[i]}"
    prompt = ("Extract to JSON with exactly the keys name, email, company, role. Reply with only "
              "the JSON object, no commentary.\n\n"
              f"'Hi, I'm {f} {l}, {role} at {COMP[i]}. Reach me at {email}.'")
    # The first version re-derived the role by splitting the PROMPT on ", " - which split on the
    # INSTRUCTION ("keys name, email, company, role.") and produced a ground truth containing half
    # the instruction text. The teacher was right and the checker was wrong, on every contact
    # instance. Ground truth is now the value we chose; KR-E in the self-test makes this class of
    # bug impossible to ship again.
    return prompt, {"name": f"{f} {l}", "email": email, "company": COMP[i], "role": role}


def _invoice(r):
    n = r.choice([2, 3])
    rows, want = [], []
    for _ in range(n):
        it, q = r.choice(ITEM), r.randrange(1, 12)
        p = round(r.uniform(4.5, 480.0), 2)
        rows.append(f"  {it}   x{q}   @ {p}")
        want.append({"item": it, "qty": q, "unit_price": p})
    prompt = ("Extract every line item to a JSON array of objects with keys item, qty, "
              "unit_price. Reply with only the JSON array.\n\nINVOICE\n" + "\n".join(rows))
    return prompt, want


def _terms(r):
    c = r.choice(COMP)
    amt = r.randrange(2, 400) * 250
    days = r.choice([14, 30, 45, 60, 90])
    prompt = ("Return only a JSON object with keys company, amount, currency, due_days.\n\n"
              f"'Per our agreement, {c} will be invoiced {amt:,} {r.choice(CUR)}, net {days}.'")
    cur = prompt.split(f"{amt:,} ")[1].split(",")[0]
    return prompt, {"company": c, "amount": amt, "currency": cur, "due_days": days}


def _ticket(r):
    ids = sorted(r.sample(range(1000, 9999), 3))
    subj = ["SSO login loop", "VAT wrong on invoices", "CSV export truncates",
            "webhook retries stall", "PDF renders blank", "search index stale"]
    picked = r.sample(subj, 3)
    lines = ", ".join(f"TCK-{i} ({s})" for i, s in zip(ids, picked))
    prompt = ("List the three ticket IDs as a JSON array of strings, in the order given. Reply "
              f"with only the array.\n\nOpen: {lines}.")
    return prompt, [f"TCK-{i}" for i in ids]


GENS = [_contact, _invoice, _terms, _ticket]


def instance(seed):
    r = random.Random(seed)
    fn = GENS[seed % len(GENS)]
    prompt, truth = fn(r)
    return {"seed": seed, "kind": fn.__name__.strip("_"), "prompt": prompt, "truth": truth,
            "hash": hashlib.sha256(prompt.encode()).hexdigest()[:16]}


def _norm(x):
    if isinstance(x, dict):
        return {str(k).strip().lower(): _norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_norm(v) for v in x]
    if isinstance(x, str):
        return x.strip().lower()
    if isinstance(x, (int, float)):
        return round(float(x), 4)
    return x


def check(truth, raw):
    """Exact match against generator-computed ground truth. No partial credit, no judgement."""
    import re
    t = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    for o, c in (("{", "}"), ("[", "]")):
        i, j = t.find(o), t.rfind(c)
        if i != -1 and j > i:
            try:
                return _norm(json.loads(t[i:j + 1])) == _norm(truth)
            except json.JSONDecodeError:
                continue
    return False


def ask(url, prompt, npredict=2048):
    import urllib.request
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": npredict, "temperature": 0.0, "top_k": 1}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as fh:
        d = json.loads(fh.read().decode("utf-8", "replace"))
    msg = d["choices"][0]["message"]
    txt = msg.get("content") or ""
    if "<think>" in txt:
        import re
        txt = re.sub(r"<think>.*?(?:</think>|$)", "", txt, flags=re.S)
    return txt.strip(), d["choices"][0].get("finish_reason", "")


SPLITS = {"train": (0, 400), "heldout": (100000, 100120)}   # DISJOINT by construction


def selftest():
    """The generator must produce checkable, distinct instances - and the checker must reject
    a wrong answer. A generator whose 'held-out' overlaps training is the one failure that
    would silently invalidate every S-1 number."""
    a, b = SPLITS["train"], SPLITS["heldout"]
    tr = {instance(s)["hash"] for s in range(*a)}
    ho = {instance(s)["hash"] for s in range(*b)}
    if tr & ho:
        print(f"  FAIL KR-C: {len(tr & ho)} prompt(s) shared between train and held-out")
        return 1
    print(f"self-test: train {len(tr)} unique prompts, held-out {len(ho)}, overlap 0")
    ex = instance(1)
    if not check(ex["truth"], json.dumps(ex["truth"])):
        print("  FAIL: checker rejects the ground truth itself")
        return 1
    if check(ex["truth"], json.dumps({"wrong": 1})):
        print("  FAIL: checker accepts a wrong answer")
        return 1
    bad = dict(ex["truth"]) if isinstance(ex["truth"], dict) else None
    if bad:
        k = list(bad)[0]
        bad[k] = str(bad[k]) + "X"
        if check(ex["truth"], json.dumps(bad)):
            print("  FAIL: checker accepts a one-field-off answer")
            return 1
    # KR-E: a ground-truth value that does not appear verbatim in its own prompt cannot be
    # extracted from it - the truth is wrong, not the model. This guard would have caught the
    # role bug on instance 0 instead of after 15 wasted teacher calls.
    def _strings(v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, dict):
            return [x for u in v.values() for x in _strings(u)]
        if isinstance(v, list):
            return [x for u in v for x in _strings(u)]
        return []
    for sd in list(range(60)) + list(range(100000, 100030)):
        ins = instance(sd)
        for val in _strings(ins["truth"]):
            if val not in ins["prompt"]:
                print(f"  FAIL KR-E: seed {sd} ({ins['kind']}) truth value absent from prompt: "
                      f"{val!r}")
                return 1
    print("self-test: every ground-truth string appears verbatim in its prompt (KR-E)")
    kinds = {instance(s)["kind"] for s in range(20)}
    if len(kinds) < 4:
        print(f"  FAIL: only {len(kinds)} instance kinds generated")
        return 1
    print(f"self-test PASS - {len(kinds)} kinds, checker discriminates, splits disjoint")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--teacher")
    ap.add_argument("--split", default="train", choices=list(SPLITS))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if a.selftest:
        return selftest()
    if a.report:
        for sp in SPLITS:
            p = os.path.join(DATA, f"s1_{sp}_teacher.json")
            if not os.path.exists(p):
                print(f"{sp}: not collected"); continue
            rows = json.load(open(p, encoding="utf-8"))
            ok = sum(1 for r in rows if r["passed"])
            print(f"{sp}: {len(rows)} instances, {ok} teacher-clean ({100*ok/len(rows):.1f}%)")
        return 0
    if a.teacher:
        if selftest():
            print("refusing: generator self-test failed")
            return 1
        lo, hi = SPLITS[a.split]
        rows = []
        for n, s in enumerate(range(lo, hi), 1):
            ins = instance(s)
            try:
                out, fin = ask(a.teacher, ins["prompt"])
            except Exception as exc:
                out, fin = "", f"ERROR {exc}"
            ins["teacher"] = out
            ins["finish"] = fin
            # A truncated answer is NOT a wrong answer: the 30B is a thinking model and spends
            # its budget before it speaks. At npredict=512 every invoice instance came back
            # empty with finish=length and would have scored as a teacher failure.
            ins["truncated"] = (fin == "length" and not out.strip())
            ins["passed"] = False if ins["truncated"] else check(ins["truth"], out)
            rows.append(ins)
            if n % 20 == 0:
                print(f"  {n}/{hi-lo}  clean so far: {sum(1 for r in rows if r['passed'])}",
                      flush=True)
            with open(os.path.join(DATA, f"s1_{a.split}_teacher.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(rows, fh, indent=1)
        ok = sum(1 for r in rows if r["passed"])
        print(f"\n{a.split}: {len(rows)} instances, {ok} clean ({100*ok/len(rows):.1f}%)")
        if a.split == "train":
            print(f"KR-B (>=300 clean pairs): {'PASS' if ok >= 300 else 'FAIL - precondition'}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
