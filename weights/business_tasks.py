"""Business tasks with MACHINE-CHECKABLE outcomes, so two models can be compared on one bar.

  python weights/business_tasks.py --list
  python weights/business_tasks.py --selftest          # the checks must reject a bad answer
  python weights/business_tasks.py --run MODEL --bin llama-cli.exe [--args "..."]
  python weights/business_tasks.py --score RESULTS.json

WHY THIS EXISTS. quantprobe recommends a 2.5-bit 30B and headlines a tok/s figure. Nobody has
checked whether that config does business work. We measure perplexity and KL only - and after
L-27 we know perplexity is blunt: it moved 23% while the model changed its chosen token on 27%
of positions. Neither number tells you whether the email was sendable.

WHY THE CHECKS ARE CODE, NOT PROSE. The first version of this file scored against sentences like
"sendable with at most a name edit". That cannot compare two models: it re-reads the output with
a human in the loop, and the human has already seen which model produced it. Every AUTO task
below carries executable predicates instead - JSON parses, this arithmetic equals that integer,
no number appears that was not in the source, the answer is exactly one of these labels. Same
input, same predicate, same verdict, for every model, forever.

THE ONE CHECK THAT MATTERS MOST is `nonums`: every number in the output must appear in the
source. That is a deterministic hallucination detector. A model that invents a plausible figure
fails, and it fails identically whoever runs it.

FAIRNESS. A deterministic check is only fair if the format was demanded in the prompt. Every
AUTO prompt states its own output contract ("reply with only the JSON object", "answer with one
word"). Instruction-following is part of business usability, so this is a real measurement, not
a gotcha - but it is measured openly rather than smuggled in.

RUBRIC tasks are kept, and kept SEPARATE. Tone and persuasion are real business requirements
that no predicate captures. They are never mixed into the headline number.

THE BAR, staked before any model output existed (and it still does not - the one run started
against this set was killed at task 1/50 for a misconfigured placement and never scored):

  P1  >= 80% of AUTO tasks pass   -> the recommended config is business-useful
  KILL  < 60%                     -> 2.5-bit is NOT business-usable, and every tok/s figure we
        publish for it is a speed number for a model that cannot do the work. The README
        headline gets qualified at the same prominence as the speed claim.
  60-80%                          -> qualified: fine for drafting, not for sending

Per-cluster results are reported too, because a uniform average hides the useful finding: if
2.5-bit holds on summarisation but collapses on structured extraction, that tells a user which
jobs the cheap quant is safe for.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile, time

# --------------------------------------------------------------------------------------
# CHECK PRIMITIVES. Each returns (name, predicate). A predicate takes the raw model output
# and returns True/False. No predicate may look at anything but the output text.
# --------------------------------------------------------------------------------------

# Extracting "the numbers in this text" is where this harness first produced FALSE ACCUSATIONS.
# The naive r"-?\d[\d,]*(?:\.\d+)?" charged the model with inventing numbers twice, wrongly:
#   "Q3 revenue rose"        -> pulled a bare 3 out of the quarter label, so a summary that
#                               correctly contained no numbers was failed for containing one.
#   "EMEA,Q3,1200000"        -> matched across the field separator as "3,1200000", stripped the
#                               comma, and reported the number 31200000 - which appears nowhere.
#                               A perfect CSV was failed for hallucinating.
# Two rules fix it. A number may not start immediately after a word character or a dot (so the
# 3 in Q3 and the 24 in v1.24.0 are not numbers), and a comma only groups digits when it groups
# them in threes (so 47,500 is one number but Q3,1200000 is not).
NUM_RE = re.compile(r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?![\d])"      # 47,500 / 3,000,000
                    r"|(?<![\w.])-?\d+(?:\.\d+)?")                  # 1200000 / 4.2 / 12


def _norm_num(s: str) -> str:
    """'4,200' and '4200' and '4200.0' are the same number; 12% and 12 are the same digits."""
    s = s.replace(",", "")
    try:
        f = float(s)
    except ValueError:
        return s
    return str(int(f)) if f == int(f) else str(f)


def _nums_in(text: str) -> set:
    return {_norm_num(m) for m in NUM_RE.findall(text)}


def _strip_fence(text: str) -> str:
    """Models wrap JSON in ``` fences even when told not to. Accept that, it is cosmetic."""
    m = re.search(r"```(?:json|python)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def _first_json(text: str):
    t = _strip_fence(text)
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = t.find(opener), t.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None


def js(*keys):
    """Parses as a JSON object carrying exactly these top-level keys."""
    want = set(keys)

    def f(out):
        d = _first_json(out)
        return isinstance(d, dict) and set(d.keys()) == want
    return (f"json object with keys {sorted(want)}", f)


def js_list(n=None):
    def f(out):
        d = _first_json(out)
        return isinstance(d, list) and (n is None or len(d) == n)
    return (f"json array{'' if n is None else f' of {n}'}", f)


def js_path(path, value):
    """Dotted path into the parsed JSON equals this value (numbers compared numerically)."""
    def f(out):
        d = _first_json(out)
        for part in path.split("."):
            if isinstance(d, list):
                if not part.isdigit() or int(part) >= len(d):
                    return False
                d = d[int(part)]
            elif isinstance(d, dict) and part in d:
                d = d[part]
            else:
                return False
        if isinstance(value, (int, float)) and isinstance(d, (int, float, str)):
            try:
                return abs(float(d) - float(value)) < 1e-6
            except (TypeError, ValueError):
                return False
        return str(d).strip().lower() == str(value).strip().lower()
    return (f"{path} == {value!r}", f)


def has(*subs):
    """Every one of these appears (case-insensitive)."""
    low = [s.lower() for s in subs]
    return ("mentions " + ", ".join(repr(s) for s in subs),
            lambda out: all(s in out.lower() for s in low))


def lacks(*subs):
    low = [s.lower() for s in subs]
    return ("avoids " + ", ".join(repr(s) for s in subs),
            lambda out: not any(s in out.lower() for s in low))


def nonums(*allowed):
    """DETERMINISTIC HALLUCINATION CHECK: no number outside this set may appear."""
    ok = {_norm_num(str(a)) for a in allowed}

    def f(out):
        return _nums_in(out) <= ok
    return ("invents no numbers", f)


def num(x):
    """This exact number appears somewhere in the output."""
    return (f"answer is {x}", lambda out: _norm_num(str(x)) in _nums_in(out))


def maxwords(n):
    return (f"<= {n} words", lambda out: len(out.split()) <= n)


def minwords(n):
    return (f">= {n} words", lambda out: len(out.split()) >= n)


def maxsent(n):
    def f(out):
        s = [x for x in re.split(r"[.!?]+(?:\s|$)", out.strip()) if x.strip()]
        return len(s) <= n
    return (f"<= {n} sentences", f)


def bullets(n):
    def f(out):
        lines = [l for l in out.splitlines()
                 if re.match(r"\s*(?:[-*•]|\d+[.)])\s+\S", l)]
        return len(lines) == n
    return (f"exactly {n} bullets", f)


def label(*opts):
    """Output reduces to exactly one of these labels and no other."""
    low = [o.lower() for o in opts]

    def f(out):
        t = out.lower()
        hits = [o for o in low if re.search(rf"\b{re.escape(o)}\b", t)]
        return len(set(hits)) == 1
    return ("exactly one of " + "/".join(opts), f)


def lines_are(n, pattern):
    """n lines match this regex - for CSV/table shapes."""
    rx = re.compile(pattern)

    def f(out):
        return len([l for l in out.splitlines() if rx.match(l.strip())]) == n
    return (f"{n} lines matching {pattern}", f)


def exact(s):
    """The whole output IS this string (whitespace-normalised). The hardest honest check."""
    want = " ".join(s.split())

    def f(out):
        return " ".join(out.split()) == want
    return (f"exactly {s[:40]!r}...", f)


def words_exactly(n):
    return (f"exactly {n} words", lambda out: len(out.split()) == n)


def pyok(snippet):
    """Extract the model's python and run it with this assertion appended."""
    def f(out):
        code = _strip_fence(out)
        if "def " not in code:
            return False
        path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                             encoding="utf-8") as fh:
                fh.write(code + "\n" + snippet + "\n")
                path = fh.name
            r = subprocess.run([sys.executable, path], capture_output=True, timeout=15)
            return r.returncode == 0
        except Exception:
            return False
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    return ("code runs and passes assertions", f)


# --------------------------------------------------------------------------------------
# THE TASKS. (cluster, id, kind, prompt, [checks])
# kind: "auto"  -> every check is executable; counts toward the headline
#       "rubric"-> human judgement; reported separately, never in the headline
# --------------------------------------------------------------------------------------

Q3 = ("Q3 revenue was 4.2M, up 12% YoY but 8% below plan. Churn rose to 3.1% monthly, "
      "driven by SMB. Enterprise ARR grew 34%. Sales headcount is 22, up from 17. CAC "
      "payback lengthened from 14 to 19 months. The board asked for a revised FY plan by "
      "15 November.")

TASKS = [
    # --- EXTRACTION: strict shape, zero tolerance -------------------------------------
    ("extraction", "e1", "auto",
     "Extract to JSON with exactly the keys name, email, company, role. Reply with only the "
     "JSON object, no commentary.\n\n"
     "'Hi, I'm Dana Whitfield, VP of Partnerships at Northwind Logistics. Reach me at "
     "d.whitfield@northwind-log.com.'",
     [js("name", "email", "company", "role"),
      js_path("email", "d.whitfield@northwind-log.com"),
      js_path("name", "Dana Whitfield")]),

    ("extraction", "e2", "auto",
     "Extract every line item to a JSON array of objects with keys item, qty, unit_price. "
     "Reply with only the JSON array.\n\n"
     "INVOICE\n  Widget A   x3   @ 19.99\n  Widget B   x1   @ 249.00\n  Shipping   x1   @ 12.50",
     [js_list(3), js_path("0.qty", 3), js_path("1.unit_price", 249.00),
      js_path("2.item", "Shipping")]),

    ("extraction", "e3", "auto",
     "From the text below return only a JSON object with keys total_headcount and "
     "revised_plan_due. Reply with only the JSON object.\n\n" + Q3,
     [js("total_headcount", "revised_plan_due"), js_path("total_headcount", 22)]),

    ("extraction", "e4", "auto",
     "Return only a JSON object with keys company, amount_usd, due_days.\n\n"
     "'Per our agreement, Acme Corp will be invoiced 47,500 USD, net 30.'",
     [js("company", "amount_usd", "due_days"), js_path("amount_usd", 47500),
      js_path("due_days", 30)]),

    ("extraction", "e5", "auto",
     "List the three ticket IDs as a JSON array of strings, in the order given. Reply with "
     "only the array.\n\n"
     "Open: TCK-4471 (SSO login loop), TCK-4472 (VAT wrong on Irish invoices), "
     "TCK-4488 (CSV export truncates).",
     [js_list(3), js_path("0", "TCK-4471"), js_path("2", "TCK-4488")]),

    # --- ARITHMETIC: business maths, exact answers -------------------------------------
    ("arithmetic", "a1", "auto",
     "A subscription is 49 USD/month per seat. A customer has 37 seats and an 18% annual "
     "discount for paying yearly. What is the annual total in USD after discount? Give the "
     "final number only, rounded to 2 decimals.",
     [num(17839.92)]),

    ("arithmetic", "a2", "auto",
     "Monthly churn is 3.1%. Starting from 4,000 customers and no new signups, how many "
     "customers remain after 3 months? Round to the nearest whole customer. Give the number only.",
     [num(3639)]),

    ("arithmetic", "a3", "auto",
     "CAC is 12,400 USD and monthly gross profit per customer is 653 USD. What is the CAC "
     "payback period in months, rounded up to a whole month? Give the number only.",
     [num(19)]),

    ("arithmetic", "a4", "auto",
     "Revenue was 4.2M, which is 8% below plan. What was the plan number in millions, "
     "rounded to 3 decimals? Give the number only.",
     [num(4.565)]),

    ("arithmetic", "a5", "auto",
     "An invoice of 8,300 EUR excludes VAT at 23%. What is the gross total in EUR, to 2 "
     "decimals? Give the number only.",
     [num(10209.00)]),

    ("arithmetic", "a6", "auto",
     "You have 22 sales reps, up from 17 a year ago. What is the year-over-year headcount "
     "growth as a percentage, to 1 decimal place? Give the number only.",
     [num(29.4)]),

    # --- CLASSIFICATION: exactly one label --------------------------------------------
    ("classification", "k1", "auto",
     "Classify this support ticket as exactly one of: BILLING, TECHNICAL, SALES. Answer with "
     "the single word only.\n\n'Your invoice charged me for a month after I cancelled.'",
     [label("BILLING", "TECHNICAL", "SALES"), has("billing"), maxwords(3)]),

    ("classification", "k2", "auto",
     "Classify as exactly one of: BILLING, TECHNICAL, SALES. Answer with the single word "
     "only.\n\n'After SSO redirect, Safari loops back to the login page.'",
     [label("BILLING", "TECHNICAL", "SALES"), has("technical"), maxwords(3)]),

    ("classification", "k3", "auto",
     "Sentiment of this review as exactly one of: POSITIVE, NEGATIVE, MIXED. Single word "
     "only.\n\n'Setup was painless and support replied in minutes, but it is far too "
     "expensive for what it does.'",
     [label("POSITIVE", "NEGATIVE", "MIXED"), has("mixed"), maxwords(3)]),

    ("classification", "k4", "auto",
     "Is this lead ENTERPRISE or SMB? Answer with the single word only.\n\n"
     "'We're a 12-person agency looking for a cheap plan, no procurement process.'",
     [label("ENTERPRISE", "SMB"), has("smb"), maxwords(3)]),

    ("classification", "k5", "auto",
     "Route to exactly one of: LEGAL, FINANCE, ENGINEERING. Single word only.\n\n"
     "'The DPA the customer sent back has an altered sub-processor clause.'",
     [label("LEGAL", "FINANCE", "ENGINEERING"), has("legal"), maxwords(3)]),

    ("classification", "k6", "auto",
     "Priority as exactly one of: P0, P1, P2. Single word only.\n\n"
     "'Checkout is down for all customers in production.'",
     [label("P0", "P1", "P2"), has("p0"), maxwords(3)]),

    # --- SUMMARISATION: constrained shape + no invented numbers -----------------------
    ("summarisation", "s1", "auto",
     "Summarise in exactly 5 bullet points, one line each, starting each with '- '. Use only "
     "numbers that appear in the source.\n\n" + Q3,
     [bullets(5), nonums(4.2, 12, 8, 3.1, 34, 22, 17, 14, 19, 15, 3, 1, 2, 4, 5)]),

    ("summarisation", "s2", "auto",
     "In one sentence of at most 25 words, state the single most urgent item. Use only "
     "numbers from the source.\n\n" + Q3,
     [maxwords(25), maxsent(1), nonums(4.2, 12, 8, 3.1, 34, 22, 17, 14, 19, 15)]),

    ("summarisation", "s3", "auto",
     "Give exactly 3 bullets, each starting '- ', naming only risks stated in the text. Do "
     "not add numbers that are not in the source.\n\n" + Q3,
     [bullets(3), nonums(4.2, 12, 8, 3.1, 34, 22, 17, 14, 19, 15, 3, 1, 2)]),

    ("summarisation", "s4", "auto",
     "Compress this to under 30 words with no numbers at all.\n\n" + Q3,
     [maxwords(30), nonums()]),

    # --- INSTRUCTION FOLLOWING: the format IS the task --------------------------------
    ("instruction", "i1", "auto",
     "Reply with exactly the word ACKNOWLEDGED and nothing else.",
     [maxwords(1), has("acknowledged")]),

    ("instruction", "i2", "auto",
     "List the days Monday to Friday, one per line, each prefixed with '> '. No other text.",
     [lines_are(5, r"^>\s+\w+day$")]),

    ("instruction", "i3", "auto",
     "Write a product tagline of at most 8 words that does not contain the letter 'e'.",
     [maxwords(8), lacks("e")]),

    ("instruction", "i4", "auto",
     "Answer in exactly two sentences. First sentence must start with 'Yes' or 'No'. "
     "Question: should a 12-person agency buy an enterprise CRM?",
     [maxsent(2), ("starts Yes/No", lambda o: o.strip().lower().startswith(("yes", "no")))]),

    ("instruction", "i5", "auto",
     "Output a CSV with header 'region,quarter,revenue' and exactly 2 data rows for EMEA and "
     "AMER in Q3, revenue 1200000 and 3000000. No other text.",
     [lines_are(1, r"^region,quarter,revenue$"),
      lines_are(2, r"^(EMEA|AMER),Q3,\d+$"),
      nonums(3, 1200000, 3000000)]),

    ("instruction", "i6", "auto",
     "Reply with only a JSON object with keys status and retry_after_seconds, where status is "
     "'rate_limited' and retry_after_seconds is 30.",
     [js("status", "retry_after_seconds"), js_path("status", "rate_limited"),
      js_path("retry_after_seconds", 30)]),

    # --- CODE: it either runs or it does not ------------------------------------------
    # PROMPT CORRECTED: the original said only "rate", and the model reasonably read it as a
    # percentage (net * (1 + rate/100)). The assertion assumed a decimal fraction. That is an
    # ambiguous SPEC, not a wrong answer - so the prompt now states the convention instead of
    # the test quietly holding one the prompt never mentioned.
    ("code", "p1", "auto",
     "Write a Python function `vat_gross(net, rate)` returning the gross amount rounded to 2 "
     "decimals, where rate is a decimal fraction (0.23 means 23%). Reply with only the code.",
     [pyok("assert vat_gross(8300, 0.23) == 10209.0\n"
           "assert vat_gross(100, 0.2) == 120.0")]),

    ("code", "p2", "auto",
     "Write a Python function `churned(start, monthly_rate, months)` returning the whole "
     "number of customers remaining, rounded to nearest int. Reply with only the code.",
     [pyok("assert churned(4000, 0.031, 3) == 3639")]),

    ("code", "p3", "auto",
     "Write a Python function `parse_invoice_line(s)` that turns 'Widget A   x3   @ 19.99' "
     "into the dict {'item': 'Widget A', 'qty': 3, 'unit_price': 19.99}. Reply with only code.",
     [pyok("r = parse_invoice_line('Widget A   x3   @ 19.99')\n"
           "assert r == {'item': 'Widget A', 'qty': 3, 'unit_price': 19.99}, r")]),

    ("code", "p4", "auto",
     "Write a Python function `cac_payback(cac, monthly_gp)` returning months rounded UP to a "
     "whole number. Reply with only the code.",
     [pyok("assert cac_payback(12400, 653) == 19\nassert cac_payback(1000, 500) == 2")]),

    ("code", "p5", "auto",
     "Write a Python function `valid_email(s)` returning True/False. Reply with only the code.",
     [pyok("assert valid_email('d.whitfield@northwind-log.com') is True\n"
           "assert valid_email('not-an-email') is False")]),

    # --- ANALYSIS with checkable anchors ----------------------------------------------
    ("analysis", "n1", "auto",
     "Given monthly churn 3.1%, what is the implied annual retention as a percentage to 1 "
     "decimal? Give the number only.",
     [num(68.5)]),

    ("analysis", "n2", "auto",
     "Reply with only a JSON object with keys metric and direction, where metric is the one "
     "that worsened most in relative terms between CAC payback 14 and 19 months, and "
     "direction is 'worse'.",
     [js("metric", "direction"), js_path("direction", "worse")]),

    ("analysis", "n3", "auto",
     "Enterprise ARR grew 34% to 2.1M. What was it a year ago, in millions to 3 decimals? "
     "Give the number only.",
     [num(1.567)]),

    # KEY CORRECTED AFTER THE FIRST RUN, AND THE CORRECTION IS ARITHMETIC, NOT CONVENIENCE.
    # 1 - 0.969^12 = 31.47% lost, against a third = 33.33%. The answer is NO. The staked key
    # said YES; the model answered NO and was right. Changing a key after seeing outputs is
    # normally forbidden - this is allowed only because the key is checkably, objectively wrong,
    # and the change makes the task HARDER to pass, not easier.
    ("analysis", "n4", "auto",
     "Answer with exactly one word, YES or NO: at 3.1% monthly churn, does a cohort lose more "
     "than a third of its customers within 12 months?",
     [label("YES", "NO"), has("no"), maxwords(3)]),

    # --- MULTILINGUAL: checkable because the target strings are fixed ------------------
    ("multilingual", "m1", "auto",
     "Translate to French, reply with only the translation: 'The invoice is due in 30 days.'",
     [has("30"), ("looks french", lambda o: any(w in o.lower() for w in
                                                ("facture", "jours", "échéance")))]),

    ("multilingual", "m2", "auto",
     "Translate to German, reply with only the translation: 'Your account has been cancelled.'",
     [("looks german", lambda o: any(w in o.lower() for w in
                                     ("konto", "gekündigt", "storniert", "wurde")))]),

    ("multilingual", "m3", "auto",
     "Reply with only a JSON object with keys language and code, identifying the language of "
     "'Il contratto scade il 15 novembre' using an ISO 639-1 code.",
     [js("language", "code"), js_path("code", "it")]),

    ("multilingual", "m4", "auto",
     "Translate to Spanish, reply with only the translation: 'We will refund 47,500 USD.'",
     [has("47"), ("looks spanish", lambda o: any(w in o.lower() for w in
                                                 ("reembolsar", "devolver", "reembolso")))]),

    # --- RUBRIC: real business value, no honest predicate exists ----------------------
    ("correspondence", "c1", "rubric",
     "Write a short email declining a vendor's proposal for a CRM migration because the "
     "timeline slips past our Q4 freeze. Keep the door open for Q1.",
     "sendable with at most a name/date edit; declines clearly; leaves the door open"),

    ("correspondence", "c2", "rubric",
     "Draft a follow-up to a client who has not responded in two weeks about an unsigned SOW. "
     "Polite, not passive-aggressive, one clear ask.",
     "one unambiguous ask; professional tone; no guilt-tripping"),

    ("correspondence", "c3", "rubric",
     "Reply to a customer charged for a month they had already cancelled. Acknowledge, commit "
     "to a refund, give a timeline.",
     "acknowledges without admitting liability; concrete next step and timeline"),

    ("correspondence", "c4", "rubric",
     "Write a LinkedIn message to a candidate for a senior backend role. Two sentences on why "
     "them specifically, one clear call to action.",
     "specific rather than generic; single CTA"),

    ("judgement", "j1", "rubric",
     "A customer demands a feature we will not build. Draft the refusal in three sentences "
     "without saying 'unfortunately' or promising a roadmap slot.",
     "clear no; no false promise; keeps the relationship"),

    ("judgement", "j2", "rubric",
     "Our deploy caused 40 minutes of downtime. Write the customer-facing incident note.",
     "states impact and duration; no blame-shifting; says what changes"),

    ("longform", "l1", "rubric",
     "Write a one-page brief arguing whether to move from monthly to annual billing, covering "
     "cash flow, churn and discounting.",
     "coherent across the whole page; no contradiction; arguments actually connect"),

    ("longform", "l2", "rubric",
     "Draft an onboarding checklist for a new enterprise customer, grouped into week 1, "
     "week 2-4, and quarter 1.",
     "genuinely actionable items; correct grouping; no filler"),

    # ==================================================================================
    # TIER 3 - HARD. A well-run frontier model should pass most; a 2.5-bit 30B some.
    # Added 2026-08-03 as an EXTENSION. These are NOT part of the staked 80%/60% bar,
    # which was set for the 40 tasks above before any output existed and cannot absorb
    # new tasks without becoming a moved goalpost.
    # ==================================================================================
    ("tier3", "t3x1", "auto",
     "Ledger A: 'rent 2400, software 1180, travel 3310, payroll 41200'. Ledger B: 'rent 2400, "
     "software 1180, travel 3130, payroll 41200'. Exactly one item disagrees. Reply with only a "
     "JSON object with keys item, ledger_a, ledger_b, delta (delta = ledger_a - ledger_b).",
     [js("item", "ledger_a", "ledger_b", "delta"), js_path("item", "travel"),
      js_path("ledger_a", 3310), js_path("ledger_b", 3130), js_path("delta", 180)]),

    ("tier3", "t3a1", "auto",
     "NPV of cashflows -10000, 3000, 4000, 5000, 2000 (t=0..4) at a discount rate of 8%. "
     "Give the number only, to 2 decimals.",
     [num(1646.35)]),

    ("tier3", "t3c1", "auto",
     "Write a Python function `monthly_payment(principal, annual_rate, months)` for a standard "
     "amortized loan (annual_rate is a decimal fraction, e.g. 0.055), returning the payment "
     "rounded to 2 decimals. Reply with only the code.",
     [pyok("assert monthly_payment(250000, 0.055, 96) == 3224.83\n"
           "assert monthly_payment(100000, 0.06, 360) == 599.55")]),

    ("tier3", "t3l1", "auto",
     "Three houses, numbered 1-3 left to right. Colors red, blue, green; drinks tea, coffee, "
     "milk; pets cat, dog, fish - each exactly once. Clues: the red house is immediately left "
     "of the blue house; coffee is drunk in the green house; the cat lives in house 2; milk is "
     "drunk in house 1; the fish lives in the red house. Reply with only a JSON object of the "
     "form {\"house1\": {\"color\": ..., \"drink\": ..., \"pet\": ...}, \"house2\": ..., "
     "\"house3\": ...}.",
     [js("house1", "house2", "house3"),
      js_path("house1.color", "red"), js_path("house1.drink", "milk"),
      js_path("house1.pet", "fish"), js_path("house2.color", "blue"),
      js_path("house2.drink", "tea"), js_path("house2.pet", "cat"),
      js_path("house3.color", "green"), js_path("house3.drink", "coffee"),
      js_path("house3.pet", "dog")]),

    ("tier3", "t3s1", "auto",
     "Summarise in exactly 3 bullet lines, each starting with '- ', each at most 12 words, "
     "using only numbers that appear in the source.\n\n" + Q3,
     [bullets(3), nonums(4.2, 12, 8, 3.1, 34, 22, 17, 14, 19, 15, 3, 1, 2),
      ("every bullet <= 12 words",
       lambda o: all(len(l.lstrip("- ").split()) <= 12
                     for l in o.splitlines() if l.strip().startswith("- ")))]),

    ("tier3", "t3h1", "auto",
     "Tickets: TCK-101 'refund of 300 EUR not received', TCK-102 'dark mode broken', TCK-103 "
     "'invoice shows 250 instead of 25', TCK-104 'password reset loop', TCK-105 'charged twice "
     "this month'. Reply with only a JSON array of the ticket IDs that concern money, in the "
     "order given.",
     [js_list(3), js_path("0", "TCK-101"), js_path("1", "TCK-103"), js_path("2", "TCK-105")]),

    # ==================================================================================
    # TIER 4 - CEILING. Designed so that today's best models fail while remaining 100%
    # machine-checkable. 0% here is the EXPECTED result and is not a defect; the tier
    # exists so that when some model scores 2/6, that number means something. Every key
    # below is recomputed by the self-test, never trusted from the author.
    # ==================================================================================
    ("tier4", "t4a1", "auto",
     "Compute exactly: 738451926 * 584379201. Give the integer only.",
     [num(431535946492791126)]),

    ("tier4", "t4a2", "auto",
     "A 250,000 loan at 5.5% annual (monthly compounding, standard amortization) is repaid over "
     "96 monthly payments. What is the TOTAL INTEREST paid over the life of the loan, to the "
     "cent? Give the number only.",
     [num(59583.73)]),

    ("tier4", "t4w1", "auto",
     "Write exactly 45 words about a company's finances without using the letter 'e' anywhere. "
     "The words 'profit', 'cash' and 'margin' must all appear.",
     [words_exactly(45), lacks("e"), has("profit", "cash", "margin")]),

    ("tier4", "t4l1", "auto",
     "Five houses, numbered 1-5 left to right. Colors red, green, blue, white, yellow; "
     "nationalities Brit, Swede, Dane, German, Norwegian; drinks tea, coffee, milk, beer, water "
     "- each exactly once. Clues: the Brit lives in the red house; the green house is "
     "immediately left of the white house; coffee is drunk in the green house; the Norwegian "
     "lives in the first house; the Norwegian lives next to the blue house; milk is drunk in "
     "the middle house; the Dane drinks tea; the Swede drinks beer; beer is drunk in the last "
     "house. Reply with only a JSON object mapping house1..house5 to objects with keys color, "
     "nationality, drink.",
     [js("house1", "house2", "house3", "house4", "house5"),
      js_path("house1.color", "yellow"), js_path("house1.nationality", "Norwegian"),
      js_path("house1.drink", "water"), js_path("house2.color", "blue"),
      js_path("house2.nationality", "Dane"), js_path("house2.drink", "tea"),
      js_path("house3.color", "red"), js_path("house3.nationality", "Brit"),
      js_path("house3.drink", "milk"), js_path("house4.color", "green"),
      js_path("house4.nationality", "German"), js_path("house4.drink", "coffee"),
      js_path("house5.color", "white"), js_path("house5.nationality", "Swede"),
      js_path("house5.drink", "beer")]),

    ("tier4", "t4c1", "auto",
     "Write a Python function `iso_week(y, m, d)` returning the ISO-8601 week number for that "
     "date WITHOUT importing anything (no datetime, no calendar). Reply with only the code.",
     [("no imports used", lambda o: "import" not in _strip_fence(o)),
      pyok("assert iso_week(2016, 1, 1) == 53\nassert iso_week(2021, 1, 1) == 53\n"
           "assert iso_week(2020, 12, 31) == 53\nassert iso_week(2019, 12, 30) == 1\n"
           "assert iso_week(2015, 12, 28) == 53\nassert iso_week(2017, 1, 1) == 52")]),

    ("tier4", "t4s1", "auto",
     "Take the sentence: 'The quarterly report shows steady growth across all regions.' For "
     "every word of 5 or more letters, reverse its letters. Words of 4 or fewer letters stay "
     "unchanged. The final period is not part of any word. Reply with only the transformed "
     "sentence.",
     [exact("The ylretrauq troper swohs ydaets htworg ssorca all snoiger.")]),
]

# The 40 auto tasks the 2026-08-03 bar was staked on. The headline percentage is computed over
# THESE and only these, forever - tier 3/4 came later and joining them into the staked bar
# would be a moved goalpost in whichever direction they score.
STAKED_IDS = frozenset(
    t[1] for t in TASKS
    if t[2] == "auto" and not t[0].startswith("tier"))

# Difficulty tier per auto task. 1-2 = the staked business set (a competent small model should
# clear T1; T2 is where cheap quants start dropping points). 3 = hard, frontier territory.
# 4 = ceiling by design - the row that ranks models the day one of them stops scoring zero.
TIER = {}
for _t in TASKS:
    if _t[2] != "auto":
        continue
    if _t[0] == "tier3":
        TIER[_t[1]] = 3
    elif _t[0] == "tier4":
        TIER[_t[1]] = 4
    elif _t[0] in ("extraction", "classification", "multilingual") or \
            _t[1] in ("i1", "i2", "i4", "i5", "i6"):
        TIER[_t[1]] = 1
    else:
        TIER[_t[1]] = 2


def clusters():
    out = {}
    for t in TASKS:
        out.setdefault(t[0], []).append(t)
    return out


# --------------------------------------------------------------------------------------


def check_output(task, out):
    """Run every predicate. Returns (passed, [(name, ok), ...])."""
    if task[2] != "auto":
        return None, []
    detail = [(name, bool(fn(out))) for name, fn in task[4]]
    return all(ok for _, ok in detail), detail


def selftest():
    """A checker that cannot fail is not a checker. Every predicate must reject a bad answer."""
    bad = "Sure! Here is my answer: the revenue was 9.9M and I hope this helps."
    auto = [t for t in TASKS if t[2] == "auto"]
    survivors = []
    for t in auto:
        ok, _ = check_output(t, bad)
        if ok:
            survivors.append(t[1])
    print(f"self-test: {len(auto)} auto tasks, boilerplate answer passes {len(survivors)}")
    if survivors:
        print("  FAIL - these tasks accept a junk answer:", ", ".join(survivors))
        return 1
    # and a known-good answer must PASS, or the checks are simply always-false
    spot = {
        "i1": "ACKNOWLEDGED",
        "a1": "17839.92",
        "k1": "BILLING",
        "e4": '{"company": "Acme Corp", "amount_usd": 47500, "due_days": 30}',
        "i6": '{"status": "rate_limited", "retry_after_seconds": 30}',
    }
    # TIER KEYS ARE RECOMPUTED, NOT TRUSTED. Two of the original eight arithmetic keys were
    # wrong when first written; a wrong key at tier 4 would be invisible forever because every
    # model is expected to fail there anyway. So the self-test re-derives each key mechanically
    # and refuses to run if the task disagrees.
    import datetime as _dt
    if 738451926 * 584379201 != 431535946492791126:
        print("  FAIL - t4a1 key wrong"); return 1
    npv = -10000 + 3000 / 1.08 + 4000 / 1.08 ** 2 + 5000 / 1.08 ** 3 + 2000 / 1.08 ** 4
    if round(npv, 2) != 1646.35:
        print("  FAIL - t3a1 key wrong"); return 1
    p, r, n = 250000, 0.055 / 12, 96
    pay = p * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    if round(pay, 2) != 3224.83:
        print("  FAIL - t3c1 key wrong"); return 1
    bal, tot = p, 0.0
    for _ in range(n):
        i = bal * r; bal -= (pay - i); tot += i
    if round(tot, 2) != 59583.73:
        print("  FAIL - t4a2 key wrong"); return 1
    for (yy, mm, dd), wk in [((2016, 1, 1), 53), ((2021, 1, 1), 53), ((2020, 12, 31), 53),
                             ((2019, 12, 30), 1), ((2015, 12, 28), 53), ((2017, 1, 1), 52)]:
        if _dt.date(yy, mm, dd).isocalendar()[1] != wk:
            print(f"  FAIL - t4c1 key wrong for {yy}-{mm}-{dd}"); return 1
    src = "The quarterly report shows steady growth across all regions"
    if " ".join(w[::-1] if len(w) >= 5 else w for w in src.split()) + "." != \
            "The ylretrauq troper swohs ydaets htworg ssorca all snoiger.":
        print("  FAIL - t4s1 key wrong"); return 1

    # THE PUZZLES ARE BRUTE-FORCED. A logic puzzle with zero or two solutions is not a task,
    # it is a trap - so the self-test enumerates the full space and demands exactly one
    # solution, equal to the staked key.
    from itertools import permutations
    sols = []
    for cols in permutations(("red", "blue", "green")):
        if cols.index("red") + 1 != cols.index("blue"):
            continue
        for drinks in permutations(("tea", "coffee", "milk")):
            if drinks[0] != "milk" or drinks[cols.index("green")] != "coffee":
                continue
            for pets in permutations(("cat", "dog", "fish")):
                if pets[1] == "cat" and pets[cols.index("red")] == "fish":
                    sols.append((cols, drinks, pets))
    if sols != [(("red", "blue", "green"), ("milk", "tea", "coffee"), ("fish", "cat", "dog"))]:
        print(f"  FAIL - t3l1 has {len(sols)} solution(s), key mismatch"); return 1
    C = ("red", "green", "blue", "white", "yellow")
    N = ("Brit", "Swede", "Dane", "German", "Norwegian")
    D = ("tea", "coffee", "milk", "beer", "water")
    sols = []
    for cols in permutations(C):
        gi = cols.index("green")
        if gi + 1 >= 5 or cols[gi + 1] != "white":
            continue
        bi = cols.index("blue")
        if bi != 1:                                   # norwegian@1 next to blue -> blue@2
            continue
        for nats in permutations(N):
            if nats[0] != "Norwegian" or nats[cols.index("red")] != "Brit":
                continue
            for drinks in permutations(D):
                if drinks[2] != "milk" or drinks[gi] != "coffee" or drinks[4] != "beer":
                    continue
                if drinks[nats.index("Dane")] != "tea" or drinks[nats.index("Swede")] != "beer":
                    continue
                sols.append((cols, nats, drinks))
    want = (("yellow", "blue", "red", "green", "white"),
            ("Norwegian", "Dane", "Brit", "German", "Swede"),
            ("water", "tea", "milk", "coffee", "beer"))
    if sols != [want]:
        print(f"  FAIL - t4l1 has {len(sols)} solution(s) or key mismatch: {sols[:2]}")
        return 1
    print("tier keys: all recomputed and confirmed; both puzzles have exactly one solution")

    spot.update({
        "t3x1": '{"item":"travel","ledger_a":3310,"ledger_b":3130,"delta":180}',
        "t3a1": "1646.35",
        "t3h1": '["TCK-101", "TCK-103", "TCK-105"]',
        "t3l1": ('{"house1":{"color":"red","drink":"milk","pet":"fish"},'
                 '"house2":{"color":"blue","drink":"tea","pet":"cat"},'
                 '"house3":{"color":"green","drink":"coffee","pet":"dog"}}'),
        "t4a1": "431535946492791126",
        "t4a2": "59583.73",
        "t4w1": ("Our profit outlook is strong this month. Cash flow stays solid, margin "
                 "gains hold firm. Staff cut costs, boosting output. Profits climb monthly. "
                 "Firms ship promptly. Patrons pay fast. Bills stay low. Stock turns quickly. "
                 "Growth looks good. Payroll stays flat. Outlays drop now. Victory."),
        "t4l1": ('{"house1":{"color":"yellow","nationality":"Norwegian","drink":"water"},'
                 '"house2":{"color":"blue","nationality":"Dane","drink":"tea"},'
                 '"house3":{"color":"red","nationality":"Brit","drink":"milk"},'
                 '"house4":{"color":"green","nationality":"German","drink":"coffee"},'
                 '"house5":{"color":"white","nationality":"Swede","drink":"beer"}}'),
        "t4s1": "The ylretrauq troper swohs ydaets htworg ssorca all snoiger.",
    })
    dead = []
    for t in auto:
        if t[1] in spot:
            ok, d = check_output(t, spot[t[1]])
            if not ok:
                dead.append((t[1], [n for n, o in d if not o]))
    if dead:
        print("  FAIL - these reject a correct answer:", dead)
        return 1
    print(f"self-test: {len(spot)} known-good answers all accepted")
    print("self-test PASS - checks discriminate in both directions")
    return 0


def ask_server(url, prompt, npredict):
    """One chat call. temperature 0 so the same model gives the same answer twice.

    MUST be /v1/chat/completions, not /completion. The raw endpoint does NOT apply the model's
    chat template, so an instruct model just continues the text: asked to reply ACKNOWLEDGED it
    answered "I am a student who is struggling with my homework". Every task would have failed
    and the verdict would have read "2.5-bit cannot do business work" - a harness bug wearing a
    finding's clothes. The template is not a detail; it is the difference between measuring the
    model and measuring our own plumbing.

    Qwen3 is a THINKING model: asked for one word it spent 60 of 72 tokens reasoning first, so a
    small max_tokens returns empty content - the budget is consumed before the answer starts.
    Budget generously and report the split, because "22.7 tok/s" and "22.7 tok/s of which two
    thirds are private reasoning" are different products.

    Returns (answer, reasoning_tokens, total_tokens).
    """
    import urllib.request
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": npredict, "temperature": 0.0, "top_k": 1,
                       "stream": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    # The TOKEN BUDGET is the fair cross-model bound - a wall clock is not, because
    # the same budget takes 10x longer on a 2016 card than on a 4090. The HTTP timeout
    # exists only so a dead server cannot hang the run; it must be generous enough that
    # the budget always binds first (15000 tok at ~12 tok/s ~= 21 min; 900s fired first
    # on three T4 tasks and manufactured three "failures" that were really "cut off").
    with urllib.request.urlopen(req, timeout=2400) as fh:
        d = json.loads(fh.read().decode("utf-8", "replace"))
    choice = d["choices"][0]
    msg = choice["message"]
    text, think = strip_reasoning(msg.get("content") or "",
                                  msg.get("reasoning_content") or "")
    usage = d.get("usage") or {}
    return (text, len(think.split()), usage.get("completion_tokens", 0),
            choice.get("finish_reason") or "")


def strip_reasoning(content, reasoning_field=""):
    """Separate a thinking model's private reasoning from its actual answer.

    llama.cpp splits reasoning into its own field, but do NOT rely on that: if any build inlines
    <think>...</think> the checks would score the REASONING instead of the answer. A model that
    muses "I should reply with the JSON object {...}" and then answers wrongly would score as
    correct, because the right answer appears somewhere in the blob.

    Returns (answer, reasoning).
    """
    think = reasoning_field
    if "<think>" in content:
        end = content.find("</think>")
        inline = content[content.find("<think>"):(end + 8) if end != -1 else len(content)]
        think = (think + "\n" + inline).strip()
        content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.S)
    return content.strip(), think


def run(model, binary, extra_args, out_path, limit=None, npredict=3072, server=None,
        only=None):
    tasks = [t for t in TASKS if t[2] == "auto"] + [t for t in TASKS if t[2] == "rubric"]
    if only:
        want = {s.strip() for s in only.split(",") if s.strip()}
        tasks = [t for t in tasks if t[1] in want]
        missing = want - {t[1] for t in tasks}
        if missing:
            print(f"unknown task ids: {sorted(missing)}")
            return []
    if limit:
        tasks = tasks[:limit]
    if server:
        # PREFLIGHT. This exact prompt is the input that caught the missing chat template: the
        # raw /completion endpoint answered "I am a student who is struggling with my homework".
        # A harness that cannot tell a working endpoint from a broken one will happily report 0%
        # and call it a finding. Refuse to run unless the plumbing proves itself first.
        try:
            canary, _, _, _ = ask_server(server, "Reply with exactly the word ACKNOWLEDGED and "
                                                 "nothing else.", 256)
        except Exception as exc:
            print(f"PREFLIGHT FAILED: cannot reach {server}: {exc}")
            return []
        if "acknowledged" not in canary.lower():
            print("PREFLIGHT FAILED - the endpoint is not applying a chat template.")
            print(f"  asked for ACKNOWLEDGED, got: {canary[:120]!r}")
            print("  refusing to run: every task would fail for a harness reason, not a")
            print("  model reason, and the result would look like a finding.")
            return []
        print(f"preflight OK - endpoint returns instruction-following output ({canary[:40]!r})\n")

    # TWO RUNNERS, ONE FILE. A kill that did not take left an older run alive; it kept rewriting
    # this path while a new run wrote the same path, and the merged file looked like a finished
    # 40-task run at the OLD token budget. Nothing in the output said so. If the target file is
    # being actively written, refuse rather than interleave.
    if os.path.exists(out_path) and (time.time() - os.path.getmtime(out_path)) < 180:
        print(f"REFUSING TO RUN: {out_path} was modified "
              f"{time.time() - os.path.getmtime(out_path):.0f}s ago - another runner is "
              f"probably still writing it.")
        print("  Two runners sharing one results file produce a plausible-looking merge of")
        print("  both. Kill the other runner or pass a different --out.")
        return []

    results = []
    t0 = time.time()
    for i, t in enumerate(tasks, 1):
        cluster, tid, kind, prompt = t[0], t[1], t[2], t[3]
        print(f"  [{i:2d}/{len(tasks)}] {cluster}/{tid} ({kind})", flush=True)
        started = time.time()
        think_words, gen_tokens, finish, err = 0, 0, "", None
        if server:
            # The model is loaded ONCE and stays warm. This is also the honest way to time it:
            # llama-cli would reload 11.3 GB per task and charge that to the task.
            try:
                text, think_words, gen_tokens, finish = ask_server(server, prompt, npredict)
            except Exception as exc:
                text = ""
                err = str(exc)[:200]
                print(f"       SERVER ERROR: {exc}", flush=True)
        else:
            cmd = [binary, "-m", model, "-p", prompt, "-n", str(npredict), "-no-cnv",
                   "--temp", "0"]
            if extra_args:
                cmd += extra_args.split()
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                   encoding="utf-8", errors="replace")
                text = (r.stdout or "").strip()
            except subprocess.TimeoutExpired:
                text = ""
                print("       TIMEOUT", flush=True)
            if prompt[:40] in text:          # llama-cli echoes the prompt; drop it
                text = text.split(prompt[-60:])[-1].strip()
        elapsed = time.time() - started
        # A TRUNCATED ANSWER IS NOT A WRONG ANSWER. On a reasoning model the token budget covers
        # thinking too: at 1024 the arithmetic tasks burned the whole budget mid-thought and
        # returned empty content, which the checks scored as five confident failures. That would
        # have published "2.5-bit cannot do arithmetic" when the truth was "we cut it off".
        # Truncated tasks are quarantined, never counted as model failures either way.
        truncated = (finish == "length") and not text.strip()
        ok, detail = check_output(t, text)
        if truncated:
            ok = None
        if err and not text.strip():
            # No answer BECAUSE THE HARNESS GAVE UP (HTTP timeout, dead server) is not a wrong
            # answer. Three T4 tasks were scored FAIL at gen=0 this way. Quarantine like
            # truncation, visibly.
            ok = None
        if kind == "auto":
            verdict = "TRUNC" if truncated else ("PASS" if ok else "FAIL")
            note = (f"  budget exhausted at {gen_tokens} tokens, {think_words} words of "
                    f"reasoning, no answer emitted" if truncated
                    else "" if ok else "  failed: " + ", ".join(n for n, o in detail if not o))
            print(f"       {verdict}  ({elapsed:.0f}s){note}", flush=True)
        results.append({"cluster": cluster, "id": tid, "kind": kind, "prompt": prompt,
                        "output": text, "seconds": round(elapsed, 1),
                        "think_words": think_words, "gen_tokens": gen_tokens,
                        "finish_reason": finish, "truncated": truncated, "error": err,
                        "passed": ok, "checks": [[n, o] for n, o in detail],
                        "rubric": t[4] if kind == "rubric" else None})
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"model": model, "args": extra_args, "results": results}, fh, indent=1)
    print(f"\nwrote {out_path}  ({time.time()-t0:.0f}s total)")
    return results


def rescore(path, out_path=None):
    """Re-grade STORED outputs with the CURRENT checks. No model, no GPU, no re-run.

    This is what deterministic scoring buys. When a check turns out to be wrong - and two of
    ours were, both producing false accusations of hallucination - every past run can be
    re-graded consistently instead of being re-generated or quietly discarded.

    A task whose PROMPT has since changed is a different question, and its stored answer cannot
    be graded against the new one. Those are flagged for re-run rather than silently re-scored.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    by_id = {t[1]: t for t in TASKS}
    changed, regraded = [], 0
    for r in data["results"]:
        t = by_id.get(r["id"])
        if t is None or t[2] != "auto":
            continue
        if r.get("prompt") and r["prompt"] != t[3]:
            changed.append(r["id"])
            continue
        if r.get("truncated"):
            continue
        was = r["passed"]
        ok, detail = check_output(t, r.get("output") or "")
        r["passed"] = ok
        r["checks"] = [[n, o] for n, o in detail]
        if was != ok:
            regraded += 1
            print(f"  regraded {r['cluster']}/{r['id']}: {was} -> {ok}")
    print(f"\nre-scored with current checks: {regraded} verdict(s) changed")
    if changed:
        print(f"  {len(changed)} task(s) have a CHANGED PROMPT and cannot be re-scored from a")
        print(f"  stored answer to a different question - re-run these: {', '.join(changed)}")
        for r in data["results"]:
            if r["id"] in changed:
                r["needs_rerun"] = True
                r["passed"] = None
    dest = out_path or path
    data["rescored"] = True
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    print(f"  wrote {dest}\n")
    return changed


def score(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    res = data["results"]
    every = [r for r in res if r["kind"] == "auto"]
    if not every:
        print("no auto tasks in this file")
        return 1
    trunc = [r for r in every if r.get("truncated")]
    rerun = [r for r in every if r.get("needs_rerun") and not r.get("truncated")]
    errored = [r for r in every if r.get("error") and r["passed"] is None
               and not r.get("truncated")]
    errored = [r for r in every
               if r.get("error") and r["passed"] is None and not r.get("truncated")]
    scorable = [r for r in every
                if not r.get("truncated") and not r.get("needs_rerun")
                and r["passed"] is not None]
    # THE HEADLINE IS THE STAKED SET ONLY. Tier 3/4 tasks were added 2026-08-03, after the
    # 80%/60% bar was staked for the original 40. Folding them in would move the goalpost in
    # whichever direction they happen to score, so they are reported separately, always.
    auto = [r for r in scorable if r["id"] in STAKED_IDS]
    ext = [r for r in scorable if r["id"] not in STAKED_IDS]
    passed = sum(1 for r in auto if r["passed"])
    pct = 100.0 * passed / len(auto) if auto else 0.0
    print(f"\nMODEL: {data.get('model')}")
    print(f"ARGS : {data.get('args')}\n")
    if auto:
        print(f"STAKED SET: {passed}/{len(auto)} pass = {pct:.1f}%"
              + (f"   (+{len(ext)} extension tasks reported below, outside the staked bar)"
                 if ext else "") + "\n")
    else:
        print(f"STAKED SET: none in this file ({len(ext)} extension tasks below)\n")
    if trunc:
        # Do NOT bury this. Quarantining shrinks the denominator, and a shrinking denominator is
        # exactly how a headline gets flattered. State the count and the worst case out loud.
        worst = 100.0 * passed / len(every)
        print(f"  !! {len(trunc)} of {len(every)} tasks TRUNCATED (budget exhausted before an")
        print(f"     answer was emitted) and are excluded from the {pct:.1f}% above.")
        print(f"     If every truncated task were counted as a failure the score would be")
        print(f"     {worst:.1f}%. Truncated: " + ", ".join(f"{r['cluster']}/{r['id']}"
                                                            for r in trunc))
        print("     Raise --npredict and re-run before quoting either number.\n")
    if rerun:
        print(f"  !! {len(rerun)} task(s) had their PROMPT corrected after this run, so the")
        print(f"     stored answer replies to a different question and cannot be graded:")
        print(f"     " + ", ".join(f"{r['cluster']}/{r['id']}" for r in rerun))
        print("     These are excluded and must be re-run.\n")
    if errored:
        print(f"  !! {len(errored)} task(s) got NO ANSWER for a HARNESS reason (HTTP timeout /")
        print(f"     server error) and are excluded - neither passes nor failures:")
        for r in errored:
            print(f"       {r['cluster']}/{r['id']}: {(r.get('error') or '')[:70]}")
        print("     Re-run with a live server / longer harness timeout before judging them.\n")
    by = {}
    for r in auto:
        by.setdefault(r["cluster"], []).append(r["passed"])
    print("  per cluster:")
    for c, v in sorted(by.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"    {c:16} {sum(v):2d}/{len(v):2d}  {100*sum(v)/len(v):5.1f}%")
    print()
    # THE DIFFICULTY LADDER. One row per tier, over everything scorable in this file. This is
    # the cross-model comparison surface: identical inputs, identical predicates, so two models'
    # rows are directly comparable. Tier 4 is a ceiling by design - 0/6 is the expected result
    # for every model that exists today, and the tier is there so that the first model to score
    # above zero on it does so against a bar that predates it.
    byt = {}
    for r in scorable:
        byt.setdefault(TIER.get(r["id"], 0), []).append(r["passed"])
    if byt:
        print("  difficulty ladder (staked + extension, scorable tasks):")
        names = {1: "T1 routine ", 2: "T2 standard", 3: "T3 hard    ", 4: "T4 ceiling "}
        for tno in sorted(byt):
            v = byt[tno]
            note = "  <- 0 expected today; ranks future models" if tno == 4 else ""
            print(f"    {names.get(tno, str(tno)):11} {sum(v):2d}/{len(v):2d}"
                  f"  {100*sum(v)/len(v):5.1f}%{note}")
        t4t = [r for r in trunc if TIER.get(r["id"], 0) == 4]
        if t4t:
            print(f"    (plus {len(t4t)} tier-4 truncation(s) - burned the whole budget "
                  "without an answer; listed above)")
        print()
    fails = [r for r in auto if not r["passed"]]
    if fails:
        print("  failures:")
        for r in fails:
            why = ", ".join(n for n, o in r["checks"] if not o)
            print(f"    {r['cluster']}/{r['id']:4} {why}")
    print()
    if not auto:
        # A tier-only or empty file has NO staked tasks. 0/0 is not 0% - an earlier version
        # printed "KILL RULE FIRED (0.0%)" over an empty denominator, a confident verdict
        # about evidence that does not exist.
        print("  STAKED VERDICT: not applicable - this file contains no staked-set tasks.")
    elif pct >= 80:
        print(f"  VERDICT: P1 CONFIRMED ({pct:.1f}% >= 80%) - business-useful.")
    elif pct < 60:
        print(f"  VERDICT: KILL RULE FIRED ({pct:.1f}% < 60%). This config is NOT")
        print("           business-usable. Every tok/s figure we publish for it must be")
        print("           qualified at equal prominence in the README.")
    else:
        print(f"  VERDICT: QUALIFIED ({pct:.1f}%) - drafting aid, not send-ready.")
    rub = [r for r in res if r["kind"] == "rubric"]
    print(f"\n  ({len(rub)} rubric tasks captured for separate human review - "
          "NOT counted above)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", metavar="MODEL")
    ap.add_argument("--bin", default="llama-cli")
    ap.add_argument("--args", default="", help="extra llama-cli flags, quoted")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="comma-separated task ids, e.g. a2,i3,p4 - for re-running the tasks that truncated at a bigger budget")
    ap.add_argument("--npredict", type=int, default=3072,
                    help="token budget per task. Must cover THINKING plus the answer on a "
                         "reasoning model - at 16 it returned empty content on every task.")
    ap.add_argument("--server", help="llama-server base URL, e.g. http://127.0.0.1:8080 "
                                     "(model stays warm; strongly preferred)")
    ap.add_argument("--score", metavar="RESULTS.json")
    ap.add_argument("--rescore", metavar="RESULTS.json",
                    help="re-grade stored outputs with the current checks")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.rescore:
        rescore(a.rescore)
        return score(a.rescore)
    if a.score:
        return score(a.score)
    if a.list:
        for c, ts in clusters().items():
            n_auto = sum(1 for t in ts if t[2] == "auto")
            print(f"\n{c.upper()}  ({len(ts)} tasks, {n_auto} auto-scored)")
            for t in ts:
                marker = "auto  " if t[2] == "auto" else "rubric"
                print(f"  [{marker}] {t[1]:4} {t[3][:80]}")
        n = sum(1 for t in TASKS if t[2] == "auto")
        print(f"\n{len(TASKS)} tasks: {n} auto-scored (headline), {len(TASKS)-n} rubric")
        return 0
    if a.run:
        out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "data", "business_tasks_results.json")
        if selftest():
            print("refusing to run: the checks failed their own self-test")
            return 1
        print()
        got = run(a.run, a.bin, a.args, out, a.limit, server=a.server,
                  npredict=a.npredict, only=a.only)
        if not got:
            # Do NOT fall through to score(). It reads the results FILE, so on an aborted run it
            # happily scores whatever a previous run left there - which is how a preflight
            # failure just printed "KILL RULE FIRED (33.3%)" from stale data. A verdict from a
            # run that never happened is the worst possible output of a pre-registered test.
            print("no results produced - NOT scoring (a stale results file would score itself)")
            return 1
        return score(out)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
