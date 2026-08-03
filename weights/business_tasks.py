"""50 business tasks, clustered by what a company actually needs done.

  python weights/business_tasks.py --list          # inspect the set
  python weights/business_tasks.py --run MODEL     # generate outputs for blind scoring

WHY THIS EXISTS. quantprobe recommends a 2.5-bit 30B and headlines 22.6 tok/s. Nobody has
checked whether that config does business work. The README says outright that we measure
perplexity and KL only - and after L-27 we know perplexity is blunt: it moved 23% while the
model changed its chosen token on 27% of positions. Neither number tells you whether the email
was sendable.

The clusters are chosen so a FAILURE IS DIAGNOSTIC, not just a lower score. Quantization damage
is not uniform: long-form generation, strict formatting, arithmetic and instruction-following
degrade at different rates. If 2.5-bit holds up on summarisation but collapses on structured
extraction, that is a far more useful finding than one aggregate percentage - it tells a user
which jobs the cheap quant is safe for.

SCORING IS BLIND AND THE BAR IS SET FIRST. Outputs are shuffled with provenance stripped, and
the pass mark is written before any output exists:

  P1  >= 80% usable with at most trivial edits  -> the recommended config is business-useful
  KILL  < 60%                                   -> 2.5-bit is NOT business-usable, and every
        tok/s figure we publish for it is a speed number for a model that cannot do the work.
        The README headline gets qualified at the same prominence as the speed claim.
  60-80%                                        -> qualified: fine for drafting, not for sending

Blindness matters because I have twice this week let knowing-which-run-was-clean contaminate a
reading. A judge who knows which output is "ours" is not scoring the output.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

# (cluster, id, prompt, what "usable" means for THIS task)
TASKS = [
    # --- 1. CORRESPONDENCE: the highest-volume business use of an LLM ------------------
    ("correspondence", "c1", "Write a short email declining a vendor's proposal for a CRM "
     "migration because the timeline slips past our Q4 freeze. Keep the door open for Q1.",
     "sendable with at most a name/date edit; declines clearly; leaves the door open"),
    ("correspondence", "c2", "Draft a follow-up email to a client who has not responded in "
     "two weeks about an unsigned SOW. Polite, not passive-aggressive, one clear ask.",
     "one unambiguous ask; tone is professional; no guilt-tripping"),
    ("correspondence", "c3", "Write an internal announcement that Friday's all-hands is moved "
     "to Monday 10:00 because the CEO is travelling. Three sentences maximum.",
     "under 3 sentences; states old time, new time, and reason"),
    ("correspondence", "c4", "Reply to a customer complaining that our invoice charged them "
     "for a month they had already cancelled. Acknowledge, commit to a refund, give a timeline.",
     "acknowledges without admitting liability language; concrete next step and timeline"),
    ("correspondence", "c5", "Write a LinkedIn message to a candidate for a senior backend "
     "role. Two sentences on why them specifically, one clear call to action.",
     "specific rather than generic; single CTA; under 80 words"),

    # --- 2. SUMMARISATION: where hallucination is most expensive -----------------------
    ("summarisation", "s1", "Summarise in 5 bullets: 'Q3 revenue was 4.2M, up 12% YoY but 8% "
     "below plan. Churn rose to 3.1% monthly, driven by SMB. Enterprise ARR grew 34%. Sales "
     "headcount is 22, up from 17. CAC payback lengthened from 14 to 19 months. The board asked "
     "for a revised FY plan by 15 November.'",
     "5 bullets; every number correct; nothing invented"),
    ("summarisation", "s2", "The following are three support tickets: (1) login loop on Safari "
     "after SSO, (2) invoice PDF shows wrong VAT for Irish customers, (3) CSV export truncates "
     "at 10k rows. Write a one-paragraph engineering standup summary grouping them by severity.",
     "all three preserved; severity ordering is defensible; one paragraph"),
    ("summarisation", "s3", "Condense this to a single sentence for a board slide: 'We evaluated "
     "four vendors, shortlisted two, ran a 6-week pilot with both, and selected Vendor B on "
     "total cost of ownership despite Vendor A scoring higher on features.'",
     "one sentence; keeps the decision AND the reason"),
    ("summarisation", "s4", "Summarise the risk in this clause for a non-lawyer: 'Either party "
     "may terminate for convenience on 30 days notice, provided that Customer shall remain "
     "liable for all fees accrued through the effective date of termination and for any "
     "committed minimum not yet consumed.'",
     "plain English; correctly identifies the committed-minimum exposure"),
    ("summarisation", "s5", "Three sentences: what changed, who is affected, what they must do. "
     "'Starting 1 December, API v1 is deprecated. v2 requires OAuth2 instead of API keys. "
     "Existing keys stop working 1 March. Migration guide is at docs/migrate-v2.'",
     "exactly the three requested facts, in the requested order"),

    # --- 3. STRUCTURED EXTRACTION: strict format, the classic quantisation failure -----
    ("extraction", "e1", "Return ONLY valid JSON with keys name, email, company, role from: "
     "'Hi, I'm Marta Ruiz, Head of Ops at Delta Logistics — reach me at m.ruiz@deltalog.es'",
     "parses as JSON; all four fields correct; NO prose around it"),
    ("extraction", "e2", "Extract every monetary amount and its currency as a JSON array from: "
     "'The contract is 45,000 EUR upfront, 3,200 EUR monthly, with a 10,000 USD success fee.'",
     "3 items; amounts and currencies correct; valid JSON"),
    ("extraction", "e3", "Return ONLY a JSON object mapping each weekday to the meeting on it: "
     "'Standup is Monday and Wednesday 9am, retro Friday 4pm, no meetings Tuesday or Thursday.'",
     "valid JSON; Tuesday/Thursday handled explicitly, not omitted silently"),
    ("extraction", "e4", "From this sentence output ONLY a CSV line 'severity,component,owner': "
     "'Critical: the payments service is dropping webhooks, assigned to the platform team.'",
     "one CSV line; no header; no quotes around the whole line; no prose"),
    ("extraction", "e5", "Return ONLY JSON {\"decision\": \"approve\"|\"reject\", \"reason\": "
     "string}: 'The applicant has 4 years experience against a 5 year minimum, but holds the "
     "required certification and an internal referral.'",
     "valid JSON; decision is one of the two allowed literals"),

    # --- 4. ANALYSIS + ARITHMETIC: where low-bit models are known to slip --------------
    ("analysis", "a1", "A subscription costs 49 EUR/month or 490 EUR/year. What is the annual "
     "discount as a percentage? Show the calculation.",
     "16.7% (or 16.67%); arithmetic shown and correct"),
    ("analysis", "a2", "We have 22 sales reps, each closing 1.4 deals/month at 8,500 EUR average. "
     "What is monthly bookings, and what headcount hits 400k/month?",
     "261,800 EUR/month; ~34 reps; both derivable from the shown work"),
    ("analysis", "a3", "Churn is 3.1% monthly. What is the approximate annual retention rate?",
     "~68-69%; method (1-0.031)^12 is visible or implied correctly"),
    ("analysis", "a4", "Server costs 1,200 EUR/month, handles 40k requests/day. A competitor "
     "charges 0.002 EUR/request. At what daily volume does the competitor become cheaper?",
     "~20,000 requests/day; correctly identifies the break-even direction"),
    ("analysis", "a5", "CAC is 4,200 EUR, ARPU 180 EUR/month, gross margin 78%. What is the CAC "
     "payback in months?",
     "~30 months; margin correctly applied (not 23 months from ignoring it)"),

    # --- 5. CLASSIFICATION + ROUTING: short output, high volume, easy to grade ---------
    ("classification", "l1", "Classify as BUG, FEATURE, or QUESTION, one word only: 'Can I "
     "export to Excel instead of CSV?'", "exactly one of the three words, nothing else"),
    ("classification", "l2", "Classify sentiment POSITIVE/NEUTRAL/NEGATIVE, one word: 'The "
     "onboarding took longer than promised but support was excellent.'",
     "one word; NEUTRAL or POSITIVE both defensible; NEGATIVE is wrong"),
    ("classification", "l3", "Route to SALES, SUPPORT, BILLING or LEGAL, one word: 'Our DPA "
     "needs updating for the new sub-processor.'", "LEGAL; one word only"),
    ("classification", "l4", "Priority P1/P2/P3, one token: 'Checkout is down for all users.'",
     "P1; single token"),
    ("classification", "l5", "Is this GDPR-relevant, YES or NO: 'We store customer email "
     "addresses in an EU datacentre.'", "YES; single word"),

    # --- 6. CODE + TECHNICAL: the second-highest-volume real use -----------------------
    ("code", "k1", "Write a Python function that takes a list of dicts with a 'date' string "
     "(ISO) and returns them sorted newest first. No imports beyond stdlib.",
     "runs; correct order; no syntax errors"),
    ("code", "k2", "Write a SQL query: total revenue per customer for 2026, only customers with "
     "more than 3 orders, highest first. Tables: orders(id, customer_id, total, created_at).",
     "valid SQL; HAVING used correctly; year filter present"),
    ("code", "k3", "This regex is meant to match EU VAT numbers but rejects valid Irish ones: "
     "^[A-Z]{2}[0-9]{8,12}$. Explain why and give a corrected version.",
     "identifies that IE VAT contains letters; corrected pattern actually allows them"),
    ("code", "k4", "Write a bash one-liner to find all .log files over 100MB modified in the "
     "last 7 days.", "syntactically valid find command; both conditions present"),
    ("code", "k5", "Explain in three sentences what this does and one risk: "
     "'docker run -v /:/host -it alpine sh'",
     "identifies the host-root mount and the security risk"),

    # --- 7. INSTRUCTION FOLLOWING: constraint adherence, degrades early ----------------
    ("instruction", "i1", "Reply with exactly three words describing a good API.",
     "exactly three words, no punctuation commentary"),
    ("instruction", "i2", "List 4 project risks. Every line must start with 'RISK:' and be under "
     "12 words.", "4 lines; all prefixed; all under 12 words"),
    ("instruction", "i3", "Rewrite this in exactly 20 words: 'Our platform helps mid-market "
     "logistics companies reduce empty miles by matching return loads automatically.'",
     "exactly 20 words when counted"),
    ("instruction", "i4", "Answer only with a number: how many days in the third quarter of a "
     "non-leap year?", "92, with no prose"),
    ("instruction", "i5", "Write a product tagline. Do NOT use the words 'AI', 'revolutionary', "
     "'seamless' or 'powerful'.", "a tagline; none of the four banned words present"),

    # --- 8. LONG-FORM: where degradation compounds over the output ---------------------
    ("longform", "g1", "Write a 300-word section for an internal wiki explaining our incident "
     "severity levels P1-P4, with a one-line example for each.",
     "coherent to the end; all four levels; no repetition loop"),
    ("longform", "g2", "Draft a 250-word job description for a Customer Success Manager at a "
     "B2B SaaS company, including responsibilities and requirements.",
     "both sections present; no drift into unrelated content"),
    ("longform", "g3", "Write a 200-word explanation of why we chose PostgreSQL over MongoDB, "
     "for a non-technical stakeholder.",
     "stays non-technical throughout; gives at least two concrete reasons"),
    ("longform", "g4", "Write meeting minutes (200 words) from these notes: 'discussed Q4 "
     "hiring, agreed 3 engineers 1 designer, budget approved 480k, start Jan, Sara owns "
     "req writing, revisit in 2 weeks'.",
     "all six facts preserved; reads as minutes; no invented attendees"),
    ("longform", "g5", "Write a 250-word competitor comparison between a self-hosted and a SaaS "
     "deployment model, balanced, for a CTO audience.",
     "genuinely balanced; no drift; ends cleanly rather than trailing off"),

    # --- 9. MULTILINGUAL: common in EU business, degrades fast at low bits -------------
    ("multilingual", "m1", "Translate to Italian, business register: 'We regret that we cannot "
     "accommodate the requested discount, but we can offer extended payment terms.'",
     "correct Italian; business register; meaning preserved"),
    ("multilingual", "m2", "Reply in French to: 'Pouvez-vous confirmer la date de livraison?' "
     "Confirm delivery for 12 March.", "correct French; date correct"),
    ("multilingual", "m3", "Translate to German: 'Please find the signed contract attached.'",
     "correct German; idiomatic not literal"),
    ("multilingual", "m4", "Summarise in Spanish, two sentences: 'The pilot succeeded, we are "
     "expanding to three more sites in January, budget is approved.'",
     "correct Spanish; two sentences; all three facts"),
    ("multilingual", "m5", "Write a one-line out-of-office in both English and Italian for "
     "10-20 August.", "both languages; dates correct in both"),

    # --- 10. JUDGEMENT: no single right answer, tests whether output is USEFUL ---------
    ("judgement", "j1", "A customer asks for a 40% discount. Our floor is 15%. Give three "
     "negotiation options that are not simply saying no.",
     "three DISTINCT options; all respect the 15% floor"),
    ("judgement", "j2", "Our deploy takes 45 minutes and blocks the team. Suggest three "
     "improvements ordered by effort-to-impact.",
     "three concrete suggestions; ordering is justified"),
    ("judgement", "j3", "Two engineers disagree on whether to rewrite or refactor a legacy "
     "service. What three questions would settle it?",
     "questions are decision-relevant, not generic"),
    ("judgement", "j4", "We missed a customer SLA by 4 hours. Draft the three-part response: "
     "what happened, what we are doing, what changes.",
     "all three parts; no over-promising; no blame-shifting"),
    ("judgement", "j5", "Rank these by business risk and justify in one line each: expired SSL "
     "cert, an unpatched CVE in a dev box, a departing employee with prod access.",
     "ranking is defensible; each justification is specific"),
]


def clusters():
    out = {}
    for c, i, p, u in TASKS:
        out.setdefault(c, []).append((i, p, u))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", metavar="GGUF")
    ap.add_argument("--bin", default=None, help="llama-cli path")
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.list or not a.run:
        cl = clusters()
        print(f"{len(TASKS)} tasks in {len(cl)} clusters\n")
        for c, items in cl.items():
            print(f"  {c:16} {len(items)} tasks")
        print("\nClusters are chosen so a failure is DIAGNOSTIC: extraction and instruction")
        print("adherence break before summarisation does, and long-form degrades last but")
        print("worst. An aggregate score would hide which jobs the cheap quant is safe for.")
        return 0

    binp = a.bin or "llama-cli"
    outp = a.out or f"weights/data/tasks_{os.path.basename(a.run)}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    res = {"model": a.run, "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
           "outputs": []}
    for n, (c, i, prompt, usable) in enumerate(TASKS, 1):
        print(f"  [{n:2}/{len(TASKS)}] {c}/{i}", flush=True)
        try:
            p = subprocess.run([binp, "-m", a.run, "-ngl", a.ngl, "-no-cnv", "-n", "400",
                                "--temp", "0", "-p", prompt],
                               capture_output=True, text=True, errors="replace", timeout=600)
            text = p.stdout
        except subprocess.TimeoutExpired:
            text = "<<TIMEOUT>>"
        res["outputs"].append({"cluster": c, "id": i, "prompt": prompt,
                               "usable_means": usable, "output": text})
        json.dump(res, open(outp, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {outp}")
    print("Score BLIND: strip 'model', shuffle, and judge each against its own usable_means.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
