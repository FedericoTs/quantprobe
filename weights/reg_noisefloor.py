"""Register the ladder noise floor + the C-17 refinement. Run once.
  python weights/reg_noisefloor.py
"""
import json

P = "findings/REGISTER.json"
d = json.load(open(P, encoding="utf-8"))
n = max(int(x["id"][2:]) for x in d["contradictions"] if x["id"].startswith("C-")) + 1

d["contradictions"].append({
    "id": f"C-{n}",
    "kind": "contradiction",
    "status": "open",
    "confidence": "measured",
    "claim": (
        "THE 14-ROW LADDER MEDIAN HAS A RUN-TO-RUN NOISE FLOOR OF ROUGHLY +/-1 POINT, SO EVERY "
        "SUB-2-POINT 'IMPROVEMENT' THIS PROJECT HAS CLAIMED FROM IT IS UNSUPPORTED - INCLUDING TWO "
        "I MADE TODAY. Four ladders measured 2026-07-31 on the SAME idle machine with functionally "
        "equivalent calibrations returned medians of 7.2%, 7.9%, 8.8% and 9.0%. The final pair is "
        "the cleanest evidence: cal c24a253b (7.9%) and cal 2dc97d41 (9.0%) differ only in a disk "
        "figure that NONE of the 14 rows reads, yet the median moved 1.1 points and individual "
        "rows wandered 1-2 points in scattered directions (gemma4-12B measured 13.23 then 12.25 "
        "tok/s - a MEASUREMENT difference, not a prediction one). RETRACTED AT EQUAL PROMINENCE: "
        "I reported '7.2% beats the 8.8% baseline' and later '7.9%, beating the 8.8% baseline' as "
        "wins. Neither clears the noise floor. The ladder cannot resolve improvements below about "
        "2 points, and any future claim from it must either exceed that or report repeated runs "
        "with a spread. This does NOT touch the large effects - RAM-only calibration at 12.5% and "
        "uncalibrated at 27.2% are far outside the floor and stand."),
    "magnitude": "four same-machine ladders span 7.2-9.0% median (1.8 points) with no real change between them",
    "evidence": (
        "weights/data/ladder_20260731_{postreboot_stalecal,uncalibrated,ramonly_37a91948,"
        "c24a253b_partialdisk}.json plus the locked 2dc97d41 ladder; all idle-gated, same box"),
    "scope": (
        "This ladder, this machine, single runs per state (r>=2 within llama-bench but ONE ladder "
        "pass per calibration). The +/-1 point figure is the observed spread across four passes, "
        "not a computed confidence interval - a proper estimate needs repeated passes under one "
        "fixed calibration, which has never been done."),
    "wired_into": (
        "nothing - this is a reporting discipline: never quote a ladder median difference under 2 "
        "points as an improvement without repeated passes. Also argues for adding a DISK-TIER row "
        "to the ladder: all 14 current rows are VRAM or RAM-split, which is precisely how C-17's "
        "6.8x-wrong disk probe survived undetected - a component no validation row reads can stay "
        "wrong indefinitely."),
})

c17 = [x for x in d["contradictions"] if x["id"] == "C-17"][0]
c17["claim"] += (
    " [REFINED by the 2dc97d41 ladder: 'calibration must be COMPLETE' is too coarse. Correcting "
    "the disk figure from 2.99 to 0.47 GB/s moved the ladder median 7.9% -> 9.0%, i.e. within "
    "noise (see the noise-floor entry), because NONE of the 14 rows reads the disk tier. The "
    "sharper rule: EVERY COMPONENT YOUR PREDICTION PATH ACTUALLY READS must be measured, and a "
    "component nobody reads can stay wrong indefinitely without surfacing. That is exactly how a "
    "6.8x error survived - the RAM-only catastrophe (12.5%) hurt because every row reads RAM and "
    "GPU. It also exposes a ladder COVERAGE gap: we ship disk-tier advice and validate it with "
    "zero disk-tier rows.]")

json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"C-{n} registered; C-17 refined")
