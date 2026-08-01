"""Register the first scored disk-tier measurement and the machine-state shift it exposed.

  python weights/reg_disktier_scored.py
"""
import json

P = "findings/REGISTER.json"
d = json.load(open(P, encoding="utf-8"))
n = max(int(x["id"][2:]) for x in d["contradictions"] if x["id"].startswith("C-")) + 1

EV = ("weights/data/unattended_20260801_002809_RESULT.json (scored verdict, gate readings, "
      "per-rep tok/s), _stake_addendum.json, _PRIMARY_laguna_tg64.log, _KR2_laguna_tg32.log, "
      "_KR3_control_7B_ngl0.log; stake frozen at weights/data/disktier_20260731_1857_staked.json; "
      "runner weights/unattended_serial.py")

d["contradictions"].append({
    "id": f"C-{n}",
    "kind": "contradiction",
    "status": "open",
    "confidence": "measured",
    "claim": (
        "THE DISK TIER IS PESSIMISTIC BY 30%: THE FIRST DISK-TIER MEASUREMENT THIS PROJECT HAS "
        "EVER TAKEN MISSED ITS OWN STAKED BAND, AND MISSED IT UPWARD. Predicted 0.331505 tok/s "
        "for Laguna-S-2.1-UD-Q2_K_XL (39.685 GB, 117.6B total / 8.14B active, -ngl 0) against a "
        "staked +/-25% band [0.265204, 0.442007]. Measured 0.476225 tok/s: err_pct -30.39%, "
        "OUTSIDE the band. KR1 FAIL, published at equal prominence and not renegotiated. "
        "THE MISS IS TRUSTWORTHY BECAUSE ALL THREE GUARD RULES PASSED FIRST - this is the "
        "difference between this run and the 2026-07-31 attempt, which produced a number 2.5% "
        "inside the band that had to be thrown away. KR3 (harness can vary): control 7B at "
        "-ngl 0 measured 7.570 tok/s against a 7.09 anchor and a >3.0 threshold, reps "
        "7.5353/7.6048. KR4 (steady state): rep spread 1.172x (reps 0.5122/0.4369/0.4796) "
        "against a >2x kill rule - the contaminated run scored 2.09x here. KR2 (decode not "
        "load): tg32 0.449311 vs tg64 0.476225, 5.7% on one denominator and 6.0% on the other, "
        "so the ambiguity that could not be resolved on 2026-07-31 is moot at this margin. "
        "One cal_id (2dc97d41) at both ends, no C-14 warning, gates opened on the first attempt "
        "with CPU mean 0.2%/max 2.0% and 14.61 GB free. INVERTING THE TERM: the I/O component "
        "is 1.485x faster than modelled (predicted 2.66599 s of a 2.86572 s token; measured "
        "token 1.99486 s implies 1.79513 s of I/O). That resolves to exactly one of two things, "
        "and this measurement CANNOT distinguish them: either effective disk bandwidth under "
        "llama.cpp's sequential streaming is ~0.698 GB/s rather than the 0.47 GB/s the "
        "random-offset cold probe measures - i.e. the probe measures the wrong ACCESS PATTERN, "
        "a different error from the page-cache one C-17 fixed - or the page-cache miss fraction "
        "is ~0.429 rather than the modelled 0.637, which is what expert-usage skew would "
        "predict if hot experts stay resident and never pay disk."),
    "magnitude": "predicted 0.331505 vs measured 0.476225 tok/s (-30.4%); I/O term 1.485x fast",
    "evidence": EV,
    "scope": (
        "ONE row, ONE model, ONE machine, ctx=0. This is a single point on a tier that had zero "
        "points an hour ago, so it establishes DIRECTION and rough magnitude and nothing more - "
        "it does not give a corrected coefficient, and fitting one to a single datapoint would "
        "repeat the error C-02 exists to warn about. Not covered: any disk-tier row at nonzero "
        "context depth (plan.py carries a KV-residency-deficit disclosure that stays untested "
        "here, KV ~12 MB at ctx 0); the printed WINNER row 'stream from disk (VRAM+RAM expert "
        "cache)' at 0.367424 tok/s, which needs a ktransformers/colibri-class runtime this box "
        "does not have and remains unmeasurable by nature rather than by scheduling. The two "
        "candidate mechanisms are DISTINGUISHABLE by experiment and have not been distinguished: "
        "an access-pattern probe (sequential vs random at equal size) separates them cleanly."),
    "wired_into": (
        "nothing yet - deliberately. The honest next step is the discriminating experiment, not "
        "a coefficient change: (a) probe sequential-vs-random read at matched size to test the "
        "access-pattern arm, (b) measure expert-usage skew on the MoE rows (task #52) against "
        "the 0.429 implied miss fraction. Whichever survives gets wired; until then plan.py "
        "keeps under-predicting the disk tier and the CHANGELOG says so."),
})

d["contradictions"].append({
    "id": f"C-{n + 1}",
    "kind": "contradiction",
    "status": "open",
    "confidence": "measured",
    "claim": (
        "A LADDER MEDIAN CAN HOLD STILL WHILE EVERY SINGLE ROW UNDER IT MOVES. On 2026-08-01, "
        "on a box scrubbed idle (CPU mean 0.7%, max 2.0%, 14.11 GB free, gate open on the first "
        "attempt), ALL 14 ladder rows measured FASTER than the reference ladder taken under the "
        "same cal_id 2dc97d41 - not 13 of 14, all of them. Median +4.6%, range +1.4% to +27.5%. "
        "This is measured-vs-measured; prediction plays no part. Yet the median |err| moved only "
        "9.0% -> 8.4%, because the errors are a MIX of signs and a uniform speed-up improves the "
        "rows we over-predict while worsening the rows we under-predict. The median was stable "
        "BY CANCELLATION, not because the machine was. CONSEQUENCE FOR C-18: the +/-1 point "
        "noise floor is real but it is the wrong instrument here - a median inside the floor is "
        "NOT evidence that the machine state is unchanged, and 'no regression' can be reported "
        "truthfully while the whole measurement basis has shifted underneath. The per-row "
        "measured-vs-measured diff is the sensitive detector; the median is not. RELEASE "
        "CONSEQUENCE, recorded so it cannot be quietly ignored: the 8.4% median PASSES its "
        "staked band [6.8, 10.8] and is being reported as UNCHANGED, not as an improvement, per "
        "C-18. SEPARATE DEFECT SURFACED: gemma4-12B has now returned 13.23, 12.25 and 15.62 "
        "tok/s across three runs on this box - a 27% spread. That row is not measuring a stable "
        "quantity and its ladder entry should not be trusted until it is."),
    "magnitude": "14/14 rows faster, median +4.6% (min +1.4%, max +27.5%), while median |err| moved 0.6 points",
    "evidence": ("weights/data/unattended_20260801_002809_ladder_result.json vs "
                 "_ladder_baseline_backup.json (both cal 2dc97d41); gate readings in "
                 "_RESULT.json; runner weights/unattended_serial.py"),
    "scope": (
        "One machine, one pair of ladder passes. The DIRECTION is unambiguous (14/14 is not a "
        "coin flip) but the CAUSE is not established: the idle scrub, thermal state, page-cache "
        "warmth and Windows service state all changed together and were not varied "
        "independently. It is NOT established that the baseline was contaminated - only that the "
        "two passes are different machine states. Which one better represents a user's box is "
        "an open question with a practical answer: the scrubbed state is the ceiling, and a user "
        "with a browser open will not reach it."),
    "wired_into": (
        "reporting discipline, immediately: (a) every ladder comparison must print the per-row "
        "measured-vs-measured diff alongside the median, since the median can hide a uniform "
        "shift; (b) published headline speeds stay at the reproducible-by-others range, NOT the "
        "scrubbed-box ceiling - README's 20.4-22.2 for Qwen3-30B-A3B is retained even though "
        "this pass measured 22.94, because a user cannot reproduce a number that required "
        "stopping services to obtain."),
})

json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"C-{n} (disk tier missed upward) and C-{n + 1} (14/14 state shift) registered")
