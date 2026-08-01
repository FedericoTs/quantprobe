"""Register prereg #94: access-pattern hypothesis refuted, disk calibration confirmed.

  python weights/reg_exp94.py
"""
import json

P = "findings/REGISTER.json"
d = json.load(open(P, encoding="utf-8"))
n = max(int(x["id"][2:]) for x in d["dead_ends"] if x["id"].startswith("D-")) + 1

d["dead_ends"].append({
    "id": f"D-{n}",
    "kind": "dead_end",
    "status": "resolved",
    "confidence": "measured",
    "claim": (
        "THE ACCESS-PATTERN EXPLANATION FOR THE DISK-TIER MISS IS DEAD, AND KILLING IT CONFIRMED "
        "THE DISK CALIBRATION IS CORRECT. C-21 measured the disk tier 30% faster than predicted "
        "and named two candidate causes it could not separate: H1 the probe's random-offset draw "
        "under-reports what sequential streaming achieves, or H2 the page-cache miss fraction is "
        "lower than modelled. Prereg #94 staked the discriminator BEFORE running, with the "
        "inconclusive band declared in advance (>=1.30x supports H1, <=1.10x refutes it, between "
        "is claimed by neither) so the answer could not be chosen for convenience. RESULT: "
        "scattered reads (8 x 512 MB at random offsets) 0.452 GB/s; one contiguous 4 GB read "
        "0.459 GB/s; ratio 1.015x. H1 REFUTED - this drive does not care about access pattern at "
        "this size. THE MORE VALUABLE HALF IS THE POSITIVE ONE: both arms land within 4% of the "
        "0.47 GB/s that `quantprobe calibrate` measures, so the C-17-corrected disk figure is now "
        "INDEPENDENTLY CONFIRMED by a probe with a working falsifier - the cannot-vary guard read "
        "2.802 GB/s on a warm re-read, well clear of its 2.0 threshold, proving the harness "
        "distinguishes RAM from disk rather than emitting a constant. The v1.24.0 changelog said "
        "'proving the old number wrong is not the same as proving the new one right'; the new one "
        "is now right. BY ELIMINATION C-21's 30% MISS BELONGS TO THE MISS FRACTION: the law "
        "assumes 0.637 of each token's bytes are re-read from disk, and the measurement implies "
        "~0.429. That is what expert-usage skew predicts - if MoE routing concentrates on a hot "
        "subset, those experts stay page-cached permanently and never pay disk."),
    "magnitude": "SEQ/RND 1.015x (0.459 vs 0.452 GB/s); both within 4% of the calibrated 0.47",
    "evidence": ("weights/exp94_access_pattern.py (stake in module docstring), "
                 "weights/data/exp94_access_pattern.json"),
    "scope": (
        "One drive (D:, SATA SSD), one file, 512 MB and 4 GB spans, cold regions assigned "
        "disjointly so no arm warms another. It says nothing about NVMe or Gen4 drives, where "
        "queue depth and pattern may well matter, and nothing about spans far below 512 MB. It "
        "also does NOT establish that llama.cpp reads sequentially - llama.cpp mmaps and touches "
        "expert tensors scattered through the file. SEQ and RND bracket that behaviour rather "
        "than reproducing it, and the finding is that the bracket is narrow enough for the "
        "distinction not to matter here. ELIMINATION IS NOT DIRECT EVIDENCE: H2 now stands "
        "because its only rival is dead, not because anyone has measured expert-usage skew. "
        "That measurement (task #52) is still owed, and the 0.429 implied miss fraction is the "
        "number it has to hit."),
    "wired_into": (
        "nothing in the shipped code - correctly. The probe is sound, so measure_disk() needs no "
        "change, and the miss-fraction term must not be re-fitted from one datapoint. Next: "
        "task #52 measures expert-usage skew directly against the 0.429 target. If skew explains "
        "it, the disk-tier term gains a residency-aware miss fraction and the same mechanism "
        "feeds the hot-expert caching work (task #55)."),
})

c21 = [x for x in d["contradictions"] if x["id"] == "C-21"][0]
c21["claim"] += (
    f" [RESOLVED HALFWAY by prereg #94, 2026-08-01: the access-pattern arm is REFUTED (see "
    f"D-{n}) - scattered and contiguous reads measure 0.452 vs 0.459 GB/s, a 1.015x ratio, and "
    f"both sit within 4 percent of the calibrated 0.47, which independently CONFIRMS the C-17 "
    f"correction. The 30 percent miss therefore belongs to the miss fraction: modelled 0.637, "
    f"implied ~0.429. Still an elimination, not a direct measurement of skew.]")
c21["wired_into"] = (
    f"nothing yet, but the fork is now closed on one side. The access-pattern arm is dead "
    f"(D-{n}) and the disk bandwidth input is confirmed correct, so the remaining suspect is "
    f"the residency model: task #52 must measure expert-usage skew against the 0.429 implied "
    f"miss fraction. No coefficient moves until it does.")

json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"D-{n} registered; C-21 updated with the elimination")
