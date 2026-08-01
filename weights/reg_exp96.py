"""Register #96/#96b: the streaming-efficiency gap, and the withdrawal of my own verdict.

  python weights/reg_exp96.py
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
        "THE DISK BANDWIDTH THE DRIVE DELIVERS AND THE DISK BANDWIDTH llama.cpp ACHIEVES ARE "
        "DIFFERENT NUMBERS, AND LAW 4 USES THE WRONG ONE. Raw reads on this drive measure "
        "0.452-0.459 GB/s (D-28, sequential and scattered alike). llama.cpp streaming the same "
        "drive through mmap achieves an effective 0.2505 GB/s - a 1.82x gap. Measured by taking "
        "Qwen3-Coder-30B-A3B Q3_K_M (13.7 GB, fits this 16 GiB box) from evicted to fully "
        "resident: cold 1.2997 tok/s with 0.16708 GB/token off disk, resident 9.09-10.00 tok/s "
        "with ZERO disk traffic. The whole difference in token time, 0.76939 s vs 0.1026 s, is "
        "attributable to 0.16708 GB of disk traffic, which prices that traffic at 0.2505 GB/s. "
        "THIS SUPPLIES THE MISSING COST C-21 AND #95 TRIANGULATED: #95 showed the law budgets "
        "~2.7x too many bytes (R=0.366) while the row is only 1.485x fast, leaving ~0.789 s per "
        "token unexplained. A streaming rate 1.82x below the raw rate is the right size and sign "
        "to be it. WHAT I WITHDRAW: I staked this as a PER-FAULT LATENCY hypothesis and the "
        "scorer printed 'H4 SUPPORTED, lambda 16.35 us/fault, r2 1.000'. That verdict is NOT "
        "EARNED and I am retracting it rather than banking it. Fault counts were DERIVED as "
        "bytes/4096, never counted, so lambda*faults is identically bytes/(4096/lambda) - the "
        "per-fault model and a lower-effective-bandwidth model are the SAME equation in "
        "different units, and no experiment of this shape can separate them. 4096/16.35us is "
        "exactly the 0.2505 GB/s above. Two further weaknesses in my own design, recorded "
        "because the printed r2 hides them: the fit had only TWO DISTINCT x values (40791 and "
        "0, three times), so it is a difference of means wearing a regression's clothes and "
        "r2=0.9999 is decorative - a line through two x values always fits; and KR-E, the "
        "out-of-sample check meant to falsify the LINE, held out a zero-fault run, so it only "
        "ever tested whether the resident time is stable run-to-run. The 6.7% it passed at is a "
        "real fact about run-to-run stability and no evidence at all for a fault model."),
    "magnitude": "llama.cpp mmap streaming 0.2505 GB/s vs 0.452-0.459 GB/s raw on the same drive (1.82x)",
    "evidence": ("weights/exp96b_fault_latency.py + weights/data/exp96b_fault_latency.json "
                 "(5 runs, evicted -> resident); weights/exp96_fault_latency.py + "
                 "weights/data/exp96_fault_latency.json (the null that forced the redesign); "
                 "raw-read baseline in weights/data/exp94_access_pattern.json"),
    "scope": (
        "One drive, one model, one runtime, cold-vs-resident rather than a graded residency "
        "sweep. The MECHANISM is undetermined and this design cannot determine it: per-fault "
        "latency, mmap page-fault overhead, readahead defeat, and simply lower achieved "
        "throughput are all indistinguishable when fault count is derived from byte count. "
        "Separating them needs actual fault counts (ETW / hard-fault counters) or a runtime "
        "knob such as --no-mmap on a model small enough to load that way. CONTENTION: an agent "
        "was active, so 0.2505 GB/s is a LOWER bound on achieved streaming and the 1.82x gap is "
        "an UPPER bound. The clean-conditions disk row (0.476 tok/s) is consistent with it but "
        "was not measured with the disk counter running. NOT a coefficient: two independent "
        "routes now point near this number, but neither was measured clean, and C-02 is the "
        "standing warning against fitting a constant to convenient points."),
    "wired_into": (
        "nothing yet. The tempting move - reprice the disk tier at ~0.25 GB/s - would fix C-21's "
        "30% miss in one line and is exactly the move this register exists to slow down: it "
        "would bake a contended measurement of one model on one drive into every disk-tier "
        "prediction. Owed first: (a) re-measure streaming efficiency under clean conditions with "
        "the disk counter live, (b) a second drive or model to show the 1.82x is not "
        "device-specific, (c) actual hard-fault counts to settle the mechanism. Until then "
        "plan.py keeps under-predicting the disk tier and the CHANGELOG says why."),
})

json.dump(d, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"C-{n} registered (streaming-efficiency gap + withdrawal of the per-fault verdict)")
