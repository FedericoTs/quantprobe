"""Audit the public llama.cpp benchmark corpus with Law 4: implied eta per entry, ceiling
validation vs optimization headroom. eta = tg x active_bytes / BW_spec; the measured reference
bands come from our ladder (dense Q4-class AIV eta 0.55-0.62 incl. Q4_0 0.619; MoE-resident
0.31-0.41; below-band = speed left on the table or a kernel gap)."""
import json, os

BW = {  # spec GB/s, documented sources; the audit reports spec-based eta (delivered varies)
    "RTX 5090": 1792, "RTX 4090": 1008, "RTX 3090": 936, "RTX 4080": 717, "H100": 2000,
    "M1 Max": 400, "M4 Max": 546, "M2 Ultra": 800, "M3 Ultra": 819,
    "DGX Spark": 273, "GB10": 273, "Thor": 273,
    "Strix Halo": 256, "8060S": 256, "MI300X": 5300, "7900 XTX": 960, "MI50": 1024,
    "9070 XT": 645, "B580": 456, "RTX 3060": 360, "RTX 2060": 336, "7900 XT": 800,
    "RTX Pro 6000": 1792, "Pro 6000": 1792, "5090 Mobile": 896, "5090 (laptop)": 896,
}
# active GB/token per model+quant (dense: file bytes; MoE: active params x bits/8 x 1.15)
ACT = {
    ("Llama 2 7B", "Q4_0"): 3.82, ("LLaMA 7B", "Q4_0"): 3.82, ("LLaMA 7B v2", "Q4_0"): 3.82,
    ("gpt-oss-20b", "MXFP4"): 2.2, ("gpt-oss 20B MXFP4 MoE", "MXFP4"): 2.2,
    ("gpt-oss 120B MXFP4 MoE", "MXFP4"): 3.1, ("gpt-oss-120b", "MXFP4"): 3.1,
    ("GPT-OSS-120B", None): 3.1, ("gpt-oss 120B", None): 3.1,
    ("Qwen3-30B-A3B", None): 1.42, ("Qwen3-Coder 30B", None): 1.42,
    ("Qwen3.5-35B-A3B", None): 1.2, ("GLM 4.5 Air", None): 4.99, ("GLM-4.5-Air (106B-A12B MoE)", None): 4.99,
}

def bw_for(gpu):
    for k, v in BW.items():
        if k.lower() in gpu.lower():
            return v
    return None

def act_for(model, quant):
    for (m, q), v in ACT.items():
        if m.lower() in model.lower() and (q is None or (quant or "").startswith(q)):
            return v
    return None

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    es = json.load(open(os.path.join(here, "data", "public_bench_corpus.json"), encoding="utf-8"))
    print(f"{'gpu':34s} {'model':28s} {'tg':>7s} {'eta':>6s}  verdict")
    print("-" * 100)
    rows = []
    for e in es:
        bw, act = bw_for(e["gpu"]), act_for(e["model"], e.get("quant"))
        if not bw or not act:
            continue
        eta = e["tg_toks"] * act / bw
        moe = any(s in e["model"].lower() for s in ("moe", "a3b", "a4b", "oss", "a12b", "a22b", "a32b", "air", "scout"))
        lo, hi = (0.28, 0.45) if moe else (0.50, 0.75)
        verdict = "at ceiling (validates the law)" if eta >= lo else f"BELOW BAND - headroom ~{(lo/eta-1)*100:.0f}%+"
        if eta > hi:
            verdict = "above band (check inputs)"
        rows.append((e, eta, verdict))
        print(f"{e['gpu'][:34]:34s} {e['model'][:28]:28s} {e['tg_toks']:7.1f} {eta:6.3f}  {verdict}")
    return rows

if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# DEPTH CAVEAT (L-19, added 2026-07-30 after preregs #73/#74/#75)
# ---------------------------------------------------------------------------
# 17 of the 61 corpus entries carry a depth signal (16k-131k), and 8 of those ALSO offload to
# CPU. The eta computed above is a d0 quantity: for any depth entry it under-states the machine,
# because part of the measured token time is KV reads (all placements) and, where whole layers
# sit on CPU, attention compute (L-19). Consequences for the headroom claims:
#
#   * Entries at depth WITHOUT cpu offload (4080/gpt-oss-20b @32k, the 4x3090 GLM rows): eta is
#     under-stated by the KV-read share only - modest, and it makes those rows look WORSE than
#     the machine is. Their "at ceiling" verdicts are therefore safe (a real ceiling is higher).
#   * Entries at depth WITH cpu offload (RTX PRO 6000 GLM rows @16k, RTX 6000 R1 @131k, 3090
#     Qwen3-235B @32k, and BOTH RTX 3060 gpt-oss rows): eta is under-stated by KV reads AND, for
#     MoE expert-offload, nothing extra (attention stays on GPU) - but for any DENSE offload the
#     L-19 term applies and the row cannot be graded on a d0 band at all.
#   * The single most-quoted headroom case (RTX 2060 + Qwen3.5-35B at ~102k, eta 0.054) is MoE
#     with --n-cpu-moe 99: attention stays on the GPU, so L-19 does NOT apply, but ~102k of KV
#     reads absolutely do. Its d0-basis eta is meaningless; what survives is the PLACEMENT
#     comparison (all experts on CPU vs partial residency), which is depth-independent advice.
#
# Bottom line for publication: the eta-generality table (dense Q4-class, all-in-VRAM, d0 entries)
# is sound and is what L-18 rests on. The HEADROOM column must not be published for depth entries
# without re-deriving them with the KV term - which needs each reporter's exact context length,
# and most did not state one. Marked here rather than silently dropped.
