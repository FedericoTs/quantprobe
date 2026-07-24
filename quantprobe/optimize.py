"""quantprobe optimize — the cheapest path to a target speed.

A pure SEARCH LAYER over the frozen decode law: enumerate lever combinations, price each with
plan.evaluate(), rank by (meets target, quality cost, euros, speed). The law is never modified —
the optimizer only reads it, so every published anchor is untouchable by construction.

Levers and their measured gates:
  bits      effective bits/weight ladder, priced with the DEPTH-AWARE quality curve (measured;
            uniform quantization costs ~1.3x more quality at <=2.5 bits — Gemma 1.91x vs 1.45x)
  kv-q8     quantized K-cache (kvp x0.75). GATED OFF on weak-decode GPUs: measured -83%
            at 16k depth on Pascal (2026-07-24, no flash attention -> per-token dequant tax).
            Offered [est] only where geta >= 0.5; verify with `bench --depth`.
  prune     REAP-class 50% expert pruning (total x0.82, file shrinks, active bytes UNCHANGED).
            Measured +39% out-of-domain perplexity (pre-registration #8) — domain-specialized,
            never ranked first without --allow-prune.
  hardware  euro-priced deltas from the projections table: XMP (free), +16GB RAM (~40), NVMe (~100).
"""
from __future__ import annotations
from . import plan as planmod

BITS_LADDER = [2.0, 2.5, 3.0, 3.5, 4.5]
REALIZE = {
    2.0: 'quantize --gguf <f16> --protect-late 8   (base Q2_K, narrow probed band)',
    2.5: 'quantize --gguf <f16> --protect-late 12  (base Q2_K - the validated default)',
    3.0: 'quantize --gguf <f16> --protect-late 20  (or base Q3_K_S)',
    3.5: 'fetch a Q3_K_M and probe --apply',
    4.5: 'fetch a Q4_K_M (straight; no probe needed at 4-bit)',
}
HW_DELTAS = [
    ("as-is", 0, {}),
    ("+16GB RAM (~40 EUR used)", 40, {"rc": +16}),
    ("NVMe SSD (~100 EUR)", 100, {"db_min": 3.5}),
    ("+RAM & NVMe (~140 EUR)", 140, {"rc": +16, "db_min": 3.5}),
]


def resolve(a):
    """Model + machine resolution, same semantics as plan.run (autospec + autodetect included)."""
    from . import spec as specmod
    specmod.apply(a, quiet=True)
    if getattr(a, "bits", None) is None:
        a.bits = 2.5
    m = dict(planmod.MODELS[a.model]) if getattr(a, "model", None) in planmod.MODELS else {}
    t = a.total or m.get("t") or 13.0
    ac = a.active or m.get("a") or t
    ne = a.always_active or m.get("ne") or (ac if ac >= t * 0.9 else ac * 0.35)
    moe = m.get("moe", ac < t * 0.9)
    hw = dict(planmod.MACHINES[a.machine]) if getattr(a, "machine", None) in planmod.MACHINES else {}
    if not hw and all(getattr(a, k, None) is None for k in ("vram", "vram_bw", "ram", "ram_bw", "disk_bw")):
        from . import detect as detmod
        auto, _ = detmod.detect()
        hw = dict(vc=auto["vram"], vb=auto["vram_bw"], rc=auto["ram"], rb=auto["ram_bw"],
                  db=auto["disk_bw"], geta=auto.get("geta", 0.45), gl=auto.get("gl"),
                  hint="THIS machine [auto-detected]")
        print("[quantprobe] hardware auto-detected (pass --machine/flags to optimize another box)")
    vc = planmod.agg_cap(a.vram) if a.vram is not None else hw.get("vc", 0)
    vb = planmod.agg_bw(a.vram_bw, 0.85) if a.vram_bw is not None else hw.get("vb", 0)
    rc = a.ram if a.ram is not None else hw.get("rc", 16)
    rb = a.ram_bw if a.ram_bw is not None else hw.get("rb", 40)
    db = planmod.agg_bw(a.disk_bw, 0.75) if a.disk_bw is not None else hw.get("db", 0.5)
    geta = hw.get("geta", 0.45); gl = hw.get("gl")
    ctx = getattr(a, "ctx", 0) or 0
    kvp = (a.kv_per_pos * 1024 if getattr(a, "kv_per_pos", None) else m.get("kvp", planmod.DEFAULT_KVP))
    return m, t, ac, ne, moe, (vc or 0, vb or 0, rc or 16, rb or 40, db or 0.5, geta, gl), ctx, kvp


def run(a):
    m, t, ac, ne, moe, (vc, vb, rc, rb, db, geta, gl), ctx, kvp = resolve(a)
    tgt = getattr(a, "tps", None)
    maxq = getattr(a, "max_quality", None) or 1.12
    allow_prune = getattr(a, "allow_prune", False)
    kvq_ok = geta >= 0.5                      # measured gate: Pascal-class collapses (-83% @16k)
    rows = []
    for hw_name, euro, delta in HW_DELTAS:
        rc2 = rc + delta.get("rc", 0)
        db2 = max(db, delta.get("db_min", db))
        if euro and rc2 == rc and db2 == db:
            continue                           # delta changes nothing on this box
        for bits in BITS_LADDER:
            for prune, pf in ((False, 1.0),) + (((True, 0.82),) if allow_prune and moe else ()):
                for kvq, kf in ((False, 1.0),) + (((True, 0.75),) if kvq_ok and ctx > 0 else ()):
                    q = planmod.qual_of(moe, bits) * (1.0 if not prune else 1.0)  # in-domain qual; OOD flagged in text
                    if q > maxq:
                        continue
                    _, _, cfgs = planmod.evaluate(t * pf, ac, ne, moe, bits, vc, vb, rc2, rb, db2,
                                                  geta, 1.0, gl, ctx=ctx, kvp=kvp * kf)
                    if not cfgs:
                        continue
                    if not getattr(a, "any_runtime", False):
                        cfgs = [c for c in cfgs if "expert cache" not in c[0]] or cfgs
                    name, tps, warn, flags = cfgs[0]
                    desc = f"{bits:g}-bit depth-aware + {name}"
                    tags = []
                    if prune:
                        tags.append("PRUNED: +39% out-of-domain ppl measured - domain use only")
                    if kvq:
                        tags.append("KV q8 [est - needs FA; measured trap on Pascal-class]")
                    if hw_name != "as-is":
                        desc += f" + {hw_name}"
                    rows.append(dict(tps=tps, q=q, euro=euro, bits=bits, desc=desc,
                                     tags=tags, flags=flags, warn=warn))
    # Pareto: drop rows beaten on every axis
    keep = []
    for r in sorted(rows, key=lambda x: (-x["tps"], x["q"], x["euro"])):
        if not any(k["tps"] >= r["tps"] and k["q"] <= r["q"] and k["euro"] <= r["euro"]
                   and (k["tps"], k["q"], k["euro"]) != (r["tps"], r["q"], r["euro"]) for k in keep):
            keep.append(r)
    if tgt:
        meeting = [r for r in keep if r["tps"] >= tgt]
        ranked = (sorted(meeting, key=lambda x: (x["q"], x["euro"], -x["tps"])) or
                  sorted(keep, key=lambda x: -x["tps"])[:1])
        headline = "cheapest configuration meeting the target" if meeting else \
                   "TARGET NOT REACHABLE on this hardware - fastest available:"
    else:
        ranked = sorted(keep, key=lambda x: (-x["tps"], x["q"], x["euro"]))
        headline = "speed frontier (quality ceiling x%.2f)" % maxq
    print(f"\nquantprobe optimize - {m.get('hint', 'custom model')} on "
          f"{'this machine' if not getattr(a, 'machine', None) else a.machine}"
          + (f" | target {tgt:g} tok/s" if tgt else "") + (f" | ctx {ctx}" if ctx else ""))
    print(f"  {headline}\n")
    for i, r in enumerate(ranked[:6]):
        star = "*" if i == 0 else " "
        euro = "free" if r["euro"] == 0 else f"~{r['euro']}EUR"
        print(f"  {star} {r['tps']:6.1f} tok/s  quality x{r['q']:.2f}  {euro:>7s}  {r['desc']}")
        for tag in r["tags"]:
            print(f"                [{tag}]")
    best = ranked[0]
    print(f"\n  realize the pick:")
    print(f"    quantprobe {REALIZE.get(best['bits'], 'quantize')}")
    print(f"    quantprobe run --gguf <the file> ...   # launches with: {best['flags']}")
    print("\n  (search over the validated law only - no new physics; quality = depth-aware recipe,")
    print("   uniform quantization costs ~1.3x more at <=2.5 bits. Estimates +/-25%.)")
    return ranked
