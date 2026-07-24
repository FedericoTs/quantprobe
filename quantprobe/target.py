"""quantprobe target — inverse planning: from a tok/s target to the smartest feasible model.

The tiered decode law solved backwards: given YOUR machine and the speed you need,
rank every (model x bits x placement) that meets the target, by intelligence
(total parameters, quality-adjusted). Then print the LADDER: at each speed band,
the best option — so trading speed for intelligence is one `quantprobe run` away.
"""
from __future__ import annotations
from . import plan as planmod

# extended catalog for the ladder (t/a/ne in B params; approximations, +/-25% banner applies)
CATALOG = dict(planmod.MODELS)
CATALOG.update({
    "qwen3-4b":    dict(t=4.0,   a=4.0,  ne=4.0,  moe=False, hint="Qwen3-4B"),
    "qwen3-14b":   dict(t=14.8,  a=14.8, ne=14.8, moe=False, hint="Qwen3-14B"),
    "mixtral-8x7b": dict(t=46.7, a=12.9, ne=1.6,  moe=True,  hint="Mixtral 8x7B"),
    "llama3-70b":  dict(t=70.6,  a=70.6, ne=70.6, moe=False, hint="Llama-3 70B"),
    "qwen3-235b":  dict(t=235,   a=22,   ne=7.0,  moe=True,  hint="Qwen3-235B-A22B"),
})
BITS_LADDER = [4.5, 3.0, 2.5, 2.0]           # prefer quality (higher bits) when target is met


def hw_of(a):
    hw = dict(planmod.MACHINES[a.machine]) if getattr(a, "machine", None) in planmod.MACHINES else {}
    vc = planmod.agg_cap(a.vram) if a.vram is not None else hw.get("vc", 0)
    vb = planmod.agg_bw(a.vram_bw, 0.85) if a.vram_bw is not None else hw.get("vb", 0)
    rc = a.ram if a.ram is not None else hw.get("rc", 16)
    rb = a.ram_bw if a.ram_bw is not None else hw.get("rb", 40)
    db = planmod.agg_bw(a.disk_bw, 0.75) if a.disk_bw is not None else hw.get("db", 0.5)
    return hw, vc, vb, rc, rb, db, hw.get("geta", 0.45), hw.get("gl", None)


def feasible(a, tps_target):
    """All (model, bits, placement) meeting the target, best-first."""
    ctx = getattr(a, "ctx", 0) or 0
    hw, vc, vb, rc, rb, db, geta, gl = hw_of(a)
    rows = []
    for key, m in CATALOG.items():
        for bits in BITS_LADDER:
            size, act, cfgs = planmod.evaluate(m["t"], m["a"], m["ne"], m["moe"],
                                               bits, vc, vb, rc, rb, db, geta, 1.0, gl,
                                               ctx=ctx, kvp=m.get("kvp", planmod.DEFAULT_KVP))
            best = cfgs[0]
            if best[1] >= tps_target:
                q = planmod.qual_of(m["moe"], bits)
                rows.append(dict(key=key, hint=m["hint"], t=m["t"], bits=bits, q=q,
                                 size=size, tps=best[1], place=best[0], warn=best[2], flags=best[3]))
                break                          # highest feasible bits for this model = its best entry
    # rank: biggest model first; tie-break lower quality cost, then speed
    rows.sort(key=lambda r: (-r["t"], r["q"], -r["tps"]))
    return rows


def run(a):
    from . import plan as planmod
    planmod.check_presets(a)
    hw, vc, vb, rc, rb, db, geta, gl = hw_of(a)
    rows = feasible(a, a.tps)
    print(f"\nquantprobe target - smartest model meeting >= {a.tps:g} tok/s "
          f"on {hw.get('hint', 'custom machine')}\n")
    if not rows:
        print("  nothing in the catalog meets that target on this machine.")
        print("  try a lower --tps, or see the upgrade paths in `quantprobe plan`.")
    else:
        w = rows[0]
        print(f"  * {w['hint']}  @ {w['bits']:g}-bit  ({w['t']:g}B params)")
        print(f"    {w['tps']:.1f} tok/s | {w['size']:.1f} GB | quality x{w['q']:.2f} | {w['place']}"
              + (f"  [{w['warn']}]" if w["warn"] else ""))
        print(f"    run it:  quantprobe run --gguf <{w['key']}.gguf> --model {w['key']}"
              f" --machine {a.machine or 'custom'} --bits {w['bits']:g}\n")
        if len(rows) > 1:
            print("  also feasible (by size):")
            for r in rows[1:6]:
                print(f"    {r['tps']:6.1f} tok/s  {r['hint']:22s} @ {r['bits']:g}-bit"
                      f"  {r['size']:5.1f} GB  x{r['q']:.2f}  {r['place']}")
    if a.ladder:
        print("\n  THE LADDER - trade speed for intelligence (best model per speed band):")
        print(f"  {'need':>8s}  {'best option':40s} {'delivers':>9s}")
        for band in [30, 20, 10, 5, 2, 0.5]:
            rr = feasible(a, band)
            if rr:
                w = rr[0]
                print(f"  {band:>5g}+   {w['hint'] + ' @ ' + format(w['bits'], 'g') + '-bit (' + format(w['t'], 'g') + 'B)':40s}"
                      f" {w['tps']:6.1f} tok/s")
            else:
                print(f"  {band:>5g}+   {'-':40s}")
        print("\n  switching rungs = one `quantprobe run` with the rung's model/bits. "
              "Downloads differ; placement flags are emitted automatically.")
    print("\n  (catalog params approximate; law estimates +/-25%. Bigger != always smarter - "
          "quality x is the gap-ratio vs each model's own fp16.)")
