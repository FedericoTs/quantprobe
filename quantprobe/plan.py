"""quantprobe plan - the tiered decode law as a CLI.

tok/s = eta(tier) x bandwidth / active-bytes-per-token.
Evaluates every placement (full-VRAM, hybrid attention->VRAM + experts->RAM, dense layer-split,
pure CPU, disk-stream), predicts speed for each, prints the winner WITH the llama.cpp command to
run it, plus an upgrade advisor. eta bands are fitted from published measurements (7B..744B),
validated by pre-registered predictions (30B hybrid: predicted 19, measured 19.30 +/- 0.88).
"""
from __future__ import annotations

# kvp = KV-cache bytes per position (K+V, f16, all layers). Exact where the architecture is known,
# [est] otherwise. MLA models (deepseek) cache the compressed latent -> ~10x smaller: placement's
# context story differs per architecture, which is why this is per-model, not a constant.
MODELS = {
    "qwen3-30b":  dict(t=30.5, a=3.3,  ne=1.2,  moe=True,  kvp=98304,  hint="Qwen3-30B-A3B"),          # 48L x 4KV x 128d (exact; calibration anchor)
    "deepseek-16b": dict(t=15.7, a=2.4, ne=1.3,  moe=True,  kvp=31104,  hint="DeepSeek-V2-Lite"),        # MLA: 27L x (512+64) latent (exact)
    "gemma-12b":  dict(t=11.9, a=11.9, ne=11.9, moe=False, kvp=65536,  hint="Gemma 4 12B"),             # [est] SWA: long-ctx slope from global layers only
    "mistral-7b": dict(t=7.2,  a=7.2,  ne=7.2,  moe=False, kvp=131072, hint="Mistral 7B"),              # 32L x 8KV x 128d (exact)
    "glm-air":    dict(t=110,  a=12,   ne=2.7,  moe=True,  kvp=94208,  hint="GLM-4.5-Air 106B"),        # [est]
    "glm-744b":   dict(t=744,  a=32,   ne=8,    moe=True,  kvp=188416, hint="GLM-5.2 744B"),            # [est]
}
DEFAULT_KVP = 98304          # custom models without --kv-per-pos: typical GQA mid-size (Qwen3-30B class)
ETA_KV = 0.70                # KV-read efficiency. Single-point calibration: measured tg32 d0->d16384
                             # 20.02 -> 16.12 on Qwen3-30B (+12.1 ms/token = 133 GB/s effective on the
                             # 192 GB/s tier). Falsify/refine it: quantprobe bench --depth N --contribute
# eta values: MEASURED on 2016-xmp/2016 (my box); ESTIMATED for the rest (help validate: run `quantprobe bench`).
# Mac uses unified memory -> modeled as one high-bandwidth pool the GPU serves from (the "all in VRAM" path).
MACHINES = {
    # --- measured on my hardware ---
    "2016-xmp":    dict(vc=6,  vb=192,  rc=16,  rb=48,  db=0.45, geta=0.35, gl=0.04, hint="2016 desktop (GTX 1060 6GB), XMP on [measured]"),
    "2016":        dict(vc=6,  vb=192,  rc=16,  rb=34,  db=0.45, geta=0.35, gl=0.04, hint="2016 desktop, XMP off [measured]"),
    # --- estimated: consumer GPUs ---
    "rtx-3060":    dict(vc=12, vb=360,  rc=32,  rb=51,  db=3.5,  geta=0.5,  gl=0.3,  hint="RTX 3060 12GB + DDR4-3200 [est]"),
    "rtx-3090":    dict(vc=24, vb=936,  rc=64,  rb=51,  db=3.5,  geta=0.6,  gl=0.4,  hint="RTX 3090 24GB + DDR4 [est]"),
    "rtx-4090":    dict(vc=24, vb=1008, rc=64,  rb=83,  db=5,    geta=0.62, gl=0.42, hint="RTX 4090 24GB + DDR5 [est]"),
    "rtx-5090":    dict(vc=32, vb=1792, rc=64,  rb=90,  db=5,    geta=0.62, gl=0.42, hint="RTX 5090 32GB + DDR5 [est]"),
    "laptop-8gb":  dict(vc=8,  vb=256,  rc=16,  rb=45,  db=2,    geta=0.45, gl=0.28, hint="gaming laptop, 8GB GPU + DDR5 [est]"),
    "gaming":      dict(vc=12, vb=360,  rc=32,  rb=51,  db=3.5,  geta=0.5,  gl=0.3,  hint="RTX 3060 12GB + DDR4-3200 [est] (alias of rtx-3060)"),
    # --- estimated: Apple Silicon (unified memory; I have NOT measured a Mac - these are predictions) ---
    "mac-m2-max":  dict(vc=64,  vb=400, rc=8,   rb=400, db=5,    geta=0.26, gl=0.24, hint="Mac M2 Max, 400 GB/s unified [est, unvalidated]"),
    "mac-m3-max":  dict(vc=96,  vb=400, rc=8,   rb=400, db=5,    geta=0.26, gl=0.24, hint="Mac M3 Max, 400 GB/s unified [est, unvalidated]"),
    "mac-m4-max":  dict(vc=128, vb=546, rc=8,   rb=546, db=5,    geta=0.26, gl=0.24, hint="Mac M4 Max, 546 GB/s unified [est, unvalidated]"),
    "mac-m2-ultra":dict(vc=192, vb=800, rc=8,   rb=800, db=5,    geta=0.25, gl=0.23, hint="Mac M2 Ultra, 800 GB/s unified [est, unvalidated]"),
    "mac-m3-ultra":dict(vc=512, vb=819, rc=8,   rb=819, db=5,    geta=0.25, gl=0.23, hint="Mac M3 Ultra 512GB, 819 GB/s [est, unvalidated]"),
    # --- estimated: big-RAM / server ---
    "ddr5":        dict(vc=0,  vb=0,    rc=64,  rb=80,  db=5,    geta=0.5,  gl=0.3,  hint="modern desktop DDR5, no GPU [est]"),
    "colibri":     dict(vc=0,  vb=0,    rc=128, rb=60,  db=5,    geta=0.5,  gl=0.3,  hint="128 GB DDR5 workstation [est]"),
    "epyc-256":    dict(vc=0,  vb=0,    rc=256, rb=200, db=5,    geta=0.5,  gl=0.3,  hint="Epyc/Threadripper, 256GB, ~200 GB/s [est]"),
    "dgx-spark":   dict(vc=128,vb=273,  rc=8,   rb=273, db=5,    geta=0.79, gl=0.6,  hint="NVIDIA DGX Spark / GB10, 128GB unified [validated vs published]"),
}
QUAL = {True:  {2.0: 1.10, 2.5: 1.07, 3.0: 1.05, 4.5: 1.02, 6.5: 1.01, 8.5: 1.00},
        False: {2.0: 1.45, 2.5: 1.30, 3.0: 1.12, 4.5: 1.03, 6.5: 1.01, 8.5: 1.00}}


def evaluate(t, a, ne, moe, bits, vc, vb, rc, rb, db, geta, act_scale=1.0, gl=None, ctx=0, kvp=0.0):
    ab = max(bits, 4.5)                                   # attention protected at ~4-bit (Law 3 recipes)
    size = (ne * ab / 8 + (t - ne) * bits / 8) * 1.08 * act_scale
    act_ne = ne * ab / 8 * 1.15 * act_scale
    act_ex = (a - ne) * bits / 8 * 1.15 * act_scale
    act = act_ne + act_ex
    # Law 4 v2 (context term, v1.1): every generated token re-reads the whole KV cache from
    # whichever tier KV lives on — kv_gb adds to BOTH the byte budget and that tier's capacity.
    kv_gb = ctx * kvp / 1e9 if ctx > 0 else 0.0
    ra = max(rc - 4, 1)
    eta_r = 0.38 if moe else 0.62
    if gl is None: gl = geta * 0.6
    geta_w = geta if bits >= 4 else gl                 # decode-util law: low-bit GPU decode collapses on weak GPUs
    out = []
    if vc > 0 and size + kv_gb <= vc * 0.90:
        out.append(("all in VRAM", 1 / (act / (geta_w * vb) + kv_gb / (ETA_KV * vb)), None,
                    "-ngl 99"))
    if moe and vc > 0:
        v_need = ne * ab / 8 * 1.08 + 1.2 + kv_gb          # KV sits with attention in VRAM
        r_need = size - ne * ab / 8 * 1.08
        if v_need <= vc * 0.95 and r_need <= ra:
            warn = "RAM boundary - needs --no-mmap; can be unstable" if r_need > ra * 0.85 else None
            out.append(("hybrid: attention->VRAM, experts->RAM",
                        1 / (act_ne / (geta * vb) + act_ex / (eta_r * rb) + kv_gb / (ETA_KV * vb)), warn,
                        '-ngl 99 -ot "exps=CPU" --no-mmap'))
    if (not moe) and vc > 0 and size + kv_gb > vc * 0.90 and size + kv_gb <= ra + vc * 0.9:
        g = min(0.95, vc * 0.9 / (size + kv_gb))           # KV splits with its layers
        kv_t = g * kv_gb / (ETA_KV * vb) + (1 - g) * kv_gb / (ETA_KV * rb)
        out.append((f"split: {g:.0%} layers->VRAM, rest->RAM",
                    1 / (g * act / (geta_w * vb) + (1 - g) * act / (eta_r * rb) + kv_t), None,
                    f"-ngl {int(g * 99)}"))
    if size + kv_gb <= ra:
        warn = "RAM boundary - expect bimodal speed" if size + kv_gb > ra * 0.85 else None
        out.append(("pure CPU (GPU idle)", 1 / (act / (eta_r * rb) + kv_gb / (ETA_KV * rb)), warn, "-ngl 0"))
    if size + kv_gb > ra:
        ra_eff = max(ra - kv_gb, 1)                        # KV crowds the expert cache
        miss = max(0.0, 1 - (ra_eff * 0.9) / size)
        hot = act_ne if moe else 0.0                       # MoE attention stays LRU-hot; dense has no hot set
        streamable = act - hot
        tps = 0.95 / (streamable * miss / db + (streamable * (1 - miss) + hot) / (eta_r * rb)
                      + kv_gb / (ETA_KV * rb))
        out.append(("stream from disk (cold experts)", tps, "exceeds RAM - capacity demo", "-ngl 0"))
    out.sort(key=lambda x: -x[1])
    return size, act, out


def qual_of(moe, bits):
    keys = sorted(QUAL[moe])
    return QUAL[moe][min(keys, key=lambda k: abs(k - bits))]


def run(args):
    from . import spec as specmod
    specmod.apply(args)
    if getattr(args, "bits", None) is None:
        args.bits = 2.5
    m = dict(MODELS[args.model]) if args.model in MODELS else {}
    t = args.total or m.get("t") or 13.0
    a = args.active or m.get("a") or t
    ne = args.always_active or m.get("ne") or (a if a >= t * 0.9 else a * 0.35)
    moe = m.get("moe", a < t * 0.9)
    hw = dict(MACHINES[args.machine]) if args.machine in MACHINES else {}
    if not hw and all(getattr(args, k, None) is None for k in ("vram", "vram_bw", "ram", "ram_bw", "disk_bw")):
        from . import detect as detmod
        auto, _ = detmod.detect()
        hw = dict(vc=auto["vram"], vb=auto["vram_bw"], rc=auto["ram"], rb=auto["ram_bw"],
                  db=auto["disk_bw"], geta=auto.get("geta", 0.45), gl=auto.get("gl"),
                  hint="THIS machine [auto-detected - run `quantprobe hw` for details]")
        print("[quantprobe] no hardware flags: auto-detected this machine "
              f"(vram {hw['vc']:g}GB@{hw['vb']:g} | ram {hw['rc']:g}GB@{hw['rb']:g} | disk {hw['db']:g} GB/s). "
              "Pass --machine/flags to estimate a different box.")
    vc = hw.get("vc", args.vram); vb = hw.get("vb", args.vram_bw)
    rc = hw.get("rc", args.ram);  rb = hw.get("rb", args.ram_bw)
    db = hw.get("db", args.disk_bw); geta = hw.get("geta", 0.45); gl = hw.get("gl", None)
    if args.vram is not None: vc = args.vram
    if args.vram_bw is not None: vb = args.vram_bw
    if args.ram is not None: rc = args.ram
    if args.ram_bw is not None: rb = args.ram_bw
    if args.disk_bw is not None: db = args.disk_bw
    vc = vc or 0; vb = vb or 0; rc = rc or 16; rb = rb or 40; db = db or 0.5
    ctx = getattr(args, "ctx", 0) or 0
    kvp = (args.kv_per_pos * 1024 if getattr(args, "kv_per_pos", None)
           else m.get("kvp", DEFAULT_KVP))

    size, act, cfgs = evaluate(t, a, ne, moe, args.bits, vc, vb, rc, rb, db, geta, gl=gl, ctx=ctx, kvp=kvp)
    q = qual_of(moe, args.bits)
    print(f"\nquantprobe plan - {m.get('hint', 'custom model')} @ {args.bits:g}-bit "
          f"on {hw.get('hint', 'custom machine')}")
    kvline = (f" | ctx {ctx}: +{ctx * kvp / 1e9:.2f} GB KV read/token"
              if ctx > 0 else "")
    print(f"  model {size:.1f} GB | active {act:.2f} GB/token{kvline} | est. quality cost x{q:.2f} "
          f"(depth-aware recipe)\n")
    for i, (name, tps, warn, flags) in enumerate(cfgs):
        star = "*" if i == 0 else " "
        w = f"   [{warn}]" if warn else ""
        print(f"  {star} {tps:6.1f} tok/s  {name}{w}")
    best = cfgs[0]
    print(f"\n  run it:  llama-server -m model.gguf {best[3]}")
    # upgrade advisor
    alts = []
    if rb < 40:
        s2, _, c2 = evaluate(t, a, ne, moe, args.bits, vc, vb, rc, 48, db, geta, gl=gl, ctx=ctx, kvp=kvp)
        if c2[0][1] > best[1] * 1.08: alts.append(("enable XMP (free)", c2[0][1]))
    s2, _, c2 = evaluate(t, a, ne, moe, args.bits, vc, vb, rc + 16, rb, db, geta, gl=gl, ctx=ctx, kvp=kvp)
    if c2[0][1] > best[1] * 1.08: alts.append(("+16 GB RAM", c2[0][1]))
    s2, _, c2 = evaluate(t, a, ne, moe, args.bits, vc, vb, rc, rb, 3.5, geta, gl=gl, ctx=ctx, kvp=kvp)
    if c2[0][1] > best[1] * 1.08: alts.append(("NVMe SSD", c2[0][1]))
    if alts:
        print("  upgrade advisor: " + " | ".join(f"{n} -> ~{v:.1f} tok/s" for n, v in alts))
    print("\n  (eta bands fitted from published measurements; estimates +/-25%. "
          "Hybrid needs --no-mmap.)")
