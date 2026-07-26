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
    "qwen3-30b":  dict(t=30.5, a=3.3,  ne=1.2,  moe=True,  kvp=98304,  nl=48,  hint="Qwen3-30B-A3B"),          # 48L x 4KV x 128d (exact; calibration anchor)
    "deepseek-16b": dict(t=15.7, a=2.4, ne=1.3,  moe=True,  kvp=31104,  nl=27,  hint="DeepSeek-V2-Lite"),        # MLA: 27L x (512+64) latent (exact)
    "gemma-12b":  dict(t=11.9, a=11.9, ne=11.9, moe=False, kvp=65536,  hint="Gemma 4 12B"),             # [est] SWA: long-ctx slope from global layers only
    "mistral-7b": dict(t=7.2,  a=7.2,  ne=7.2,  moe=False, kvp=131072, hint="Mistral 7B"),              # 32L x 8KV x 128d (exact)
    "glm-air":    dict(t=110,  a=12,   ne=2.7,  moe=True,  kvp=94208,  hint="GLM-4.5-Air 106B"),        # [est]
    "glm-744b":   dict(t=753.3, a=32,  ne=8,    moe=True,  kvp=188416, hint="GLM-5.2 (753B)"),          # total exact (HF safetensors); a/ne/kvp [est]
    "qwen3-235b": dict(t=235.1, a=22,  ne=7.5,  moe=True,  kvp=192512, hint="Qwen3-235B-A22B"),         # total exact; kvp: 94L x 4KV x 128d [est]
    "kimi-k2.6":  dict(t=1058.6, a=32, ne=6,    moe=True,  kvp=70272,  hint="Kimi K2.6 (1T MLA)"),      # total exact; kvp: 61L x 576 MLA latent [est]
    "gpt-oss-120b": dict(t=120.4, a=5.1, ne=1.8, moe=True, kvp=73728,  hint="gpt-oss-120b"),            # total+active official; kvp: 36L x 8KV x 64d [est]
    "llama-70b":  dict(t=70.6, a=70.6, ne=70.6, moe=False, kvp=327680, hint="Llama-3.3-70B"),           # dense; kvp: 80L x 8KV x 128d
}
DESKTOP_VRAM_RESERVE = 1.0   # GB held by the OS/desktop on a real machine, not the model's to use.
                             # Measured 0.8-1.5 GB on this box (pre-registration #13 sweep).
                             # Applied ONLY to the new MoE split row: existing rows keep their
                             # published anchors untouched.
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
def numlist(x):
    """CLI type: '24,24' -> [24.0, 24.0]; '24' -> 24.0. Lists = multiple devices."""
    if isinstance(x, (int, float)):
        return float(x)
    if "," in str(x):
        return [float(v) for v in str(x).split(",") if v.strip()]
    return float(x)


def agg_cap(v):
    return sum(v) if isinstance(v, list) else v


def agg_bw(v, eff):
    """Aggregate bandwidth of multiple devices: sum x efficiency (TP loss for GPUs [est],
    stripe loss for disks [est from the RAID-0 Gen5 datapoint, eta 0.66 vs single ~0.88])."""
    if isinstance(v, list):
        return sum(v) * (eff if len(v) > 1 else 1.0)
    return v


QUAL = {True:  {2.0: 1.10, 2.5: 1.07, 3.0: 1.05, 4.5: 1.02, 6.5: 1.01, 8.5: 1.00},
        False: {2.0: 1.45, 2.5: 1.30, 3.0: 1.12, 4.5: 1.03, 6.5: 1.01, 8.5: 1.00}}


def effective_n_layer(args=None, model=None):
    """THE resolver for "how many layers does this model have". Do not hand-write this again.

    Precedence: an explicit --n-layer, then a preset's `nl` (verified against a real GGUF).
    Callers that read a GGUF thread the file's own count onto args before calling.

    This function exists because the fallback was hand-written at each call site and was wrong
    FOUR times: v1.9.0 (target.py), v1.10.5 (runtime.py), and twice more found by auditing for
    the shape - plan's layer-count note, and auto.py, which omitted the preset step entirely and
    so could never offer the MoE split placement for a preset model. Each was fixed where it was
    found; the shape was never fixed. One resolver, one behaviour, one place to be wrong.

    `model` accepts a MODELS dict or a preset name, so callers can pass whichever they hold.
    """
    n = getattr(args, "n_layer", None)
    if n:
        return int(n)
    if isinstance(model, str):
        model = MODELS.get(model)
    if isinstance(model, dict) and model.get("nl"):
        return int(model["nl"])
    return None


def moe_split_flags(frac, n_layer):
    """-ot regex placing the FIRST ceil(frac*L) layers' experts on GPU, the rest on CPU.
    Measured 2026-07-26 (pre-registration #13, corrected): +12.4% decode and ~2-3x prefill
    on a 6 GB card, against a properly configured baseline.
    Returns None when the layer count is unknown - we will not emit a regex we cannot ground."""
    if not n_layer or frac <= 0:
        return None
    k = max(1, min(n_layer - 1, int(frac * n_layer)))
    # --no-mmap for the same reason the all-experts row carries it: llama.cpp itself warns that
    # tensor overrides to CPU with mmap enabled cost performance. Measured 2026-07-26 on this
    # split placement: 16.45 tok/s with mmap vs 18.70 without (+13.7%).
    return ('-ngl 99 -ot "blk\\.(%s)\\.ffn_.*_exps\\.=CPU" --no-mmap'
            % "|".join(str(i) for i in range(k, n_layer)))


def fits_in_vram_advice(placement, bits):
    """Once a model fits in VRAM, quantizing it further buys almost no speed - measured.

    The law prices decode as bandwidth, so it predicts that halving the bits nearly halves the
    time. In the fits-entirely-in-VRAM regime that is simply not what happens. Same 7B, same
    card, all in VRAM, only the quantization changed (pre-registration #16, r=3):

        Q4_K_M  4.5 bits  4.68 GB  20.03 +/- 0.04 tok/s
        Q2_K    2.8 bits  3.01 GB  19.17 +/- 0.03 tok/s     36% smaller, 4% SLOWER

    Decode there is not bandwidth-bound, so bytes stop predicting speed. Until that regime is
    modelled properly the law over-rewards low bits here, and a user following the ranking alone
    would trade real quality for nothing. Say so, rather than silently ranking on a number known
    to be wrong in this direction.
    """
    if placement != "all in VRAM" or bits >= 4.5:
        return None
    return ("it already fits in VRAM, so going lower-bit buys you almost nothing. Measured on "
            "this class of card: the same 7B at Q2_K vs Q4_K_M is 36% smaller and 4% SLOWER "
            "(19.17 vs 20.03 tok/s). The speeds ranked above assume decode is bandwidth-bound, "
            "which it is not once the whole model sits in VRAM. Quantize to make a model FIT - "
            "once it fits, take the highest bits that still fit.")


def speculation_advice(moe, placement):
    """What speculative decoding is worth for THIS model and placement.

    Every number here is measured on the reference box (Law 6 arms S-a/S-b/S-e/S-f, 3 runs per
    cell, raw logs in weights/data/). Speculation drafts tokens and verifies them, so output is
    identical - the only question is whether the verify batch costs more than the pass it saves.

    dense       code  2.10x | prose 1.01x   (ngram; copyability is the whole mechanism)
    MoE offload code  1.03x                 (the verify batch unions experts - the tax eats it)
    dense       MTP   1.17x GPU-resident, 1.046x CPU
    MoE offload MTP   0.76x                 (an actual LOSS)

    Returns None when we have no measurement for the case rather than guessing.
    """
    # Only the fully-offloaded case is measured (exps=CPU). A partial split puts some experts
    # on the fast tier, which we have NOT measured - so it gets no claim either way.
    experts_offloaded = "exps=CPU" in (placement or "")
    if moe and experts_offloaded:
        return ("speculation will NOT pay here: measured +3% (ngram) and -24% (MTP) with experts "
                "offloaded - a verify batch unions experts, and every extra one is a slow read.")
    if not moe:
        return ("if you write CODE, add `--spec-type ngram-simple` - measured **2.10x decode** "
                "(17.7 -> 37.2 tok/s), one flag, no download, identical output. Prose gains "
                "nothing (1.01x): it drafts by copying spans from your context.")
    return None                    # MoE fully resident: untested here, so we say nothing


def evaluate(t, a, ne, moe, bits, vc, vb, rc, rb, db, geta, act_scale=1.0, gl=None, ctx=0, kvp=0.0,
             n_layer=None, true_size_gb=None):
    ab = max(bits, 4.5)                                   # attention protected at ~4-bit (Law 3 recipes)
    size = (ne * ab / 8 + (t - ne) * bits / 8) * 1.08 * act_scale
    # CAPACITY uses the real file size when we have the file. The estimate above assumes the
    # depth-aware recipe (attention held at >=4.5 bits), which for a DENSE model - where the
    # tables set ne = t - inflates size by 4.5/bits: a dense 7B at 2 bits came out 125% too big,
    # and a real 12B at 3.51 bits read as 7.2 GB against an actual 5.2 GB. That wrongly evicted
    # the all-in-VRAM row and recommended a placement measured 2.4x slower (3.9 vs 9.56 tok/s).
    # Only `size` is corrected: the activation terms below are empirically calibrated and
    # accurate as-is (scaling them too is what made bench 11% optimistic before v1.10.5).
    if true_size_gb:
        size = true_size_gb
    act_ne = ne * ab / 8 * 1.15 * act_scale
    act_ex = (a - ne) * bits / 8 * 1.15 * act_scale
    act = act_ne + act_ex
    # Law 4 v2 (context term, v1.1): every generated token re-reads the whole KV cache from
    # whichever tier KV lives on - kv_gb adds to BOTH the byte budget and that tier's capacity.
    kv_gb = ctx * kvp / 1e9 if ctx > 0 else 0.0
    ra = max(rc - 4, 1)
    eta_r = 0.38 if moe else 0.62
    if gl is None: gl = geta * 0.6
    # The sub-4-bit GPU DECODE collapse does not exist. It was gated on bit-width (`bits >= 4`),
    # which made 3.99 bits predict 8.75x slower than 4.00. Pre-registration #16 measured decode
    # all-in-VRAM across three formats and 1.13-3.51 bits and found no collapse anywhere:
    #   Bonsai-27B Q1_0 @1.13b  11.94   |  gemma4-12b K-mix @3.51b  9.56  |  Qwen2.5-7B IQ3_XS @3.3b  18.11
    # The lowest efficiency ever measured on this path is 0.272; gl = 0.04 sits 6.8x below that
    # floor. Worse than inaccurate, it inverted advice: gemma was predicted at 1.0 tok/s so the
    # planner recommended pure CPU (3.9) over a placement that actually runs 9.56.
    # What IS real is a PREFILL effect, and it is format-dependent, not bit-width-dependent: on a
    # matched pair (same model, same card) IQ3_XS costs 6.80x in prefill but only 1.55x in decode
    # - IQ dequant is compute, prefill is compute-bound, decode is bandwidth-bound and hides it.
    # `gl` is retained in the machine table for the prefill model; it must not gate decode.
    geta_w = geta
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
        # MoE partial expert offload: attention+KV in VRAM, then as many EXPERT layers as still
        # fit, remainder on CPU. Measured pre-registration #13 (2026-07-26, corrected): +12.4% decode,
        # ~2-3x prefill vs all-experts-to-CPU, with a hard CLIFF on overcommit (-29% at one step
        # past the ceiling) - so the free-VRAM headroom is deliberately conservative.
        # Desktop reserve: a real machine is not an empty GPU. Measured on this box during the
        # pre-registration #13 sweep: Explorer + compositor + browser held 0.8-1.5 GB throughout.
        # Overshooting the cutoff costs -29% (measured cliff), undershooting costs a few percent,
        # so the asymmetry is deliberately resolved toward caution.
        v_free = vc * 0.90 - v_need - DESKTOP_VRAM_RESERVE
        experts_gb = size - ne * ab / 8 * 1.08
        if v_free > 0.3 and experts_gb > 0:
            f = min(1.0, v_free / experts_gb)
            ram_left = experts_gb * (1 - f)
            if f > 0.05 and ram_left <= ra:
                t_split = (act_ne / (geta * vb) + f * act_ex / (geta * vb)
                           + (1 - f) * act_ex / (eta_r * rb) + kv_gb / (ETA_KV * vb))
                fl = moe_split_flags(f, n_layer)
                # Only offer it if we can emit the EXACT command. Advertising a speed the
                # printed flags cannot deliver is the v1.6.5 bug class; without a layer count
                # the row is suppressed and the footer tells the user how to unlock it.
                if fl:
                    out.append((f"split experts: {f:.0%}->VRAM, rest->RAM", 1 / t_split, None, fl))
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
    if moe and vc > 0 and size + kv_gb > ra:
        # three-tier expert cache (VRAM + RAM + disk): what expert-caching runtimes achieve.
        # llama.cpp mainline cannot LRU experts in VRAM - its number is the row above.
        v_res = ne * ab / 8 * 1.08 + 1.2 + kv_gb           # attention + KV live in VRAM here
        vfree = max(0.0, vc * 0.90 - v_res)
        if vfree > 0.5:
            cache = ra * 0.9 + vfree
            miss3 = max(0.0, 1 - cache / size)
            hot = act_ne
            streamable = act - hot
            vshare = vfree / cache if cache > 0 else 0.0
            hit = streamable * (1 - miss3)
            tps3 = 0.95 / (streamable * miss3 / db
                           + hit * (1 - vshare) / (eta_r * rb)
                           + hit * vshare / (geta * vb)
                           + hot / (geta * vb)
                           + kv_gb / (ETA_KV * vb))
            out.append(("stream from disk (VRAM+RAM expert cache)", tps3,
                        "needs an expert-caching runtime (ktransformers/colibri-class) - stock llama.cpp gets the RAM-cache row",
                        "-ngl 99 + runtime-managed expert cache"))
    out.sort(key=lambda x: -x[1])
    return size, act, out


def qual_of(moe, bits):
    keys = sorted(QUAL[moe])
    return QUAL[moe][min(keys, key=lambda k: abs(k - bits))]


def check_presets(args):
    """Refuse unknown preset names LOUDLY instead of silently falling back to defaults."""
    mdl = getattr(args, "model", None)
    if mdl and mdl not in MODELS and not getattr(args, "total", None):
        raise SystemExit(
            "unknown --model '%s' (and no --total to describe it).\n"
            "  presets: %s\n"
            "  or describe it:      --total <B> --active <B> [--always-active <B>]\n"
            "  or point at a file:  --gguf model.gguf   (exact spec read from the GGUF)" % (mdl, ", ".join(sorted(MODELS))))
    mac = getattr(args, "machine", None)
    if mac and mac not in MACHINES:
        raise SystemExit(
            "unknown --machine '%s'.\n"
            "  presets: %s\n"
            "  or pass flags: --vram/--vram-bw/--ram/--ram-bw/--disk-bw   (no flags = auto-detect this box)" % (mac, ", ".join(sorted(MACHINES))))


def run(args):
    from . import spec as specmod
    specmod.apply(args)
    check_presets(args)
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
    vc = agg_cap(vc) or 0; vb = agg_bw(vb, 0.85) or 0
    rc = rc or 16; rb = rb or 40
    db = agg_bw(db, 0.75) or 0.5
    ctx = getattr(args, "ctx", 0) or 0
    kvp = (args.kv_per_pos * 1024 if getattr(args, "kv_per_pos", None)
           else m.get("kvp", DEFAULT_KVP))

    nlay = effective_n_layer(args, m)
    import os as _os
    _g = getattr(args, 'gguf', None)
    true_size = _os.path.getsize(_g) / 1e9 if _g and _os.path.isfile(_g) else None
    size, act, cfgs = evaluate(t, a, ne, moe, args.bits, vc, vb, rc, rb, db, geta, gl=gl, ctx=ctx,
                               kvp=kvp, n_layer=nlay, true_size_gb=true_size)
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
    fit_adv = fits_in_vram_advice(best[0], args.bits)
    if fit_adv:
        print(f"\n  note: {fit_adv}")
    adv = speculation_advice(moe, best[0])
    if adv:
        print(f"\n  speculation: {adv}")
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
    # tier-boundary advisor: the biggest cliffs sit at tier boundaries (REAP finding, pre-reg #8 P-c).
    # If a modest shave of the FILE crosses one, say so and price it.
    kvg = ctx * kvp / 1e9 if ctx > 0 else 0.0
    size_now = size
    for tier_name, cap, lever in (("VRAM (all-in-VRAM)", vc * 0.90, "next quant step down / tighter probed band / pruned variant"),
                                  ("RAM (pure-CPU)", max(rc - 4, 1), "next quant step down / tighter probed band")):
        if cap <= 0 or size_now + kvg <= cap:
            continue
        gap = size_now + kvg - cap
        if gap > size_now * 0.30:
            continue                                   # not a "shave" - a different model class
        fit_scale = max(0.05, (cap - kvg) / size_now) * 0.995
        _, _, c9 = evaluate(t, a, ne, moe, args.bits, vc, vb, rc, rb, db, geta, fit_scale, gl,
                            ctx=ctx, kvp=kvp)
        promoted = [x for x in c9 if x[0].startswith("all in VRAM")] if "VRAM" in tier_name else                    [x for x in c9 if x[0].startswith("pure CPU")]
        if promoted and promoted[0][1] > best[1] * 1.15:
            print(f"  tier-boundary advisor: this config is {gap:.1f} GB over the {tier_name} boundary - "
                  f"shave it ({lever}) -> ~{promoted[0][1]:.1f} tok/s (x{promoted[0][1]/best[1]:.1f})")
        break                                          # nearest boundary only
    # Fires only when the layer count is genuinely unknown. It used to test args.n_layer - the raw
    # CLI flag - and so told users to "re-run with --gguf to unlock it" while the exact -ot flags
    # for layers 10-47 sat printed directly above, because presets supply `nl` through the same
    # fallback the placement rows already use. Same class as the v1.10.5 n_layer divergence: a
    # second reader of a value that has a fallback, written without the fallback.
    if moe and vc > 0 and not nlay:
        print("\n  note: MoE partial expert offload (measured +12.4% decode, ~2-3x prefill) needs this\n"
              "  model's layer count to emit exact -ot flags - re-run with --gguf <file> to unlock it.")
    print("\n  (eta bands fitted from published measurements; estimates +/-25%. "
          "Hybrid needs --no-mmap.)")
