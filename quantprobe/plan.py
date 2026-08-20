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
    "mistral-7b": dict(t=7.2,  a=7.2,  ne=7.2,  moe=False, kvp=131072, nl=32, hint="Mistral 7B"),              # 32L x 8KV x 128d (exact)
    "glm-air":    dict(t=110,  a=12,   ne=2.7,  moe=True,  kvp=188416, nl=46, hint="GLM-4.5-Air 106B"), # 46L x 8KV x 128d GQA (exact, zai-org/GLM-4.5-Air config.json)
    "glm-744b":   dict(t=753.3, a=32,  ne=8,    moe=True,  kvp=89856,  nl=78, hint="GLM-5.2 (753B)"),   # MLA: 78L x (512+64) latent (exact, zai-org/GLM-5.2 config.json); a/ne [est]
    "qwen3-235b": dict(t=235.1, a=22,  ne=7.5,  moe=True,  kvp=192512, hint="Qwen3-235B-A22B"),         # total exact; kvp: 94L x 4KV x 128d [est]
    "kimi-k2.6":  dict(t=1058.6, a=32, ne=6,    moe=True,  kvp=70272,  hint="Kimi K2.6 (1T MLA)"),      # total exact; kvp: 61L x 576 MLA latent [est]
    "gpt-oss-120b": dict(t=120.4, a=5.1, ne=1.8, moe=True, kvp=73728,  hint="gpt-oss-120b"),            # total+active official; kvp: 36L x 8KV x 64d [est]
    "llama-70b":  dict(t=70.6, a=70.6, ne=70.6, moe=False, kvp=327680, nl=80, hint="Llama-3.3-70B"),           # dense; kvp: 80L x 8KV x 128d
}
DESKTOP_VRAM_RESERVE = 1.0   # GB held by the OS/desktop on a real machine, not the model's to use.
                             # Measured 0.8-1.5 GB on this box (pre-registration #13 sweep).
                             # Applied ONLY to the new MoE split row: existing rows keep their
                             # published anchors untouched.
DENSE_PROTECTED_SHARE = 0.214  # fraction of a DENSE model the depth-aware recipe holds at >=4.5
                               # bits. The recipe protects attn_/ssm_ only - not embeddings, not
                               # the FFN. Mean of the attention share measured from real GGUF
                               # tensor shapes: 10.8% (Qwen2.5-7B), 20.2% (gemma4-12B), 25.1%
                               # (Qwen3.5-4B), 29.6% (Qwen3-0.6B). MoE is unaffected: there `ne`
                               # already names the protected set exactly.
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


UBATCH_HEADROOM_GB = 1.5   # VRAM the compute buffer needs before -ub 2048 is safe. NOT yet
                           # characterised as a curve - measured only as a boundary: with experts
                           # on CPU (little in VRAM) -ub 2048 gave +73% prefill, while a 4.36 GiB
                           # model filling a 6 GB card LOST 39% at the same setting. The ceiling
                           # is real; its shape is an open measurement (pre-registration #19).


def moe_vram_resident_gb(ne, bits, moe):
    """GB a MoE split keeps resident in VRAM (attention + shared experts, bits floored at the
    recipe's protected 4.5, x1.08 overhead) - the input ubatch_flags and phase_advice price.
    ONE spelling on purpose: report.py renders the same -ub decision from the same number, and
    a constant drifting here must drift for both or plan and report emit different commands
    for the same file (the exact disagree-class the report's no-second-physics invariant
    forbids)."""
    return ne * max(bits, 4.5) / 8 * 1.08 if moe else 0.0


def ubatch_flags(placement, vram_resident_gb, vc):
    """-b/-ub for placements that leave weights in HOST memory. Prefill-only lever.

    With `-ot ...=CPU` the expert tensors live in a host buffer, so CUDA is offered the op and
    accepts it once the ubatch clears 32 tokens (ggml-cuda.cu: MUL_MAT_ID -> op->ne[2]). Those
    weights then cross PCIe ONCE PER UBATCH instead of once per token, so the per-token transfer
    cost falls as 1/ub. Measured on the reference box (pre-registration #19, r=3):

        Qwen3-30B-A3B, -ot exps=CPU   pp2048  199.90 -> 345.89   +73%   at ub 512 -> 2048
        dense 7B fully in VRAM        pp2048  329.80 -> 200.31   -39%   same flag, opposite sign

    The control is why this is gated rather than defaulted: with nothing host-resident there is no
    transfer to amortise and a larger ubatch only inflates the compute buffer, which on a tight
    card costs more than it saves. Decode is unaffected either way (18.46 -> 18.76), because a
    ubatch cannot be filled one token at a time.
    """
    # The SPLIT placement is excluded even though it is partly host-resident, and this correction
    # comes from measurement (pre-registration #20) after v1.13.0 shipped the wrong gate:
    #
    #   placement                    pp2048 @ub512   pp2048 @ub2048
    #   all experts -> CPU              199.90          349.59   (+75%)
    #   split, K=16 experts -> VRAM     279.07          161.87   (-42%)
    #
    # The split exists to fill spare VRAM with experts - so by construction it consumes the very
    # headroom the larger compute buffer needs, and the flag that pays 75% on one placement costs
    # 42% on the other. v1.13.0 gated on "is anything host-resident", which is true for the split
    # ("...->VRAM, rest->RAM"), and so recommended it there. Wrong, and measured wrong.
    #
    # Note this also INVERTS which placement is fastest at prefill: at the default ub the split
    # wins (279 vs 200), at ub 2048 the all-CPU placement wins (350 vs 162). Placement and batch
    # are not independent dimensions - they compete for the same VRAM.
    if vc <= 0:
        return None
    if "split experts" in placement:
        # U-16, closed by measurement: the #20-era total exclusion of the split was over-broad.
        # #20's harm was ub 2048 (the compute-buffer cliff: 279 -> 162); #62 measured the SAME
        # split placement at ub 1024 with pp 393.7 AND tg 22.21 - no cliff, both metrics at
        # their best - while the excluded emitted config measured pp 301 (#66). The split gets
        # ub capped at 1024 HARD (never 2048, where the measured cliff lives), still sized by
        # the buffer math against the residual headroom.
        ub = safe_ubatch(vc * 0.90 - vram_resident_gb, cap=1024)
        return "-b %d -ub %d" % (ub, ub) if ub else None
    if "experts->RAM" in placement:
        # L-26 (out-of-sample run; predictions staked in the header of
        # weights/data/prereg92b_ub_oos.log BEFORE it ran - there is no prereg DOCUMENT for
        # it, and the '92b' in that filename must not be read as prereg #92, which is a
        # different, unrelated experiment): the CPU-expert
        # placement (all routed experts in host RAM, attention+KV in VRAM) follows
        # sec/token = A + X/C, validated with staked predictions at C=256 (-0.27%) and C=4096
        # (-8.2%, inside the 10% kill band).
        #
        # WHAT THE ub-4096 ARM ACTUALLY MEASURED, stated precisely because the first write-up of
        # it was not: 360.76 tok/s at **pp4096**, against prereg #19's 345.89 at **pp2048**/ub2048.
        # Those are two different prompt lengths. The like-for-like control (pp4096 at ub 2048)
        # was never run, and it CANNOT be inferred from #19 except through the very law under
        # test - which missed by -8.2% at that same C. So "+4.3% for one flag" is a
        # LAW-MEDIATED comparison, not a measured A/B, and it is quoted that way everywhere it
        # is printed. Note also that -ub 4096 is a CAP: on a prompt of 2048 tokens or fewer it
        # changes nothing at all, so whatever it is worth, it is worth it only past 2048 tokens.
        # ON THIS 6GB REFERENCE CARD. The generic half-budget margin below was
        # calibrated on the SPLIT (#23), where resident experts compete with the compute buffer
        # for the same VRAM; on this placement nothing else grows, and the measured ub-4096
        # buffer (2406 MiB) ran clean at 51.5% of headroom - which the half rule rejects by
        # 15 MiB. So this branch sizes against the FULL budget after reserving the same
        # UBATCH_HEADROOM_GB of absolute slack the generic gate demands, takes the generic
        # answer as a floor (a tight card keeps exactly what it had), and HARD-CAPS at 4096:
        # the fit's asymptote bends below 1/A, so extrapolation past C=4096 is not licensed.
        # Scope (L-26): CPU-expert MoE tier ONLY - the dense-in-VRAM control VIOLATES the form
        # (prereg #19 P-2, -39% at ub 2048), which is why dense rows never reach this branch.
        head = vc * 0.90 - vram_resident_gb
        if head < UBATCH_HEADROOM_GB:
            return None                   # no room for the bigger compute buffer; the -39% case
        ub = max(safe_ubatch(head, cap=SAFE_UBATCH_CAP),
                 safe_ubatch(head - UBATCH_HEADROOM_GB, cap=SAFE_UBATCH_CAP, margin=1.0))
        return "-b %d -ub %d" % (ub, ub) if ub else None
    if not any(k in placement for k in ("->RAM", "CPU", "disk")):
        return None                       # nothing host-resident: no transfer to amortise
    if vc * 0.90 - vram_resident_gb < UBATCH_HEADROOM_GB:
        return None                       # no room for the bigger compute buffer; the -39% case
    # Dense splits, pure CPU and disk rows cap at the 2048 the half-budget rule was measured
    # with. They must NEVER see SAFE_UBATCH_CAP's 4096: that number is licensed by L-26 for the
    # CPU-expert MoE tier only (the branch above), and the dense control in the same experiment
    # violates the very law that licenses it (prereg #19 P-2). Before this gate, a dense
    # layer-split on a roomy card was quietly handed -ub 4096 on a MoE datapoint's authority.
    ub = safe_ubatch(vc * 0.90 - vram_resident_gb, cap=2048)
    return "-b %d -ub %d" % (ub, ub) if ub else None


# llama.cpp's CUDA compute buffer is EXACTLY LINEAR in the ubatch. Measured (pre-registration #23,
# Qwen3-30B-A3B-Q2_K, llama-bench -v, the runtime's own report):
#
#   -ub 1024   601.50 MiB      -ub 1536   902.25 MiB      -ub 2048  1203.00 MiB
#
# 0.5874 MiB per ubatch token at all three points - no curvature, no step. That linearity is what
# makes the cliff predictable rather than merely observable: the buffer grows smoothly, VRAM runs
# out abruptly, and the speed falls off where the two cross.
COMPUTE_BUFFER_MIB_PER_UB_TOKEN = 0.5874


def safe_ubatch(headroom_gb, cap=2048, margin=0.5):
    """Largest power-of-two ubatch whose compute buffer fits in `headroom_gb`, or 0.

    This replaces a hard-coded `-ub 2048`, and the replacement is a bug fix, not a refinement.
    Pre-registration #23 swept the ubatch on the split placement with KV evicted - the exact
    configuration v1.14.x shipped as the prefill champion - and found a cliff, not a plateau:

        -ub  512  303.32      -ub 1536   381.21  <- buffer 902 MiB, still fits
        -ub 1024  387.37      -ub 2048   209.64  <- buffer 1203 MiB, does not     -45% in one step
                              -ub 3072   209.46
                              -ub 4096   210.84

    Past the edge the runtime does not fail; it silently spills and holds that degraded speed for
    every larger ubatch. So the failure is invisible to anyone who does not sweep, which is why we
    shipped it: v1.14.x quoted 391.72 tok/s for a command that delivers 209 on the same box the
    391.72 was measured on. The difference was ~250 MiB of desktop VRAM - one browser window.

    A HALF of the nominal headroom is required, not all of it: the buffer is not the only thing
    that grows, and the measured margin between the last good point (902 MiB) and the first bad one
    (1203 MiB) is under 300 MiB on a 6 GB card. Sizing to the last byte would re-create the cliff
    one step further out.

    The cap: 2048 was the ceiling this 6 GB box could validate - past it the buffer math already
    said no, so larger values were never reachable here. On big-VRAM cards that cap was doing the
    limiting itself: the first external replication (RTX 3090 24GB, 117B MoE) reported prefill
    jumping 90 -> 470 tok/s at -b 4096 -ub 4096 [external datapoint, analogalok's 4090 numbers via
    u/MoneroApe - NOT validated on our hardware; the buffer-fit math still gates it, so a card
    without the headroom never sees 4096]. MoE prefill benefits disproportionately: each layer's
    expert dispatch is one big matmul, and the batch amortizes it.

    `margin` is the fraction of nominal headroom the buffer may claim. The default 0.5 IS the
    half-budget rule above and no caller may weaken it casually: the single exception is the
    L-26 CPU-expert MoE branch of ubatch_flags, which passes margin=1.0 against a headroom it
    has ALREADY debited by UBATCH_HEADROOM_GB of absolute slack - grounded there by a direct
    measurement of the ub-4096 buffer running clean on the reference card (the ub-OOS run above; staked in its log header, not in a prereg document), not by
    optimism about the cliff #23 mapped.
    """
    budget_mib = max(0.0, headroom_gb) * 1024 * margin
    ub = 0
    n = 256
    while n <= cap:
        if n * COMPUTE_BUFFER_MIB_PER_UB_TOKEN <= budget_mib:
            ub = n
        n *= 2
    return ub


# prereg #19 P-1, verbatim: pp2048 on the CPU-expert MoE placement (-ot exps=CPU), one session,
# r=3, from weights/data/prereg19_ubatch.log. THREE CELLS - that is the whole sweep, and the
# headline "+73%" is the 2048 one against the 512 baseline. There is no 4096 cell here: the
# out-of-sample run measured 360.76 at pp4096, a different workload (see ubatch_flags).
UB_SWEEP_PP2048 = {512: 199.90, 1024: 277.17, 2048: 345.89}

SAFE_UBATCH_CAP = 4096      # reachable ONLY on the CPU-expert MoE branch of ubatch_flags: L-26
                            # validated sec/tok = A + X/C out of sample to C=4096 there (and the
                            # 3090 external datapoint was the same placement). Dense/other
                            # host-resident rows cap at 2048 - the dense-in-VRAM control violates
                            # the form (prereg #19 P-2), so 4096 was never licensed for them.


# The measured Pareto frontier for a MoE whose experts do not fit VRAM, on the reference box
# (Qwen3-30B-A3B Q2_K, pre-registration #21, one session, r=3, -ub 2048). There is no single best
# placement: three configurations are Pareto-optimal and the right one depends on how many prompt
# tokens you read per token you generate.
#
#   (label, pp2048, tg128, extra flags beyond the placement's own)
MOE_FRONTIER = [
    ("KV in VRAM, safe batch",          386.04, 21.58, "-b 1024 -ub 1024"),
]
# Re-measured 2026-07-27 (pre-registration #25), ALL FOUR CELLS IN ONE SESSION with matched flags,
# because pre-registration #24 established 10-13% of drift BETWEEN sessions against sub-1% error
# bars within one. The previous table compared numbers from different days.
#
#   placement / batch                    pp2048    tg128
#   split, ub 512                        307.13    22.02   <- kept, but see the margin note
#   split, ub 1024, KV in VRAM           386.04    21.58   <- NEW, and it dominates two shipped rows
#   all experts to CPU, ub 2048          381.82    19.79   <- DROPPED: dominated on both axes
#   split, ub 1024, KV evicted (-nkvo)   381.60    17.95   <- DROPPED: dominated on both axes
#
# Two of the three rows we shipped were dominated by a cell nobody had measured, and the reason is
# instructive. "split + KV in VRAM" was only ever benchmarked at ub 2048 (161.59 pp - past the
# compute-buffer cliff, see safe_ubatch) and at ub 512 (307.13 pp). At ub 1024 it wins outright.
# So the whole "evict KV to buy prompt speed" trade was an ARTEFACT of measuring one cell past a
# cliff: keeping KV in VRAM gives MORE prefill (386.04 vs 381.60) and 20% more decode. The frontier
# had three points because the search never tried the one that beats them.
#
# THE FRONTIER IS GONE, and dropping it is the honest reading of our own numbers. The ub-512 row
# (307.13 / 22.02) survives Pareto only on decode, by 22.02 vs 21.58 - a 0.44 tok/s gap against
# combined error bars of 0.456, i.e. 0.96 sigma. That is not a measurement, it is noise, and a row
# retained on it manufactures exactly the false frontier this table has already produced twice.
#
# So Law 7 - "there is no single best placement, there is a frontier selected by the prompt:
# generation ratio" - is REFUTED for this model on this box. There IS a single best configuration.
# The frontier existed because three of its four cells were measured in the wrong place: one past
# the compute-buffer cliff, one on a different day, and one under a flag (-nkvo) that was never
# needed. Re-measured together, one point wins everything.
#
# The claim shrank three times before it died: 2.25x spread, then 1.33x once a dominated point was
# corrected, then 1.23x once all cells shared a session - and at that width the surviving choice is
# inside its own error bars. Each shrink came from fixing one of our own measurement errors, which
# is the pattern worth remembering: a feature that only looks valuable through a flawed measurement
# stops looking valuable as the measurement improves.
#
# Flash attention was tested and does NOTHING here: 387.24/21.69 with -fa 1 against 386.04/21.58
# without, both inside their error bars. The entire win is the ubatch and KV residency.
# Row 3 shipped as `-ub 2048 / 391.72 / 16.54` in v1.14.0-v1.14.1 and that command was ON THE WRONG
# SIDE OF A CLIFF (pre-registration #23). Re-measured across the ubatch on the same box:
#
#   -ub 1024  386.14 pp  18.06 tg   <- shipped now
#   -ub 1536  381.21 pp
#   -ub 2048  209.64 pp             <- shipped before: -45.7% prefill, and 16.54 tg
#
# The correction costs 1.4% of the quoted prefill number and BUYS 9.2% of decode, so the row is
# strictly better than the one it replaces on one axis and within noise on the other. The 391.72
# was not fabricated - it is reproducible on a card with nothing else on it. It is simply not
# reproducible on a card with a browser open, and advice that needs a clean desktop to hold is
# advice we should not have given.
# Two cells are deliberately EXCLUDED because they are dominated on both axes:
#   all experts to CPU + KV evicted     336.31 / 15.82  - beaten by row 2
#   split + KV in VRAM + ub 2048        163.39 / 20.13  - beaten by row 1
#
# That second one shipped ON this frontier in v1.14.0, as the "decode champion", because the
# frontier was built with -ub pinned at 2048 while only placement and KV were varied. Measured
# afterwards: the same configuration at ub 512 gives 280.64 pp for the same 20.25 tg - 72% more
# prompt processing, free. Pinning one dimension while sweeping the others is precisely the error
# the Law 4 fungibility corollary warns about, made in the code that implements the corollary.
# A frontier is only Pareto-optimal with respect to the dimensions actually swept.
#
# EPILOGUE, 2026-07-27: that same excluded cell - "split + KV in VRAM" - is now the WINNER. It was
# excluded on a measurement at ub 2048, which is past the compute-buffer cliff; at ub 1024 it beats
# every row we shipped. The lesson repeated itself one level up. Excluding a configuration is also
# a claim, and it inherits the scope of the sweep that produced it: this one was really "dominated
# AT ub 2048", which is not the same statement and should never have been recorded as though it was.


def workload_frontier(prompt_to_gen):
    """Pick the configuration that minimises total time at a given prompt:generation ratio.

    Kept as a function, and it now returns the SAME row at every ratio, because pre-registration
    #25 collapsed the frontier to one point. Total time for P prompt tokens and G generated is
    still `P/pp + G/tg`, so the machinery is correct and would select again the moment a second
    configuration is measured that beats this one somewhere - which is why this is not deleted.

    What it must NOT do is present a choice that does not exist. Every earlier version of this
    docstring quoted a workload-dependent crossover (2.25x, then 1.33x, then 1.23x); all three were
    artefacts of comparing cells measured past a cliff, on different days, or under a flag that was
    never needed. See MOE_FRONTIER for the numbers that replaced them.
    """
    G = 1.0
    P = max(prompt_to_gen, 0.0) * G
    scored = [(P / pp + G / tg, lab, pp, tg, fl) for lab, pp, tg, fl in MOE_FRONTIER]
    scored.sort()
    best = scored[0]
    worst = scored[-1]
    return dict(label=best[1], pp=best[2], tg=best[3], flags=best[4],
                speedup_vs_worst=worst[0] / best[0])


def phase_advice(placement, rows, vram_resident_gb=None, vc=None):
    """Which phase does the recommended command actually optimise, and what does it cost?

    `plan` prints ONE command, and the placement that maximises generation is not the placement
    that maximises prompt processing. Measured on the reference box (pre-registration #20,
    Qwen3-30B-A3B, one session, r=3):

        placement                  pp2048 @ub2048   tg128
        all experts -> CPU            349.59        18.54
        split, K=16 -> VRAM           161.87        20.16

    The split wins generation by 9% and loses prompt processing by 2.16x. Ranking by decode - which
    is what this tool does - silently hands long-prompt users the worse configuration by a factor
    of two, so it has to say so.

    Not a law change and not a second command: llama.cpp is started once, with one placement. This
    tells the user which one they are getting and what the other is worth.
    """
    if "split experts" not in placement:
        return None
    alt = next((r for r in rows if r[0].startswith("hybrid")), None)
    if not alt:
        return None
    # L-26: the batch flags quoted for the alternative are SIZED by the same gate that would emit
    # them if this placement won, never re-hardcoded here (the hardcoded "-b 2048 -ub 2048" this
    # replaces predates the out-of-sample validation of the prefill law at C=4096). Callers that
    # do not know the card keep the ub-2048 wording the numbers in this sentence were measured at.
    alt_ub = (ubatch_flags(alt[0], vram_resident_gb or 0.0, vc) if vc else None) \
        or "-b 2048 -ub 2048"
    tail = ""
    if "-ub 4096" in alt_ub:
        tail = (" At -b 4096 -ub 4096 that placement measured 360.76 tok/s AT pp4096 out of "
                "sample (L-26; predictions staked in weights/data/prereg92b_ub_oos.log before the "
                "run). The nearest published baseline is 345.89 at "
                "pp2048/ub2048 (prereg #19) - a DIFFERENT prompt length, so the ~+4% between "
                "them is what the prefill law sec/tok = A + X/C implies, not a measured A/B; "
                "the pp4096-at-ub2048 control was never run. The law is validated to C=4096 and "
                "NOT past it, so do not raise the batch further. -ub is a cap: below ~2048 "
                "prompt tokens it changes nothing.")
    return ("this command is tuned for GENERATION. If your prompts are long - agent transcripts, "
            "RAG context, whole files - the other placement is far better at reading them: "
            "measured 349.6 vs 161.9 tok/s prompt processing (2.2x), for 8% less generation. "
            f"Use `{alt[3]} {alt_ub}` instead. Prompt processing and generation genuinely "
            "want different placements here, and one command cannot serve both." + tail)


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


# U-23 (E-08, a 32 GB box whose 20 GB working set left 330 MiB free): --no-mmap buys measured
# speed but makes the model's pages NON-EVICTABLE. On a RAM-tight box that is an OOM risk under
# desktop load, and the reporter was right to override us. Keep mmap when the host share leaves
# less than this much RAM free, and say which way the trade went either way.
# Threshold on the SAME basis as the pinning warning above it (share of the post-reserve pool),
# deliberately NOT a second absolute reserve - stacking one on `ra` would repeat the double-count
# #62 removed. 0.75 keeps --no-mmap on every configuration this box has measured (the flagship
# split holds ~58% of the pool and gained +13.7% from it) and drops it only where the placement
# dominates RAM, which is E-08's regime.
MMAP_HOST_SHARE_CAP = 0.75


def mmap_decision(host_gb, ram_gb):
    """(use_no_mmap, note). host_gb = how much this placement holds in RAM.

    U-23 v1 (v1.23) DROPPED --no-mmap on RAM-tight placements to keep pages evictable. The full
    ladder measured what that costs and the remedy was REFUTED on its own box: the flagship split
    ran 22.20 tok/s with --no-mmap and 7.70 without it - a 2.9x collapse, not the +13.7% the note
    promised. The mechanism is exactly the RAM-tightness the gate was reacting to: near the RAM
    boundary the OS pages the mapped file in and out every token. So the honest answer is not to
    pick for the user - it is to give them both measured numbers and default to the fast one.
    """
    if not ram_gb or (host_gb or 0) <= ram_gb * MMAP_HOST_SHARE_CAP:
        return True, None
    return True, (f"RAM-TIGHT TRADE-OFF, both sides measured: this placement holds ~{host_gb:.0f}GB "
                  f"of your ~{ram_gb:.0f}GB usable RAM ({host_gb / ram_gb:.0%}). We keep "
                  f"--no-mmap because dropping it measured **2.9x SLOWER** here (22.20 -> 7.70 "
                  f"tok/s: near the RAM boundary the OS pages the mapped file in and out every "
                  f"token). The risk you accept: --no-mmap pages are NOT evictable, so under "
                  f"desktop memory pressure the process can OOM instead of slowing down (U-23, "
                  f"reported by a user whose 32GB box was left with 330 MiB free). If your box "
                  f"OOMs, remove --no-mmap and expect roughly a third of the speed, or offload "
                  f"fewer layers so less sits in RAM.")


def moe_split_flags(frac, n_layer, host_gb=None, ram_gb=None):
    """-ot regex placing the FIRST ceil(frac*L) layers' experts on GPU, the rest on CPU.
    Measured 2026-07-26 (pre-registration #13, corrected): +12.4% decode and ~2-3x prefill
    on a 6 GB card, against a properly configured baseline.
    Returns None when the layer count is unknown - we will not emit a regex we cannot ground."""
    if not n_layer or frac <= 0:
        return None
    k = max(1, min(n_layer - 1, int(frac * n_layer)))
    # --no-mmap for the same reason the all-experts row carries it: llama.cpp itself warns that
    # tensor overrides to CPU with mmap enabled cost performance. Measured 2026-07-26 on this
    # split placement: 16.45 tok/s with mmap vs 18.70 without (+13.7%). Gated by U-23 above.
    use_nm = True if host_gb is None or ram_gb is None else mmap_decision(host_gb, ram_gb)[0]
    return ('-ngl 99 -ot "blk\\.(%s)\\.ffn_.*_exps\\.=CPU"%s'
            % ("|".join(str(i) for i in range(k, n_layer)), " --no-mmap" if use_nm else ""))


# L-19, validated out-of-sample by prereg #74. Seconds per KV position per CPU-RESIDENT layer
# per token: CPU attention over the whole cache. Measured 1.43 (7B, 8 layers, d16k), 1.42 (7B,
# 11, d32k), 1.76 (14B, 30, d8k) us; this is their MEAN, not the value that flatters one arm.
# Scales with THIS box's 4 threads - a wider CPU pays less, which is why the note below still
# says the number is soft even though it is now priced.
CPU_ATTN_S_PER_POS_LAYER = 1.55e-6


def depth_scope_warning(placement, moe, ctx):
    """L-19 (prereg #73): the context term is a BANDWIDTH model, true only where attention runs
    on the GPU. Measured to 32k: all-in-VRAM exact (-0.9% at d4096), MoE expert-splits in band at
    every depth (-10%..-23% - a MoE split keeps attention and KV in VRAM, only expert FFNs go to
    CPU). DENSE splits put whole layers INCLUDING attention on the CPU and break: +182% over at
    d16384, +381% at d32768, because CPU attention over the whole cache is compute on N threads,
    not a byte read. Rather than print a number we cannot stand behind, say so."""
    if ctx <= 4096 or moe or "layers->VRAM" not in (placement or ""):
        return None
    return (f"DEPTH NOTE (preregs #73/#74, L-19): this is a DENSE split at {ctx} context, where "
            f"the CPU-resident layers run attention over the whole KV cache every token - compute "
            f"on your threads, not a byte read. The number above INCLUDES that cost "
            f"(1.55 us/position/layer, measured across 8-30 CPU layers and 8k-32k depth, "
            f"validated out-of-sample); without it we over-promised this regime by 2-5x. Two "
            f"caveats: the constant is from ONE 4-thread CPU (a wider CPU pays less, so this "
            f"reads pessimistic there), and the honest fix is to avoid the regime - raise -ngl "
            f"until attention fits in VRAM, quantize the KV cache (-ctk q8_0 -ctv q8_0: +37% SPEED "
            f"at d16384 measured, prereg #25; quality measured CLEAN at the deepest depth f16 even "
            f"runs on our card - PPL ratio 1.00031 +/- 0.0188 at c7168 against a <=1.02 gate, "
            f"prereg #91/L-24. Scope: wikitext at ~7k, and an external report calls the "
            f"difference large on niche-domain code at 100k+ context, so judge your own domain, "
            f"E-10), or use a MoE model, whose splits keep attention on the GPU and "
            f"are validated to 32k here. Do NOT evict the KV cache to host RAM (-nkvo 1) to make "
            f"room: at d16384 it measured 3.48 tok/s decode against 10.59 for q8_0 KV in VRAM - "
            f"3.04x WORSE - for a prefill difference inside the error bar (382.17 vs 386.14 "
            f"pp2048). Deep contexts (RAG, document QA) are exactly where -nkvo loses hardest; "
            f"that advice is WITHDRAWN per prereg #25's pre-commitment (L-24).")


def fits_in_vram_advice(placement, bits):
    """What we know, and admit we don't know, about the all-in-VRAM row.

    This is the most common configuration for anyone with adequate VRAM, and it is where the law
    is least trustworthy. Two separate facts, both measured, both worth telling the user:

    1. There is NO point prediction for this regime - and pretending otherwise is worse than
       saying so. Measured efficiency varies 0.32-0.56 across 8 models (six candidate
       explanations refuted, prereg #15/#24); a number quoted with a band that wide is not a
       prediction. What survives is a ONE-SIDED BOUND with the same logical form as the law's
       own +/-25% claim, just asymmetric: real speed >= 0.90x the printed number, 13 of 13
       benchmarks, no exceptions, falsified by any single measurement below it. In 12 of the 13
       it was strictly above, typically 1.1x-1.8x.

    2. Below 4.5 bits, quantizing further buys almost nothing. Same 7B, same card, only the
       quantization changed (pre-registration #16, r=3):
           Q4_K_M  4.5 bits  4.68 GB  20.03 +/- 0.04 tok/s
           Q2_K    2.8 bits  3.01 GB  19.17 +/- 0.03 tok/s    36% smaller, 4% SLOWER
       Decode here is not bandwidth-bound, so bytes stop predicting speed and the ranking
       over-rewards low bits.

    Why no fix instead of a disclaimer: seven points on ONE GPU do not identify a functional
    form, and the last two times a constant moved on thin evidence it cost a public correction.
    So the honest move is to say what we know, and ask for the datapoint that would settle it -
    which is why this note ends in a request rather than an apology. The regime is pinned by a
    ratchet in the test suite so it can improve but never silently worsen.
    """
    if placement != "all in VRAM":
        return None
    note = ("we do not have a point prediction for this placement - measured efficiency varies "
            "too much across models for the number above to carry a band (0.32-0.56, six candidate "
            "explanations refuted). What we can state is one-sided and has no exceptions: in 13 of "
            "13 benchmarks real speed was >= 0.90x this number, and in 12 of 13 it was HIGHER, "
            "typically 1.1x-1.8x. Read it as a floor, not a ceiling - a single measurement below "
            "0.90x would falsify this and we ask for exactly that measurement below. "
            "The spread is NOT our old GPU: 61 public benchmarks across 8 architectures "
            "(hopper/ada/blackwell/ampere/apple/rdna3/rdna4/intel-xe) span efficiency "
            "0.51-0.81 on one basis - a 1.6x spread that our own card sits INSIDE - so no "
            "per-architecture constant can replace this floor (prereg #72, L-18). Only a "
            "measurement of YOUR box collapses it: `quantprobe calibrate` does that in ~5 min.")
    if bits < 4.5:
        note += (" It also already fits, and going lower-bit buys almost nothing: the same 7B at "
                 "Q2_K vs Q4_K_M is 36% smaller and 4% SLOWER (19.17 vs 20.03 tok/s). Quantize "
                 "to make a model FIT - once it fits, take the highest bits that still fit.")
    fmt = format_advice(placement, bits)
    if fmt:
        note += "\n  " + fmt
    note += ("\n  We only have one GPU, and one GPU cannot fix this. If you run this model, "
             "`quantprobe bench --contribute` turns your machine into the datapoint that does - "
             "it prints exactly what would be shared and you submit it yourself. Results that "
             "land OUTSIDE our predicted band are the most valuable ones we can receive.")
    return note


def format_advice(placement, bits):
    """The format lever: on an ALU-weak GPU, the quantization FORMAT sets decode speed, not just
    the byte count. Measured (preregs #52/#53/#54, kernelprobe controls, 2026-07-28):

        Qwen2.5-7B, all-in-VRAM, same card, same session, interleaved:
            Q4_K_M  4.36 GB   22.72 tok/s   eta 0.553
            Q4_0    4.12 GB   26.87 tok/s   eta 0.619   <- +19% where bytes explain +5.7%
            Q2_K    2.80 GB   21.67 tok/s   eta 0.340   <- SLOWER than Q4_0 while 32% smaller

    Mechanism, isolated at the metal (own CUDA, zero llama.cpp): a matvec with NO unpacking runs
    at 95% of the streaming ceiling; the same bytes unpacked naively run at 42%. The cost is the
    per-block metadata decode - K-quants unpack a 6-bit scale AND a 6-bit min before any dot
    product, Q4_0 reads one fp16 scale. So eta is a property of (format x kernel), not the tier.

    Scope, stated rather than implied: measured on ONE Pascal-class card (GTX 1060, cc 6.1),
    where ALU is scarce relative to bandwidth. On Ampere+ the unpack has headroom to hide and
    the ranking may invert - we say so and ask for the datapoint. SPEED-only claim: Q4_0 is
    lower-quality per byte than Q4_K_M; at equal fit take K-quants on a modern card, Q4_0 on an
    old one, and never Q2_K when a 4-bit file also fits (it is smaller, slower AND lower quality
    there - strictly dominated).
    """
    if placement != "all in VRAM":
        return None
    if bits <= 3.0:
        return ("FORMAT LEVER (pre-Ampere cards): this bit-width is in the regime where unpack "
                "cost has REVERSED the byte ordering - measured Q2_K decodes 19% slower than Q4_0 "
                "on the same model while being 32% smaller, and CODEBOOK IQ formats pay even more "
                "(IQ2_XS/IQ3 measured 36-52% below Q4_K per byte, prereg #70). If a ~4.5-bit file "
                "fits in VRAM, use it instead; prefer Q4_0 or IQ4_NL over Q4_K_M on pre-Ampere "
                "(+19%/+14% measured). On Ampere+ this is unverified and may invert.")
    if bits <= 5.0:
        return ("FORMAT LEVER (pre-Ampere cards): at this bit-width the FORMAT is worth more than "
                "the bytes - Q4_0 measured +19% over Q4_K_M on the same model (26.87 vs 22.72 "
                "tok/s), and IQ4_NL measured +14% (25.63) at the same size class WITH imatrix "
                "quality: its kernel is Q4_0-class, not codebook-class (prereg #70), so it is the "
                "quality-per-byte pick of the fast formats here. Mechanism (measured, preregs "
                "#56/#57): metadata application DENSITY - K-quants apply fine-grained scale+min "
                "chains far more often per byte, and that cost is intrinsic to the format "
                "definition, not a kernel bug. Avoid codebook IQ (IQ2/IQ3) for VRAM decode on "
                "this class of card: measured 36-52% below Q4_K per byte. On Ampere+ this is "
                "unverified and may invert.")
    return None


def dense_draft_note(moe, placement):
    """The dense-target draft cells, measured (preregs #67/#69). All-in-VRAM: 0.5B drafting the
    dense 7B, NOVEL prompts: +11% on code at --spec-draft-n-max 2 (79% acceptance), net NEGATIVE
    on prose at every draft length, and llama.cpp's default K=3 is past the optimum. SPLIT: the
    best speculation cell on this box - the K+1 verify batch reads each CPU-resident layer once,
    so the CPU share of the token amortizes: +33% on the 14B at K=2 (76% acceptance) with the
    draft itself on CPU (-ngld 0, zero VRAM cost). The K-cliff is HARDER on split (K=3+ lands at
    or below baseline) because the CPU draft pays from the same bandwidth pocket the
    amortization saves."""
    if moe:
        return None
    if "layers->VRAM" in (placement or ""):
        return ("dense-SPLIT speculation (measured, prereg #69): a small same-family draft is "
                "the best speculation cell on this box - `-md draft.gguf -ngld 0 "
                "--spec-draft-n-max 2` bought **+33%** decode on a 14B split target (5.5 -> 7.4 "
                "tok/s, 76% acceptance, novel code), and the draft costs ZERO VRAM because it "
                "runs on CPU. Mechanism: the K+1-token verify batch reads each CPU-resident "
                "layer once, so the CPU share of every token amortizes. Keep K=2: every "
                "measured K>=3 landed AT OR BELOW no-draft baseline (the CPU draft spends the "
                "same RAM bandwidth the amortization saves), so llama.cpp's default 3 already "
                "loses here.")
    if "all in VRAM" not in placement:
        return None
    return ("dense-model speculation (measured, prereg #67): pairing this model with a small "
            "same-family draft (-md draft.gguf -ngld 99 --spec-draft-n-max 2) buys ~+11% on "
            "CODE-style novel output at 79% acceptance - and LOSES speed on prose and at "
            "draft lengths >=4 (llama.cpp's default 3 is past the optimum here). Copy-regime "
            "ngram speculation remains the big multiplier where output copies context.")


# ------------------------------------------------------------------------------------------------
# The two measured speculation multipliers this block quotes, as NUMBERS rather than prose.
#
# C-15 (2026-07-31): each of these is also the per-verify-round AMORTIZATION FACTOR of the drafter
# that produced it. Both were measured on rows whose token is ~100% weight bandwidth, where a
# verify round costs exactly one weight read, so measured_x = accepted tokens per round = R. That
# is what makes them carryable: on a row that spends a fraction c of its token on CPU attention -
# which does NOT amortize, because every drafted position still attends over the whole cache - the
# SAME drafter at the SAME acceptance can reach at most 1/((1-c)/R + c). Printing R itself on such
# a row is printing a number the row's own arithmetic forbids. No new constant is introduced here:
# R is the published measurement, written as a ratio so the source pair stays visible.
SPEC_X_NGRAM_TUNED = 98.80 / 21.32      # 4.63x, flagship split experts @ ~3-bit (preregs #28/#36/#37)
SPEC_X_NGRAM_DENSE = 37.2 / 17.7        # 2.10x, dense 7B all-in-VRAM, code (Law 6 arm S-a)

# How far below its own measurement the row's bound must fall before the headline is replaced.
# The headlines are printed to two significant figures ("4.7x", "2.10x"), so a bound within a
# printed half-ulp of the measurement IS that number to a reader and keeps the measured text
# verbatim. This is a display threshold, not a physical one - it decides which sentence prints,
# never which number is true.
SPEC_HEADLINE_TOL = 0.01


def cpu_attention_share(row):
    """Fraction of this row's token spent on CPU attention over the KV cache, or None."""
    tt = resource_times(row)
    if not tt:
        return None
    total = sum(tt.values())
    return (tt.get("cpu_compute", 0.0) / total) if total > 0 else None


def spec_headline_x(moe, placement):
    """The measured multiplier `speculation_advice` quotes as a HEADLINE for this branch.

    None where the branch quotes no headline multiplier (MoE offload quotes +3%/+11%, and the
    fully-resident MoE case prints nothing at all). Kept next to `speculation_advice` and driven
    by the same two conditions so the two can never disagree about which number is on the page.
    """
    if moe and "split experts" in (placement or ""):
        return SPEC_X_NGRAM_TUNED
    if moe:
        return None
    return SPEC_X_NGRAM_DENSE


def spec_reachable_x(row, measured_x):
    """The most the drafter that measured `measured_x` can deliver ON THIS ROW.

    `measured_x` is a ratio measured where the token is ~100% weight bandwidth, so it IS that
    drafter's per-round amortization factor R. Carrying R to a row whose token is a fraction c of
    non-amortizing CPU attention gives x(c) = 1/((1-c)/R + c): equal to R at c=0, 1.0 at c=1,
    monotone between. Optimistic on purpose (100% acceptance, no MoE union tax), because it exists
    to BOUND an over-sold headline, and an optimistic bound can only ever keep a claim we might
    otherwise have dropped.

    Returns None when the row carries no decomposition, or when there is no headline to bound.
    """
    if not measured_x or measured_x <= 1.0:
        return None
    return spec_ceiling(row, k=measured_x - 1.0)


def speculation_advice(moe, placement, row=None):
    """What speculative decoding is worth for THIS model and placement.

    Every number here is measured on the reference box (Law 6 arms S-a/S-b/S-e/S-f, 3 runs per
    cell, raw logs in weights/data/). Speculation drafts tokens and verifies them, so output is
    identical - the only question is whether the verify batch costs more than the pass it saves.

    dense       code  2.10x | prose 1.01x   (ngram; copyability is the whole mechanism)
    MoE offload code  1.03x                 (the verify batch unions experts - the tax eats it)
    dense       MTP   1.17x GPU-resident, 1.046x CPU
    MoE offload MTP   0.76x                 (an actual LOSS)

    C-15: `row` is the winning Row, and when it is supplied the HEADLINE is bounded by that row's
    own decomposition instead of being quoted flat. It stays optional so the six modules that call
    this with a placement string keep working - but a caller that omits it gets the measured
    number on a row that may not be able to reach it, which is the defect this argument exists to
    close. Every call site inside run() passes it.

    Returns None when we have no measurement for the case rather than guessing.
    """
    experts_offloaded = "exps=CPU" in (placement or "")
    # The headline, priced against the row it is about to be printed on. `head` is prose, not a
    # number, so a branch that quotes no headline (MoE offload) is unaffected.
    hx = spec_headline_x(moe, placement)
    reach = spec_reachable_x(row, hx) if (row is not None and hx) else None
    oversold = bool(reach and hx and reach < hx * (1.0 - SPEC_HEADLINE_TOL))
    cpu_c = cpu_attention_share(row) if row is not None else None
    if moe and "split experts" in (placement or ""):
        # Measured on the flagship ITSELF, on this exact placement (pre-registration #28). The
        # axis that matters is COPY vs NOVEL, not code vs prose: the prize attaches to output
        # that reproduces context spans (edits, refactors, quoting) - most of what coding agents
        # emit, and none of what fresh generation emits.
        head = ("if your output REUSES its context - edits, refactors, RAG quoting - add "
                "`--spec-type ngram-simple --spec-ngram-simple-size-m 384 "
                "--spec-ngram-simple-size-n 4` to llama-server: measured **4.7x decode at ~3-bit** "
                "(21.3 -> 98.8 tok/s), no download. Do NOT shrink m below 8: the verify pass "
                "switches kernels at width 9 on this card class (X-1), and m=4-7 measured 2.5x "
                "slower than m>=8 on the dense control - the large m is why this works. ")
        if oversold:
            head = ("if your output REUSES its context - edits, refactors, RAG quoting - add "
                    "`--spec-type ngram-simple --spec-ngram-simple-size-m 384 "
                    "--spec-ngram-simple-size-n 4` to llama-server - but the 4.7x we measured with "
                    f"it (21.3 -> 98.8 tok/s at ~3-bit) is **NOT REACHABLE ON THIS ROW**: that was "
                    f"a row whose token is ~100% weight bandwidth, and {cpu_c:.0%} of THIS token is "
                    f"CPU attention over the KV cache, which a verify batch does not amortize - "
                    f"every drafted position still attends over the whole cache. Same drafter, same "
                    f"~{SPEC_X_NGRAM_TUNED:.1f} accepted tokens per verify round, this row's own "
                    f"decomposition: at most **{reach:.2f}x**, at 100% acceptance and before the MoE "
                    f"union tax. No download either way. ")
        return (head + "REQUIRES A LONG PROMPT: with these values the drafter "
                "cannot fire until the context exceeds size_m+size_n+1 = 389 tokens, so on short "
                "prompts it does NOTHING and llama.cpp's default m=48 (61-token floor) is better. "
                "Output was byte-identical in every copy-regime test we ran, but that is evidence, "
                "not a guarantee - a verify pass batches several positions and batched reductions "
                "are not bit-identical to single-token decode, which at temp 0 can flip an argmax "
                "(measured: a rejected-draft case diverged reproducibly). The multiplier SHRINKS "
                "with bit-width - the SAME model at Q3_K_M gives 3.4x (17.5 -> 59.2) - because a "
                "verify round is compute-bound, so extra bits cost dequantisation on every drafted "
                "token, not just extra bytes moved. Speculation and quantization are NOT "
                "independent levers. "
                "BOTH tunables matter and llama.cpp's defaults (m=48, n=12) capture only 2.3x of "
                "it: the cost unit is the VERIFY ROUND (one full weight read), not the token, so "
                "what pays is delivering the same accepted tokens in fewer, longer runs. 108 tok/s "
                "is 2.6x this box's raw-decode wall, which no runtime, fork or rewrite can pass. "
                "Two traps, both measured: do NOT tune on acceptance rate (it falls 89%->68% while "
                "throughput doubles), and do NOT over-shorten n (n=2 drafts 36% MORE and runs 25% "
                "SLOWER). Novel generation gains nothing (0% acceptance), a 0.6B DRAFT MODEL is net "
                "negative here (0.72x), and stacking a second drafter costs 10%. Note llama-cli "
                "ignores --spec-type silently; the flag only works on llama-server.")
    if moe and experts_offloaded:
        return ("speculation will NOT pay here with external drafts: measured +3% (ngram) and "
                "-24% (MTP, old-generation) with experts offloaded - a verify batch unions "
                "experts, and every extra one is a slow read. ONE measured exception (prereg "
                "#71): models with a TRAINED-IN MTP head (Qwen3.6 class) reach ~93% acceptance, "
                "enough for `--spec-type draft-mtp --spec-draft-n-max 2` to buy +11% - and K "
                "MUST stay at 2: K=4 measured 0.61x on the same pair (the union tax is alive, "
                "the head just out-accepts it at short drafts).")
    if not moe:
        if oversold:
            return ("if you write CODE, add `--spec-type ngram-simple` - one flag, no download, "
                    "identical output. But the **2.10x** we measured with it (17.7 -> 37.2 tok/s) "
                    "was measured all-in-VRAM, where the token is ~100% weight bandwidth, and it "
                    f"is **NOT REACHABLE ON THIS ROW**: {cpu_c:.0%} of this token is CPU attention "
                    "over the KV cache, which a verify batch does not amortize - every drafted "
                    "position still attends over the whole cache. Same drafter, this row's own "
                    f"decomposition: at most **{reach:.2f}x**, at 100% acceptance. Prose gains "
                    "nothing anywhere (1.01x): it drafts by copying spans from your context.")
        # X-1 (preregistrations/2026-08-04-x1-verify-width-cliff.md): the verify pass is a fused
        # multi-token step, and this card class serves widths <=8 with mat-vec kernels and >=9
        # with mat-mat. Measured on the same 7B this note used to quote: draft 4-7 sits at
        # 48-51 tok/s, draft 8 jumps to 88.5, draft 24 reaches 132.1 against a 22.8 baseline -
        # the acceptance confound is excluded by bound (100% acceptance in the slow kernel caps
        # at 60.8). So DRAFT LENGTH IS A KERNEL DECISION FIRST; the old 2.10x default parked
        # users in the slow kernel and left ~2.5x on the table.
        return ("if you write CODE, add `--spec-type ngram-simple "
                "--spec-ngram-simple-size-m 12 --spec-ngram-simple-size-n 4` - one flag family, "
                "no download, identical output. THE DRAFT LENGTH IS THE LEVER (X-1, staked and "
                "measured): on pre-Ampere cards the verify pass switches kernels at width 9, so "
                "drafts of 4-7 are strictly dominated - measured 48-51 tok/s at m=4-7 vs "
                "**88.5 at m=8 and 124-132 at m=12-24**, against a 22.8 baseline: **up to 5.8x "
                "single-stream** on copy-heavy work, byte-identical output. Never draft 4-7 on "
                "this card class: m>=8 or leave speculation off. On Ampere+ the boundary is "
                "unverified (`bench --contribute` settles it). Prose gains nothing at any m "
                "(1.01x): the drafter copies spans from your context, and prose has none.")
    return None                    # MoE fully resident: untested here, so we say nothing



# U-17 (prereg #66): the per-IQ-byte CPU decode penalty. Both IQ2_XS arms of the #66 ladder
# over-predicted while the tool's IQ warning stayed prose-only. Calibrated on the pure-CPU arm
# (predicted 14.1 vs measured 11.44 tok/s at iq_share 0.962 -> k = 0.242), cross-validated on
# the independent expert-split arm (see the retrodiction in tests/smoke.py). NOT V-11's 2.7x:
# that number is a different regime (isolated prefill-heavy comparison) and would overshoot
# this correction ~7x - the retrodiction gate rejected it. Scope: weight bytes read from RAM
# during token generation; KV terms are format-independent and untouched.
#
# C-13 RE-DERIVATION (2026-07-30): this constant was fitted against `iq_share`, which counts
# IQ4_NL as codebook although #70 measured it Q4_0-class. Switching the driver to the true
# codebook share changes the BASIS, so the constant must move with it or the calibration point
# breaks. On the arm it was fitted to (DS-Lite pure CPU, exact at 11.4 vs 11.44): iq_share 0.962
# x 0.242 = 0.2328 of penalty, delivered by a codebook share of 0.510 -> k = 0.2328 / 0.510.
# The number changed; the measurement it encodes did not.
IQ_CPU_TG_PENALTY = 0.456


# ==================================================================================================
# WHICH RESOURCE BINDS (pre-registration #88, experiment #54)
# ==================================================================================================
# BigMoeOnEdge's sharpest observation is not a number, it is a distinction: the same model and the
# same config is I/O-bound on their phone and DRAM-bandwidth-bound on their laptop, and WHICH KNOB
# PAYS depends entirely on which one binds. On their laptop the compute time per token sits at
# ~0.11 s in every cell of three rounds, so more I/O lanes, more threads and a faster NVMe all
# measured NEUTRAL and ~9 tok/s is a ceiling even at zero I/O.
#
# `evaluate` has always built each placement's per-token time as a sum of physical terms and then
# thrown the decomposition away, returning only 1/T. Everything below keeps it. Recommending
# faster storage to a DRAM-bandwidth-bound user is not weak advice, it is WRONG advice, and until
# now this tool had no way to tell the two users apart.
#
# THIS IS A REPORTING LAYER. No term changes, no constant moves, no predicted tok/s moves - that
# is kill rule K-3 of prereg #88, checked cell-by-cell against the previous commit by
# weights/exp54_binding_constraint.py. If a number moved, the change is wrong.
RESOURCES = ("vram_bw", "ram_bw", "io", "cpu_compute")

# Deterministic tie-break order, fixed in the prereg so two equal shares cannot produce two
# different answers on two runs.
TIE_ORDER = ("io", "ram_bw", "vram_bw", "cpu_compute")

RESOURCE_LABEL = {"vram_bw": "VRAM bandwidth", "ram_bw": "system RAM bandwidth",
                  "io": "disk I/O", "cpu_compute": "CPU compute (attention over the KV cache)"}

# The four classes the user sees. BOTH bandwidth pools collapse to one class deliberately: it
# makes prereg #88's P-1 (no class may cover >85% of the shipped grid) HARDER to pass, because
# almost every row we model is bandwidth-limited in one pool or the other.
RESOURCE_CLASS = {"vram_bw": "bandwidth-bound", "ram_bw": "bandwidth-bound",
                  "io": "IO-bound", "cpu_compute": "compute-bound"}

# The lever each resource corresponds to, for the "what is dead here" print. No threshold decides
# whether a lever is dead - the exact Amdahl ceiling 1/(1-share) is printed instead, so a reader
# sees "capped at 1.05x" rather than our opinion of what counts as small.
RESOURCE_LEVER = {"vram_bw": "a faster GPU memory bus", "ram_bw": "faster RAM / XMP / more channels",
                  "io": "faster storage, more I/O lanes", "cpu_compute": "more/faster CPU threads"}

# Capacity thresholds. These are NOT new constants: both are lifted verbatim from the
# tier-boundary advisor that has shipped in run() since v1.8, and that advisor now calls
# capacity_probe() so the two can never disagree about whether capacity is the story.
CAP_PROMOTION_MIN = 1.15        # was: `promoted[0][1] > best[1] * 1.15`
CAP_SHAVE_MAX_SHARE = 0.30      # was: `if gap > size_now * 0.30: continue`

# CPU-tier bandwidth efficiency, named rather than inline. Same two values evaluate() has always
# used - nothing moves - but prereg #88's external arm (P-5) has to price the MoE RAM tier
# OUTSIDE evaluate (it substitutes BigMoeOnEdge's measured 80% hot-expert hit rate, which this
# tool cannot yet express). That scoring script had 0.38 hard-coded, so a change here would have
# left it silently pricing the old constant and still reporting P-5 as reproduced. Named so the
# script imports the number instead of copying it.
ETA_R_MOE = 0.38
ETA_R_DENSE = 0.62

# L-24/L-25 capacity lever: q8_0 KV bytes per f16 KV byte. FORMAT ARITHMETIC, not a tuned
# threshold - q8_0 stores 32 values in 34 bytes (32 int8 + one f16 scale) = 8.5 bits/value
# against f16's 16 - and it never gates whether capacity_probe fires or what the weight levers
# report (prereg #88's thresholds are untouched). It only sizes the KV half of the gap so the
# probe can OFFER the one capacity lever that costs no weight quant tier: +37% SPEED at d16384
# (prereg #25), quality measured CLEAN at the deepest depth f16 even runs (PPL ratio
# 1.00031 +/- 0.0188 at d7168 against a <=1.02 gate, prereg #91/L-24), and f16 KV CUDA-OOMs
# past ~7k on a 6GB card where q8_0 completes - there the lever is ENABLING, not an optimization.
KV_Q8_BYTES_PER_F16_BYTE = 8.5 / 16.0


class Row(tuple):
    """A placement row: still exactly the 4-tuple (name, tok_s, warn, flags) every caller unpacks.

    Six modules index or unpack these rows (auto, optimize, runtime, target, dashboard, the smoke
    suite). Adding a fifth element would have broken every one of them, and adding a parallel
    dict keyed by row name would have gone stale the first time two rows shared a prefix. A tuple
    subclass carries the decomposition on the side and is invisible to everything that does not
    ask for it - `len(row) == 4`, sorting, indexing and unpacking are unchanged.

    `terms`: {resource: seconds}, only the resources this row actually uses.
    `eff`  : the row's numerator, so `tok_s == eff / sum(terms.values())` EXACTLY. The disk rows
             carry 0.95 there; everything else 1.0. Checked to 1e-9 by the smoke suite - a term
             that does not reconstruct its own row's speed means the attribution is wrong, and a
             wrong attribution is worse than none because it is confident.
    """
    # No __slots__: a subclass of a variable-length built-in cannot declare them. The two
    # attributes live in a per-instance dict, which costs nothing at the handful of rows we build.

    def __new__(cls, name, tok_s, warn, flags, terms=None, eff=1.0, runnable=True):
        r = super().__new__(cls, (name, tok_s, warn, flags))
        r.terms = dict(terms or {})
        r.eff = eff
        # `runnable`: can a user execute this row from the emitted command TODAY? A row whose
        # speed is an UPPER BOUND with named unpriced terms is not comparable to a fully priced
        # one, and a row needing a runtime they do not have is worse than useless as a winner -
        # `plan` prints ONE `run it:` line, so an unrunnable winner emits a non-command. That is
        # the trap the disk-tier stake caught for the expert-cache row; adding a second such row
        # made it live. These rows are still SHOWN - the capacity claim is their whole point -
        # but they sort below every runnable row instead of competing on tok/s.
        r.runnable = runnable
        return r


def resource_times(row):
    """{resource: seconds} for a row, or None if this row carries no decomposition.

    Returns None rather than guessing. A plain 4-tuple reaching here (a caller that rebuilt the
    row, an older pickle) must produce NO classification, not a classification of zero terms."""
    terms = getattr(row, "terms", None)
    if not isinstance(terms, dict) or not terms:
        return None
    live = {k: v for k, v in terms.items() if v and v > 0}
    return live or None


def binding_constraint(row, capacity=None):
    """Which resource limits THIS row, by how much, and what the ceiling is if you fix it.

    Rule R1-R5 of pre-registration #88, in order:
      R2  bind = argmax over resources of accumulated seconds (ties by TIE_ORDER).
      R3  margin_x  = t_bind / t_second  - how much faster the binding resource must get before
                      the next one takes over.
          ceiling_x = 1/(1-share)        - the most ANY lever touching only the binding resource
                      can ever buy. This is the printable form of "9 tok/s even at zero I/O".
      R4  capacity is not a term in the time sum - it is what SELECTED the placement - so it is
          passed in from capacity_probe() rather than derived here.
      R5  capacity wins when it fires, because the action it implies (make it fit) is different in
          kind from the action a bandwidth verdict implies (make the tier faster).

    Returns None when the row carries no decomposition. Never returns a guess.
    """
    tt = resource_times(row)
    if not tt:
        return None
    total = sum(tt.values())
    if total <= 0:
        return None
    ordered = sorted(tt.items(), key=lambda kv: (-kv[1], TIE_ORDER.index(kv[0])))
    bind, t_bind = ordered[0]
    second, t_second = ordered[1] if len(ordered) > 1 else (None, 0.0)
    share = t_bind / total
    out = dict(
        resource=bind,
        klass=RESOURCE_CLASS[bind],
        label=RESOURCE_LABEL[bind],
        share=share,
        shares={k: v / total for k, v in tt.items()},
        seconds=dict(tt),
        total_s=total,
        next_resource=second,
        next_share=(t_second / total) if second else 0.0,
        margin_x=(t_bind / t_second) if t_second > 0 else None,
        ceiling_x=(1.0 / (1.0 - share)) if share < 1.0 else None,
        capacity=None,
    )
    # Amdahl per resource, including the ones that are NOT binding: this is the line that tells a
    # DRAM-bound user that the NVMe upgrade is capped at 1.05x. No threshold, just the arithmetic.
    out["lever_ceiling"] = {r: (1.0 / (1.0 - out["shares"].get(r, 0.0))
                                if out["shares"].get(r, 0.0) < 1.0 else None)
                            for r in RESOURCES}
    if capacity:
        out["capacity"] = capacity
        out["klass"] = "capacity-bound"
        out["label"] = f"capacity-bound ({capacity['tier']})"
        out["time_resource"] = bind          # what binds once it fits
    return out


def capacity_probe(ev, best_tps, size, kv_gb, vc, rc):
    """Is the reason this configuration is slow that it does not FIT?

    `ev(**overrides) -> rows` re-runs evaluate with the baseline's own arguments and a few
    replaced. Two probes, both leaving every bandwidth untouched:
      lift  - give the tier exactly enough capacity to hold the model (the hardware answer)
      shave - shrink the file until it crosses the boundary (the file answer, and the probe the
              tier-boundary advisor has always used)

    Returns at most ONE finding, from the nearest over-capacity tier that is within shave range -
    the same tier order and the same `break` the shipped advisor used, so its printed line is
    unchanged. Both thresholds are the advisor's own (CAP_PROMOTION_MIN, CAP_SHAVE_MAX_SHARE);
    this refactor exists so that "the advisor fired" and "the classifier says capacity-bound" are
    the SAME event rather than two rules that drift apart.
    """
    for tier, cap, want, lever in (
            ("VRAM", vc * 0.90, "all in VRAM",
             "next quant step down / tighter probed band / pruned variant"),
            ("RAM", max(rc - 4, 1), "pure CPU",
             "next quant step down / tighter probed band")):
        if cap <= 0 or size + kv_gb <= cap:
            continue
        gap = size + kv_gb - cap
        if gap > size * CAP_SHAVE_MAX_SHARE:
            continue                       # not a shave - a different model class
        fit_scale = max(0.05, (cap - kv_gb) / size) * 0.995
        shaved = [r for r in ev(act_scale=fit_scale, true_size_gb_scale=fit_scale)
                  if r[0].startswith(want)]
        shave_tps = shaved[0][1] if shaved else 0.0
        # 1.001: the capacity gates are `<=` comparisons against a product, so handing back
        # EXACTLY the required capacity lands on the boundary and loses to float rounding - the
        # probe then reports "more VRAM buys nothing" for a model that misses by an ULP. A hair of
        # headroom makes the answer the arithmetic one instead of the rounding one.
        need = (size + kv_gb) * 1.001
        lift_kw = {"vc": need / 0.90} if tier == "VRAM" else {"rc": need + 4}
        lifted = ev(**lift_kw)
        lift_tps = max([r[1] for r in lifted], default=0.0)
        gain_shave = shave_tps / best_tps if best_tps > 0 else 0.0
        gain_lift = lift_tps / best_tps if best_tps > 0 else 0.0
        if max(gain_shave, gain_lift) < CAP_PROMOTION_MIN:
            return None
        # fit_scale travels IN the finding: report's quality section prices the shave's implied
        # bits (bits x fit_scale), and re-deriving the tier boundary there would be a second
        # copy of this arithmetic - the drift class this probe exists to prevent.
        find = dict(tier=tier, gap_gb=gap, lever=lever, shave_tps=shave_tps, lift_tps=lift_tps,
                    gain_shave=gain_shave, gain_lift=gain_lift, fit_scale=fit_scale,
                    need_gb=(lift_kw.get("vc") or lift_kw.get("rc")))
        # THE KV HALF OF THE GAP (L-24/L-25). This probe used to shave WEIGHTS only, though for
        # a deep-context config the cheapest GB on the table is often the KV cache itself:
        # -ctk q8_0 -ctv q8_0 returns (1 - 8.5/16) of the f16 cache at a MEASURED quality cost
        # indistinguishable from zero (PPL ratio 1.00031 +/- 0.0188 at d7168, gate <=1.02,
        # prereg #91), where shaving the file means dropping a quant tier, which is NOT free
        # below ~3 bits (Laws 1-2). L-25's audit named this gap explicitly. Strictly ADDITIVE:
        # whether the finding fires, and every weight-lever number above, are untouched.
        if kv_gb > 0:
            kv_q8 = kv_gb * KV_Q8_BYTES_PER_F16_BYTE
            find["kv_gb"] = kv_gb
            find["kv_saved_gb"] = kv_gb - kv_q8
            find["kv_closes"] = size + kv_q8 <= cap
            if find["kv_closes"]:
                kv_rows = [r for r in ev(kvp_scale=KV_Q8_BYTES_PER_F16_BYTE)
                           if r[0].startswith(want)]
                find["kv_tps"] = kv_rows[0][1] if kv_rows else 0.0
                find["kv_gain"] = find["kv_tps"] / best_tps if best_tps > 0 else 0.0
        return find
    return None


def spec_ceiling(row, k=2):
    """UPPER bound on what speculative decoding can buy on THIS row, from the row's own terms.

    A verify batch of K+1 tokens reads the weights ONCE, so every BANDWIDTH term amortizes by
    1/(K+1). CPU attention does not: each drafted position still attends over the whole cache, so
    that term is paid K+1 times per verify round and cancels out of the ratio. Hence

        ceiling = T / ( T_bandwidth/(K+1) + T_compute )

    at 100% acceptance. Deliberately optimistic in two ways, both stated where it is printed:
    acceptance is never 100%, and on MoE offload placements the verify batch UNIONS the experts
    of K+1 tokens, which is why the measured ngram gain there is +3% and MTP measured -24%. So a
    row whose UPPER bound is already small cannot be sold the 4.7x headline - which is the point.

    `k` is a REAL number, not an integer count. Integer K is the draft-model case (`k=2` is the
    `--spec-draft-n-max 2` this tool recommends). The ngram drafter has no fixed K at all - m=384
    is a per-round BUDGET and the accepted run length is whatever the context yields - so its
    amortization enters here as the measured tokens-per-round it actually delivered
    (`spec_reachable_x`), which is fractional. Nothing in the arithmetic below needs K integral.
    """
    tt = resource_times(row)
    if not tt:
        return None
    cpu = tt.get("cpu_compute", 0.0)
    bw = sum(v for r, v in tt.items() if r != "cpu_compute")
    total = bw + cpu
    denom = bw / (k + 1) + cpu
    return (total / denom) if denom > 0 else None


def codebook_bounded_gain(row, share):
    """What is re-downloading a codebook-quant file as K-quant actually worth ON THIS ROW?

    The shipped warning promises "~2.7x slower than K-quants" and tells the user to re-download.
    That number is a property of the RAM WEIGHT READ, and this tool already prices it: eta_r is
    divided by (1 + share*IQ_CPU_TG_PENALTY). Removing the penalty scales that term back, so the
    honest bound on the whole token is

        speedup <= T / (T - T_ram * f/(1+f)),   f = share * IQ_CPU_TG_PENALTY

    Conservative on purpose: T_ram here includes the KV read, which carries no format penalty, so
    this OVERSTATES the gain. It is used only to SUPPRESS or bound an over-sold warning, and an
    overstated bound can only ever make us keep a warning we might have dropped.
    """
    tt = resource_times(row)
    if not tt or not share or share <= 0:
        return None
    t_ram = tt.get("ram_bw", 0.0)
    total = sum(tt.values())
    f = share * IQ_CPU_TG_PENALTY
    saved = t_ram * (f / (1.0 + f))
    return (total / (total - saved)) if total > saved > 0 else None


def evaluate(t, a, ne, moe, bits, vc, vb, rc, rb, db, geta, act_scale=1.0, gl=None, ctx=0, kvp=0.0,
             n_layer=None, true_size_gb=None, codebook_share=0.0):
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
    # How much of the model does the depth-aware recipe actually hold at >=4.5 bits?
    #
    # For a MoE, `ne` is exactly that set - attention plus shared experts - and the routed experts
    # scale with `bits`. For a DENSE model the tables set ne = t, which is true for ACTIVATION
    # (every parameter is read per token) and false for QUANTIZATION: it priced the entire model
    # at max(bits, 4.5), so a dense model's predicted speed did not respond to its bit-width AT
    # ALL. Gemma 4 12B came out at 7.70 GB/token, and therefore identical tok/s, at 2.5 bits and
    # at 4.5 - which is how a 106B MoE ended up predicted FASTER than a 12B dense on the same
    # hardware, on the public calculator (pre-registration #17, found by a user).
    #
    # The recipe protects `attn_.*` and `ssm_.*`, not embeddings and not the FFN. Measured from
    # real GGUF tensor shapes, the attention share of a dense model is 10.8% (Qwen2.5-7B), 20.2%
    # (gemma4-12B), 25.1% (Qwen3.5-4B), 29.6% (Qwen3-0.6B). Reading each model's true share from
    # its file scores no better than the mean (11% vs 10% error over six points), so the constant
    # ships and the machinery does not.
    prot = ne if moe else min(ne, t * DENSE_PROTECTED_SHARE)
    act_ne = prot * ab / 8 * 1.15 * act_scale
    act_ex = (a - prot) * bits / 8 * 1.15 * act_scale
    act = act_ne + act_ex
    # Law 4 v2 (context term, v1.1): every generated token re-reads the whole KV cache from
    # whichever tier KV lives on - kv_gb adds to BOTH the byte budget and that tier's capacity.
    kv_gb = ctx * kvp / 1e9 if ctx > 0 else 0.0
    ra = max(rc - 4, 1)
    eta_r = ETA_R_MOE if moe else ETA_R_DENSE
    # U-17: price the IQ share into every RAM weight read (pure CPU, hybrid experts, both split
    # kinds, waterfall). 0.0 unless a real GGUF was scanned, so presets are untouched. C-13:
    # this is the CODEBOOK share, not every format named IQ - IQ4_NL is Q4_0-class (#70).
    eta_r /= (1.0 + codebook_share * IQ_CPU_TG_PENALTY)
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
        # Every row below also records WHICH RESOURCE each term spends its seconds on (prereg #88).
        # The arithmetic is untouched - the terms are the same expressions, named.
        t_vram = act / (geta_w * vb) + kv_gb / (ETA_KV * vb)
        out.append(Row("all in VRAM", 1 / t_vram, None, "-ngl 99", {"vram_bw": t_vram}))
    if moe and vc > 0:
        # KV sits with attention in VRAM. The buffer term was a flat 1.2 GB estimate from before
        # the compute buffer was MEASURED (#23: exactly 0.5874 MiB per ubatch token -> 0.60 GB at
        # the ub-1024 the split now emits). 0.9 = measured buffer + 0.3 margin; the flat 1.2 was
        # the same over-reserve class #62 removed once already, and it held the emitted split at
        # 13 residents where the sweep measured 16 as optimal on both metrics (22.21 tg/393.7 pp)
        # with the soft edge two notches away at 18 (-6.6%). This lands the emit at 15 - both
        # sweep neighbors (14: 391/21.67, 16: 394/22.21) measured healthy.
        v_need = ne * ab / 8 * 1.08 + 0.9 + kv_gb
        r_need = size - ne * ab / 8 * 1.08
        if v_need <= vc * 0.95 and r_need <= ra:
            warn = "RAM boundary - needs --no-mmap; can be unstable" if r_need > ra * 0.85 else None
            # PINNED-MEMORY CAPACITY RISK (first external replication, u/MoneroApe, extends D-08):
            # `-ot ...=CPU` host buffers are CUDA-pinned. Pinning is why the PCIe path is fast -
            # and pinning a large share of system RAM fails outright under memory pressure. On his
            # 64 GB box the advice tried to pin 36.5 GB. The fallback when this bites: drop -ot
            # entirely and let llama.cpp auto-placement decide (slower, but it starts). APPENDED
            # to any existing warning, never masked by it - his rig had both problems at once.
            if r_need > ra * 0.45:
                pin = (f"pins {r_need:.0f}GB of {ra:.0f}GB RAM (CUDA host memory) - fails under "
                       "memory pressure; if it does, drop -ot and let auto-placement decide")
                warn = f"{warn}; {pin}" if warn else pin
            h_vram = act_ne / (geta * vb) + kv_gb / (ETA_KV * vb)
            h_ram = act_ex / (eta_r * rb)
            out.append(Row("hybrid: attention->VRAM, experts->RAM",
                           1 / (act_ne / (geta * vb) + act_ex / (eta_r * rb) + kv_gb / (ETA_KV * vb)), warn,
                           '-ngl 99 -ot "exps=CPU" --no-mmap',
                           {"vram_bw": h_vram, "ram_bw": h_ram}))
        # MoE partial expert offload: attention+KV in VRAM, then as many EXPERT layers as still
        # fit, remainder on CPU. Measured pre-registration #13 (2026-07-26, corrected): +12.4% decode,
        # ~2-3x prefill vs all-experts-to-CPU, with a hard CLIFF on overcommit (-29% at one step
        # past the ceiling) - so the free-VRAM headroom is deliberately conservative.
        # Desktop reserve: a real machine is not an empty GPU. Measured on this box during the
        # pre-registration #13 sweep: Explorer + compositor + browser held 0.8-1.5 GB throughout.
        # The 0.90 multiplier that used to stack on top of the reserve was a DOUBLE discount:
        # the #62 resident-expert sweep measured the config it forbade (16 of 48 resident) as
        # strictly better than the one it emitted (11 resident) - +15.3% prompt processing AND
        # +6.3% generation, fitting fine at ub 1024 - while the overshoot it feared is now a
        # SOFT edge (-6.6% pp at 18 resident, not the historical -29% hard cliff, which was the
        # ubatch compute-buffer cliff and is guarded by safe_ubatch since v1.15). One reserve,
        # counted once.
        v_free = vc - v_need - DESKTOP_VRAM_RESERVE
        experts_gb = size - ne * ab / 8 * 1.08
        if v_free > 0.3 and experts_gb > 0:
            f = min(1.0, v_free / experts_gb)
            ram_left = experts_gb * (1 - f)
            if f > 0.05 and ram_left <= ra:
                t_split = (act_ne / (geta * vb) + f * act_ex / (geta * vb)
                           + (1 - f) * act_ex / (eta_r * rb) + kv_gb / (ETA_KV * vb))
                fl = moe_split_flags(f, n_layer, host_gb=ram_left, ram_gb=ra)
                # Only offer it if we can emit the EXACT command. Advertising a speed the
                # printed flags cannot deliver is the v1.6.5 bug class; without a layer count
                # the row is suppressed and the footer tells the user how to unlock it.
                if fl:
                    pin_warn = ((f"pins {ram_left:.0f}GB of {ra:.0f}GB RAM (CUDA host memory) - "
                                 "fails under memory pressure; if it does, drop -ot and let "
                                 "auto-placement decide") if ram_left > ra * 0.45 else None)
                    mm_note = mmap_decision(ram_left, ra)[1]
                    if mm_note:
                        pin_warn = f"{pin_warn}; {mm_note}" if pin_warn else mm_note
                    s_vram = (act_ne / (geta * vb) + f * act_ex / (geta * vb)
                              + kv_gb / (ETA_KV * vb))
                    s_ram = (1 - f) * act_ex / (eta_r * rb)
                    out.append(Row(f"split experts: {f:.0%}->VRAM, rest->RAM", 1 / t_split,
                                   pin_warn, fl, {"vram_bw": s_vram, "ram_bw": s_ram}))
    if (not moe) and vc > 0 and size + kv_gb > vc * 0.90 and size + kv_gb <= ra + vc * 0.9:
        # C-11 (prereg #66): the old budget handed the model vc*0.9 outright - no desktop
        # reserve, no compute buffer - so at d16384 the emitted 26/28-layer config overcommitted
        # VRAM and measured -58% (driver memory fallback, the silent Windows thrash). Budget =
        # VRAM minus the reserve minus the default-ub compute buffer (measured slope, #23),
        # counted once, same discipline as the MoE v_free path above.
        cbuf = 512 * COMPUTE_BUFFER_MIB_PER_UB_TOKEN / 1024
        v_budget = max(0.0, vc * 0.9 - DESKTOP_VRAM_RESERVE - cbuf)
        g = min(0.95, v_budget / (size + kv_gb))           # KV splits with its layers
        # -ngl takes a LAYER COUNT. This emitted `int(g * 99)`, treating 99 - the all-layers
        # sentinel used elsewhere in this file - as if it were a layer count. Two failures, and
        # the second is severe:
        #   * the split is misreported: llama-70b (80 layers) printed "50% layers->VRAM" and
        #     emitted -ngl 49, which is 61%.
        #   * for any model with <= 99*g layers the flag EXCEEDS the layer count and llama.cpp
        #     puts EVERYTHING on the GPU - on a row that exists only because the model does NOT
        #     fit in VRAM. A 32-layer model does this for any g > 0.32. That is an OOM, or a
        #     silent thrash on Windows with driver memory fallback.
        # Same discipline as moe_split_flags: without a grounded layer count we suppress the row
        # rather than print a command we cannot honour (the v1.6.5 bug class). --gguf always
        # unlocks it, since autospec reads block_count from the file.
        gpu_layers = int(g * n_layer) if n_layer else 0
        if 0 < gpu_layers < n_layer:
            kv_t = g * kv_gb / (ETA_KV * vb) + (1 - g) * kv_gb / (ETA_KV * rb)
            # L-19 / prereg #74: the CPU-resident layers of a DENSE split run attention over the
            # whole KV cache every token. That is compute on this machine's threads, not the byte
            # read the kv_t term above prices - and it dominates at depth. Validated out-of-sample
            # (14B, 30 CPU layers, d8192: staked 1.53 against the un-termed 3.30, measured 1.36).
            cpu_attn_t = (n_layer - gpu_layers) * ctx * CPU_ATTN_S_PER_POS_LAYER
            d_vram = g * act / (geta_w * vb) + g * kv_gb / (ETA_KV * vb)
            d_ram = (1 - g) * act / (eta_r * rb) + (1 - g) * kv_gb / (ETA_KV * rb)
            out.append(Row(f"split: {gpu_layers}/{n_layer} layers->VRAM, rest->RAM",
                           1 / (g * act / (geta_w * vb) + (1 - g) * act / (eta_r * rb) + kv_t
                                + cpu_attn_t), None,
                           f"-ngl {gpu_layers}",
                           {"vram_bw": d_vram, "ram_bw": d_ram, "cpu_compute": cpu_attn_t}))
    if size + kv_gb <= ra:
        warn = "RAM boundary - expect bimodal speed" if size + kv_gb > ra * 0.85 else None
        # L-19 applies here MORE than to a split: every layer's attention runs on the CPU, so
        # this row pays the term over ALL layers. Leaving it un-termed while the split carries it
        # would rank pure CPU above a config with strictly fewer CPU layers - a ranking inversion
        # rather than a prediction. (Its own out-of-sample check: prereg #75.)
        cpu_attn_all = (n_layer or 32) * ctx * CPU_ATTN_S_PER_POS_LAYER
        c_ram = act / (eta_r * rb) + kv_gb / (ETA_KV * rb)
        out.append(Row("pure CPU (GPU idle)",
                       1 / (act / (eta_r * rb) + kv_gb / (ETA_KV * rb) + cpu_attn_all), warn,
                       "-ngl 0", {"ram_bw": c_ram, "cpu_compute": cpu_attn_all}))
    if size + kv_gb > ra:
        ra_eff = max(ra - kv_gb, 1)                        # KV crowds the expert cache
        miss = max(0.0, 1 - (ra_eff * 0.9) / size)
        hot = act_ne if moe else 0.0                       # MoE attention stays LRU-hot; dense has no hot set
        streamable = act - hot
        # L-19 applies here too: this row is `-ngl 0`, so every layer's attention runs on the CPU
        # exactly as in the pure-CPU row. Pricing one and not the other would let a disk-streaming
        # config outrank the same config WITHOUT disk misses - the inversion class #75 caught.
        tps = 0.95 / (streamable * miss / db + (streamable * (1 - miss) + hot) / (eta_r * rb)
                      + kv_gb / (ETA_KV * rb) + (n_layer or 32) * ctx * CPU_ATTN_S_PER_POS_LAYER)
        k_io = streamable * miss / db
        k_ram = (streamable * (1 - miss) + hot) / (eta_r * rb) + kv_gb / (ETA_KV * rb)
        k_cpu = (n_layer or 32) * ctx * CPU_ATTN_S_PER_POS_LAYER
        # THE KV RESIDENCY DEFICIT IS DISCLOSED, NOT SILENTLY ABSORBED (v1.21 integration audit).
        # The kv_gb/(ETA_KV*rb) term above prices every KV read at RAM bandwidth - i.e. this row
        # ASSUMES the whole KV cache is host-RAM-resident. When the 1 GB floor on ra_eff binds
        # (ra - kv_gb < 1), that assumption is false: KV plus the minimum expert cache do not fit
        # in the RAM budget, and before this fix the row silently granted KV the residency it does
        # not have. We have NO measured model of KV paging through disk (every disk-tier anchor -
        # 110B 0.19, laguna 0.38 - ran at ctx where KV fits), so repricing the term would be an
        # invented number, the exact failure class the anchors exist to prevent. The row therefore
        # KEEPS its arithmetic - the printed speed is an upper bound - and the warn names the
        # shortfall and its consequence, so the disclosure travels ON the row it qualifies rather
        # than in prose ten paragraphs down (the D-10 lesson).
        disk_warn = "exceeds RAM - capacity demo"
        kv_deficit = kv_gb + 1.0 - ra
        if kv_deficit > 0:
            disk_warn += (
                f"; KV DEFICIT {kv_deficit:.1f} GB: the {kv_gb:.1f} GB KV cache plus the 1 GB "
                f"minimum expert cache overshoot the {ra:.0f} GB RAM budget, so this row prices "
                f"KV at RAM bandwidth it does not have - the deficit pages through disk every "
                f"token and the printed tok/s is an UPPER BOUND, not a prediction. Shrink ctx "
                f"or quantize KV (-ctk q8_0 -ctv q8_0) to restore residency")
        out.append(Row("stream from disk (cold experts)", tps, disk_warn,
                       "-ngl 0", {"io": k_io, "ram_bw": k_ram, "cpu_compute": k_cpu}, eff=0.95))
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
            out.append(Row("stream from disk (VRAM+RAM expert cache)", tps3,
                           "needs an expert-caching runtime (ktransformers/colibri-class) - stock llama.cpp gets the RAM-cache row",
                           "-ngl 99 + runtime-managed expert cache",
                           {"io": streamable * miss3 / db,
                            "ram_bw": hit * (1 - vshare) / (eta_r * rb),
                            "vram_bw": (hit * vshare / (geta * vb) + hot / (geta * vb)
                                        + kv_gb / (ETA_KV * vb))},
                           eff=0.95, runnable=False))
    # LAYER-BY-LAYER STREAMING (E-11, the airllm placement). One transformer layer resident on
    # the GPU at a time; the rest of the model streams past it. VRAM stops scaling with model
    # size and starts scaling with LAYER size, which is what makes "70B on a 4 GB card" true.
    #
    # THE DEFINING TERM IS `size`, NOT `act`. Every other row in this menu reads only the weights
    # a token activates. This one visits every layer, so it reads the WHOLE model per token. For a
    # dense model that is nearly the same thing; for a MoE it is catastrophically not - a 235B MoE
    # with 22B active reads all 235B here, roughly 10x the traffic of the expert-offload rows
    # above. That is why the warn says so on the row rather than leaving it to be discovered.
    nl = n_layer or 32
    layer_gb = size / nl
    # Gate on the SINGLE-buffered requirement: one layer plus KV plus working space. Prefetch
    # (overlapping the next layer's fetch with this layer's compute) needs 2x layer_gb and is
    # what airllm's docs price at ~10% - a real but optional bonus, so requiring room for it
    # would wrongly suppress the row on exactly the tiny cards the placement exists for. A 70B
    # fp16 layer is 1.9 GB: single-buffered it fits a 4 GB card, double-buffered it does not,
    # and "70B on 4 GB" is the claim being modelled.
    v_need = layer_gb + kv_gb + 0.5
    if vc > 0 and size + kv_gb > vc * 0.90 and v_need <= vc * 0.90:
        cache_ll = max(0.0, ra * 0.9 - kv_gb)
        miss_ll = max(0.0, 1 - cache_ll / size)
        t_ll = (size * miss_ll / db                        # the part that must come off disk
                + size * (1 - miss_ll) / (eta_r * rb)      # the part host RAM can hold
                + kv_gb / (ETA_KV * vb))                   # KV stays on the GPU with the layer
        warn = (f"one layer resident ({layer_gb:.2f} GB of {size:.1f} GB); needs a "
                f"layer-streaming runtime (airllm/accelerate-class) - llama.cpp has no such mode")
        if layer_gb * 2 + kv_gb + 0.5 > vc * 0.90:
            warn += (f"; no room to PREFETCH (needs {layer_gb*2+kv_gb+0.5:.1f} GB for "
                     f"double-buffering, you have {vc*0.90:.1f}) - the ~10% overlap gain is off")
        if moe:
            warn += ("; MoE PENALTY: this reads ALL experts every token, not just the active "
                     "ones - the expert-offload rows above move far fewer bytes and this row "
                     "should not be preferred on a MoE unless VRAM genuinely cannot hold their "
                     "resident set")
        # UPPER BOUND, and the two reasons are named rather than absorbed. This is the same
        # treatment the KV-deficit disclosure gets above: keep the arithmetic, say what it omits.
        warn += ("; UPPER BOUND - two costs are unpriced: host-to-device PCIe transfer of every "
                 "layer every token (we have no measured PCIe anchor and will not invent one), "
                 "and the streaming-efficiency gap C-23 measured at 1.82x on llama.cpp "
                 "(0.2505 GB/s achieved against 0.452-0.459 raw). Expect materially slower")
        out.append(Row("layer-by-layer streaming (one layer resident)", 0.95 / t_ll, warn,
                       "runtime-managed layer streaming (not a llama.cpp flag)",
                       {"io": size * miss_ll / db,
                        "ram_bw": size * (1 - miss_ll) / (eta_r * rb),
                        "vram_bw": kv_gb / (ETA_KV * vb)}, eff=0.95, runnable=False))
    out.sort(key=lambda x: (not getattr(x, 'runnable', True), -x[1]))
    return size, act, out


def qual_of(moe, bits):
    keys = sorted(QUAL[moe])
    return QUAL[moe][min(keys, key=lambda k: abs(k - bits))]


def check_presets(args):
    """Refuse unknown preset names LOUDLY, and never predict from a silently-fabricated model.

    The whole product is a trustworthy number, so every prediction entry point calls this -
    plan/optimize/target AND run/bench/report/dashboard/audit-ollama. It refuses a typo'd
    --machine/--model that used to quietly predict for the WRONG box or model (e.g.
    `--machine rtx4090` for `rtx-4090` fell through to auto-detect and nobody said so). The
    companion warn_if_no_model() labels the separate generic-13B fallback for the commands that
    actually reach it."""
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


def warn_if_no_model(args):
    """Label the generic-13B assumption, unmissably, when the user described no model at all.

    Separate from check_presets because only the commands that actually FALL BACK to 13B should
    say so: `plan` can (no required --gguf), but audit-ollama prices real stored models and
    run/bench/dashboard require --gguf, so none of them assume 13B and none should print this."""
    if not (getattr(args, "model", None) or getattr(args, "total", None)
            or getattr(args, "gguf", None)):
        print("[quantprobe] NO MODEL GIVEN - assuming a generic 13B DENSE model. This number is "
              "illustrative, not about your model.\n"
              "             Pass --gguf model.gguf (exact, read from the file), --model <preset>, "
              "or --total <B> for yours.")


CAL_COMPONENTS = (("RAM stream", "ram_bw_measured"), ("disk", "disk_bw_measured"),
                  ("decode anchors", "anchors"))


def calibration_gap_warning(cal):
    """C-17 defect (2): PARTIAL CALIBRATION IS WORSE THAN NONE for the components you skipped.

    Five ladders on one idle box, same 14 rows, medians of |err|: full calibration 7.9%,
    fully-preset baseline 8.8%, RAM-ONLY 12.5%, uncalibrated 27.2%. A RAM-only state scored
    WORSE than either complete state - the presets are mutually consistent, and replacing one of
    them breaks that consistency. Yet the tool printed the identical `calibration applied [...]`
    line for a complete state and for a one-component state, so a user could not tell the 12.5%
    case from the 7.9% one.

    Returned as lines rather than printed so the property is testable without capturing stdout -
    the shape the P-88 report block already uses. Empty list when every component is measured.
    """
    missing = [name for name, key in CAL_COMPONENTS if not cal.get(key)]
    if not missing:
        return []
    return [f"[quantprobe] MIXED calibration state: {', '.join(missing)} still preset/spec-sheet, "
            f"NOT measured on this box. Partial calibration measured WORSE than none on the "
            f"components it skipped (RAM-only 12.5% median |err| vs 8.8% for the fully-preset "
            f"baseline, C-17). Complete it with `quantprobe calibrate --model your.gguf`, or "
            f"read the rows that lean on the missing component as uncalibrated."]


def apply_calibration_overrides(hw, args):
    """MEASURED beats spec-sheet, on this machine only - the ONE place calibration and anchors
    touch the constants, shared by plan AND runtime.best_flags so the two commands can never
    disagree about the same input again (the v1.10.5 bug class; layer 3 enforces it, and caught
    this function's absence the same day the anchors shipped).

    Calibration (rb/db): calibrate measures the DELIVERED stream; the law's constants were fitted
    against theoretical-peak inputs on the reference box, whose measured stream was 26.1 of a
    48 GB/s peak (0.544). Expressing the user's measurement in the same peak-units keeps every
    fitted eta valid. Assumption stated: stream-realism fraction is machine-similar.

    Anchors (prereg #64, gate: LOO median error 19.0% -> 5.8%): each anchor becomes a TIER-LOCAL
    ratio between what the anchor arm measured and what the law predicts for that same arm - the
    ratio then scales that tier's bandwidth for every OTHER prediction. The target's own
    measurement is never consulted (that would be a lookup, the #59 failure class). Ratios are
    clamped 0.70-1.40; anchor models are dense, so dense etas price their arms; the clamp absorbs
    anchor-format bias (L-16). `--no-anchors` disables.
    """
    from . import calibrate as calmod
    cal, age = calmod.load()
    if not cal:
        return hw
    hw["_cal_active"] = True
    applied = []
    if cal.get("ram_bw_measured"):
        hw["rb"] = round(cal["ram_bw_measured"] / 0.544)
        applied.append(f"ram {cal['ram_bw_measured']:g} GB/s measured")
    if cal.get("disk_bw_measured"):
        hw["db"] = cal["disk_bw_measured"]
        applied.append(f"disk {cal['disk_bw_measured']:g} GB/s measured")
    if applied:
        stale = f"; {age:.0f} days old - re-run `quantprobe calibrate`" if age and age > calmod.STALE_DAYS else ""
        _cid = cal.get("cal_id")
        print(f"[quantprobe] calibration applied [{'; '.join(applied)}] ({cal.get('date','?')}"
              + (f", state {_cid}" if _cid else "") + f"{stale})")
        for _line in calibration_gap_warning(cal):
            print(_line)
    if cal.get("boost_verdict") and "healthy" not in cal["boost_verdict"]:
        print(f"[quantprobe] GPU health at last calibration: {cal['boost_verdict']}")
    if not getattr(args, "no_anchors", False):
        ratios, notes = {}, []
        for an in (cal.get("anchors") or []):
            try:
                ts = float(an["tok_s"])
                # act_gb prices the anchor with the SAME active-byte convention every row uses
                # (params x bits/8 x 1.08). Old calibrations without it fall back to file size -
                # a known ~8% low bias, which the clamp bounds and a re-calibrate removes.
                act_a = float(an.get("act_gb") or an["model_gb"])
                if act_a <= 0 or ts <= 0:
                    continue
                if "pure CPU" in an.get("placement", ""):
                    pred = 0.62 * hw["rb"] / act_a           # dense CPU eta, same as evaluate()
                    ratios["cpu"] = max(0.70, min(1.40, ts / pred))
                    hw["_anchor_cpu_act"] = act_a
                elif "all-in-VRAM" in an.get("placement", "") and hw.get("vb"):
                    eta_a = (an["fmt_bw"] / 192.2) if an.get("fmt_bw") else hw.get("geta", 0.45)
                    pred = eta_a * hw["vb"] / act_a
                    ratios["gpu"] = max(0.70, min(1.40, ts / pred))
                    hw["_anchor_gpu_act"] = act_a       # for the size-class dispatcher below
                    # FORMAT RESCALING (L-16): the anchor measured ONE format's eta; the target
                    # may decode another. Both sides priced from the measured per-format ladder
                    # (spec.FORMAT_EBW); missing data -> no rescale.
                    tgt_bw = getattr(args, "fmt_bw", None)
                    anc_bw = an.get("fmt_bw")
                    if tgt_bw and anc_bw:
                        f = tgt_bw / anc_bw
                        ratios["gpu"] = max(0.60, min(1.50, ratios["gpu"] * f))
                        if abs(f - 1.0) > 0.03:
                            notes.append(f"format x{f:.2f}")
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
        if ratios:
            if "cpu" in ratios:
                # stored, not applied: validity depends on the TARGET's size class, resolved in
                # resolve_gpu_eta (a 0.5B CPU anchor measures 16% hotter per byte than the
                # flagship's CPU path - the small-model bias exists on BOTH tiers)
                hw["_cpu_ratio"] = ratios["cpu"]
            if "gpu" in ratios:
                # NOT applied to vb here: whether the GPU ratio is valid depends on the TARGET's
                # size class (resolve_gpu_eta below), which is not known yet. Stored instead.
                hw["_gpu_ratio"] = ratios["gpu"]
            extra = ("; " + ", ".join(notes)) if notes else ""
            print("[quantprobe] anchored: "
                  + ", ".join(f"{k.upper()} x{v:.2f}" for k, v in sorted(ratios.items()))
                  + f" from your calibrate anchor runs [tier ratios{extra}; --no-anchors disables]")
    return hw


def resolve_gpu_eta(hw, args, active_b, bits, vb, geta):
    """The size-classed GPU dispatcher - ONE function, used by plan AND runtime.best_flags.

    The #65 ladder measured the failure this prevents the same day anchors shipped: a 0.5B
    anchor priced a 7B GPU arm 34% low, because small models pay a size floor (#59) that big
    models do not - the anchor conflates format eta with that floor. Rules, each measured:
      - target active >= 1.5 GB and format known: eta comes from the per-format ladder
        (spec.FORMAT_EBW / 192.2 ref) - the reference box's own big-model GPU calibration,
        format-resolved (L-16). The anchor's machine ratio applies ONLY if the anchor sits
        within 4x of the target's active bytes (same size class).
      - smaller targets, or unknown format: the anchor ratio applies as before (a same-class
        small anchor measured -8%/+10%).
    Returns (vb, geta) with the choice printed.
    """
    ratio = hw.get("_gpu_ratio")
    # the format-ladder eta is a CALIBRATION-class refinement: it applies only on the
    # calibrated auto-detect path. Preset machines keep their fitted geta and the one-sided
    # floor semantics - the ratchet test caught exactly this scope creep (+47% on gemma's
    # custom mixed-quant file when the ladder eta overrode a preset).
    fmt_bw = getattr(args, "fmt_bw", None) if hw.get("_cal_active") else None
    act_est = (active_b or 0) * (bits or 4.5) / 8 * 1.08
    anchor_act = hw.get("_anchor_gpu_act")
    same_class = bool(anchor_act and act_est and act_est / 2 <= anchor_act <= act_est * 2)
    big = act_est >= 1.5
    # The size-class guard holds with OR without format knowledge: an out-of-class anchor ratio
    # never touches a big target (that misapplication is what the ratchet test caught).
    use_ratio = bool(ratio) and (same_class or not big)
    # The ladder eta is a BIG-MODEL quantity - #59 measured that small models pay a size floor
    # big models do not, and #65 measured a 0.5B anchor mis-pricing a 7B by -34%. This function's
    # docstring has always said "target active >= 1.5 GB and format known"; the code computed
    # `big` and then never used it HERE, so a 0.5B Q8_0 was handed a big-model efficiency
    # (0.60 instead of ~0.35). Caught by verify layer 3 on the v1.24 full-ladder re-run: the
    # small arms over-promised 56-81%. The gate the docstring promised, now actually applied:
    if fmt_bw and not big:
        fmt_bw = None
    if fmt_bw:
        # format-eta on BOTH sides: the anchor ratio is priced against the anchor's format eta,
        # so targets must use format eta too or the two references diverge (measured: pricing
        # the ratio on fmt-eta while the row used table-eta halved the small-model predictions).
        # Within the anchor's class the ratio then encodes the measured size floor and the
        # anchored prediction of the anchor's own arm is exact by construction.
        eta_fmt = fmt_bw / 192.2
        if not use_ratio and ratio:
            print(f"[quantprobe] GPU eta from the measured format ladder: {eta_fmt:.2f} "
                  f"(fmt {fmt_bw:.0f} GB/s ref; anchor outside this size class, machine ratio not applied)")
        return (vb * ratio, eta_fmt) if use_ratio else (vb, eta_fmt)
    if ratio and not use_ratio:
        print("[quantprobe] GPU anchor is outside this model's size class - machine ratio not "
              "applied (small anchors mis-price big models; #65).")
    return (vb * ratio, geta) if use_ratio else (vb, geta)


def resolve_cpu_bw(hw, args, active_b, bits, rb):
    """CPU-tier twin of resolve_gpu_eta: the anchor's machine ratio applies only within the
    anchor's size class. Same measured basis: the 0.5B CPU anchor runs 32.6 GB/s effective where
    the flagship's CPU expert path measured 28.0 (E3) - a +16% small-model bias that, applied to
    a 30B, over-promised the split row by 25% (caught by the post-ship consistency check)."""
    ratio = hw.get("_cpu_ratio")
    if not ratio:
        return rb
    act_est = (active_b or 0) * (bits or 4.5) / 8 * 1.08
    anchor_act = hw.get("_anchor_cpu_act")
    same_class = bool(anchor_act and act_est and act_est / 2 <= anchor_act <= act_est * 2)
    if same_class or act_est < 1.5:
        return rb * ratio
    print("[quantprobe] CPU anchor is outside this model's size class - machine ratio not "
          "applied (small anchors mis-price big models; #65).")
    return rb


def append_threads_flag(flags, placement):
    """--threads for CPU-resident placements - ONE helper for plan's printout AND runtime's
    launched command, so the command a user sees is the command run/bench executes (E-06's
    fork auto-detected 6 of 12 threads; C-07 measured the 40% class of swing)."""
    import os as _os
    n = _os.cpu_count() or 0
    if n and any(k in placement for k in ("->RAM", "CPU", "disk", "split experts")) \
            and "--threads" not in flags:
        return f"{flags} --threads {n}", n
    return flags, None


# The three hardware upgrades the tool has always advised, each tagged with the resource it
# actually attacks. Tagging is not decoration: it is what lets the printout say "this upgrade is
# capped at 1.05x here" instead of silently not appearing.
UPGRADES = (("enable XMP (free)", "ram_bw", dict(rb=48)),
            ("+16 GB RAM", "ram_capacity", dict(rc_delta=16)),
            ("NVMe SSD", "io", dict(db=3.5)))
UPGRADE_MIN_GAIN = 1.08          # unchanged: the shipped `c2[0][1] > best[1] * 1.08`


def upgrade_advisor(ev, best_tps, rb):
    """Which hardware upgrade would actually move this configuration.

    The logic (re-evaluate with one resource improved, print only if it beats the baseline by 8%)
    is unchanged and was already regime-aware by construction: an upgrade that does not help does
    not appear. THE INPUTS WERE NOT. The three shipped call sites passed neither `n_layer` nor
    `true_size_gb`, so every counterfactual was drawn from

      * a SMALLER ROW MENU - both split placements refuse to exist without a layer count, so the
        upgrade was compared against a baseline containing rows the counterfactual could not
        contain; and
      * a DIFFERENT MODEL - without `true_size_gb` the size is re-estimated from bits, which for a
        dense model inflates it by 4.5/bits, the exact error this file's own comment says
        "wrongly evicted the all-in-VRAM row and recommended a placement measured 2.4x slower"; and
      * a WRONG CPU-ATTENTION TERM at depth - `(n_layer or 32)` silently prices an 80-layer model
        as 32 layers, so at ctx>0 the counterfactual under-charges CPU attention for six of the
        ten shipped presets.

    THE DIRECTION CLAIM THAT SHIPPED HERE WAS WRONG, and is corrected rather than quietly
    deleted. This docstring, prereg #88 §4 P-3a and the L-23 register entry all asserted the bug
    was "one-directional: it can only ever SUPPRESS an upgrade that would genuinely help". Replay
    over the same unselected 10x17x2 grid, injecting the defect into the real `run()` and reading
    the advisor's own return, says the dominant direction is the OPPOSITE one. Of the 31 cells
    whose advice moves: the bug INVENTED 30 upgrade recommendations that the fixed advisor
    retracts, and suppressed 3; it overstated the counterfactual on 14 upgrades (worst x2.16) and
    understated it on 7. Every invented one is at ctx>0.

    Two mechanisms with OPPOSITE signs, which is why "one-directional" was never available:
      * the row menu suppresses - a counterfactual missing the split rows can only lose speed
        (all 3 suppressions are `enable XMP (free)` on the slow-RAM `2016` box: deepseek-16b at
        ctx 0 and at 16384, qwen3-30b at ctx 0); while
      * the depth term INVENTS - `(n_layer or 32)` prices the counterfactual's CPU attention as a
        32-layer model while the baseline pays for 80, so the "gain" is a comparison between two
        DIFFERENT MODELS, not between two machines. On `llama-70b` + `colibri` at ctx 16384 that
        artifact printed BOTH `+16 GB RAM` and `NVMe SSD` at an identical x1.70 on a box where
        neither resource is even reachable - the tell being that a RAM-capacity lever and an I/O
        lever cannot honestly buy the same number.

    Suppressed advice costs a user speed they could have had; invented advice costs them money on
    hardware that will not move their token. The second is the worse failure, and it was the one
    the shipped rationale said could not happen.

    The fix is structural rather than three more keyword arguments - `ev` closes
    over the baseline's own argument dict, so a counterfactual cannot differ from the baseline in
    anything the caller did not deliberately override. That is what the three call sites lacked:
    not the arguments, but the impossibility of forgetting them. (prereg #88 P-3a + §8.6)
    """
    out = []
    for name, resource, over in UPGRADES:
        if resource == "ram_bw" and rb >= 40:
            continue                            # XMP is only a lever on a machine running slow RAM
        rows = ev(**over)
        if rows and rows[0][1] > best_tps * UPGRADE_MIN_GAIN:
            out.append(dict(name=name, resource=resource, tps=rows[0][1], row=rows[0][0],
                            gain=rows[0][1] / best_tps if best_tps > 0 else 0.0))
    return out


def _pct(x):
    """Share as a percentage. Sub-1% shares keep a decimal instead of rounding to a bare '0%' -
    'disk I/O 0%' next to 'margin 347x' reads as a contradiction when it is really 0.3%."""
    return f"{x * 100:.1f}%" if 0 < x < 0.01 else f"{x:.0%}"


def binding_report(bc, bits=None, placement=None):
    """The printed 'which resource binds' block, as a list of lines.

    Returned rather than printed so the smoke suite can assert on it without capturing stdout.

    R6/K-4 of prereg #88 lives here: an `all in VRAM` row below 4.5 bits MUST NOT be shown a bare
    'bandwidth-bound' label. Our geta fuses VRAM bandwidth and unpack ALU into one number and
    cannot separate them - and in that exact cell we have MEASURED the opposite of what the naive
    label implies (Q2_K is 36% smaller and 4% SLOWER than Q4_K_M; Q4_0 is +19% over Q4_K_M at
    equal bytes). Printing the label without that caveat would contradict our own evidence.
    """
    if not bc:
        return []
    # Prereg #95 P-3 FAILED (scored 2026-08-17, staked 2026-08-07): Sobol variance attribution
    # put the PLACEMENT lever (-ngl on the dense regime, decided 1000/1000 bootstrap) on top,
    # not the flag the staked mapping assigned to the binding resource. The time decomposition
    # below is untouched arithmetic and still stands; what is NOT measurement-confirmed is the
    # flag-level tuning implication of the class name. The staked kill rule prices that
    # honestly: this label ships directly under the headline, at full prominence, until a
    # re-derivation survives a Taguchi arm.
    scope_label = ("  validation       derived from the law, not confirmed by variance "
                   "attribution (prereg #95 P-3: the top variance carrier was the placement "
                   "lever, not the mapped flag)")
    cap = bc.get("capacity")
    lines = []
    if cap:
        best_gain = max(cap["gain_shave"], cap["gain_lift"])
        lines.append(f"binding constraint: CAPACITY-BOUND ({cap['tier']}) - this configuration is "
                     f"{cap['gap_gb']:.1f} GB over the {cap['tier']} boundary, and crossing it is "
                     f"worth {best_gain:.1f}x.")
        lines.append(scope_label)
        lines.append(f"  cross it by      shaving the file ({cap['lever']}) -> "
                     f"~{cap['shave_tps']:.1f} tok/s, or fitting {cap['need_gb']:.1f} GB of "
                     f"{cap['tier']} -> ~{cap['lift_tps']:.1f} tok/s")
        # THE KV CAPACITY LEVER (L-24/L-25). Deep-context configs get a third way across the
        # boundary, and at depth it is the PREFERRED one: quantizing the KV cache costs no
        # weight quant tier, and both of its halves are measured - +37% speed at d16384
        # (prereg #25) and quality clean at the deepest depth f16 even runs (prereg #91).
        if cap.get("kv_closes"):
            lines.append(f"  KV lever         PREFER THIS at depth over dropping a quant tier: "
                         f"quantize the KV cache (-ctk q8_0 -ctv q8_0) - it returns "
                         f"{cap['kv_saved_gb']:.1f} GB of the {cap['kv_gb']:.1f} GB f16 cache and "
                         f"crosses the boundary WITHOUT touching the weights -> "
                         f"~{cap['kv_tps']:.1f} tok/s. Both halves of the lever are measured, "
                         f"not assumed: speed +37% decode at d16384 vs f16 KV (prereg #25), and "
                         f"quality PPL ratio 1.00031 +/- 0.0188 at d7168 against a <=1.02 gate "
                         f"(prereg #91, L-24) - and f16 KV OOMs past ~7k on a 6GB card where "
                         f"q8_0 runs, so there the lever is ENABLING, not an optimization. "
                         f"Permanent caveat: wikitext cannot see the niche-domain degradation "
                         f"reported at 100k+ context (E-10) - judge your own domain.")
        elif cap.get("kv_gb"):
            lines.append(f"  KV lever         partial: quantizing the KV cache (-ctk q8_0 "
                         f"-ctv q8_0) returns {cap['kv_saved_gb']:.1f} GB of the "
                         f"{cap['gap_gb']:.1f} GB gap at a measured-clean quality cost (PPL ratio "
                         f"1.00031 +/- 0.0188 at d7168, prereg #91/L-24; speed half also "
                         f"measured, +37% decode at d16384 vs f16 KV, prereg #25; E-10 "
                         f"niche-domain caveat applies) - it does not cross the boundary alone "
                         f"here, so pair it with a smaller file shave.")
        lines.append(f"  until then       {RESOURCE_LABEL[bc['resource']]} is where the time goes "
                     f"({_pct(bc['share'])} of the token) - but making it faster does not solve "
                     f"the reason you are on this placement")
    else:
        lines.append(f"binding constraint: {bc['klass'].upper()} ({bc['label']}) - "
                     f"{_pct(bc['share'])} of every decode token is spent there.")
        lines.append(scope_label)
        if bc.get("margin_x") and bc.get("next_resource"):
            lines.append(f"  margin           {bc['margin_x']:.2f}x - {RESOURCE_LABEL[bc['resource']]} "
                         f"must get that much faster before {RESOURCE_LABEL[bc['next_resource']]} "
                         f"({_pct(bc['next_share'])}) takes over")
        else:
            lines.append(f"  margin           no second constraint in this model - "
                         f"{RESOURCE_LABEL[bc['resource']]} is 100% of the modelled token")
        if bc.get("ceiling_x"):
            # DEMOTED, per the staked consequence of prereg #88's P-4 - which MISSED. P-4
            # predicted that >=10% of the 340-cell preset grid would have a binding ceiling under
            # 2x (a token split three ways). Measured: 0.6% - two cells. The binding resource
            # holds a median 99.6% of the token across all 340 cells, and 91.2% among the 172
            # rows that use more than one resource at all (median ceiling there: 11.4x). So this
            # ceiling is >= 2x almost everywhere and is rarely the surprising number. The caps
            # that DO surprise are on the next line, and the prereg said IN ADVANCE that a miss
            # here costs this number its headline space.
            lines.append(f"  ceiling          {bc['ceiling_x']:.2f}x if you fixed "
                         f"{RESOURCE_LABEL[bc['resource']]} outright - but that is the expected "
                         f"shape, not news: across our own 340-cell preset grid the binding "
                         f"resource holds a median 91% of the token wherever a second one exists "
                         f"at all (prereg #88 P-4 MISSED: 0.6% of cells below 2x vs a staked 10%). "
                         f"The caps worth reading are the ones below.")
    # Amdahl for the levers that do NOT bind. No threshold decides what counts as dead: the exact
    # cap is printed, which is the difference between advice and an opinion.
    dead = []
    for r in RESOURCES:
        if r == bc["resource"]:
            continue
        sh = bc["shares"].get(r, 0.0)
        if sh <= 0:
            dead.append(f"{RESOURCE_LEVER[r]}: NO effect (0% of the token)")
        else:
            dead.append(f"{RESOURCE_LEVER[r]}: <={1.0 / (1.0 - sh):.2f}x ({_pct(sh)})")
    lines.append("  other levers     " + " | ".join(dead))
    if placement == "all in VRAM" and bits is not None and bits < 4.5:
        lines.append("  CAVEAT (#16/#52): our GPU efficiency fuses bandwidth and UNPACK ALU into one "
                     "number, and in this exact cell we measured the byte ordering REVERSED - Q2_K "
                     "is 36% smaller and 4% SLOWER than Q4_K_M, Q4_0 is +19% over Q4_K_M at equal "
                     "bytes. Read the label as 'the GPU memory path', not 'the bytes'.")
    # The illustration used to name "+73%" flat, on EVERY row this block prints on - including
    # the all-in-VRAM rows where the same flag MEASURED -39% (prereg #19 P-2). Same leak class as
    # the one the v1.23 validation pass caught two paragraphs lower: a measured number keeps the
    # placement it was measured on, even when it is only being used to make a point about decode.
    lines.append("  scope            DECODE only. Prefill has a different binding constraint "
                 "(compute) which this classification does not model: on the CPU-expert MoE "
                 "placement -ub 2048 measured +73% on prefill and 0% on decode, while on an "
                 "all-in-VRAM row the same flag measured -39% prefill (prereg #19). Neither "
                 "number is a prediction for this row.")
    lines += serving_advisory(placement)
    return lines


def serving_advisory(placement):
    """Multi-user advice from the U-38/U-39 sweeps - the half of task #66 the X-1 rule left open.

    Everything here is MEASURED on one Pascal box (GTX 1060) and labelled so. Two families only,
    because two families are what we measured; a MoE fully resident in VRAM gets NOTHING printed
    rather than a guess. The single most useful sentence for anyone serving more than one user:
    the best model INVERTS with concurrency - measured 219 aggregate tok/s (dense 7B in VRAM)
    against 40 (30B MoE, experts in RAM) at 32 streams on the same box, while at 1 stream the
    MoE is the better model. Law 4 stays a single-stream law; these lines are its measured edge.
    """
    if placement == "all in VRAM":
        return [
            "  serving          MULTI-USER (measured, U-38/U-39, one Pascal box): aggregate decode",
            "    scales far past the single-stream number - 23.1 tok/s at 1 stream -> 175.6 at 16",
            "    -> 219.4 at 32 (7B Q4, llama-server -np). The jump sits at the width-8->9 kernel",
            "    boundary, so widths 2-8 are STRICTLY DOMINATED: a batch-8 step costs 148 ms, a",
            "    batch-9 step 84 ms. Serve 1 stream or >=9, never in between, on this card class.",
            "    Ampere+ boundary unverified - `bench --contribute` settles it.",
        ]
    if placement and "experts" in placement and ("CPU" in placement or "RAM" in placement):
        return [
            "  serving          MULTI-USER (measured, U-39, one Pascal box): batching does NOT",
            "    survive this placement - routed-expert reads from system RAM do not amortise",
            "    across streams (aggregate 19.7 -> 40.0 tok/s from 1 -> 32 streams, a 2.0x cap,",
            "    against 9.5x for a dense model in VRAM on the same box). One user: this row is",
            "    the right call. Many users: a dense model that fits VRAM inverts the choice -",
            "    measured 219 vs 40 aggregate at 32 streams. Prefill is the exception and batches",
            "    fine here (45 -> 225 tok/s by 8 streams): compute amortises, expert bandwidth",
            "    does not.",
        ]
    return []



def resolve_hw(args, announce=True):
    """The one place hardware inputs are resolved: preset -> auto-detect -> calibration -> flags.

    Extracted from `run` so `audit-ollama` cannot drift from `plan`. Two commands quoting
    different tok/s for the same file on the same box would be worse than either being wrong,
    because a user would have no way to tell which to believe - the C-15 auto-vs-plan
    divergence (26.2 vs 19.5 on one model) is exactly that failure, and it took a measurement
    to notice. Returns the same tuple `run` used to build inline, plus the hw dict for its hint.

    `announce=False` suppresses the auto-detect banner for callers that print their own header.
    """
    hw = dict(MACHINES[args.machine]) if getattr(args, "machine", None) in MACHINES else {}
    if not hw and all(getattr(args, k, None) is None
                      for k in ("vram", "vram_bw", "ram", "ram_bw", "disk_bw")):
        from . import detect as detmod
        auto, _ = detmod.detect()
        hw = dict(vc=auto["vram"], vb=auto["vram_bw"], rc=auto["ram"], rb=auto["ram_bw"],
                  db=auto["disk_bw"], geta=auto.get("geta", 0.45), gl=auto.get("gl"),
                  hint="THIS machine [auto-detected - run `quantprobe hw` for details]")
        if announce:
            print("[quantprobe] no hardware flags: auto-detected this machine "
                  f"(vram {hw['vc']:g}GB@{hw['vb']:g} | ram {hw['rc']:g}GB@{hw['rb']:g} | "
                  f"disk {hw['db']:g} GB/s). Pass --machine/flags to estimate a different box.")
        apply_calibration_overrides(hw, args)
    vc = hw.get("vc", getattr(args, "vram", None)); vb = hw.get("vb", getattr(args, "vram_bw", None))
    rc = hw.get("rc", getattr(args, "ram", None)); rb = hw.get("rb", getattr(args, "ram_bw", None))
    db = hw.get("db", getattr(args, "disk_bw", None))
    geta = hw.get("geta", 0.45); gl = hw.get("gl", None)
    if getattr(args, "vram", None) is not None: vc = args.vram
    if getattr(args, "vram_bw", None) is not None: vb = args.vram_bw
    if getattr(args, "ram", None) is not None: rc = args.ram
    if getattr(args, "ram_bw", None) is not None: rb = args.ram_bw
    if getattr(args, "disk_bw", None) is not None: db = args.disk_bw
    vc = agg_cap(vc) or 0; vb = agg_bw(vb, 0.85) or 0
    rc = rc or 16; rb = rb or 40
    db = agg_bw(db, 0.75) or 0.5
    # A card with capacity but no bandwidth is not a machine - it is a missing flag. Passing
    # ANY hw flag skips auto-detect (you are describing a different box), and RAM/disk carry
    # fallbacks while VRAM fell through to 0 - so `plan --vram 24` (sizing a card you do not
    # own yet) divided by zero three frames down. Borrow THIS box's detected GPU bandwidth and
    # SAY so - a stated borrowed number beats both a crash and a silent invention.
    if vc > 0 and vb <= 0:
        from . import detect as detmod
        try:
            auto, _ = detmod.detect()
        except Exception:
            auto = {}
        vb = agg_bw(auto.get("vram_bw"), 0.85) or 0
        if vb <= 0:
            raise SystemExit(
                f"--vram {vc:g} was given without --vram-bw, and no GPU was detected here to "
                f"borrow a bandwidth from.\nPass --vram-bw GB/s (the card's spec figure is "
                f"fine) so the VRAM tier has a rate.")
        if announce:
            print(f"[quantprobe] --vram {vc:g} given without --vram-bw: using {vb:g} GB/s "
                  f"detected on THIS box. Pass --vram-bw to model a different card.")
    return vc, vb, rc, rb, db, geta, gl, hw


def build_rows(args):
    """Everything plan.run COMPUTES, none of what it prints.

    Extracted for `report` (docs/DESIGN_REPORT_CMD.md section 2): the report is a RENDERER over
    this engine, exactly as binding_report() is a renderer over binding_constraint(). If report
    and plan could disagree about the same file on the same box the feature would be wrong (the
    C-15 auto-vs-plan divergence, 26.2 vs 19.5 on one model, is the failure class) - so there is
    ONE compute path, and run() prints from its return value.

    Stdout-safe by construction: every print that happens in this prefix happens inside nested
    calls that moved with it (spec.apply's autospec banner, resolve_hw's auto-detect banner and
    calibration lines, resolve_gpu_eta/resolve_cpu_bw's size-class notes), and evaluate /
    capacity_probe / binding_constraint print nothing. The moved block is a strict prefix of the
    old run() and executes first in both versions, so the byte order of stdout cannot change.

    Returns a dict:
      inputs  t, a, ne, moe, bits, ctx, kvp, nlay, true_size_gb, gguf, arch, kv_layers,
              iq_share, codebook_share, model_hint, machine_hint
      hw      vc, vb, rc, rb, db, geta, gl, cal_active, gpu_ratio, cpu_ratio
      size, act, q                      # file GB, active GB/token, QUAL multiplier
      rows    [Row]                     # terms/eff/runnable intact, already sorted
      best    rows[0] (None only if evaluate ever returned no rows, which it cannot today)
      cap     capacity_probe finding or None
      bc      binding_constraint dict or None
      ev      the counterfactual closure - it shares the baseline's own argument dict
              (prereg #88 P-3a); advisors MUST use this, never a re-spelled kwarg set
    """
    from . import spec as specmod
    specmod.apply(args)
    check_presets(args)
    warn_if_no_model(args)
    if getattr(args, "bits", None) is None:
        args.bits = 2.5
    m = dict(MODELS[args.model]) if args.model in MODELS else {}
    t = args.total or m.get("t") or 13.0
    a = args.active or m.get("a") or t
    ne = args.always_active or m.get("ne") or (a if a >= t * 0.9 else a * 0.35)
    moe = m.get("moe", a < t * 0.9)
    vc, vb, rc, rb, db, geta, gl, hw = resolve_hw(args)
    ctx = getattr(args, "ctx", 0) or 0
    kvp = (args.kv_per_pos * 1024 if getattr(args, "kv_per_pos", None)
           else m.get("kvp", DEFAULT_KVP))

    nlay = effective_n_layer(args, m)
    import os as _os
    from . import spec as _specmod
    _g = getattr(args, 'gguf', None)
    true_size = _specmod.gguf_size(_g) / 1e9 if _g and _os.path.isfile(_g) else None
    vb, geta = resolve_gpu_eta(hw, args, a, args.bits, vb, geta)
    rb = resolve_cpu_bw(hw, args, a, args.bits, rb)
    # ONE argument dict for the baseline AND every counterfactual (upgrades, capacity probes,
    # tier boundary). Three call sites used to spell these out by hand and three of them drifted -
    # see upgrade_advisor's docstring. A counterfactual now differs from the baseline in exactly
    # what the caller overrides and in nothing else, structurally.
    ev_kw = dict(t=t, a=a, ne=ne, moe=moe, bits=args.bits, vc=vc, vb=vb, rc=rc, rb=rb, db=db,
                 geta=geta, gl=gl, ctx=ctx, kvp=kvp, n_layer=nlay, true_size_gb=true_size,
                 codebook_share=getattr(args, "codebook_share", 0.0))

    def ev(**over):
        kw = dict(ev_kw)
        if "rc_delta" in over:
            kw["rc"] = kw["rc"] + over.pop("rc_delta")
        if "kvp_scale" in over:
            # The q8_0 KV counterfactual (L-24): scaling kvp shrinks BOTH the capacity crowding
            # and the per-token KV read, which is exactly what -ctk/-ctv q8_0 does to the model.
            kw["kvp"] = kw["kvp"] * over.pop("kvp_scale")
        if "true_size_gb_scale" in over:
            # A shave probe scales act_scale; when a REAL file size is pinned it must be scaled
            # too, or `true_size_gb` overrides the shave and the probe silently measures nothing.
            s = over.pop("true_size_gb_scale")
            if kw.get("true_size_gb"):
                kw["true_size_gb"] = kw["true_size_gb"] * s
        kw.update(over)
        return evaluate(**kw)[2]

    size, act, cfgs = evaluate(**ev_kw)
    q = qual_of(moe, args.bits)
    best = cfgs[0] if cfgs else None
    cap_find = (capacity_probe(ev, best[1], size, ctx * kvp / 1e9 if ctx > 0 else 0.0, vc, rc)
                if best else None)
    bc = binding_constraint(best, capacity=cap_find) if best else None
    return dict(
        inputs=dict(t=t, a=a, ne=ne, moe=moe, bits=args.bits, ctx=ctx, kvp=kvp, nlay=nlay,
                    true_size_gb=true_size, gguf=_g, arch=getattr(args, "arch", None),
                    kv_layers=getattr(args, "kv_layers", None),
                    iq_share=getattr(args, "iq_share", 0.0),
                    codebook_share=getattr(args, "codebook_share", 0.0),
                    model_hint=m.get("hint"), machine_hint=hw.get("hint")),
        hw=dict(vc=vc, vb=vb, rc=rc, rb=rb, db=db, geta=geta, gl=gl,
                cal_active=bool(hw.get("_cal_active")),
                gpu_ratio=hw.get("_gpu_ratio"), cpu_ratio=hw.get("_cpu_ratio")),
        size=size, act=act, q=q, rows=cfgs, best=best, cap=cap_find, bc=bc, ev=ev)


def run(args):
    px = build_rows(args)
    inp, hwd = px["inputs"], px["hw"]
    ne, moe, ctx, kvp, nlay = inp["ne"], inp["moe"], inp["ctx"], inp["kvp"], inp["nlay"]
    vc, rb = hwd["vc"], hwd["rb"]
    size, act, q, cfgs, ev = px["size"], px["act"], px["q"], px["rows"], px["ev"]
    print(f"\nquantprobe plan - {inp['model_hint'] or 'custom model'} @ {args.bits:g}-bit "
          f"on {inp['machine_hint'] or 'custom machine'}")
    kvline = (f" | ctx {ctx}: +{ctx * kvp / 1e9:.2f} GB KV read/token"
              if ctx > 0 else "")
    print(f"  model {size:.1f} GB | active {act:.2f} GB/token{kvline} | est. quality cost x{q:.2f} "
          f"(depth-aware recipe)\n")
    for i, (name, tps, warn, flags) in enumerate(cfgs):
        star = "*" if i == 0 else " "
        w = f"   [{warn}]" if warn else ""
        print(f"  {star} {tps:6.1f} tok/s  {name}{w}")
    best = px["best"]
    # WHICH RESOURCE BINDS (prereg #88). Computed in build_rows, printed directly under the rows,
    # because everything below it - speculation, the IQ warning, the upgrade advisor, the
    # concurrency note - is advice whose payoff is conditional on this answer, and until now the
    # tool asserted all of it unconditionally.
    cap_find = px["cap"]
    bc = px["bc"]
    rep = binding_report(bc, bits=args.bits, placement=best[0])
    if rep:
        print()
    for line in rep:
        print("  " + line)
    # L-30 / prereg #107: the expert-count dial is a bounded lever, and the bound is arithmetic
    # on THIS file. Printing the ceiling is the useful output; offering the dial is not, because
    # every measured point on the curve costs more quality than the speed is worth. Only shown
    # when a real GGUF was scanned - a preset has no expert byte split to read.
    _sp = getattr(args, "_spec", None)
    if isinstance(_sp, dict) and _sp.get("moe"):
        from . import spec as _specmod
        _note = _specmod.expert_ceiling_note(_sp)
        if _note:
            print(_note)
    # Speculation reality, TOP-LINE (first external replication, u/MoneroApe: the buried version
    # of this note cost him a debugging session - the drafter silently produced 0 drafts on a
    # novel prompt, exactly as D-10 measured, and the warning was ten paragraphs down).
    print("\n  speculation: pays ONLY when output copies its context (edits, refactors, RAG"
          " quoting)\n  - on novel generation the ngram drafter produces 0 drafts and changes"
          " nothing (D-10,\n  independently replicated on an RTX 3090). Details below.")
    # Prefill lever, appended only where the measurement says it pays (host-resident weights with
    # VRAM headroom). Not part of the law: `evaluate` is untouched and no anchor can move.
    ub = ubatch_flags(best[0], moe_vram_resident_gb(ne, args.bits, moe), vc)
    run_flags = f"{best[3]} {ub}" if ub else best[3]
    run_flags, nthreads = append_threads_flag(run_flags, best[0])
    print(f"\n  run it:  llama-server -m model.gguf {run_flags}")
    if "--threads" in run_flags:
        print(f"           (--threads {nthreads} = this machine's LOGICAL cores; llama.cpp's own"
              " auto-detect may pick\n           physical-only and cost 2x on CPU-bound decode -"
              " verify with your fork if unsure)")
    # I-quant files on a host tier: measured 2.7x slower than K-quants at the same size
    # (pre-registration #31: IQ3_XS 10.6 GB/s vs Q2_K 28.4 / Q4_K_M 29.7 on pure-CPU decode).
    # The warning fires only when weights actually land on the CPU - in VRAM the IQ formats
    # are fine, so warning there would be crying wolf.
    if getattr(args, "iq_share", 0.0) > 0.3 and any(
            k in best[0] for k in ("->RAM", "CPU", "disk", "split experts")):
        # CONDITIONAL on the binding constraint (prereg #88 P-3b). "~2.7x slower" is a property of
        # the RAM WEIGHT READ, not of the token, and this row may spend most of its token
        # elsewhere - on CPU attention at depth, or on the disk. Telling someone to re-download
        # 20 GB for a 3% gain is the same error class as recommending an NVMe to a DRAM-bound
        # user. The bound below is exact arithmetic from the penalty this tool already prices.
        cs = getattr(args, "codebook_share", 0.0) or args.iq_share
        bound = codebook_bounded_gain(best, cs)
        head = "WARNING" if (bound or 1.0) >= 1.05 else "note"
        print(f"\n  {head}: this file is {args.iq_share*100:.0f}% I-quant (IQ*) tensors and this"
              f" placement puts weights\n  on the CPU tier, where IQ formats decode ~2.7x slower"
              f" than K-quants of the same size\n  (measured: 10.6 vs ~29 GB/s effective).")
        if bound:
            ram_share = (bc or {}).get("shares", {}).get("ram_bw", 0.0)
            if bound >= 1.05:
                print(f"  ON THIS ROW it is worth at most **{(bound - 1) * 100:.0f}%**"
                      f" (RAM weight reads are {ram_share:.0%} of the token):"
                      f" re-download as Q_K\n  (e.g. Q3_K_M instead of IQ3_XS) before trusting"
                      f" the speed above.")
            else:
                print(f"  ON THIS ROW the fix is worth at most **{(bound - 1) * 100:.1f}%** -"
                      f" RAM weight reads are only {ram_share:.0%} of\n  the token here, so"
                      f" re-downloading the model would buy you almost nothing. The 2.7x is real"
                      f"\n  and it is not what limits you: see the binding constraint above.")
    ph = phase_advice(best[0], cfgs,
                      vram_resident_gb=moe_vram_resident_gb(ne, args.bits, moe), vc=vc)
    if ph:
        print(f"\n  phase: {ph}")
        chat, rag = workload_frontier(0.5), workload_frontier(200)
        if chat["label"] != rag["label"]:
            print("\n  workload: the best setup depends on how much prompt you read per token you write:")
            print(f"    chat, short prompts     -> {chat['label']:<26} {chat['pp']:>6.0f} pp / {chat['tg']:>5.1f} tg")
            print(f"    long prompts, RAG, docs -> {rag['label']:<26} {rag['pp']:>6.0f} pp / {rag['tg']:>5.1f} tg")
            print(f"  Choosing wrong costs up to {rag['speedup_vs_worst']:.2f}x on a long-prompt workload.")
        else:
            print(f"\n  workload: one setup wins at every prompt:generation ratio - {rag['label']}")
            print(f"  ({rag['pp']:.0f} pp2048 / {rag['tg']:.1f} tg128 - REFERENCE-BOX numbers at "
                  "those exact batch sizes;")
            print("  a 22-token prompt measures startup, not prefill). Earlier versions offered a "
                  "choice here; re-measuring")
            print("  every cell in ONE session (pre-registration #25) showed the alternatives were "
                  "dominated,")
            print("  so there is nothing to choose. One fewer knob, and the honest number.")
            print()
            print("  honesty on GENERATION speed (preregs #43/#60/#61): a plain fixed -ngl split measures")
            print("  EQUAL tg to this -ot placement - three times, at degraded AND full clocks. The -ot")
            print("  advice earns its keep on PROMPT PROCESSING (2.2x measured), on keeping KV in VRAM,")
            print("  and on enabling the speculation numbers below - not on raw generation.")
            print("  KV stays in VRAM at EVERY depth: evicting it (-nkvo 1) measured 3.48 vs 10.59")
            print("  tok/s decode against q8_0 KV at d16384 - 3.04x WORSE, for a prefill difference")
            print("  inside the error bar (prereg #25). Deep workloads (RAG at 50:1-200:1) are where")
            print("  it loses hardest; that advice is WITHDRAWN per prereg #25's pre-commitment (L-24).")
            print("  IF YOUR NUMBERS SAG 25-30% after hours of GPU churn: the cause on our box was a")
            print("  STUCK BOOST STATE (SM 1506 vs 1835+ MHz at cool temps - diagnosed and confirmed by")
            print("  cold-boot A/B, preregs #60/#61). Check `nvidia-smi --query-gpu=clocks.sm` (NVIDIA)")
            print("  or `rocm-smi --showclocks` (AMD) under load; consumer cards cannot reset it -")
            print("  reboot restores full speed. Not thermal, not your config.")
            print()
            print("  long prompts, same document: send cache_prompt=true to llama-server. Measured")
            print("  here (pre-registration #29): the second question against a 2k-token document")
            print("  pays 183 ms of prompt time instead of 5381 - 29x - because the document KV is")
            print("  reused and only the new question is processed. Restarting the server is cold.")
    if ub:
        # The +73% is a property of the PLACEMENT it was measured on (-ot exps=CPU, prereg #19),
        # not of the flags. This paragraph used to print it unconditionally under every sized
        # batch: the winning split-experts row claimed "+73% on this placement" two paragraphs
        # after the tool itself printed that row's measured 161.9 pp2048, and a dense split was
        # told "your experts sit in RAM" about a model with no experts. Same scope boundary as
        # the emission gate above (L-26): the claim travels only with the experts->RAM tier.
        if "experts->RAM" in best[0]:
            # THE NUMBER MUST MATCH THE UBATCH WE ARE ACTUALLY EMITTING. "+73%" is the ub-2048
            # cell of prereg #19's sweep; the sizer often emits 1024 on a tight card, where the
            # SAME sweep measured 277.17 (+38.7%), and this paragraph printed +73% anyway. That is
            # the leak the v1.23 pass caught between placements, repeated one level down between
            # cells of one sweep - so the emitted ubatch now selects its own measured cell, and a
            # ubatch with no cell in the sweep gets prose instead of a borrowed number.
            _ubn = int(ub.rsplit(" ", 1)[1])
            _cell = UB_SWEEP_PP2048.get(_ubn)
            if _cell:
                _gain = _cell / UB_SWEEP_PP2048[512] - 1.0
                _worth = (f"is worth **{_gain:+.0%} prefill** on this placement (measured "
                          f"{UB_SWEEP_PP2048[512]:.1f} -> {_cell:.1f} pp2048 against the ub-512 "
                          f"baseline, pre-registration #19)")
            else:
                _worth = ("is the sized safe batch for this card; prereg #19 swept this placement "
                          "at ub 512/1024/2048 only (199.9 / 277.2 / 345.9 pp2048) and measured no "
                          "cell at this ubatch, so no percentage is quoted for it")
            print(f"\n  prompt speed: `{ub}` {_worth}. It costs nothing on generation "
                  f"(18.5 -> 18.8) and applies because your experts sit in RAM: they then cross PCIe "
                  f"once per batch instead of once per token. Do NOT set it when a model is fully in "
                  f"VRAM - measured there, the same flag LOSES 39%.")
            if "-ub 4096" in ub:
                print(f"  L-26 (out of sample; predictions staked in "
                      f"weights/data/prereg92b_ub_oos.log before the run):\n  at "
                      f"-b 4096 -ub 4096 this placement "
                      f"measured 360.76 tok/s\n  AT pp4096. The prefill law sec/tok = A + X/C "
                      f"held at C=256 (-0.27%) and C=4096 (-8.2%,\n  in band). READ THE COMPARISON "
                      f"CAREFULLY: the published ub-2048 number, 345.89, is a\n  pp2048 "
                      f"measurement (prereg #19) - a different prompt length - and the "
                      f"pp4096-at-ub2048\n  control was never run, so the ~+4% between them is "
                      f"what the law implies, not a measured\n  A/B. -ub is a CAP: on prompts of "
                      f"2048 tokens or fewer this flag changes nothing. The law\n  is validated "
                      f"TO C=4096 and not past it (the asymptote bends below 1/A), so do not "
                      f"raise\n  the batch further on its authority.")
        else:
            print(f"\n  prompt speed: `{ub}` is the sized safe batch for this card (buffer-fit "
                  f"rule, prereg #23) - a bigger ubatch batches host-resident weights across "
                  f"PCIe, but the measured numbers behind this lever (+73%, 199.9 -> 345.9 "
                  f"pp2048, prereg #19; 360.76 at pp4096/ub4096, L-26) are from the "
                  f"CPU-expert MoE placement (-ot exps=CPU) and are NOT a prediction for this "
                  f"row - the prefill law that licenses them is scoped to that tier (L-26; the "
                  f"dense control violates its form). Known boundary: fully in VRAM the same "
                  f"flag LOSES 39% (prereg #19 P-2), so the batch is sized against VRAM "
                  f"headroom and capped at 2048 outside the measured tier.")
    dsw = depth_scope_warning(best[0], moe, ctx)
    if dsw:
        print(f"\n  {dsw}")
    fit_adv = fits_in_vram_advice(best[0], args.bits)
    if fit_adv:
        print(f"\n  note: {fit_adv}")
    # C-15: the winning ROW goes in, so the headline is priced against the token it is printed
    # next to instead of being quoted flat on every row the branch fires on.
    adv = speculation_advice(moe, best[0], row=best)
    dn = dense_draft_note(moe, best[0])
    if dn:
        print("\n  " + dn)
    if adv:
        print(f"\n  speculation: {adv}")
        # CONDITIONAL on the binding constraint (prereg #88 P-3c). Every speculation number above
        # was measured on a BANDWIDTH-BOUND row, and the mechanism is why: a verify batch reads
        # the weights once for K+1 tokens. CPU attention does not amortize - each drafted position
        # still attends over the whole cache - so on a compute-heavy row the headline multiplier
        # is unreachable ARITHMETICALLY, before acceptance rate is even discussed.
        #
        # C-15: this line used to print "upper bound ... 3.00x at K=2" directly underneath a 4.7x
        # headline and call that a bound on it. It is not: K=2 is the DRAFT-MODEL config this tool
        # recommends (`--spec-draft-n-max 2`), which verifies 3 tokens per weight read and is
        # therefore capped at 3x by construction, while the ngram drafter above has no fixed K and
        # measured 4.63 accepted tokens per verify round. Two different drafters, two different
        # bounds, and printing one under the other made the page contradict itself on all 25 MoE
        # split-expert cells. Both are named now, each against its own drafter.
        sc = spec_ceiling(best, k=2)
        hx = spec_headline_x(moe, best[0])
        reach = spec_reachable_x(best, hx) if hx else None
        if sc:
            cpu_share = (bc or {}).get("shares", {}).get("cpu_compute", 0.0)
            if sc < 1.5:
                print(f"\n  BUT NOT ON THIS ROW: with a DRAFT MODEL at K=2 the arithmetic ceiling "
                      f"here is **{sc:.2f}x**,\n  at 100% acceptance, because {cpu_share:.0%} of "
                      f"this token is CPU attention, which a verify batch\n  does NOT amortize "
                      f"(each drafted position still attends over the whole cache). The\n  numbers "
                      f"above were all measured on bandwidth-bound rows. Fix the binding "
                      f"constraint first.")
            elif reach:
                print(f"  (upper bounds from this row's own decomposition, at 100% acceptance: "
                      f"{sc:.2f}x for a DRAFT\n  MODEL at K=2 - it verifies K+1 tokens per weight "
                      f"read - and {reach:.2f}x for the ngram drafter\n  above, which measured "
                      f"~{hx:.1f} accepted tokens per verify round. A verify batch amortizes "
                      f"weight\n  READS only; measured gains are below both and MoE offload pays a "
                      f"union tax on top.)")
            else:
                print(f"  (upper bound on this row from its own decomposition: {sc:.2f}x for a "
                      f"draft model at K=2\n  and 100% acceptance - a verify batch amortizes weight "
                      f"READS only; measured gains are below\n  this and MoE offload pays a union "
                      f"tax on top.)")
    # UPGRADE ADVISOR - same 8% gate, same three levers, but the counterfactual now shares the
    # baseline's arguments instead of re-spelling a subset of them (see upgrade_advisor).
    alts = upgrade_advisor(ev, best[1], rb)
    if alts:
        print("  upgrade advisor: " + " | ".join(
            f"{u['name']} -> ~{u['tps']:.1f} tok/s (x{u['gain']:.2f}, {u['row']})" for u in alts))
    # Name the upgrades that are DEAD here, with the exact cap. Silence used to be the only signal
    # that an upgrade would not help, and silence is indistinguishable from an oversight. This is
    # the BigMoeOnEdge lesson made printable: on their laptop more I/O lanes, more threads and a
    # faster NVMe all measured NEUTRAL, and nothing in any tool told them so in advance.
    dead = []
    for name, resource, _over in UPGRADES:
        if any(u["name"] == name for u in alts):
            continue
        if resource == "ram_bw" and rb >= 40:
            continue                                   # already fast RAM: not applicable, not dead
        if resource.endswith("_capacity") or not bc:
            dead.append(f"{name}: re-evaluated, does not move this placement")
            continue
        sh = bc["shares"].get(resource, 0.0)
        dead.append(f"{name}: " + (f"{RESOURCE_LABEL[resource]} is 0% of this token"
                                   if sh <= 0 else
                                   f"capped at {1.0 / (1.0 - sh):.2f}x ({RESOURCE_LABEL[resource]} "
                                   f"is {_pct(sh)} of this token)"))
    if dead:
        print("  upgrade advisor - WON'T HELP HERE: " + " | ".join(dead))
    # TIER-BOUNDARY ADVISOR: the biggest cliffs sit at tier boundaries (REAP finding, pre-reg #8
    # P-c). Same thresholds, same nearest-boundary-only rule - it now reads the SAME probe the
    # binding-constraint classifier used, so "the advisor fired" and "the classifier says
    # capacity-bound" cannot disagree (prereg #88 R4).
    if cap_find and max(cap_find["gain_shave"], cap_find["gain_lift"]) >= CAP_PROMOTION_MIN:
        if cap_find["gain_shave"] >= CAP_PROMOTION_MIN:
            print(f"  tier-boundary advisor: this config is {cap_find['gap_gb']:.1f} GB over the "
                  f"{cap_find['tier']} boundary - shave it ({cap_find['lever']}) -> "
                  f"~{cap_find['shave_tps']:.1f} tok/s (x{cap_find['gain_shave']:.1f})")
        else:
            print(f"  tier-boundary advisor: this config is {cap_find['gap_gb']:.1f} GB over the "
                  f"{cap_find['tier']} boundary. Shaving the file does not pay here "
                  f"(x{cap_find['gain_shave']:.1f}), but fitting {cap_find['need_gb']:.1f} GB of "
                  f"{cap_find['tier']} does: ~{cap_find['lift_tps']:.1f} tok/s "
                  f"(x{cap_find['gain_lift']:.1f})")
        # L-24: the same KV lever the binding report offers, at the advisor's own event - the
        # two read ONE probe result, so they cannot disagree about whether the lever exists.
        if cap_find.get("kv_closes"):
            print(f"  tier-boundary advisor: deep-context preference (L-24) - quantize the KV "
                  f"cache (-ctk q8_0 -ctv q8_0)\n  instead of dropping a quant tier: frees "
                  f"{cap_find['kv_saved_gb']:.1f} GB, crosses the boundary -> "
                  f"~{cap_find['kv_tps']:.1f} tok/s (x{cap_find['kv_gain']:.2f}).\n  Speed "
                  f"measured +37% decode at d16384 vs f16 KV (prereg #25); quality measured "
                  f"CLEAN at d7168\n  (PPL ratio 1.00031 +/- 0.0188, gate <=1.02, prereg #91) "
                  f"and f16 KV OOMs past ~7k on 6GB\n  where q8_0 runs; E-10 niche-domain "
                  f"caveat applies.")
    # Fires only when the layer count is genuinely unknown. It used to test args.n_layer - the raw
    # CLI flag - and so told users to "re-run with --gguf to unlock it" while the exact -ot flags
    # for layers 10-47 sat printed directly above, because presets supply `nl` through the same
    # fallback the placement rows already use. Same class as the v1.10.5 n_layer divergence: a
    # second reader of a value that has a fallback, written without the fallback.
    if moe and vc > 0 and not nlay:
        print("\n  note: MoE partial expert offload (measured +12.4% decode, ~2-3x prefill) needs this\n"
              "  model's layer count to emit exact -ot flags - re-run with --gguf <file> to unlock it.")
    print("\n  concurrency: every number above is SINGLE-STREAM. Measured with 8 parallel slots, "
          "aggregate\n  throughput is ~2x higher and saturates by about 4 slots "
          "(pre-registration #26). The same\n  ratio appeared on every placement AND on a dense "
          "control, so it is a ceiling we do not model -\n  read our figures as one user's speed, not a server's capacity.")
    # CONDITIONAL (prereg #88): the ~2x is the same mechanism as speculation - slots share one
    # weight read. A row whose token is mostly CPU attention shares nothing, because every slot
    # attends over its OWN cache.
    if bc and bc["shares"].get("cpu_compute", 0.0) > bc["shares"].get("ram_bw", 0.0) \
            and bc["shares"].get("cpu_compute", 0.0) > bc["shares"].get("vram_bw", 0.0):
        print(f"  ...but NOT on this row: {bc['shares']['cpu_compute']:.0%} of this token is CPU "
              f"attention, and parallel slots\n  share a weight read, not a KV scan. The #26 "
              f"measurement was taken on a bandwidth-bound\n  placement; do not carry it here.")
    print("\n  (eta bands fitted from published measurements; estimates +/-25%. "
          "Hybrid needs --no-mmap.)")
