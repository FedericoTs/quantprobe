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
    if "split experts" in placement or vc <= 0:
        return None
    if not any(k in placement for k in ("->RAM", "CPU", "disk")):
        return None                       # nothing host-resident: no transfer to amortise
    if vc * 0.90 - vram_resident_gb < UBATCH_HEADROOM_GB:
        return None                       # no room for the bigger compute buffer; the -39% case
    ub = safe_ubatch(vc * 0.90 - vram_resident_gb)
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


def safe_ubatch(headroom_gb, cap=2048):
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
    """
    budget_mib = max(0.0, headroom_gb) * 1024 * 0.5
    ub = 0
    n = 256
    while n <= cap:
        if n * COMPUTE_BUFFER_MIB_PER_UB_TOKEN <= budget_mib:
            ub = n
        n *= 2
    return ub


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


def phase_advice(placement, rows):
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
    return ("this command is tuned for GENERATION. If your prompts are long - agent transcripts, "
            "RAG context, whole files - the other placement is far better at reading them: "
            "measured 349.6 vs 161.9 tok/s prompt processing (2.2x), for 8% less generation. "
            f"Use `{alt[3]} -b 2048 -ub 2048` instead. Prompt processing and generation genuinely "
            "want different placements here, and one command cannot serve both.")


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
            "0.90x would falsify this and we ask for exactly that measurement below.")
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
                "on the same model while being 32% smaller. If a ~4.5-bit file fits in VRAM, use "
                "it instead; prefer Q4_0 over Q4_K_M on pre-Ampere (+19% measured, bytes explain "
                "only 5.7%). On Ampere+ this is unverified and may invert.")
    if bits <= 5.0:
        return ("FORMAT LEVER (pre-Ampere cards): at this bit-width the FORMAT is worth more than "
                "the bytes - Q4_0 measured +19% over Q4_K_M on the same model (26.87 vs 22.72 "
                "tok/s). The deficit is intrinsic to the K-quant format/kernel pair (cost-model "
                "kernels acquit the arithmetic itself, prereg #56 - the suspects are layout walk "
                "and occupancy). Speed-only: Q4_K_M is higher quality per byte. On Ampere+ this "
                "is unverified and may invert.")
    return None


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
    experts_offloaded = "exps=CPU" in (placement or "")
    if moe and "split experts" in (placement or ""):
        # Measured on the flagship ITSELF, on this exact placement (pre-registration #28). The
        # axis that matters is COPY vs NOVEL, not code vs prose: the prize attaches to output
        # that reproduces context spans (edits, refactors, quoting) - most of what coding agents
        # emit, and none of what fresh generation emits.
        return ("if your output REUSES its context - edits, refactors, RAG quoting - add "
                "`--spec-type ngram-simple --spec-ngram-simple-size-m 384 "
                "--spec-ngram-simple-size-n 4` to llama-server: measured **4.7x decode at ~3-bit** "
                "(21.3 -> 98.8 tok/s), no download. REQUIRES A LONG PROMPT: with these values the drafter "
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
            out.append((f"split: {gpu_layers}/{n_layer} layers->VRAM, rest->RAM",
                        1 / (g * act / (geta_w * vb) + (1 - g) * act / (eta_r * rb) + kv_t), None,
                        f"-ngl {gpu_layers}"))
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
    # Prefill lever, appended only where the measurement says it pays (host-resident weights with
    # VRAM headroom). Not part of the law: `evaluate` is untouched and no anchor can move.
    ub = ubatch_flags(best[0], ne * max(args.bits, 4.5) / 8 * 1.08 if moe else 0.0, vc)
    run_flags = f"{best[3]} {ub}" if ub else best[3]
    print(f"\n  run it:  llama-server -m model.gguf {run_flags}")
    # I-quant files on a host tier: measured 2.7x slower than K-quants at the same size
    # (pre-registration #31: IQ3_XS 10.6 GB/s vs Q2_K 28.4 / Q4_K_M 29.7 on pure-CPU decode).
    # The warning fires only when weights actually land on the CPU - in VRAM the IQ formats
    # are fine, so warning there would be crying wolf.
    if getattr(args, "iq_share", 0.0) > 0.3 and any(
            k in best[0] for k in ("->RAM", "CPU", "disk", "split experts")):
        print(f"\n  WARNING: this file is {args.iq_share*100:.0f}% I-quant (IQ*) tensors and this"
              f" placement puts weights\n  on the CPU tier, where IQ formats decode ~2.7x slower"
              f" than K-quants of the same size\n  (measured: 10.6 vs ~29 GB/s effective)."
              f" Re-download the model as Q_K (e.g. Q3_K_M\n  instead of IQ3_XS) before"
              f" trusting the speed above.")
    ph = phase_advice(best[0], cfgs)
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
            print(f"  ({rag['pp']:.0f} pp / {rag['tg']:.1f} tg). Earlier versions offered a choice here; "
                  "re-measuring")
            print("  every cell in ONE session (pre-registration #25) showed the alternatives were "
                  "dominated,")
            print("  so there is nothing to choose. One fewer knob, and the honest number.")
            print()
            print("  long prompts, same document: send cache_prompt=true to llama-server. Measured")
            print("  here (pre-registration #29): the second question against a 2k-token document")
            print("  pays 183 ms of prompt time instead of 5381 - 29x - because the document KV is")
            print("  reused and only the new question is processed. Restarting the server is cold.")
    if ub:
        print(f"\n  prompt speed: `{ub}` is worth **+73% prefill** on this placement (measured "
              f"199.9 -> 345.9 tok/s, pre-registration #19). It costs nothing on generation "
              f"(18.5 -> 18.8) and applies because your experts sit in RAM: they then cross PCIe "
              f"once per batch instead of once per token. Do NOT set it when a model is fully in "
              f"VRAM - measured there, the same flag LOSES 39%.")
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
    print("\n  concurrency: every number above is SINGLE-STREAM. Measured with 8 parallel slots, "
          "aggregate\n  throughput is ~2x higher and saturates by about 4 slots "
          "(pre-registration #26). The same\n  ratio appeared on every placement AND on a dense "
          "control, so it is a ceiling we do not model -\n  read our figures as one user's speed, not a server's capacity.")
    print("\n  (eta bands fitted from published measurements; estimates +/-25%. "
          "Hybrid needs --no-mmap.)")
