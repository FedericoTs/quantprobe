"""quantprobe autospec - read the MODEL's law-parameters from the GGUF itself.

A GGUF already contains everything the decode law needs:
  total params    = sum of tensor element counts
  routed/active   = expert tensor split + expert_used/expert_count from metadata
  effective bits  = file bytes x 8 / total params (the real number, not the type name)
  kv bytes/pos    = exact, from layer count x KV heads x head dims (MLA-aware)
So `--gguf model.gguf` alone fully specifies the model; flags remain as overrides.
"""
from __future__ import annotations
import os
import re


# Multi-part GGUFs (llama.cpp gguf-split naming: model-00001-of-00002.gguf). The first external
# contribution the tool ever received (issue #1, RX 5700 XT) arrived as "total=None active=None
# @ 2.5-bit" for a Q4_0 7.6B - because the file was a 2-part split, from_gguf saw one part,
# and every downstream consumer fell back to defaults. A split part is a fully valid GGUF, so
# the fix is enumeration, not parsing: spec from ALL parts, size from ALL parts.
_SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def split_siblings(path):
    """All parts of a split GGUF (sorted), or [path] for a normal file. Missing parts raise:
    a spec computed from a subset of the model is wrong, not approximate."""
    m = _SPLIT_RE.match(os.path.basename(path))
    if not m:
        return [path]
    stem, _no, cnt = m.group(1), int(m.group(2)), int(m.group(3))
    d = os.path.dirname(path) or "."
    parts = [os.path.join(d, f"{stem}-{i:05d}-of-{cnt:05d}.gguf") for i in range(1, cnt + 1)]
    missing = [p for p in parts if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"split GGUF: {len(parts) - len(missing)} of {cnt} parts present "
            f"(first missing: {os.path.basename(missing[0])})")
    return parts


def gguf_size(path):
    """Bytes on disk for the WHOLE model - sums split parts. os.path.getsize on part 1 of a
    2-part file halves every size-derived quantity (bits, act_scale, tier placement)."""
    return sum(os.path.getsize(p) for p in split_siblings(path))


def _field(r, *names):
    for f in r.fields.values():
        for n in names:
            if f.name.endswith(n):
                try:
                    return int(f.parts[f.data[0]][0])
                except Exception:
                    pass
    return None


# Effective decode bandwidth BY FORMAT on the reference box, e2e (GB/s). This is L-16 made
# actionable: eta is a property of the format, so an anchor measured on one format must be
# rescaled before it prices another (prereg #65 measured the cost of NOT doing this: -34% on
# Q4_K_M from a Q8_0 anchor). 'measured' = e2e on a real 7B (#52/#53) or in-situ per-op (#58);
# 'derived' = scaled from the kernelprobe ladder. Formats not listed (IQ*, exotic) are excluded
# from the weighting; if coverage falls below 60% of bytes the factor is withheld entirely -
# no number beats a wrong number.
FORMAT_EBW = {
    "Q4_0": 119.1, "Q4_K": 106.4, "Q2_K": 65.4, "Q3_K": 57.3, "Q6_K": 100.0,   # measured
    "Q8_0": 115.0, "Q5_K": 103.0, "Q5_0": 117.0, "Q5_1": 115.0,                # derived
    "F16": 150.0, "F32": 150.0, "BF16": 150.0,                                  # derived
    # IQ entries measured in prereg #70 (same-session matched set vs the Q4_K control, pure-type
    # values solved from file blends). The divide is CODEBOOK vs not, not IQ vs K: the codebook
    # formats (IQ2/IQ3) pay their lookup in decode; IQ4_NL's kernel is Q4_0-class and lands
    # beside it. IQ2/IQ3 variants not listed stay excluded (coverage rule withholds fmt_bw).
    "IQ2_XS": 51.1, "IQ3_S": 61.1, "IQ3_XXS": 68.3,                             # measured (#70)
    "IQ4_NL": 117.0,                                                            # measured (#70)
    "IQ2_XXS": 46.0,                                                            # measured (#77)
}

# The slowest codebook format we have MEASURED. Prereg #77: when a file's unpriced formats are
# codebook-named and material, pricing them here is conservative by construction - the entire
# measured codebook ladder (IQ2_XXS 46.0 < IQ2_XS 51.1 < IQ3_S 61.1 < IQ3_XXS 68.3) descends with
# bit-width, so an unmeasured IQ2/IQ1 variant cannot plausibly sit above the floor of what we
# have measured. Withholding a number is only honest when the FALLBACK is conservative; the old
# fallback assumed K-quant-class decode and over-promised a 59%-codebook file by 50%.
K_CLASS_IQ = {"IQ4_NL"}   # IQ by name, Q4_0-class by measurement (#70)
WORST_MEASURED_CODEBOOK = 46.0
CODEBOOK_FALLBACK_MIN_SHARE = 0.25


def from_gguf(path):
    from gguf import GGUFReader
    paths = split_siblings(path)
    # Part 1 carries the full metadata (gguf-split copies the KV store there; later parts hold
    # only split bookkeeping). Tensors, however, live where they live: every part contributes.
    r = GGUFReader(paths[0])
    n_layer = _field(r, ".block_count") or 32
    total = 0
    routed = 0
    iq_bytes = all_bytes = unpriced_cb = codebook_bytes = 0
    fmt_wsum = fmt_bytes = 0.0
    tier_exp = {'bytes': 0, 'wsum': 0.0, 'wb': 0, 'cb': 0}   # prereg #79: per-tier format stats
    tier_att = {'bytes': 0, 'wsum': 0.0, 'wb': 0, 'cb': 0}
    embd_params = 0          # token_embd: a GATHER at decode, not a read (U-26 / prereg #76)
    embd_bytes = 0           # the same exclusion in bytes, for L-30's routed share
    has_output = False       # a separate output/lm_head means embeddings are NOT tied
    kv_blocks = set()        # U-51: blocks that carry a K projection, i.e. FULL attention
    tensors = list(r.tensors)
    for extra in paths[1:]:
        tensors.extend(GGUFReader(extra).tensors)
    for t in tensors:
        if ".attn_k." in t.name and t.name.startswith("blk."):
            kv_blocks.add(t.name.split(".")[1])
        n = 1
        for d in t.shape:
            n *= int(d)
        total += n
        if "exps" in t.name or "_expert" in t.name:
            routed += n
        if "token_embd" in t.name:
            embd_params += n
            embd_bytes += int(t.n_bytes)   # L-30 needs the BYTE version of the same exclusion
        if t.name.startswith("output.") or "lm_head" in t.name:
            has_output = True
        # bytes-weighted I-quant share. Measured (pre-registration #31): on the CPU tier the
        # IQ formats deliver 10.6 GB/s against ~29 for K-quants at the same size - a 2.7x
        # decode penalty for any host-resident placement. The K-format dequant is
        # bandwidth-shaped on AVX2; the IQ codebook lookup is compute-shaped, and 4 cores
        # cannot hide it.
        nb = int(getattr(t, "n_bytes", 0) or 0)
        all_bytes += nb
        if t.tensor_type.name.startswith("IQ"):
            iq_bytes += nb
            # C-13: "IQ" is a NAME, not a kernel class. Prereg #70 measured IQ4_NL at 117.0 GB/s -
            # Q4_0-class, 2.3x faster than IQ2_XS - so charging it the codebook penalty applied
            # the tax to up to 89x the bytes that earn it. The penalty follows the CODEBOOK
            # formats (grid lookup in decode), which is what #31 and #70 both actually measured.
            if t.tensor_type.name not in K_CLASS_IQ:
                codebook_bytes += nb
        tn = t.tensor_type.name
        if tn in FORMAT_EBW:
            fmt_wsum += nb * FORMAT_EBW[tn]
            fmt_bytes += nb
        elif tn.startswith("IQ"):
            unpriced_cb += nb        # a codebook format we have not measured (prereg #77)
        # U-28 / prereg #79: the same bytes, split by WHICH TIER holds them. On a MoE split the
        # GPU holds attention and the CPU holds offloaded experts, and their formats differ
        # sharply (measured: attention 82-106 GB/s-equivalent vs experts 50-69 on the Qwen MoEs).
        # One file-wide number cannot price both.
        _t = tier_exp if ("exps" in t.name or "_expert" in t.name) else tier_att
        _t["bytes"] += nb
        if tn in FORMAT_EBW:
            _t["wsum"] += nb * FORMAT_EBW[tn]; _t["wb"] += nb
        elif tn.startswith("IQ"):
            _t["wsum"] += nb * WORST_MEASURED_CODEBOOK; _t["wb"] += nb
        if tn.startswith("IQ") and tn not in K_CLASS_IQ:
            _t["cb"] += nb
    # U-26 / prereg #76: token_embd is GATHERED at decode - one row of a ~150k-row matrix, i.e.
    # ~zero bytes - but `total - routed` charged the whole matrix at >=4.5 bits every token.
    # When embeddings are TIED the same tensor IS the output projection and is fully read, so it
    # must stay counted; only the untied case (a separate output/lm_head exists) double-charges.
    # Measured share of active bytes on untied models: 4.2-17.2%, the sign and size of the
    # MoE-K-quant under-prediction family.
    gather_only = embd_params if has_output else 0
    ne_params = total - routed - gather_only

    n_exp = _field(r, ".expert_count")
    n_used = _field(r, ".expert_used_count")
    if routed and n_exp and n_used:
        active = ne_params + routed * n_used / n_exp
        moe = True
    else:
        active, moe = total - gather_only, False    # same gather correction on the dense path

    # exact KV bytes/pos (f16): MLA caches the latent; GQA caches heads x dims, K+V
    #
    # U-51 (prereg #101 P-5): only FULL-attention layers grow a KV cache with position. Hybrid
    # models (Qwen3.8-27B: 48 of 64 layers are linear attention with fixed-size state) were
    # priced as if every layer cached K+V - a measured 4x over-estimate (260 KB/pos read vs ~64
    # real). The count comes from the FILE, not a per-arch table: a block that carries an
    # `attn_k` projection caches K+V; a linear/SSM block has no attn_k and caches nothing that
    # grows. On every full-attention model each block has attn_k, so kv_layers == n_layer and
    # the output is byte-identical to the old formula (the regression guard in tests/smoke.py
    # pins this). Deliberately UNCHANGED here: the n_layer convention includes an MTP block
    # where one exists (it carries attn_k too) - pre-existing behavior, one change at a time.
    # MLA models keep the n_layer path: their cache is the latent, not per-head K+V, and no
    # MLA hybrid has been observed - guessing a rule for one would be invention, not reading.
    kv_layers = len(kv_blocks) if kv_blocks else n_layer
    kv_lora = _field(r, ".attention.kv_lora_rank")
    if kv_lora:
        rope = _field(r, ".rope.dimension_count") or 64
        kvp = n_layer * (kv_lora + rope) * 2
    else:
        kv_heads = _field(r, ".attention.head_count_kv") or 8
        k_dim = _field(r, ".attention.key_length")
        v_dim = _field(r, ".attention.value_length")
        if not k_dim:
            emb = _field(r, ".embedding_length") or 4096
            heads = _field(r, ".attention.head_count") or 32
            k_dim = v_dim = emb // heads
        kvp = kv_layers * kv_heads * ((k_dim or 128) + (v_dim or k_dim or 128)) * 2

    bits = sum(os.path.getsize(p) for p in paths) * 8 / total
    arch = None
    for field in r.fields.values():        # recipe matching needs (arch, n_layer), not layers alone
        if field.name == "general.architecture":
            try:
                arch = bytes(field.parts[field.data[0]]).decode("utf-8")
            except Exception:
                arch = None
            break
    # prereg #77: unmeasured CODEBOOK bytes get the slowest measured codebook rather than being
    # dropped. Dropping them withheld fmt_bw entirely, and the fallback path then assumed
    # K-quant-class decode - over-promising a 59%-codebook file by 50%. Conservative by
    # construction: the measured ladder descends with bit-width, so an unmeasured IQ2/IQ1 variant
    # cannot plausibly beat the floor of what we have measured.
    ws, wb = fmt_wsum, fmt_bytes
    if all_bytes and unpriced_cb / all_bytes >= CODEBOOK_FALLBACK_MIN_SHARE:
        ws += unpriced_cb * WORST_MEASURED_CODEBOOK
        wb += unpriced_cb
    fmt_bw = (ws / wb) if (all_bytes and wb / all_bytes >= 0.6) else None
    # L-30 (prereg #107): the ceiling of the expert-count knob is decided by the routed share of
    # ACTIVE BYTES, and that is readable here. Params are the wrong unit - experts are usually
    # quantized harder than the always-active path, so their byte share is smaller than their
    # param share, and predicting from params overstates the lever.
    routed_active_b = tier_exp["bytes"] * (n_used / n_exp) if (moe and n_exp and n_used) else 0
    always_b = all_bytes - tier_exp["bytes"] - (embd_bytes if has_output else 0)
    return dict(t=total / 1e9, a=active / 1e9, ne=ne_params / 1e9, moe=moe,
                routed_byte_share=(routed_active_b / (routed_active_b + always_b))
                if (moe and (routed_active_b + always_b)) else 0.0,
                n_expert=n_exp, n_expert_used=n_used,
                bits=round(bits, 2), kvp=int(kvp), n_layer=n_layer, arch=arch,
                kv_layers=kv_layers,     # U-51: < n_layer marks a hybrid (linear-attn) model
                iq_share=(iq_bytes / all_bytes) if all_bytes else 0.0,
                codebook_share=(codebook_bytes / all_bytes) if all_bytes else 0.0,
                # prereg #79: what each TIER actually holds. None when a tier has no tensors.
                fmt_bw_attn=round(tier_att["wsum"] / tier_att["wb"], 1) if tier_att["wb"] else None,
                fmt_bw_exp=round(tier_exp["wsum"] / tier_exp["wb"], 1) if tier_exp["wb"] else None,
                codebook_share_exp=(tier_exp["cb"] / tier_exp["bytes"]) if tier_exp["bytes"] else 0.0,
                fmt_bw=round(fmt_bw, 1) if fmt_bw else None)


def expert_ceiling(s):
    """L-30: the most the expert-count knob can ever buy on this file. -> (share, ceiling) or None.

    `--override-kv <arch>.expert_used_count=int:K` is widely traded as a free speed dial for MoE
    on small hardware. It is bounded, and the bound is arithmetic on the file: routed experts own
    `routed_byte_share` of the active bytes AT THE DEFAULT k, so driving k to 1 removes all but
    1/k_default of that share and Law 4 caps the speedup at

        1 / (1 - share * (1 - 1/k_default))

    The divisor is the DEFAULT k, not the expert count. Using n_expert instead reads as "drop from
    all 256 experts to one" and inflates the ceiling - on the file below, 1.28x against the true
    1.24x. Checked against prereg #107's staked byte table, which is why the slip was caught.

    Measured against this prediction on Qwen3.6-35B-A3B (prereg #107, 4/4): k=4 1.146x against
    1.125x predicted, k=2 1.175x against 1.200x. It is exact where the working set is unchanged.
    At k=1 the measurement overshot by 17% - on a model larger than free RAM the touched-expert
    set collapses and the page cache starts paying (L-29). So this is a CEILING for the regime the
    law covers, and the honest word is "about", not a guarantee.

    Returns None for dense models and for any MoE whose expert metadata is missing - a ceiling we
    cannot compute is not a ceiling we should print."""
    if not s.get("moe"):
        return None
    share, k_def = s.get("routed_byte_share") or 0.0, s.get("n_expert_used") or 0
    if not share or k_def < 2:
        return None                       # k=1 already: there is nothing left to turn down
    return share, _ceil(share, k_def)


def _ceil(share, k_def):
    return 1.0 / (1.0 - share * (1.0 - 1.0 / k_def))


def expert_ceiling_prefill(s):
    """The same ceiling for PREFILL -> (share, ceiling) or None.

    L-30 prices decode, which is bandwidth-bound, so it divides BYTES. Prefill is compute-bound
    and compute scales with PARAMETERS - quantization shrinks bytes, not FLOPs. The routed share
    is therefore larger here, and so is the ceiling.

    Measured (prereg #108, V-23): prefill moved 1.613x at k=4 and 3.766x at k=2, where decode
    moved 1.146x and 1.175x. The direction and ordering are confirmed. The MAGNITUDE is not -
    the param share predicted 1.206x at k=4 and the measurement came in 34% above it, so this
    number is a FLOOR on what prefill does, not an estimate of it. Said that way in the note."""
    a, ne, k_def = s.get("a") or 0.0, s.get("ne") or 0.0, s.get("n_expert_used") or 0
    if not s.get("moe") or not a or a <= ne or k_def < 2:
        return None
    share = (a - ne) / a
    return share, _ceil(share, k_def)


def expert_ceiling_note(s):
    """One line for `plan`/`report`, or None when there is nothing honest to say."""
    r = expert_ceiling(s)
    if not r:
        return None
    share, ceil = r
    if ceil < 1.05:                       # below a 5% ceiling the knob is not worth a line
        return (f"  experts        routed experts are only {share*100:.0f}% of the active bytes - "
                f"cutting expert_used_count cannot help here (L-30)")
    out = (f"  experts        routed experts are {share*100:.0f}% of the active bytes, so "
           f"lowering expert_used_count\n"
           f"                 buys at most ~{ceil:.2f}x DECODE even at k=1 - and quality falls "
           f"long before that (L-30, prereg #107)")
    pf = expert_ceiling_prefill(s)
    if pf:
        pshare, pceil = pf
        out += (f"\n                 PREFILL is the better place for it: {pshare*100:.0f}% of "
                f"active PARAMS are routed, and\n"
                f"                 measured gains beat this {pceil:.2f}x floor (1.6x at k=4, "
                f"3.8x at k=2 - V-23, prereg #108).\n"
                f"                 Neither is free: halving the experts cost +1.51 perplexity "
                f"on the measured model.")
    return out


def apply(a, quiet=False):
    """Fill law-parameters from a.gguf for anything the user didn't set. Explicit flags win."""
    g = getattr(a, "gguf", None)
    if not g or not os.path.isfile(g):
        return False
    try:
        s = from_gguf(g)
    except Exception as e:
        if not quiet:
            print(f"[quantprobe] autospec skipped ({e}); using flags/presets")
        return False
    used = []
    if getattr(a, "total", None) is None and getattr(a, "model", None) is None:
        a.total = s["t"]; a.active = a.active or s["a"]; a.always_active = a.always_active or s["ne"]
        used.append(f"{s['t']:.1f}B total, {s['a']:.1f}B active")
    if getattr(a, "bits", None) is None:
        a.bits = s["bits"]; used.append(f"{s['bits']:g} effective bits")
    if getattr(a, "kv_per_pos", None) is None:
        a.kv_per_pos = s["kvp"] / 1024
        if s.get("kv_layers", s["n_layer"]) < s["n_layer"]:
            used.append(f"KV {s['kvp']/1024:.0f} KB/pos (hybrid: {s['kv_layers']} of "
                        f"{s['n_layer']} layers cache KV)")
        else:
            used.append(f"KV {s['kvp']/1024:.0f} KB/pos")
    if getattr(a, "n_layer", None) is None:
        a.n_layer = s["n_layer"]        # enables the MoE partial-offload -ot regex (needs real layer indices)
    a.iq_share = s.get("iq_share", 0.0)  # read-only: lets plan warn when IQ weights land on a CPU tier
    a.arch = s.get("arch")               # read-only: report's recipe matching needs (arch, n_layer)
    a.kv_layers = s.get("kv_layers")     # read-only: report's hybrid-KV line ("N of L layers cache KV")
    a.codebook_share = s.get("codebook_share", 0.0)   # the share that actually pays the tax (C-13)
    a.fmt_bw_attn = s.get("fmt_bw_attn")              # prereg #79: per-tier format pricing
    a.fmt_bw_exp = s.get("fmt_bw_exp")
    a.codebook_share_exp = s.get("codebook_share_exp", 0.0)
    a.fmt_bw = s.get("fmt_bw")           # read-only: lets anchored predictions rescale by format (L-16)
    a._spec = s                          # read-only: L-30's expert ceiling needs the byte split,
    #                                      and re-scanning the GGUF to recover it would be silly
    if used and not quiet:
        print(f"[quantprobe] read from GGUF: " + ", ".join(used))
    return True


# --- tensor-role registry -----------------------------------------------------------------
# WHAT TRANSFERS between models is structure, not fragility. Which tensor classes exist and
# which are always-active is a property of the ARCHITECTURE, so it is knowable for a model
# nobody has ever probed. Which LAYERS are fragile is not (Law 3; Mistral is early-fragile
# where its near-twin Qwen is late, a 25x error if you guess).
#
# This registry exists because we shipped that bug: hybrid SSM architectures name their
# recurrent-state tensors ssm_*, our protection pattern only matched attn_*, and every SSM
# tensor silently landed at the aggressive base level (fixed in v1.6.4, cost -24% ppl).
# Anything unrecognised is now REPORTED rather than silently compressed.
TENSOR_ROLES = [
    ("routed-expert", r"ffn_(gate|up|down)_exps",  "compressible: ~8 of N experts fire per token"),
    ("shared-expert", r"ffn_.*_shexp",             "ALWAYS ACTIVE on every token - protect"),
    ("attention",     r"attn_",                     "ALWAYS ACTIVE - protect"),
    ("recurrent/SSM", r"ssm_",                      "ALWAYS ACTIVE - protect"),
    ("embedding",     r"(token_embd|output\.)",     "ALWAYS ACTIVE - protect"),
    ("mtp-head",      r"nextn",                      "ALWAYS ACTIVE when MTP is on - protect"),
    ("router",        r"ffn_gate_inp",              "tiny, kept at full precision"),
    ("norm",          r"(_norm|norm\.)",            "tiny, kept at full precision"),
    ("dense-ffn",     r"ffn_(gate|up|down)\.",      "dense FFN - the depth-aware band applies here"),
]


def tensor_roles(path):
    """Classify a GGUF's tensors by role. Returns (roles, unknown) with byte totals, so the
    builder can warn about weight classes it has no protection rule for."""
    import re
    from gguf import GGUFReader
    tensors = []
    for p in split_siblings(path):
        tensors.extend(GGUFReader(p).tensors)
    roles, unknown = {}, {}
    for t in tensors:
        nbytes = int(t.n_bytes) if hasattr(t, "n_bytes") else 0
        for name, pat, _ in TENSOR_ROLES:
            if re.search(pat, t.name):
                roles[name] = roles.get(name, 0) + nbytes
                break
        else:
            key = re.sub(r"blk\.\d+\.", "", t.name)
            unknown[key] = unknown.get(key, 0) + nbytes
    return roles, unknown
