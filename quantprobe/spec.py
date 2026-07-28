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
}


def from_gguf(path):
    from gguf import GGUFReader
    r = GGUFReader(path)
    n_layer = _field(r, ".block_count") or 32
    total = 0
    routed = 0
    iq_bytes = all_bytes = 0
    fmt_wsum = fmt_bytes = 0.0
    for t in r.tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        total += n
        if "exps" in t.name or "_expert" in t.name:
            routed += n
        # bytes-weighted I-quant share. Measured (pre-registration #31): on the CPU tier the
        # IQ formats deliver 10.6 GB/s against ~29 for K-quants at the same size - a 2.7x
        # decode penalty for any host-resident placement. The K-format dequant is
        # bandwidth-shaped on AVX2; the IQ codebook lookup is compute-shaped, and 4 cores
        # cannot hide it.
        nb = int(getattr(t, "n_bytes", 0) or 0)
        all_bytes += nb
        if t.tensor_type.name.startswith("IQ"):
            iq_bytes += nb
        tn = t.tensor_type.name
        if tn in FORMAT_EBW:
            fmt_wsum += nb * FORMAT_EBW[tn]
            fmt_bytes += nb
    ne_params = total - routed

    n_exp = _field(r, ".expert_count")
    n_used = _field(r, ".expert_used_count")
    if routed and n_exp and n_used:
        active = ne_params + routed * n_used / n_exp
        moe = True
    else:
        active, moe = total, False

    # exact KV bytes/pos (f16): MLA caches the latent; GQA caches heads x dims, K+V
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
        kvp = n_layer * kv_heads * ((k_dim or 128) + (v_dim or k_dim or 128)) * 2

    bits = os.path.getsize(path) * 8 / total
    arch = None
    for field in r.fields.values():        # recipe matching needs (arch, n_layer), not layers alone
        if field.name == "general.architecture":
            try:
                arch = bytes(field.parts[field.data[0]]).decode("utf-8")
            except Exception:
                arch = None
            break
    fmt_bw = (fmt_wsum / fmt_bytes) if (all_bytes and fmt_bytes / all_bytes >= 0.6) else None
    return dict(t=total / 1e9, a=active / 1e9, ne=ne_params / 1e9, moe=moe,
                bits=round(bits, 2), kvp=int(kvp), n_layer=n_layer, arch=arch,
                iq_share=(iq_bytes / all_bytes) if all_bytes else 0.0,
                fmt_bw=round(fmt_bw, 1) if fmt_bw else None)


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
        a.kv_per_pos = s["kvp"] / 1024; used.append(f"KV {s['kvp']/1024:.0f} KB/pos")
    if getattr(a, "n_layer", None) is None:
        a.n_layer = s["n_layer"]        # enables the MoE partial-offload -ot regex (needs real layer indices)
    a.iq_share = s.get("iq_share", 0.0)  # read-only: lets plan warn when IQ weights land on a CPU tier
    a.fmt_bw = s.get("fmt_bw")           # read-only: lets anchored predictions rescale by format (L-16)
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
    r = GGUFReader(path)
    roles, unknown = {}, {}
    for t in r.tensors:
        nbytes = int(t.n_bytes) if hasattr(t, "n_bytes") else 0
        for name, pat, _ in TENSOR_ROLES:
            if re.search(pat, t.name):
                roles[name] = roles.get(name, 0) + nbytes
                break
        else:
            key = re.sub(r"blk\.\d+\.", "", t.name)
            unknown[key] = unknown.get(key, 0) + nbytes
    return roles, unknown
