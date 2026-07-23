"""quantprobe autospec — read the MODEL's law-parameters from the GGUF itself.

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


def from_gguf(path):
    from gguf import GGUFReader
    r = GGUFReader(path)
    n_layer = _field(r, ".block_count") or 32
    total = 0
    routed = 0
    for t in r.tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        total += n
        if "exps" in t.name or "_expert" in t.name:
            routed += n
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
    return dict(t=total / 1e9, a=active / 1e9, ne=ne_params / 1e9, moe=moe,
                bits=round(bits, 2), kvp=int(kvp), n_layer=n_layer)


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
    if used and not quiet:
        print(f"[quantprobe] read from GGUF: " + ", ".join(used))
    return True
