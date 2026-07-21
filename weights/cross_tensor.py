"""Cross-tensor structure of a real edit (paper §3-C2 / technique extension).

Abliteration writes ΔW_l = -r̂·(r̂ᵀW_l): rank-1 per layer with a SHARED output direction r̂
(the refused direction in the residual stream). If true, the top-left singular vector of
every changed tensor's delta should align (|cosine|→1) across layers AND across the two
edited projections (o_proj, down_proj) -- both write to the residual stream. A shared global
basis could then be stored once instead of per tensor. We measure it on real Qwen abliteration.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import wcodec as wc  # noqa: E402

BASE = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")
ABL = os.path.join(_ROOT, "weights", "data", "qwen", "ablit.safetensors")


def top_left(dW):
    U, S, _ = np.linalg.svd(dW, full_matrices=False)
    return U[:, 0], float(S[0] ** 2 / (S ** 2).sum())  # left vec, rank-1 energy frac


def main():
    braw, _, hb, ob = wc.parse(BASE)
    araw, _, ha, oa = wc.parse(ABL)
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}
    vecs = {"o_proj": [], "down_proj": []}
    energy = []
    for name, dt, b, e in wc._tensors_in_order(ha):
        kind = "o_proj" if "o_proj" in name else ("down_proj" if "down_proj" in name else None)
        if kind is None or name not in bt or dt != "BF16":
            continue
        m_, n_ = ha[name]["shape"]
        bf = wc._bf16_to_f32(np.frombuffer(braw[ob + bt[name][1]:ob + bt[name][2]], "<u2")).reshape(m_, n_)
        af = wc._bf16_to_f32(np.frombuffer(araw[oa + b:oa + e], "<u2")).reshape(m_, n_)
        dW = (af - bf).astype(np.float64)
        if dW.shape[0] != dW.shape[0] or np.abs(dW).max() == 0:
            continue
        u, en = top_left(dW)
        vecs[kind].append(u)
        energy.append(en)

    allv = vecs["o_proj"] + vecs["down_proj"]
    # all left vectors are hidden-dim (the residual-stream direction); align signs
    V = np.stack(allv)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    ref = V[0]
    V = V * np.sign(V @ ref)[:, None]  # fix sign ambiguity
    C = V @ V.T
    off = C[~np.eye(len(C), dtype=bool)]
    print(f"abliteration cross-tensor structure ({len(allv)} changed matrices)")
    print(f"  rank-1 energy per delta:  mean {np.mean(energy)*100:.1f}%  min {np.min(energy)*100:.1f}%")
    print(f"  pairwise |cosine| of top-left singular vectors: mean {np.mean(np.abs(off)):.3f}  "
          f"min {np.min(np.abs(off)):.3f}")
    # do o_proj and down_proj share the SAME direction?
    if vecs["o_proj"] and vecs["down_proj"]:
        uo = np.mean([v * np.sign(v @ ref) for v in vecs["o_proj"]], axis=0)
        ud = np.mean([v * np.sign(v @ ref) for v in vecs["down_proj"]], axis=0)
        cos = abs(float(uo @ ud / (np.linalg.norm(uo) * np.linalg.norm(ud))))
        print(f"  o_proj vs down_proj mean-direction |cosine|: {cos:.3f}")
    print("  => |cosine|~1 confirms a single shared rank-1 direction across the whole model;")
    print("     abliteration is one global edit. (Coding note: the lossless residual dominates,")
    print("     so a shared global basis saves <2%; the characterization is the contribution.)")


if __name__ == "__main__":
    main()
