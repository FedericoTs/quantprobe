"""Quality of a dp4a-native low-bit format vs Q2_K, measured on real model weights.

The speed case for a 2-bit dp4a format is settled (#53: 2.07x headroom at equal bits/weight).
The open question is whether such a format can carry enough metadata to match Q2_K's fidelity,
because a fast format that degrades the model is worth nothing.

This answers it WITHOUT touching llama.cpp: read real fp-ish weights out of a Q8_0 GGUF (Q8_0 is
near-lossless, so it stands in for the original), quantize each candidate scheme, reconstruct, and
compare error against Q2_K's own scheme implemented to the same spec llama.cpp uses.

Everything is reported per tensor AND aggregated, because a scheme can look fine on average and
destroy one critical tensor.
"""
import sys, numpy as np

sys.path.insert(0, r"C:/Users/Federico/Documents/evo-compress/tools/llama.cpp-src/gguf-py")
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType

QK_K = 256


def q2k_roundtrip(w):
    """Q2_K as llama.cpp defines it: 256-weight superblock, 16 sub-blocks of 16.
    Each sub-block has a 4-bit scale and a 4-bit min, themselves scaled by superblock fp16
    d and dmin. 2 bits per weight + 16 bytes of packed scales + 2 fp16 = 2.625 bits/weight.
    Reconstruction: x = d*scale*q - dmin*min, with q in [0,3].
    """
    n = (w.size // QK_K) * QK_K
    x = w[:n].reshape(-1, QK_K, )
    sb = x.reshape(-1, 16, 16)                      # [nblk, 16 sub-blocks, 16 weights]

    # per sub-block asymmetric min/max fit
    lo = sb.min(axis=2)
    hi = sb.max(axis=2)
    scale = (hi - lo) / 3.0                          # 4 levels
    mn = -lo                                          # stored as a positive offset

    # quantize the 16 sub-block scales and mins to 4 bits each, relative to a superblock fp16
    d = scale.max(axis=1) / 15.0
    dmin = mn.max(axis=1) / 15.0
    d = np.where(d == 0, 1e-30, d)
    dmin = np.where(dmin == 0, 1e-30, dmin)
    ls = np.clip(np.rint(scale / d[:, None]), 0, 15)
    lm = np.clip(np.rint(mn / dmin[:, None]), 0, 15)

    rs = ls * d[:, None]                             # reconstructed sub-block scale
    rm = lm * dmin[:, None]                          # reconstructed sub-block min
    rs_s = np.where(rs == 0, 1e-30, rs)
    q = np.clip(np.rint((sb + rm[:, :, None]) / rs_s[:, :, None]), 0, 3)
    rec = q * rs[:, :, None] - rm[:, :, None]
    return rec.reshape(-1)[: n], n


def sym2_roundtrip(w, group=32):
    """The probe format as benchmarked: symmetric 2-bit, one fp16 scale per `group` weights.
    q in [0,3] with an implicit -2 offset. 2 + 16/group bits per weight.
    """
    n = (w.size // group) * group
    g = w[:n].reshape(-1, group)
    amax = np.abs(g).max(axis=1)
    s = np.where(amax == 0, 1e-30, amax / 2.0)       # levels -2,-1,0,1 scaled
    q = np.clip(np.rint(g / s[:, None]) + 2, 0, 3)
    rec = (q - 2) * s[:, None]
    return rec.reshape(-1)[:n], n


def asym2_roundtrip(w, group=16):
    """Candidate: asymmetric 2-bit with fp16 scale AND fp16 min per `group`.
    Keeps the dp4a unpack trivial (the offset folds into a hoisted sum(x) term, exactly as the
    probe kernel already does) while restoring the asymmetry Q2_K has.
    2 + 32/group bits per weight -> at group=16 that is 4.0, at group=32 it is 3.0.
    """
    n = (w.size // group) * group
    g = w[:n].reshape(-1, group)
    lo = g.min(axis=1)
    hi = g.max(axis=1)
    s = (hi - lo) / 3.0
    s = np.where(s == 0, 1e-30, s)
    q = np.clip(np.rint((g - lo[:, None]) / s[:, None]), 0, 3)
    rec = q * s[:, None] + lo[:, None]
    return rec.reshape(-1)[:n], n


def asym2_i8scale(w, group=16, sup=256):
    """Candidate: asymmetric 2-bit, per-group scale and min stored as BYTES (not 4-bit packed),
    each times a superblock fp16. Byte-aligned metadata is the whole point: it removes the
    shift/mask work that makes Q2_K's unpack expensive, at the cost of 4 extra bits per group.
    2 + 16/group bits per weight -> group=16: 3.0, group=32: 2.5.
    """
    n = (w.size // sup) * sup
    x = w[:n].reshape(-1, sup)
    per = sup // group
    g = x.reshape(-1, per, group)
    lo = g.min(axis=2)
    hi = g.max(axis=2)
    scale = (hi - lo) / 3.0
    mn = -lo
    d = np.where(scale.max(axis=1) == 0, 1e-30, scale.max(axis=1) / 255.0)
    dmin = np.where(np.abs(mn).max(axis=1) == 0, 1e-30, np.abs(mn).max(axis=1) / 127.0)
    ls = np.clip(np.rint(scale / d[:, None]), 0, 255)
    lm = np.clip(np.rint(mn / dmin[:, None]), -128, 127)
    rs = ls * d[:, None]
    rm = lm * dmin[:, None]
    rs_s = np.where(rs == 0, 1e-30, rs)
    q = np.clip(np.rint((g + rm[:, :, None]) / rs_s[:, :, None]), 0, 3)
    rec = q * rs[:, :, None] - rm[:, :, None]
    return rec.reshape(-1)[:n], n


def sym2_i8scale(w, group=16, sup=256):
    """THE CANDIDATE. Symmetric 2-bit, levels q-2 in {-2,-1,0,1} (a zero level, which matters for
    zero-peaked weights), one BYTE of scale per `group`, times a superblock fp16.

    bits/weight = 2 + 8/group + 16/sup.  group=16 -> 2.5625, group=32 -> 2.3125.

    Unpack cost on the GPU is the minimum possible: (v >> 2j) & 0x03030303 gives a dp4a operand
    directly, the -2 offset folds into a sum(x) term that is row-independent and hoisted out of the
    loop entirely, and the scale is a byte load with no shift or mask. There is no min to decode.
    """
    n = (w.size // sup) * sup
    x = w[:n].reshape(-1, sup)
    per = sup // group
    g = x.reshape(-1, per, group)
    amax = np.abs(g).max(axis=2)
    scale = amax / 2.0                                  # levels -2..1 scaled
    d = np.where(scale.max(axis=1) == 0, 1e-30, scale.max(axis=1) / 255.0)
    ls = np.clip(np.rint(scale / d[:, None]), 0, 255)
    rs = ls * d[:, None]
    rs_s = np.where(rs == 0, 1e-30, rs)
    q = np.clip(np.rint(g / rs_s[:, :, None]) + 2, 0, 3)
    rec = (q - 2) * rs[:, :, None]
    return rec.reshape(-1)[:n], n


def rmse(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt((d * d).mean()))


def main(path, max_tensors=12):
    r = GGUFReader(path)
    # the tensors that actually matter for decode: the big 2-D projection matrices
    # the tensors that actually matter for decode are the projection matrices; skip token_embd,
    # whose distribution is unlike the weight matrices and which dominates by sheer size
    cands = [t for t in r.tensors
             if len(t.shape) == 2 and t.n_elements >= 1 << 20 and not t.name.startswith("token_embd")]
    cands = cands[:max_tensors]
    print(f"model: {path}")
    print(f"tensors examined: {len(cands)} (2-D projection matrices, >= 1M elements)\n")

    schemes = [
        ("Q2_K             (2.625 bit)", lambda w: q2k_roundtrip(w)),
        ("sym2  g32 fp16   (2.500 bit)", lambda w: sym2_roundtrip(w, 32)),
        ("sym2  g16 fp16   (3.000 bit)", lambda w: sym2_roundtrip(w, 16)),
        ("sym2  g32 i8sc   (2.312 bit)", lambda w: sym2_i8scale(w, 32)),
        ("sym2  g16 i8sc   (2.562 bit)", lambda w: sym2_i8scale(w, 16)),
        ("sym2  g8  i8sc   (3.062 bit)", lambda w: sym2_i8scale(w, 8)),
        ("asym2 g16 i8sc   (3.125 bit)", lambda w: asym2_i8scale(w, 16)),
    ]

    tot = {name: [0.0, 0.0] for name, _ in schemes}   # [sum sq err, count]
    for t in cands:
        # t.data is the RAW packed block bytes for a quantized tensor. It must be dequantized,
        # or every number below is the quantization error of random bytes rather than of weights.
        if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            w = np.array(t.data, dtype=np.float32).reshape(-1)
        else:
            w = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)
        line = f"  {t.name:<34s} n={w.size:>9d} sd={w.std():.4f}"
        for name, fn in schemes:
            rec, n = fn(w)
            e = rec - w[:n].astype(np.float64)
            tot[name][0] += float((e * e).sum())
            tot[name][1] += n
            line += f"  {rmse(rec, w[:n]):.5f}"
        print(line)

    ref = np.sqrt(tot[schemes[0][0]][0] / tot[schemes[0][0]][1])
    print("\n" + "=" * 78)
    print(f"{'scheme':<30s} {'aggregate RMSE':>15s} {'vs Q2_K':>10s}   verdict")
    print("-" * 78)
    for name, _ in schemes:
        s, c = tot[name]
        v = np.sqrt(s / c)
        ratio = v / ref
        verdict = ("BETTER than Q2_K" if ratio < 0.98 else
                   "parity" if ratio < 1.05 else
                   f"{ratio:.2f}x worse")
        print(f"{name:<30s} {v:>15.6f} {ratio:>9.3f}x   {verdict}")
    print("=" * 78)
    print("\nRMSE is on the raw weights. Lower is better. Q2_K is the bar to beat at 2.625 bit;")
    print("a candidate only matters if it is BOTH dp4a-cheap to unpack AND at or under that bar.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "D:/evo-compress-data/gguf/Qwen2.5-0.5B-Instruct-Q8_0.gguf")
