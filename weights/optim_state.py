"""Do OPTIMIZER-STATE deltas compress well? (the 'expand the data' direction)

A training checkpoint is weights + Adam state (m, v) -- the optimizer is ~2x the model.
Unlike weights (whose step deltas are independent, momentum dead), the optimizer moments
are EMAs:  m_t = b1*m_{t-1} + (1-b1)*g_t ,  v_t = b2*v_{t-1} + (1-b2)*g_t^2 .
So m_t - m_{t-1} = (1-b1)(g_t - m_{t-1})  and  v_t - v_{t-1} = (1-b2)(g_t^2 - v_{t-1}) --
a fixed small fraction (10% / 0.1%) of the scale per step. They should delta-compress far
better than weights. We run real AdamW dynamics (real weight tensor, temporally-correlated
synthetic gradients) and compress each state's step-delta with wcodec. Synthetic gradients
UNDERSTATE m's compressibility (real grads are more correlated), so these are lower bounds.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import wcodec as wc  # noqa: E402

QBASE = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")


def real_weight(n_elems=4_000_000):
    raw, _, h, off = wc.parse(QBASE)
    for name, dt, b, e in wc._tensors_in_order(h):
        if dt == "BF16" and (e - b) // 2 >= n_elems:
            u = np.frombuffer(raw[off + b:off + e], "<u2")[:n_elems]
            return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64).copy()
    raise RuntimeError("no big tensor")


_CT, _CP = None, None


def _codecs():
    global _CT, _CP
    if _CT is None:
        _CT, _CP = wc._codecs(19), wc._codecs(3)
    return _CT, _CP


def fp32_delta_save(cur, ref):
    cb = cur.astype(np.float32).tobytes()
    rb = ref.astype(np.float32).tobytes()
    ct, cp = _codecs()
    _, blob = wc._enc_tensor(cb, "F32", rb, ct, cp, 19, 3, list(cur.shape))
    return (1 - len(blob) / len(cb)) * 100, len(blob)


def bf16_delta_save(cur, ref):
    cu = wc._to_bf16(cur.astype(np.float32))
    ru = wc._to_bf16(ref.astype(np.float32))
    ct, cp = _codecs()
    _, blob = wc._enc_tensor(cu.tobytes(), "BF16", ru.tobytes(), ct, cp, 19, 3, list(cur.shape))
    return (1 - len(blob) / (cur.size * 2)) * 100, len(blob)


def standalone_save(cur):
    sb = wc._SMART.decompress  # noqa: just to ensure import; use smart compress below
    blob = wc.cd.SplitSmartCodec("zstd", 19).compress(cur.astype(np.float32).tobytes(), "fp32")
    return (1 - len(blob) / (cur.size * 4)) * 100


def main():
    W = real_weight().reshape(2000, 2000)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    b1, b2, lr, eps = 0.9, 0.999, 1e-4, 1e-8
    rs = np.random.RandomState(0)
    g = np.zeros_like(W)
    gscale = float(W.std()) * 0.02
    steps = 40
    snap = {}
    for t in range(1, steps + 1):
        g = 0.9 * g + 0.1 * rs.standard_normal(W.shape) * gscale  # correlated grad
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        W = W - lr * m / (np.sqrt(v) + eps)
        if t >= steps - 1:
            snap[t] = (W.copy(), m.copy(), v.copy())

    print(f"OPTIMIZER-STATE delta compression after {steps} AdamW steps (converged regime)")
    print(f"  tensor: {W.size:,} params\n")
    (W1, m1, v1) = snap[steps]
    (W0, m0, v0) = snap[steps - 1]
    print(f"{'dtype':<7}{'W save':>10}{'m save':>10}{'v save':>10}{'optim(m+v)':>13}{'full(W+m+v)':>13}")
    print("-" * 64)
    for name, fn, elem in (("fp32", fp32_delta_save, 4), ("bf16", bf16_delta_save, 2)):
        sW, bW = fn(W1, W0)
        sm, bm = fn(m1, m0)
        sv, bv = fn(v1, v0)
        raw1 = W1.size * elem
        s_optim = (1 - (bm + bv) / (2 * raw1)) * 100
        s_full = (1 - (bW + bm + bv) / (3 * raw1)) * 100
        print(f"{name:<7}{sW:>9.1f}%{sm:>9.1f}%{sv:>9.1f}%{s_optim:>12.1f}%{s_full:>12.1f}%")
    print(f"\n  standalone (no delta): W {standalone_save(W1):.1f}%  m {standalone_save(m1):.1f}%  "
          f"v {standalone_save(v1):.1f}%")


if __name__ == "__main__":
    main()
