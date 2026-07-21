"""Is the real fine-tune delta LOW-RANK? (the structure per-element coders can't see)

Per-element entropy coders are at the floor. But fine-tuning updates are often
approximately low-rank (the LoRA premise). If a delta matrix dW = W_ft - W_base has
its energy concentrated in the top-r singular values, we can store a small low-rank
term + an exact, much-smaller residual -> a lever invisible to byte/ULP coders.

We SVD the real SmolLM delta matrices and report energy captured vs rank, and the
implied residual RMS (sqrt of leftover energy) which sets how much the residual's
ULP-moves shrink. Decides whether to build a low-rank residual coder.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from weights import wcodec as wc  # noqa: E402

BB = os.path.join(_ROOT, "weights", "data", "smollm", "base_bf16.safetensors")
INST = os.path.join(_ROOT, "weights", "data", "smollm", "instruct.safetensors")


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main(base=BB, ft=INST):
    braw = open(base, "rb").read()
    iraw = open(ft, "rb").read()
    _, _, hb, ob = wc.parse(base)
    _, _, hi, oi = wc.parse(ft)
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}

    print(f"{'tensor':<42}{'shape':>14}{'r4':>6}{'r16':>6}{'r64':>7}{'rnd':>7}")
    print("-" * 82)
    shown = 0
    agg = {}
    for name, dt, b, e in wc._tensors_in_order(hi):
        if name not in bt:
            continue
        shape = hi[name]["shape"]
        if len(shape) != 2 or min(shape) < 64:
            continue
        m, n = shape
        u = np.frombuffer(iraw[oi + b:oi + e], "<u2")
        r = np.frombuffer(braw[ob + bt[name][1]:ob + bt[name][2]], "<u2")
        if u.size != m * n:
            continue
        dW = (bf16_to_f32(u) - bf16_to_f32(r)).reshape(m, n).astype(np.float64)
        if not np.isfinite(dW).all() or dW.std() == 0:
            continue
        s = np.linalg.svd(dW, compute_uv=False)
        e2 = s ** 2
        tot = e2.sum()
        k = min(m, n)

        def energy(rr):
            return float(e2[:min(rr, k)].sum() / tot)

        # random-matrix baseline: rank-r of a full-rank random matrix ~ r/k energy
        rnd16 = 16 / k
        kind = name.split(".")[-2] if "." in name else name
        agg.setdefault(kind, []).append((energy(4), energy(16), energy(64), rnd16))
        if shown < 16:
            print(f"{name[:42]:<42}{str(tuple(shape)):>14}"
                  f"{energy(4)*100:>5.0f}%{energy(16)*100:>5.0f}%{energy(64)*100:>6.0f}%"
                  f"{rnd16*100:>6.1f}%")
            shown += 1

    print("\nby layer-kind (mean energy captured):")
    print(f"{'kind':<22}{'r4':>7}{'r16':>7}{'r64':>7}{'rand-r16':>10}{'count':>7}")
    for kind, rows in sorted(agg.items()):
        a = np.array(rows)
        print(f"{kind:<22}{a[:,0].mean()*100:>6.0f}%{a[:,1].mean()*100:>6.0f}%"
              f"{a[:,2].mean()*100:>6.0f}%{a[:,3].mean()*100:>9.1f}%{len(rows):>7}")
    print("\nIf r16/r64 energy >> rand baseline => delta is low-rank => residual coder helps.")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
