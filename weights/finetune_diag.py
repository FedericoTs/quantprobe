"""Per-layer structure of a real base->fine-tune delta.

Given two .safetensors files (base + fine-tune of the same architecture), report for
each matching tensor: fraction of elements changed, and the per-plane delta save%.
Reveals whether instruction-tuning leaves many layers nearly frozen (=> sparse mode
wins) or moves everything a little (=> dense per-plane, like training deltas).

  python -m weights.finetune_diag BASE.safetensors FINETUNE.safetensors
"""

from __future__ import annotations

import sys

import numpy as np

from weights import wcodec as wc


def main(base_path, ft_path):
    _, _, hb, ob = wc.parse(base_path)
    _, _, hf, of = wc.parse(ft_path)
    braw = open(base_path, "rb").read()
    fraw = open(ft_path, "rb").read()
    bt = {n: (dt, b, e) for n, dt, b, e in wc._tensors_in_order(hb)}
    smart, perplane = wc._codecs(19)

    rows = []
    tot_raw = tot_delta = tot_changed = tot_elems = 0
    matched = unmatched = 0
    for name, dt, b, e in wc._tensors_in_order(hf):
        fbuf = fraw[of + b:of + e]
        if name not in bt or bt[name][0] != dt:
            unmatched += 1
            continue
        _, bb, be = bt[name]
        bbuf = braw[ob + bb:ob + be]
        if len(bbuf) != len(fbuf):
            unmatched += 1
            continue
        matched += 1
        cdt = wc.ST2DT.get(dt)
        if cdt is None:
            continue
        u = np.frombuffer(fbuf, wc.UVIEW[cdt])
        r = np.frombuffer(bbuf, wc.UVIEW[cdt])
        changed = int((u != r).sum())
        frac = changed / max(u.size, 1)
        x = (u ^ r).astype(wc.UVIEW[cdt])
        dsz = len(perplane.compress(x.tobytes(), cdt))
        save = (1 - dsz / len(fbuf)) * 100
        rows.append((name, u.size, frac, save))
        tot_raw += len(fbuf)
        tot_delta += dsz
        tot_changed += changed
        tot_elems += u.size

    rows.sort(key=lambda r: r[2])
    print(f"matched tensors: {matched}, unmatched: {unmatched}, dtype fp32/other")
    print(f"\n{'fraction changed bucket':<26}{'#tensors':>9}{'%of params':>12}")
    buckets = [(0, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.01)]
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r[2] < hi]
        params = sum(r[1] for r in sel)
        print(f"  [{lo:>5.1%}-{hi:>5.1%})            {len(sel):>9}{params/max(tot_elems,1)*100:>11.1f}%")

    print(f"\nleast-changed tensors:")
    for n, sz, frac, save in rows[:5]:
        print(f"  {frac:>6.2%} changed  save {save:>5.1f}%  {n}")
    print(f"most-changed tensors:")
    for n, sz, frac, save in rows[-5:]:
        print(f"  {frac:>6.2%} changed  save {save:>5.1f}%  {n}")

    print(f"\noverall delta: raw {tot_raw/1e6:.1f} MB -> {tot_delta/1e6:.1f} MB  "
          f"save {(1-tot_delta/tot_raw)*100:.1f}%   "
          f"({tot_changed/max(tot_elems,1)*100:.1f}% of all elements changed)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
