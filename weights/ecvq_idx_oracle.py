"""R25v2-R1 (A5-X1) -- conditional-entropy ORACLE for the champion's ECVQ index stream.
Computes D_idx = order-0 b/w minus best conditional-coding b/w, WITHOUT coder extrapolation.

Contexts (all decoder-reconstructable): prev-index (within group row), position-in-group (0..127),
group-amax octile, and conjunctions. Tables are ROLE-shared (7 linear roles), TRAINED on even
layers, EVALUATED as cross-entropy on odd layers (train/code split kills bucket-overfit flattery);
add-0.5 smoothing; per-role table bytes counted against the bank.

Decision rule: D_idx <= 0.01 -> CASE A (indices iid, order-0 accounting right, lattice comparison
stands); D_idx >= 0.05 -> CASE B (champion banks D_idx at every point, lattice edge shrinks);
else CASE C (borderline, default stay-closed).

Run:  python -m weights.ecvq_idx_oracle [lam=0.008]
"""
from __future__ import annotations

import re
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors import safe_open
from weights.codec_zoo import _ecvq_levels, _nearest_idx
from weights.quant_lab import WPATH, quant_keys
from weights.quant_sota import _fwht_rows

G = 128
P_OUT = 0.005
K = 64            # max levels (uint8 indices)
PREV0 = 64        # sentinel for "no previous symbol"

CTXS = {
    "none":        1,
    "prev":        K + 1,
    "pos":         G,
    "octile":      8,
    "pos*octile":  G * 8,
    "prev*octile": (K + 1) * 8,
}


def features(idx2d, amax, atile):
    """idx2d:[M,G] uint8; returns dict ctx_name -> [M,G] int32 context ids."""
    M = idx2d.shape[0]
    pos = np.broadcast_to(np.arange(G, dtype=np.int32), (M, G))
    prev = np.empty_like(idx2d, dtype=np.int32)
    prev[:, 0] = PREV0
    prev[:, 1:] = idx2d[:, :-1]
    oct_ = np.broadcast_to(atile[:, None].astype(np.int32), (M, G))
    return {
        "none": np.zeros((M, G), np.int32),
        "prev": prev,
        "pos": pos,
        "octile": oct_,
        "pos*octile": pos * 8 + oct_,
        "prev*octile": prev * 8 + oct_,
    }


def main():
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.008
    # accumulators: per (role, ctx): train counts + held joint counts
    train = {}
    held = {}
    n_train = n_held = 0
    h0_bits_held = 0.0          # champion's current accounting on held half (per-tensor order-0)

    with safe_open(WPATH, framework="pt") as f:
        for k in sorted(quant_keys(f)):
            m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.weight", k)
            layer, role = int(m.group(1)), m.group(2)
            is_train = (layer % 2 == 0)
            W = f.get_tensor(k).float().numpy()
            rows, cols = W.shape
            n_out = max(1, int(round(W.size * P_OUT)))
            thr = np.partition(np.abs(W).ravel(), W.size - n_out)[W.size - n_out]
            mask = np.abs(W) >= thr
            base = W.copy(); base[mask] = 0.0
            pad = (-cols) % G
            A = np.pad(base, ((0, 0), (0, pad))) if pad else base
            N = A.reshape(rows, -1, G).reshape(-1, G)
            signs = np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1
            R = _fwht_rows(N * signs) / np.sqrt(G)
            amax = np.abs(R).max(1); amax[amax == 0] = 1e-9
            Rn = R / amax[:, None]
            rng = np.random.default_rng(1)
            samp = Rn.ravel()[rng.integers(0, Rn.size, min(20000, Rn.size))]
            lv = _ecvq_levels(samp, K, lam)
            idx2d = _nearest_idx(Rn.ravel(), lv).reshape(Rn.shape).astype(np.uint8)
            # group-amax octiles (per tensor)
            qs = np.quantile(amax, np.arange(1, 8) / 8.0)
            atile = np.digitize(amax, qs)
            F = features(idx2d, amax, atile)
            flat_idx = idx2d.reshape(-1).astype(np.int64)
            for cname, card in CTXS.items():
                joint = np.bincount(F[cname].reshape(-1).astype(np.int64) * K + flat_idx,
                                    minlength=card * K).reshape(card, K)
                tgt = train if is_train else held
                key = (role, cname)
                tgt[key] = joint if key not in tgt else tgt[key] + joint
            # champion per-tensor order-0 on held half
            if not is_train:
                cnt = np.bincount(flat_idx, minlength=K).astype(np.float64)
                p = cnt[cnt > 0] / cnt.sum()
                h0_bits_held += float(-(p * np.log2(p)).sum()) * flat_idx.size
                n_held += flat_idx.size
            else:
                n_train += flat_idx.size

    h0 = h0_bits_held / n_held
    print(f"held-half (odd layers) champion order-0 accounting H0 = {h0:.4f} b/w  "
          f"(train {n_train:,} / held {n_held:,} weights)\n")
    print(f"{'context':<14}{'CE held|train-tables':>22}{'bank vs H0':>12}{'tables KB':>11}")
    print("-" * 60)
    best = ("none", 0.0)
    roles = sorted({r for (r, _) in train})
    for cname, card in CTXS.items():
        ce_bits = 0.0
        tot = 0
        tbl_bytes = 0
        for role in roles:
            tr = train.get((role, cname))
            he = held.get((role, cname))
            if tr is None or he is None:
                continue
            pt = (tr + 0.5) / (tr + 0.5).sum(1, keepdims=True)   # add-0.5 smoothed train tables
            lg = -np.log2(pt)
            ce_bits += float((he * lg).sum())
            tot += int(he.sum())
            tbl_bytes += (tr > 0).sum() * 2                       # ~2B per nonzero table entry
        ce = ce_bits / tot + (tbl_bytes * 8) / n_held
        bank = h0 - ce
        if bank > best[1]:
            best = (cname, bank)
        print(f"{cname:<14}{ce:>22.4f}{bank:>12.4f}{tbl_bytes/1024:>11.0f}", flush=True)

    D = max(0.0, best[1])
    case = "A (iid, order-0 right)" if D <= 0.01 else ("B (NOT iid, bank at every point)" if D >= 0.05
            else "C (borderline, default stay-closed)")
    print(f"\nD_idx = {D:.4f} b/w via context '{best[0]}'  ->  CASE {case}")
    print(f"champion 3.00 b/w -> {3.00 - D:.3f} b/w if realized (lossless, ppl-invariant)")


if __name__ == "__main__":
    main()
