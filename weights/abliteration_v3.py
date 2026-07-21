"""v3 - harden the weight-space abliteration detector against the frontiers the
phase-2 gauntlet exposed:

  (1) SUBSPACE-CONSISTENCY  - generalize single-direction detection to a shared
      r-dim residual subspace, defeating an adaptive adversary who spreads the
      safety removal across a consistent subspace across layers.
  (2) BLOCK-WISE QUANT      - test realistic GGUF-style block int4 (not the harsh
      per-tensor int4 that evaded v2).
  (3) BASE IDENTIFICATION   - provenance: pick the true parent from a pool of
      candidate bases (a real scanner gets an unknown model with no metadata).
"""
from __future__ import annotations

import glob
import os

import numpy as np

from weights.abliteration_detect import BASE, D, load_writers


def subspace_sig(base, cand, r=1):
    """Detector over a shared r-dim residual subspace.
    Returns (cons_r, energy_r): subspace colinearity in [~r/m,1], median captured energy."""
    Bs, deltas = [], []
    for k in base:
        if k not in cand:
            continue
        d = cand[k] - base[k]
        if np.abs(d).max() == 0:
            continue
        U, _, _ = np.linalg.svd(d, full_matrices=False)
        Bs.append(U[:, :r])          # top-r left subspace (m x r)
        deltas.append(d)
    n = len(Bs)
    if n < 2:
        return 0.0, 0.0
    m = Bs[0].shape[0]
    M = np.zeros((m, m), dtype=np.float64)
    for B in Bs:
        M += B @ B.T
    w, V = np.linalg.eigh(M)
    cons_r = float(w[-r:].sum()) / (r * n)        # 1.0 = all share one r-dim subspace
    S = V[:, -r:]                                  # shared subspace (m x r)
    ps = []
    for d in deltas:
        proj = S.T @ d
        num = float((proj * proj).sum())
        den = float((d * d).sum())
        ps.append(num / den if den > 0 else 0.0)
    return cons_r, float(np.median(ps))


def consistent_subspace_attack(base, ablit, r_attack):
    """Adaptive adversary: spread the safety edit across a FIXED r_attack-dim
    subspace SHARED by all layers (a realistic multi-direction refusal removal),
    with per-layer-varying directions inside that subspace."""
    rng = np.random.default_rng(0)
    deltas = {k: ablit[k] - base[k] for k in ablit}
    nz = [d for d in deltas.values() if np.abs(d).max() > 0]
    m = nz[0].shape[0]
    U0, _, _ = np.linalg.svd(nz[0], full_matrices=False)
    Qsub, _ = np.linalg.qr(np.column_stack([U0[:, 0], rng.standard_normal((m, r_attack - 1))]))
    cand = {}
    for k, d in deltas.items():
        if np.abs(d).max() == 0:
            cand[k] = ablit[k]
            continue
        _, S, _ = np.linalg.svd(d, full_matrices=False)
        ncol = d.shape[1]
        edit = Qsub @ rng.standard_normal((r_attack, ncol))
        edit *= S[0] / (np.linalg.norm(edit) + 1e-12)   # match original top-energy scale
        cand[k] = (base[k] + edit).astype(np.float32)
    return cand


def q_blockwise(x, bits=4, block=32):
    """GGUF-style block-wise symmetric absmax quantize->dequantize."""
    flat = x.reshape(-1).astype(np.float32)
    pad = (-len(flat)) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    fb = flat.reshape(-1, block)
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(fb).max(axis=1, keepdims=True)
    amax[amax == 0] = 1.0
    s = amax / qmax
    q = (np.round(fb / s).clip(-qmax - 1, qmax) * s).reshape(-1)[:x.size]
    return q.reshape(x.shape).astype(np.float32)


def identify_base(cand, pool_paths):
    """Provenance: rank candidate parents by mean writer-delta norm (smallest = parent)."""
    out = []
    for name, p in pool_paths:
        w = load_writers(p)
        diffs = [float(np.linalg.norm(cand[k] - w[k])) for k in cand if k in w]
        out.append((name, float(np.mean(diffs)) if diffs else 9e9))
        del w
    return sorted(out, key=lambda x: x[1])


def main():
    base = load_writers(BASE)
    ablit = load_writers(os.path.join(D, "qwen", "ablit.safetensors"))

    print("=== (1) SUBSPACE-CONSISTENCY vs adaptive distributed removal ===")
    print("  attacker spreads the edit across a fixed r_attack-dim shared subspace;")
    print("  detector tests shared subspaces of dim r. Need detector r >= attack r.\n")
    print(f"{'candidate':<22}" + "".join(f"  det r={r}" for r in (1, 2, 4, 8)))
    print("-" * 60)

    def fmt_row(label, cand):
        cells = []
        for r in (1, 2, 4, 8):
            c, e = subspace_sig(base, cand, r=r)
            hit = (c > 0.85 and e > 0.03)
            cells.append(f"{('Y' if hit else '.')}{c:.2f}")
        print(f"{label:<22}" + "".join(f"  {x:>7}" for x in cells), flush=True)

    fmt_row("true ablit (rank-1)", ablit)
    for ra in (2, 4, 8):
        fmt_row(f"attack r={ra} spread", consistent_subspace_attack(base, ablit, ra))
    print("  (cell = colinearity; 'Y' = flagged at that detector rank)")

    print("\n=== (2) BLOCK-WISE int4 quantization (realistic GGUF) ===")
    print(f"{'transform':<30}{'cons':>8}{'shared_E':>10}  result")
    print("-" * 56)
    for block in (32, 64):
        cq = {k: q_blockwise(ablit[k], 4, block) for k in ablit}
        bq = {k: q_blockwise(base[k], 4, block) for k in base}
        c, e = subspace_sig(bq, cq, r=1)
        res = "DETECTED" if (c > 0.85 and e > 0.03) else "evaded"
        print(f"block-{block} int4 (both)        {c:>8.3f}{e:>10.3f}  {res}", flush=True)

    print("\n=== (3) BASE IDENTIFICATION (provenance) ===")
    print("  given the abliterated model, find its true parent among a pool:")
    pool = [("base(TRUE)", os.path.join(D, "qwen", "base.safetensors"))]
    for nm in ("mathphd", "reasoning", "vikhr", "dpo-halueval", "neon-sft"):
        pool.append((nm, os.path.join(D, "qwen_family", f"{nm}.safetensors")))
    ranking = identify_base(ablit, pool)
    for i, (nm, sc) in enumerate(ranking):
        mark = "  <-- identified parent" if i == 0 else ""
        print(f"  {nm:<14} mean delta-norm {sc:>8.3f}{mark}", flush=True)
    ok = ranking[0][0].startswith("base")
    print(f"  => {'CORRECT: true base identified' if ok else 'WRONG parent'}")


if __name__ == "__main__":
    main()
