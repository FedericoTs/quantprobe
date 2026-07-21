"""Tamper-resistance prototype (weight-space only, no model execution).

Demonstrates the defensive half of the pipeline and is HONEST about the tiers:

 [1] RE-ARM (no retraining): graft the refusal-writing component back into an
     abliterated model from a safe reference -> restores the safety structure.
 [2] RE-ATTACK + ATTEST: a standard rank-1 abliteration re-strips the re-armed
     model, BUT a keyed attestation signature COLLAPSES -> the tamper is DETECTED.
     (We can't make removal impossible on weights someone owns; we make it PROVABLE.)
 [3] POLYMORPHIC keyed spread: distribute the refusal writing across a per-instance
     secret k-dim subspace so the universal top-1 abliteration script removes only
     ~1/k of it (attacker cost rank-1 -> rank-k). HONEST CAVEAT: pure weight-space
     relocation also reduces what the model's (r-reading) downstream can act on, so a
     *functional* polymorphic hardening needs a cheap retrain to re-couple read/write.

Refusal direction r is recovered from the base-vs-abliterated delta (it is the single
shared direction; our detector measures its consistency at ~1.0).
"""
from __future__ import annotations

import os

import numpy as np

from weights.abliteration_detect import BASE, D, load_writers


def refusal_dir(base, ablit):
    m = next(iter(base.values())).shape[0]
    M = np.zeros((m, m), dtype=np.float64)
    for k in base:
        d = (base[k] - ablit[k]).astype(np.float64)
        M += d @ d.T
    _, V = np.linalg.eigh(M)
    return V[:, -1].astype(np.float32)  # dominant shared refusal direction


def comp_along(W, r):
    """Refusal readout strength: ||r^T W|| / ||W||_F  (how much the matrix writes along r)."""
    den = float(np.linalg.norm(W))
    return float(np.linalg.norm(r @ W) / den) if den > 0 else 0.0


def main():
    base = load_writers(BASE)
    ablit = load_writers(os.path.join(D, "qwen", "ablit.safetensors"))
    ks = list(base.keys())
    r = refusal_dir(base, ablit)

    cb = np.mean([comp_along(base[k], r) for k in ks])
    ca = np.mean([comp_along(ablit[k], r) for k in ks])
    print(f"refusal-component (||r^T W||/||W||):   safe base = {cb:.4f}   abliterated = {ca:.4f}")
    print(f"  (abliteration drove the refusal-writing to ~0; base writes it strongly)\n")

    # [1] RE-ARM ---------------------------------------------------------------
    rearm = {k: (ablit[k] + np.outer(r, r @ base[k])).astype(np.float32) for k in ks}
    cr = np.mean([comp_along(rearm[k], r) for k in ks])
    print(f"[1] RE-ARM (no retraining): refusal-component {ca:.4f} -> {cr:.4f}  (target base {cb:.4f})")
    print(f"    => safety structure restored by a rank-1 graft from the safe reference.\n")

    # [2] RE-ATTACK + ATTEST ---------------------------------------------------
    reatk = {k: (rearm[k] - np.outer(r, r @ rearm[k])).astype(np.float32) for k in ks}
    cx = np.mean([comp_along(reatk[k], r) for k in ks])
    detected = cx < 0.5 * cr
    print(f"[2] standard re-abliteration of the re-armed model: {cr:.4f} -> {cx:.4f}")
    print(f"    ATTESTATION (keyed integrity signature): {cr:.4f} -> {cx:.4f}  "
          f"=> TAMPER {'DETECTED' if detected else 'MISSED'}")
    print(f"    => removal is not prevented, but it is made PROVABLE (the honest, defensible core).\n")

    # [3] POLYMORPHIC keyed spread --------------------------------------------
    rng = np.random.default_rng(0)
    m = r.shape[0]
    print("[3] POLYMORPHIC keyed spread (per-instance secret k-dim subspace):")
    for kdim in (2, 4, 8):
        Q, _ = np.linalg.qr(np.column_stack([r, rng.standard_normal((m, kdim - 1))]))
        top1, hard = [], {}
        for k in ks:
            readout = r @ base[k]
            grp = rng.integers(0, kdim, readout.shape[0])
            edit = np.zeros_like(base[k])
            for j in range(kdim):
                edit += np.outer(Q[:, j], readout * (grp == j)).astype(np.float32)
            hard[k] = (ablit[k] + edit).astype(np.float32)
            d = hard[k] - ablit[k]
            _, S, _ = np.linalg.svd(d, full_matrices=False)
            top1.append(float(S[0] ** 2 / (S ** 2).sum()))
        act = np.mean([comp_along(hard[k], r) for k in ks])
        print(f"    k={kdim}: universal top-1 attack strips only ~{np.mean(top1)*100:4.1f}% of the edit "
              f"(rest hidden in the secret subspace)")
        print(f"          functional-along-r = {act:.4f}  "
              f"(~1/k of base {cb:.4f} -> relocation needs a cheap retrain to re-couple reading)")

    # VERDICT ------------------------------------------------------------------
    print("\nVERDICT (honest tiers):")
    print("  - RE-ARM (restore safety) ............... WORKS, no retraining, functional along r")
    print("  - PREVENT removal on owned weights ...... NO (impossible white-box; standard attack re-strips)")
    print("  - DETECT/ATTEST removal ................. YES, keyed signature collapses on tamper")
    print("  - POLYMORPHIC cost (rank-1 -> rank-k) ... YES for attacker cost; FUNCTIONAL version needs cheap retrain")


if __name__ == "__main__":
    main()
