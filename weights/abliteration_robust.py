"""Phase 2 - robustness gauntlet for the weight-space abliteration detector.

The Aug-2025 SoK on LLM fingerprinting flagged the open problem: signatures that
work on pristine models collapse under "diverse model modifications". So we stress
the detector against the transformations an abliterated model actually undergoes in
the wild, and against an adaptive evader:

  (A) FINE-TUNE ON TOP   - someone keeps training the abliterated model (adds a
                            diffuse benign delta over the rank-1 safety edit).
  (B) QUANTIZATION       - the model is shipped as int8 / int4 (GGUF-style),
                            adding broadband quant noise to every weight.
  (C) ADAPTIVE ADVERSARY - the attacker spreads the safety removal over rank-k
                            instead of rank-1 to dodge a rank-1 detector.

For each we recompute the weight-only score and check it stays above the benign
ceiling measured in phase 1.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from safetensors import safe_open

from weights.abliteration_detect import (BASE, CONS_THRESH, D, ENERGY_THRESH,
                                          analyze, load_writers, verdict)


def q_dequant(x, bits):
    """Per-tensor symmetric absmax quantize->dequantize to `bits` (GGUF-ish)."""
    qmax = 2 ** (bits - 1) - 1
    amax = float(np.abs(x).max())
    if amax == 0:
        return x.copy()
    s = amax / qmax
    return np.round(x / s).clip(-qmax - 1, qmax) * s


def detect(base, cand):
    """Return (flagged_bool, cons, shared_energy) using the robust v2 detector."""
    a = analyze(base, cand)
    _, flag = verdict(a)
    return flag.startswith("*"), a["cons"], a["shared_energy"]


def row(label, base, cand):
    flagged, cons, se = detect(base, cand)
    res = "DETECTED" if flagged else "evaded"
    print(f"{label:<32}{cons:>8.3f}{se:>10.3f}  {res}", flush=True)


def main():
    base = load_writers(BASE)
    ablit = load_writers(os.path.join(D, "qwen", "ablit.safetensors"))

    print(f"detector thresholds: cons>{CONS_THRESH}, shared_energy>{ENERGY_THRESH}\n")
    f0, c0, s0 = detect(base, ablit)
    print(f"baseline abliterated: cons={c0:.3f} shared_E={s0:.3f} -> "
          f"{'DETECTED' if f0 else 'evaded'}\n")
    HEAD = f"{'transform':<32}{'cons':>8}{'shared_E':>10}  result"

    # task vectors from real benign fine-tunes, restricted to writer matrices
    def taskvec(name):
        fp = os.path.join(D, "qwen_family", f"{name}.safetensors")
        ft = load_writers(fp)
        return {k: ft[k] - base[k] for k in base if k in ft}

    print("=== (A) FINE-TUNE ON TOP of the abliterated model ===")
    print(HEAD); print("-" * 60)
    for tvname in ("mathphd", "reasoning"):
        tv = taskvec(tvname)
        for a in (0.5, 1.0, 2.0):
            cand = {k: ablit[k] + a * tv[k] for k in ablit if k in tv}
            row(f"ablit + {a}x({tvname})", base, cand)

    print("\n=== (B) QUANTIZATION of the abliterated model ===")
    print(HEAD); print("-" * 60)
    for bits, tag in ((8, "int8"), (4, "int4")):
        cand = {k: q_dequant(ablit[k], bits) for k in ablit}
        row(f"quantize ablit -> {tag}", base, cand)
    # harder: candidate quantized AND base quantized (scanner only has quant base)
    for bits, tag in ((8, "int8"), (4, "int4")):
        bq = {k: q_dequant(base[k], bits) for k in base}
        cq = {k: q_dequant(ablit[k], bits) for k in ablit}
        row(f"both base+ablit -> {tag}", bq, cq)

    print("\n=== (C) ADAPTIVE ADVERSARY: spread safety-removal over rank-k ===")
    print("  (re-spread the edit's energy across k random orthogonal directions")
    print("   per matrix, to defeat a single-shared-direction detector)")
    print(HEAD); print("-" * 60)
    rng = np.random.default_rng(0)
    deltas = {k: ablit[k] - base[k] for k in ablit}
    for k_rank in (1, 2, 4, 8):
        cand = {}
        for k, d in deltas.items():
            if np.abs(d).max() == 0:
                cand[k] = ablit[k]
                continue
            U, S, Vt = np.linalg.svd(d, full_matrices=False)
            m = U.shape[0]
            Q, _ = np.linalg.qr(rng.standard_normal((m, k_rank)))
            s_each = S[0] / np.sqrt(k_rank)
            spread = sum(s_each * np.outer(Q[:, j], Vt[0]) for j in range(k_rank))
            d2 = spread + (d - S[0] * np.outer(U[:, 0], Vt[0]))  # replace rank-1 part
            cand[k] = base[k] + d2.astype(np.float32)
        row(f"rank-{k_rank} random spread", base, cand)


if __name__ == "__main__":
    main()
