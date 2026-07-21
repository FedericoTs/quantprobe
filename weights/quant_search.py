"""Autonomous evolutionary codec search (the engine, not hand-picking).

A codec is a GENOME of composable operators:
    rotate   : incoherence Hadamard rotation (on/off)
    act      : activation-aware scaling (AWQ-style, calibration) (on/off)
    codebook : uniform | nf | vq2 | vq4   (scalar uniform, normal-float, vector-quant d=2/4)
    outlier  : 0 | 0.005 | 0.01           (fraction kept in fp16)
The loop mutates/selects genomes scored by the held-out verifier (perplexity + bits),
seeded with the two known-good hand codecs, and tries to DISCOVER a better one.
Fixed codebooks (no per-layer k-means) keep each evaluation fast + deployable.
"""
from __future__ import annotations

import gc

import numpy as np

from weights.quant_evolve import _assign
from weights.quant_lab import (_awq_scale, build_model, calibrate, load_fp16,
                               load_quant, ppl)
from weights.quant_sota import _fwht_rows, _group_codebook, _levels, _nearest
from transformers import AutoTokenizer
from weights.quant_lab import CFG

BITS, GROUP, ALPHA = 3, 128, 0.5
SPACE = {"rotate": [False, True], "act": [False, True],
         "codebook": ["uniform", "nf", "vq2"], "outlier": [0.0, 0.005, 0.01]}
_CB = {}


def _codebook(d, K, seed=0):
    if (d, K) not in _CB:
        rng = np.random.default_rng(seed)
        C = rng.standard_normal((K, d)).astype(np.float32)
        _CB[(d, K)] = C / np.abs(C).max()
    return _CB[(d, K)]


def _scalar_core(W, bits, g, rotate, kind):
    levels = _levels(bits, "uniform" if kind == "uniform" else "normal")
    if not rotate:
        return _group_codebook(W, bits, g, levels)
    rows, cols = W.shape
    pad = (-cols) % g
    A = np.pad(W, ((0, 0), (0, pad))) if pad else W
    N = A.reshape(rows, -1, g).reshape(-1, g)
    signs = np.random.default_rng(0).integers(0, 2, g).astype(np.float32) * 2 - 1
    R = _fwht_rows(N * signs) / np.sqrt(g)
    amax = np.abs(R).max(1, keepdims=True); amax[amax == 0] = 1.0
    Rh = _nearest((R / amax).ravel(), levels).reshape(R.shape) * amax
    back = (_fwht_rows(Rh) / np.sqrt(g)) * signs
    return back.reshape(rows, -1)[:, :cols].astype(np.float32), bits * W.size + 16 * N.shape[0]


def _vq_core(W, bits, d, g, rotate):
    rows, cols = W.shape
    pad = (-cols) % g
    A = np.pad(W, ((0, 0), (0, pad))) if pad else W
    N = A.reshape(rows, -1, g).reshape(-1, g)
    if rotate:
        signs = np.random.default_rng(0).integers(0, 2, g).astype(np.float32) * 2 - 1
        N = _fwht_rows(N * signs) / np.sqrt(g)
    amax = np.abs(N).max(1, keepdims=True); amax[amax == 0] = 1.0
    Nn = N / amax
    flat = Nn.ravel().astype(np.float32)
    padv = (-len(flat)) % d
    if padv:
        flat = np.concatenate([flat, np.zeros(padv, np.float32)])
    V = flat.reshape(-1, d)
    K = int(2 ** round(bits * d))
    Vh = _codebook(d, K)[_assign(V, _codebook(d, K))]
    Nnh = Vh.ravel()[:Nn.size].reshape(Nn.shape) * amax
    if rotate:
        Nnh = (_fwht_rows(Nnh) / np.sqrt(g)) * signs
    return Nnh.reshape(rows, -1)[:, :cols].astype(np.float32), len(V) * np.log2(K) + 16 * N.shape[0] + K * d * 16


def make_codec(gm, calib):
    def codec(a, key):
        extra = 0.0
        W = a
        if gm["act"]:
            s = _awq_scale(calib, key, ALPHA)
            W = W * s[None, :]
            extra += 16 * W.shape[1]
        mask = None
        if gm["outlier"] > 0:
            thr = np.quantile(np.abs(W), 1.0 - gm["outlier"])
            mask = np.abs(W) >= thr
            outv = W[mask].copy()
            W = W.copy(); W[mask] = 0.0
            extra += int(mask.sum()) * 32
        cb = gm["codebook"]
        if cb.startswith("vq"):
            Wh, cbits = _vq_core(W, BITS, int(cb[2]), GROUP, gm["rotate"])
        else:
            Wh, cbits = _scalar_core(W, BITS, GROUP, gm["rotate"], cb)
        if mask is not None:
            Wh[mask] = outv
        if gm["act"]:
            Wh = Wh / s[None, :]
        return Wh.astype(np.float32), cbits + extra
    return codec


def key(gm):
    return (gm["rotate"], gm["act"], gm["codebook"], gm["outlier"])


def mutate(gm, rng):
    g2 = dict(gm)
    f = rng.choice(list(SPACE))
    g2[f] = SPACE[f][rng.integers(len(SPACE[f]))]
    return g2


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    print("calibrating ...", flush=True)
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    print(f"fp16 held-out ppl = {fp16:.3f}  (target {BITS}-bit)\n", flush=True)

    rng = np.random.default_rng(0)
    pop = [
        {"rotate": True, "act": False, "codebook": "nf", "outlier": 0.005},   # Had+NF+outlier (6.604)
        {"rotate": True, "act": True, "codebook": "nf", "outlier": 0.005},    # hand "evolved" (4.934)
        {"rotate": True, "act": True, "codebook": "vq2", "outlier": 0.005},
        {"rotate": True, "act": True, "codebook": "nf", "outlier": 0.0},
        {"rotate": False, "act": True, "codebook": "nf", "outlier": 0.01},
        {"rotate": True, "act": True, "codebook": "vq2", "outlier": 0.01},
    ]
    cache, GENS = {}, 3
    best = None
    for gen in range(GENS):
        for gm in pop:
            k = key(gm)
            if k in cache:
                continue
            tag = f"rot={int(gm['rotate'])} act={int(gm['act'])} cb={gm['codebook']:<7} out={gm['outlier']}"
            try:
                bpw = load_quant(model, make_codec(gm, calib))
                p = ppl(model, tok)
            except Exception as e:
                print(f"  gen{gen} {tag}  FAILED: {type(e).__name__}", flush=True)
                cache[k] = (9.9, 9e9, 9e9, gm)
                gc.collect()
                continue
            gc.collect()
            score = p + 5.0 * max(0.0, bpw - 3.4)
            cache[k] = (bpw, p, score, gm)
            print(f"  gen{gen} {tag}  {bpw:5.3f}b  ppl {p:8.3f}", flush=True)
        ranked = sorted(cache.values(), key=lambda x: x[2])
        best = ranked[0]
        print(f"  -- gen{gen} best: ppl {best[1]:.3f} @ {best[0]:.3f}b  {key(best[3])}\n", flush=True)
        survivors = [r[3] for r in ranked[:3]]
        pop = survivors + [mutate(rng.choice(survivors), rng) for _ in range(3)]

    print(f"BEST DISCOVERED CODEC: ppl {best[1]:.3f} @ {best[0]:.3f} bits  genome={key(best[3])}")
    print(f"  fp16 {fp16:.3f}  |  hand Had+NF+outlier 6.604  |  hand evolved 4.934")
    if best[1] < 4.934:
        print("  => SEARCH BEAT the best hand-designed codec.")


if __name__ == "__main__":
    main()
