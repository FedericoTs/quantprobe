"""rank_bands.py -- pre-registration #102 measurement harness (U-46, task #84).

THE STAKE.  preregistrations/2026-08-16-rank-predicts-fragility.md froze, before a single
singular value had been computed, the one signal this file measures:

    erank(W)    = (sum_i s_i^2)^2 / (sum_i s_i^4)     participation ratio of squared SVs
    layer score = mean over the layer's FFN weight matrices of  erank / min(rows, cols)
    band score  = mean layer score over the probe's own four bands, identical boundaries

Staked direction: LOWER score -> MORE fragile.  This file takes no position on that: it only
measures.

KR-3 ORDERING.  Per the prereg's kill rules, this harness and the scorer are committed BEFORE
the first Gram/singular value is computed on any real GGUF; the P-A/P-B/P-C verdict is printed
mechanically by prereg102_score.py, not by a human reading these numbers.  KR-2 separation:
this file reads only the (lo, hi) band BOUNDARIES from the committed recipes -- never a
delta_ppl -- so nothing here can condition on the ground truth it will be scored against.

NO SVD IS RUN, EVER.  Both sums in the participation ratio are Frobenius norms in disguise:

    sum_i s_i^2 = ||W||_F^2          and          sum_i s_i^4 = ||G||_F^2

where G is the Gram of W over its smaller side (W W^T if rows <= cols, else W^T W).  These are
exact identities, not approximations, so the signal costs one GEMM per matrix instead of a
LAPACK SVD that takes minutes on a 14336x4096 dense FFN -- and there are hundreds of such
matrices per run on this box (GTX 1060 / i5-7600K, the same hardware as every probe log).

TENSOR POPULATION.  Exactly what the probe's band regex pushed to q2_k
(quantprobe/probe.py:78, "blk\\.(N|..)\\.ffn_.*"), minus two matches llama-quantize never
actually quantizes and which therefore carry no fragility signal:
  - *norm* weights: 1D vectors, kept float by llama.cpp (and the prereg freezes FFN MATRICES);
  - ffn_gate_inp (MoE router): llama.cpp hard-skips expert gating tensors at quantize time
    (quantprobe/spec.py:283, "router: tiny, kept at full precision").
Routed-expert 3D tensors are included per the prereg's MoE clause: PR per expert slice, mean
over experts, and the tensor enters the layer mean as ONE matrix like any other.

SOURCE FILES.  The highest-precision file held per model (BF16 where held, else the best
quant), recorded verbatim in the output JSON -- the prereg accepts quantized sources: the
signal must survive quantization to be useful, because the tool's users hold quants.

Output:  weights/data/rank_bands.json =
    {model: {file, bands: [{lo, hi, score}], layer_scores: {layer: score}, n_matrices}}
n_matrices counts scored FFN weight tensors (a routed-expert 3D tensor counts once).
The JSON is rewritten atomically after EACH model: the 27B/35B make this a multi-hour run and
a crash must not eat finished models (checkpoint discipline).  A finished model is skipped on
re-run -- KR-1 says the signal is computed exactly once per model; --force exists for a
conscious redo after a verified failure, not for retries.

Run (from anywhere; resumable):
  python weights/rank_bands.py
  python weights/rank_bands.py --models qwen3.5-4b,qwen3-30b --out weights/data/rank_bands.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# "python weights/rank_bands.py" puts weights/ (the script dir), not the repo root, first on
# sys.path; quantprobe lives at the root, so the documented run line needs this shim.
sys.path.insert(0, ROOT)

from gguf import GGUFReader, quants  # noqa: E402
from quantprobe.spec import split_siblings  # noqa: E402

PREREG = "preregistrations/2026-08-16-rank-predicts-fragility.md"

# The highest-precision file held per model (prereg #102 source-file clause).  Mistral is not
# on disk at staking time (download in flight); KR-4 says scoring WAITS for it -- a missing
# file refuses loudly rather than shrinking the model set to the four back-fragile rows.
# Source files are the HIGHEST-PRECISION file held per model (the prereg's frozen clause),
# fixed by the verify pass BEFORE any measurement ran:
#   - qwen2.5-7b uses the fp16 SPLIT (on disk since 08-04), not the Q4_K_M first written here;
#     split_siblings picks up parts 2-4.
#   - qwen3-30b was WRONGLY pointed at the Coder finetune - a different model from the one the
#     recipe's ground truth was probed on (the repo's own ladders list them as separate rows).
#     Repointed at the plain Qwen3-30B-A3B. Its highest-precision file held is Q2_K; a 2-bit
#     source is the roughest signal in the set and is disclosed in the prereg amendment.
MODELS = {
    "mistral-7b":  "D:/evo-compress-data/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    "qwen2.5-7b":  "D:/evo-compress-data/gguf/qwen2.5-7b-instruct-fp16-00001-of-00004.gguf",
    "qwen3.5-4b":  "D:/evo-compress-data/gguf/Qwen3.5-4B-BF16.gguf",
    "qwen3.5-35b": "D:/evo-compress-data/gguf/Qwen3.5-35B-A3B-Q8_0.gguf",
    "qwen3-30b":   "D:/evo-compress-data/gguf/Qwen3-30B-A3B-Q2_K.gguf",
    "qwen3.8-27b": "D:/evo-compress-data/gguf/qwen38/Qwen3.8-27B-Q4_K_M.gguf",
}

# Models with a committed recipe: band boundaries come from the recipe JSON (KR-2: ground
# truth is read-only; the probe is never re-run to make a row agree).
_RECIPE_MODELS = ("mistral-7b", "qwen2.5-7b", "qwen3-30b", "qwen3.5-35b")

# No recipe JSON is committed for these two; boundaries are transcribed verbatim from their
# probe logs' own headers:
#   weights/data/prereg100_probe_4b.log:       "32 layers -> 4 bands [(0,7),(8,15),(16,23),(24,31)]"
#   weights/data/prereg101_probe_qwen38_q4.log: "65 layers -> 4 bands [(0,16),(17,33),(34,50),(51,64)]"
_PROBE_LOG_BANDS = {
    "qwen3.5-4b":  [(0, 7), (8, 15), (16, 23), (24, 31)],
    "qwen3.8-27b": [(0, 16), (17, 33), (34, 50), (51, 64)],
}


def _recipe_bands(key):
    """(lo, hi) pairs from the committed recipe.  Only the boundaries are taken: delta_ppl
    stays unread here -- joining scores with deltas is the scorer's job (KR-3), and keeping
    this file blind to the deltas is what makes the measurement unable to peek."""
    path = os.path.join(ROOT, "quantprobe", "recipes", key + ".json")
    with open(path, encoding="utf-8") as f:
        recipe = json.load(f)
    bands = [(int(b["lo"]), int(b["hi"])) for b in recipe["probe"]["band_deltas"]]
    if len(bands) != 4:
        raise SystemExit(f"{path}: {len(bands)} bands, expected 4 - recipe does not match the "
                         f"frozen 4-band formula in {PREREG}")
    return bands


BANDS = {k: _recipe_bands(k) for k in _RECIPE_MODELS}
BANDS.update(_PROBE_LOG_BANDS)


def selected(name):
    """The probe's tensor population (see module docstring): FFN weight matrices only."""
    return (name.startswith("blk.") and ".ffn_" in name and name.endswith(".weight")
            and "norm" not in name and "ffn_gate_inp" not in name)


def pr_score(w):
    """erank(W) / min(rows, cols) of one 2D float32 matrix -- the frozen per-matrix signal.

    Max-abs normalization first: PR is scale-invariant, so this changes nothing in exact
    arithmetic; it is overflow protection only, as staked.  The two sums are computed as
    Frobenius norms (module docstring) with float64 ACCUMULATION via einsum's dtype; the GEMM
    itself stays float32 because a float64 Gram would double RAM on 200 MB+ dense FFNs for
    digits the band contrast (tens of percent) does not need."""
    m = float(np.abs(w).max())
    if m == 0.0:
        # A zero FFN matrix means a broken file, not a low-rank layer.  Refuse, never score.
        raise SystemExit("all-zero FFN matrix encountered - source file is broken")
    w = (w / m).astype(np.float32, copy=False)
    g = w @ w.T if w.shape[0] <= w.shape[1] else w.T @ w
    s2 = float(np.einsum("ij,ij->", w, w, dtype=np.float64))
    s4 = float(np.einsum("ij,ij->", g, g, dtype=np.float64))
    return (s2 * s2 / s4) / min(w.shape)


def tensor_score(t):
    """One score per GGUF tensor.  gguf-py returns data in numpy order (GGUF ne reversed),
    so 3D routed-expert tensors arrive as (n_expert, rows, cols): expert axis first."""
    if t.data.ndim == 2:
        return pr_score(quants.dequantize(t.data, t.tensor_type))
    if t.data.ndim == 3:
        # PR per expert slice, mean over experts (prereg #102 MoE clause).  Dequantized one
        # slice at a time: a 128-expert tensor is ~800 MB as float32 and this box has 16 GB.
        # Quant block boundaries never span slices, so per-slice dequantize is bit-identical
        # to slicing a whole-tensor dequantize.
        n_exp = t.data.shape[0]
        return sum(pr_score(quants.dequantize(t.data[e], t.tensor_type))
                   for e in range(n_exp)) / n_exp
    raise SystemExit(f"{t.name}: ndim={t.data.ndim} is not an FFN weight matrix - the "
                     f"selection predicate should not have matched it; fix selected(), do not "
                     f"special-case here")


def measure_model(key, path):
    """The complete frozen signal for one model: per-layer scores, then band means."""
    parts = split_siblings(path)
    if not os.path.isfile(parts[0]):
        extra = ("  KR-4: Mistral is not optional - the experiment waits for the download; "
                 "measure the other models meanwhile with --models." if key == "mistral-7b"
                 else "")
        raise SystemExit(f"{key}: missing {parts[0]}.{extra}")

    # Readers stay alive for the whole pass: tensor .data memmaps into them.  Tensors are
    # grouped by layer first (metadata only, cheap) so progress prints one ORDERED line per
    # layer even when a split file scatters a layer across parts.
    readers = [GGUFReader(p, "r") for p in parts]
    layers = {}
    for r in readers:
        for t in r.tensors:
            if selected(t.name):
                layers.setdefault(int(t.name.split(".")[1]), []).append(t)
    if not layers:
        raise SystemExit(f"{key}: no FFN weight tensors matched in {path} - wrong file?")

    n_sel = sum(len(v) for v in layers.values())
    print(f"== {key}: {path} ({len(parts)} file(s), {n_sel} FFN tensors over "
          f"{len(layers)} layers) ==", flush=True)

    t0 = time.time()
    layer_scores = {}
    for lay in sorted(layers):
        scores = [tensor_score(t) for t in layers[lay]]
        layer_scores[lay] = sum(scores) / len(scores)
        print(f"  blk.{lay:<3d} {len(scores)} matrices  score {layer_scores[lay]:.6f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    bands, edges = [], []
    for lo, hi in BANDS[key]:
        in_band = [layer_scores[l] for l in layer_scores if lo <= l <= hi]
        if not in_band:
            # An empty band means the file's layer count disagrees with the probe log the
            # boundaries came from - a wrong-file error, not a zero.
            raise SystemExit(f"{key}: band {lo}-{hi} matched no layers (file has layers "
                             f"{min(layer_scores)}-{max(layer_scores)})")
        score = sum(in_band) / len(in_band)
        if not math.isfinite(score):
            # A NaN/Inf here means a corrupt source tensor made it through dequantize.
            # Refuse rather than write poison the scorer would only reject downstream.
            raise SystemExit(f"{key}: band {lo}-{hi} score is not finite - corrupt source?")
        bands.append(score)
        edges.append([lo, hi])
        print(f"  band {lo:>2d}-{hi:<2d}  score {score:.6f}  ({len(in_band)} layers)",
              flush=True)

    # The row shape is the SCORER'S contract (prereg102_score._load_bands): "bands" is a flat
    # list of 4 floats and the file travels under "source". The verify pass caught the first
    # draft emitting {lo,hi,score} dicts under "bands" and the path under "file" - a shape the
    # scorer rejects, which would have forced a post-measurement code edit and voided KR-3.
    return {"source": path,
            "bands": bands,
            "band_edges": edges,
            "layer_scores": {str(l): layer_scores[l] for l in sorted(layer_scores)},
            "n_matrices": n_sel}


def main():
    ap = argparse.ArgumentParser(description=f"prereg #102 rank-band harness ({PREREG})")
    ap.add_argument("--models", nargs="+", metavar="KEY",
                    help="subset of model keys (space- or comma-separated); default: all")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "rank_bands.json"),
                    help="output JSON (default: weights/data/rank_bands.json)")
    ap.add_argument("--force", action="store_true",
                    help="recompute models already present in --out (KR-1: justify in the log)")
    a = ap.parse_args()

    want = ([m for tok in a.models for m in tok.split(",") if m] if a.models
            else list(MODELS))
    unknown = [m for m in want if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model key(s) {unknown}; known: {', '.join(MODELS)}")

    results = {}
    if os.path.isfile(a.out):
        try:
            with open(a.out, encoding="utf-8") as f:
                results = json.load(f)
        except ValueError as e:
            # Never clobber a file we cannot parse - it may hold hours of finished models.
            raise SystemExit(f"{a.out} exists but is not valid JSON ({e}); inspect or move it")

    print(f"prereg #102 measurement harness -- {PREREG}", flush=True)
    print(f"numpy {np.__version__}, gguf-py from {os.path.dirname(quants.__file__)}",
          flush=True)

    for key in want:
        if key in results and not a.force:
            print(f"== {key}: already measured (KR-1: computed exactly once) - skipping; "
                  f"--force to redo ==", flush=True)
            continue
        results[key] = measure_model(key, MODELS[key])
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, a.out)
        print(f"  -> {a.out} updated ({len(results)} model(s))", flush=True)

    print("measurement complete; the verdict belongs to prereg102_score.py (KR-3)",
          flush=True)


if __name__ == "__main__":
    main()
