"""R25v2-R2 (clean) -- VERIFIER POSITIVE CONTROL / SEED-SELECTION NOISE FLOOR.

The central methodological claim ("a cheap single-slice ppl verifier guided 30 rounds without being
gamed") was undefended: C1's Spearman=1.000 used only PRINCIPLED codecs, none loop-selected. This
isolates the real risk -- SELECTION NOISE. We fix lambda (iso-bit ~3.13 b/w) and vary ONLY the
incoherence rotation SEED across N candidates; the spread is pure selection noise. Then we test
whether the argmin on the WORKED slice (the 120k prose slice all 24 rounds selected on) transfers
to a disjoint PROSE ESCROW slice.

If worked-ranking transfers to escrow (high Spearman) -> seed selection is benign, margins real.
If not -> any historical margin below the seed-spread was luck; report the honest decidable margin.

Run:  python -m weights.overfit_control [N=16] [lam=0.008]
"""
from __future__ import annotations

import gc
import sys

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights import codec_zoo
from weights.quant_lab import CFG, build_model, calibrate, load_fp16, load_quant

_RAW = open("data/corpora/generic-text/enwik8_256k", "rb").read()
WORKED = _RAW[120000:128000].decode("latin-1")       # the slice all rounds selected on (prose, fp16 3.93)
ESCROW = _RAW[176000:184000].decode("latin-1")       # disjoint PROSE escrow (fp16 ~2.54), never selected on


def ppl_on(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    with torch.no_grad():
        return float(torch.exp(model(ids, labels=ids).loss))


def seeded(a, k, calib, lam, seed):
    """champion ECVQ at fixed lam, but with rotation/sample SEED varied (iso-bit; only luck differs)."""
    Ws, sc = codec_zoo._act(a, k, calib)
    wh, b = codec_zoo._had_ecvq(Ws, lam, codec_zoo.G, 0.005, seed=seed)
    return (wh / sc[None, :]).astype(np.float32), b + 16 * a.shape[1]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    n = len(a)
    return 1 - 6 * float(((ra - rb) ** 2).sum()) / (n * (n * n - 1))


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.008
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    calib = calibrate(model, tok)
    print(f"fp16: worked {ppl_on(model, tok, WORKED):.4f} | escrow {ppl_on(model, tok, ESCROW):.4f}")
    print(f"iso-bit seed sweep: lam={lam} fixed, {N} rotation seeds (~3.13 b/w each)\n", flush=True)

    rows = []
    for s in range(N):
        load_fp16(model)
        bpw = load_quant(model, lambda a, k: seeded(a, k, calib, lam, s))
        pw = ppl_on(model, tok, WORKED)
        pe = ppl_on(model, tok, ESCROW)
        rows.append((s, bpw, pw, pe))
        print(f"  seed {s:2d}  {bpw:.3f}b  worked {pw:.4f}  escrow {pe:.4f}", flush=True)
        gc.collect()

    bpw = np.array([r[1] for r in rows]); w = np.array([r[2] for r in rows]); e = np.array([r[3] for r in rows])
    rho = spearman(w, e)
    wi = int(w.argmin()); ei = int(e.argmin())
    esc_rank_of_worked_winner = int(np.argsort(e).tolist().index(wi)) + 1
    print(f"\nbits spread (should be ~0): {bpw.max()-bpw.min():.4f} b/w  [iso-bit confirmed]")
    print(f"WORKED ppl: mean {w.mean():.4f}  std {w.std():.4f}  spread {w.max()-w.min():.4f}")
    print(f"ESCROW ppl: mean {e.mean():.4f}  std {e.std():.4f}  spread {e.max()-e.min():.4f}")
    print(f"Spearman(worked-rank, escrow-rank) = {rho:.3f}")
    print(f"worked-argmin = seed {wi} (worked {w[wi]:.4f}); its escrow rank = {esc_rank_of_worked_winner}/{N} "
          f"(escrow-argmin = seed {ei})")
    print(f"\n=> SEED-SELECTION NOISE FLOOR: ~+/-{w.std():.4f} ppl on a fixed slice at iso-bit.")
    if rho >= 0.6:
        print(f"   worked-ranking TRANSFERS to escrow (rho={rho:.2f}): seed selection is largely benign; "
              f"margins above ~{2*w.std():.3f} ppl are real.")
    else:
        print(f"   worked-ranking does NOT transfer (rho={rho:.2f}): single-slice seed selection is mostly "
              f"NOISE. Any historical margin below ~{w.std():.3f}-{w.max()-w.min():.3f} ppl was luck -> "
              f"report multi-slice means; this is the verifier's true decidable resolution.")


if __name__ == "__main__":
    main()
