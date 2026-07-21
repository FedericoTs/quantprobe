"""Smoke test for AlphaEvolve opportunity #1: can search over QUANTIZATION codes
beat the naive human baselines on the bits-vs-quality frontier?

This validates the premise (cheap verifier + real headroom + evolvable artifact)
on LLM-inference compression:
  - VERIFIER (cheap, CPU): quantize all linear weights of Qwen2.5-0.5B at a target
    bit budget, load into the model, measure held-out perplexity + average bits/weight.
  - BASELINES (human): RTN per-tensor int8/int4, RTN per-group-128 int4.
  - EVOLVE CANDIDATE: an outlier-aware per-group scheme (the kind of thing the
    LLM-mutation loop would propose) — if it Pareto-beats RTN at equal bits, there
    is headroom for evolutionary search.

If the candidate beats RTN-group at <= bits and lower ppl, the premise holds.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(_ROOT, "weights", "data", "qwen_cfg")
WPATH = os.path.join(_ROOT, "weights", "data", "qwen", "base.safetensors")
torch.set_num_threads(os.cpu_count() or 4)

LINEAR = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
          "gate_proj.weight", "up_proj.weight", "down_proj.weight")

TEXT = open(os.path.join(_ROOT, "data/corpora/generic-text/enwik8_256k"),
            "rb").read()[:8000].decode("latin-1")


# ---- quantizers: return (W_hat float32, total_bits for this tensor) ----
def q_none(a):
    return a.astype(np.float32), 16.0 * a.size


def rtn(a, bits):  # per-tensor symmetric absmax
    qmax = 2 ** (bits - 1) - 1
    amax = float(np.abs(a).max()) or 1.0
    s = amax / qmax
    ah = np.clip(np.round(a / s), -qmax - 1, qmax) * s
    return ah.astype(np.float32), bits * a.size + 16.0


def rtn_group(a, bits=4, g=128):  # per-(row,group) symmetric absmax along columns
    rows, cols = a.shape
    qmax = 2 ** (bits - 1) - 1
    ah = np.empty_like(a, dtype=np.float32)
    nscale = 0
    for c0 in range(0, cols, g):
        blk = a[:, c0:c0 + g]
        amax = np.abs(blk).max(axis=1, keepdims=True)
        amax[amax == 0] = 1.0
        s = amax / qmax
        ah[:, c0:c0 + g] = np.clip(np.round(blk / s), -qmax - 1, qmax) * s
        nscale += rows
    return ah, bits * a.size + 16.0 * nscale


def outlier_group(a, bits=4, g=128, p=0.005):  # EVOLVE CANDIDATE
    """Keep the top-p magnitude weights exact (fp16); quantize the rest per-group int4.
    Outliers are what wreck RTN; preserving a tiny fraction should beat it at ~equal bits."""
    thr = np.quantile(np.abs(a), 1.0 - p)
    mask = np.abs(a) >= thr
    base = a.copy()
    base[mask] = 0.0
    ah, bits_base = rtn_group(base, bits, g)
    ah[mask] = a[mask]
    nout = int(mask.sum())
    total = bits_base + nout * (16 + 16) - nout * bits  # value+index for outliers
    return ah.astype(np.float32), total


def evaluate(name, quantizer):
    cfg = AutoConfig.from_pretrained(CFG)
    tok = AutoTokenizer.from_pretrained(CFG)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    bits_tot, elem_tot = 0.0, 0
    sd = {}
    with safe_open(WPATH, framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k.endswith(LINEAR):
                a = t.float().numpy()
                ah, bits = quantizer(a)
                sd[k] = torch.from_numpy(ah).to(t.dtype)
                bits_tot += bits
                elem_tot += a.size
            else:
                sd[k] = t
    model.load_state_dict(sd, strict=False)
    model.tie_weights()
    ids = tok(TEXT, return_tensors="pt").input_ids[:, :1024]
    with torch.no_grad():
        ppl = float(torch.exp(model(ids, labels=ids).loss))
    bpw = bits_tot / elem_tot
    del model
    return bpw, ppl


def main():
    schemes = [
        ("fp16 (reference)", q_none),
        ("RTN int8 per-tensor", lambda a: rtn(a, 8)),
        ("RTN int4 per-tensor", lambda a: rtn(a, 4)),
        ("RTN int4 per-group128", lambda a: rtn_group(a, 4, 128)),
        ("EVOLVE: outlier+group int4", lambda a: outlier_group(a, 4, 128, 0.005)),
    ]
    print(f"{'scheme':<32}{'bits/wt':>9}{'ppl':>9}")
    print("-" * 50)
    rows = []
    for name, q in schemes:
        bpw, ppl = evaluate(name, q)
        rows.append((name, bpw, ppl))
        print(f"{name:<32}{bpw:>9.3f}{ppl:>9.3f}", flush=True)

    ref = rows[0][2]
    grp = [r for r in rows if "per-group128" in r[0]][0]
    cand = [r for r in rows if r[0].startswith("EVOLVE")][0]
    print(f"\nfp16 reference ppl = {ref:.3f}")
    print(f"RTN int4-group : {grp[1]:.3f} bits, ppl {grp[2]:.3f}  (+{grp[2]-ref:.3f} vs fp16)")
    print(f"EVOLVE candidate: {cand[1]:.3f} bits, ppl {cand[2]:.3f}  (+{cand[2]-ref:.3f} vs fp16)")
    if cand[2] < grp[2] and cand[1] <= grp[1] * 1.05:
        print("=> HEADROOM CONFIRMED: a smarter code beats RTN at ~equal bits -> evolution premise holds.")
    else:
        print("=> candidate did not clearly beat RTN; needs a better evolved scheme / verifier.")


if __name__ == "__main__":
    main()
