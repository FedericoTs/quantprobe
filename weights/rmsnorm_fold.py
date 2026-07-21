"""R25v2-R3 (E1) -- RMSNorm-GAIN ABSORPTION canonicalization (exact, function-preserving symmetry).

Qwen2 blocks: q/k/v_proj consume input_layernorm(x) = (x/rms)*gamma_in; gate/up_proj consume
post_attention_layernorm(.)*gamma_post. Folding gamma into those linears' INPUT COLUMNS
(W' = W @ diag(gamma); norm weight := 1) is EXACTLY function-preserving, yet changes the
quantizer's view: smoother per-group amax field + AWQ no longer re-encodes gamma.

Test: (1) fold-only ppl == fp16 ppl (exactness gate); (2) champion ECVQ.008 on folded vs original:
ppl + arena bits (gamma side stream 2*24*896 params fp16 counted). Win if ppl drops at ~equal bits
or bits drop at ~equal ppl.

Run:  python -m weights.rmsnorm_fold
"""
from __future__ import annotations

import gc
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights import codec_zoo
from weights.quant_lab import CFG, build_model, calibrate, load_fp16, load_quant, ppl

FOLD = {  # norm gamma -> linears whose input columns absorb it
    "input_layernorm.weight": ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                               "self_attn.v_proj.weight"],
    "post_attention_layernorm.weight": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
}


def fold_gammas(model):
    """NOTE: model params are bf16 -> the fold rounds W*g to bf16 (2^-9 rel/weight), so it is
    exact-in-math but NOT byte-exact; expected ppl jitter ~0.01. Arithmetic in fp32, single cast."""
    msd = model.state_dict()
    n_side = 0
    with torch.no_grad():
        for li in range(model.config.num_hidden_layers):
            for nk, targets in FOLD.items():
                gk = f"model.layers.{li}.{nk}"
                g32 = msd[gk].clone().float()
                for t in targets:
                    wk = f"model.layers.{li}.{t}"
                    msd[wk].copy_((msd[wk].float() * g32[None, :]).to(msd[wk].dtype))
                msd[gk].fill_(1.0)
                n_side += g32.numel()
    return n_side                                     # gamma params shipped as side stream


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    fp16 = ppl(model, tok)

    # --- exactness gate ---
    n_side = fold_gammas(model)
    pf = ppl(model, tok)
    print(f"fp16 {fp16:.4f} | folded(no quant) {pf:.4f}  (delta {pf-fp16:+.4f}; "
          f"gate <0.015, bf16-rounding-limited)")
    if abs(pf - fp16) > 0.015:
        print("FOLD NOT EXACT beyond bf16 noise -> bug, aborting"); return

    # --- champion on FOLDED weights (recalibrate on folded model) ---
    calib = calibrate(model, tok)
    bpw = load_quant(model, lambda a, k: codec_zoo.ecvq_mid(a, k, calib))
    side = n_side * 16.0 / 357_826_560               # gamma side stream b/w
    p_folded = ppl(model, tok)
    print(f"\nfolded + ECVQ.008 : arena {bpw+side:.4f} b/w (incl gamma {side:.5f})  ppl {p_folded:.4f}")
    print(f"baseline champion : arena 3.1253 b/w                     ppl 4.4826")
    d = 4.4826 - p_folded
    print(f"\n=> ppl delta at ~equal bits: {d:+.4f}  "
          f"({'WIN' if d > 0.02 else 'KILL (gamma already captured by per-group amax)' if d > -0.02 else 'HURTS'})")
    gc.collect()


if __name__ == "__main__":
    main()
