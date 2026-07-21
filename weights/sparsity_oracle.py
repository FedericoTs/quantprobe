"""Campaign2-B1 day-1 ORACLE -- structured-sparsity quality law on Qwen2.5-0.5B (pure torch).

Per token, keep only the top-f fraction of MLP intermediate neurons (dim 4864) ranked by a cheap
on-device statistic, zero the rest BEFORE down_proj (== skip those gate/up rows + down columns ==
the bytes a skip-decode kernel would never decode). Granularity-STRATIFIED per the roadmap:
  g=1    unstructured (upper bound, not kernel-realizable)
  g=32   32-neuron blocks
  g=128  128-neuron blocks  <-- matches our substream/group layout; BINDING stratum
Statistic: block energy of h = act(gate(x))*up(x) (sum h^2 per block) -- computable in-kernel.

Output: held-out ppl vs decoded fraction f per granularity (+ dense ref). Roadmap kill (binding
for the C2 skip-decode kernel): no arm >= ~20-40%% sparsity at <= +0.05 ppl in the >=32 stratum.
(0.5B intermediate = 4864 = 38 blocks of 128 -- adequate stratum; 1.5B confirmation later.)

Run:  python -m weights.sparsity_oracle
"""
from __future__ import annotations

import gc
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights.quant_lab import CFG, build_model, load_fp16, ppl

FRACS = (0.7, 0.5, 0.3)
GRANS = (1, 32, 128)


class MaskState:
    g = 128
    f = 1.0
    enabled = False


def mk_hook():
    def pre_hook(mod, args):
        if not MaskState.enabled or MaskState.f >= 0.999:
            return None
        (h,) = args                                  # [B, T, I] input to down_proj
        B, T, I = h.shape
        g = MaskState.g
        nb = I // g
        e = (h.float() ** 2).reshape(B, T, nb, g).sum(-1)        # block energy [B,T,nb]
        k = max(1, int(round(nb * MaskState.f)))
        thr = e.topk(k, dim=-1).values[..., -1:]                 # kth largest per token
        keep = (e >= thr).unsqueeze(-1)                          # [B,T,nb,1]
        hm = (h.reshape(B, T, nb, g) * keep.to(h.dtype)).reshape(B, T, I)
        return (hm,)
    return pre_hook


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model()
    load_fp16(model)
    hooks = [layer.mlp.down_proj.register_forward_pre_hook(mk_hook())
             for layer in model.model.layers]
    MaskState.enabled = False
    p0 = ppl(model, tok)
    print(f"dense reference ppl = {p0:.4f}  (noise floor ~0.07 from R2)\n")
    print(f"{'gran':>5}{'frac':>6}{'ppl':>9}{'d_ppl':>8}   verdict-vs(+0.05)")
    print("-" * 46)
    MaskState.enabled = True
    for g in GRANS:
        for f in FRACS:
            MaskState.g, MaskState.f = g, f
            t0 = time.time()
            p = ppl(model, tok)
            d = p - p0
            verdict = "PASS" if d <= 0.05 else ("noise-band" if d <= 0.10 else "fail")
            print(f"{g:>5}{f:>6.1f}{p:>9.4f}{d:>+8.4f}   {verdict}  ({time.time()-t0:.0f}s)",
                  flush=True)
            gc.collect()
    for h in hooks:
        h.remove()
    print("\nBINDING read (roadmap B1): need >=1 cell in g>=32 with f<=0.8-0.6 at d_ppl<=+0.05 "
          "for the skip-decode kernel branch to stay alive; g=128 is the kernel-native stratum.")


if __name__ == "__main__":
    main()
