"""router_confidence.py -- can DYNAMIC TOP-K beat the bandwidth ceiling?

The ~220 tok/s ceiling assumes reading top-6 experts EVERY token. But if the renormalized routing weight
is concentrated in the top-1/2 on most tokens, reading fewer experts there is near-free -> fewer bytes/
token -> average throughput ABOVE the static ceiling. This is the one lever that attacks the ceiling, not
just reaches it. Measure how much of the top-6 routing mass sits in the top-1/2/3, and the fraction of
tokens where top-2 already covers >=90%. Data-light (a few windows, router stats only).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import weights.evoq_moe as em

CAP = {3, 13, 24}
TOPK = 6


def run():
    nwin = int(os.environ.get("EVOQ_NWIN", "6"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(nwin)
    caught = {}
    last = max(CAP)
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        hk = None
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if li in CAP and gate is not None and hasattr(gate, "weight"):
            def mk(idx, W):
                def hook(m, args):
                    x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                    logits = x @ W.t().float()
                    caught[idx] = logits.cpu().numpy()              # [tokens, n_experts]
                return hook
            hk = gate.register_forward_pre_hook(mk(li, gate.weight.detach()))
        h = em._run_layer(layer, h, mask, pos)
        if hk is not None:
            hk.remove()
        em.free_layer(layer)
        if li >= last:
            break

    print(f"ROUTER CONFIDENCE (renormalized top-{TOPK} routing mass concentration)")
    print("  if top-1/2 hold most of the mass on most tokens, dynamic top-k reads fewer experts -> beats ceiling")
    for li in sorted(caught):
        lg = caught[li]                                            # [tokens, n_experts]
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        idx = np.argsort(-p, axis=1)[:, :TOPK]
        topw = np.take_along_axis(p, idx, axis=1)                 # [tokens, 6]
        topw = topw / topw.sum(1, keepdims=True)                  # renorm over the top-6 (as the model does)
        m1 = topw[:, 0].mean()
        m2 = topw[:, :2].sum(1).mean()
        m3 = topw[:, :3].sum(1).mean()
        frac_top2_90 = float((topw[:, :2].sum(1) >= 0.90).mean())
        frac_top1_50 = float((topw[:, 0] >= 0.50).mean())
        # expected experts needed to reach 90% mass, per token (the dynamic-k that would suffice)
        cum = np.cumsum(topw, axis=1)
        k90 = (cum < 0.90).sum(1) + 1
        print(f"  layer {li:2d}: top1={m1:.2f} top2={m2:.2f} top3={m3:.2f} of mass | "
              f"tokens top2>=90%: {frac_top2_90:.0%} | top1>=50%: {frac_top1_50:.0%} | mean k for 90% = {k90.mean():.2f}/{TOPK}")
    print("\n  verdict: low mean-k-for-90% (e.g. ~2-3 vs 6) => dynamic top-k cuts bytes/token => throughput above the static ceiling.")


if __name__ == "__main__":
    run()
