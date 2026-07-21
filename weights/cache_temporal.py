"""cache_temporal.py -- does the MLA KV cache (c_KV latent) have TEMPORAL structure to exploit?

Everyone quantizes the KV cache statically, per token. But in decode the cache is a TRAJECTORY: c_KV(t)
is the running compression of a coherent context, so consecutive latents may be correlated. If so,
delta/predictive coding compresses the cache BELOW its per-token 8-bit floor -- a new axis the static
framing can't see. Capture c_KV across token positions, measure the delta-vs-raw variance ratio and lag-1
autocorrelation. ratio≈2 => i.i.d., nothing there; ratio<1 => temporally compressible.
Data-light (a few windows, activation stats only, no ppl).
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
D_C = 512


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
        if li in CAP:
            def mk(idx):
                def hook(m, inp, out):
                    caught[idx] = out[..., :D_C].detach().float().cpu().numpy()   # [nwin, seq, 512]
                return hook
            hk = layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(mk(li))
        h = em._run_layer(layer, h, mask, pos)
        if hk is not None:
            hk.remove()
        em.free_layer(layer)
        if li >= last:
            break

    print(f"CACHE TEMPORAL STRUCTURE (c_KV latent across {nwin} windows x {seqlen} tokens)")
    print("  ratio = Var(c[t]-c[t-1]) / Var(c);  2.00 = i.i.d./no structure,  <1.0 = temporally compressible")
    for li in sorted(caught):
        c = caught[li]                                   # [nwin, seq, 512]
        raw_var = float(c.var())
        delta = c[:, 1:, :] - c[:, :-1, :]
        ratio = float(delta.var()) / raw_var
        cc = c - c.mean(axis=1, keepdims=True)
        ac = float((cc[:, 1:, :] * cc[:, :-1, :]).mean() / (cc * cc).mean())   # lag-1 autocorr
        save = 0.5 * np.log2(1.0 / ratio) if ratio > 0 else 0.0               # bits saved by delta-coding
        # also: AR(1) one-step predictor residual (predict c[t] = ac * c[t-1])
        pred_res = c[:, 1:, :] - ac * c[:, :-1, :]
        pred_ratio = float(pred_res.var()) / raw_var
        psave = 0.5 * np.log2(1.0 / pred_ratio) if pred_ratio > 0 else 0.0
        print(f"  layer {li:2d}: delta/raw={ratio:.3f} (lag-1 autocorr={ac:+.3f}) -> delta-code saves ~{save:+.2f} bits"
              f" | AR(1)-predict resid/raw={pred_ratio:.3f} -> ~{psave:+.2f} bits")
    print("\n  verdict: if ratio<1 / autocorr>0.5 on any layer, the cache has exploitable temporal redundancy below 8-bit.")


if __name__ == "__main__":
    run()
