"""lookahead_probe.py -- T2: is colibri's router-lookahead (71.6% on GLM-5.2) architecture-general?

At each layer boundary of DeepSeek-V2-Lite we predict layer i's top-6 experts from the BOUNDARY hidden
state (before layer i's attention runs) -- pred = top6(gate_i(rmsnorm_i(h_boundary))) -- and compare with
the TRUE router decision captured by a hook during the real forward. Reports per-layer and overall recall,
plus prefetch hit-rate when prefetching top-8/top-12 predicted experts. fp16 streaming, ~10 min.
"""
from __future__ import annotations
import sys
import numpy as np
import torch
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import weights.evoq_moe as em

TOPK = 6


def rms(h, w, eps):
    v = h.pow(2).mean(-1, keepdim=True)
    return (h * torch.rsqrt(v + eps)) * w


def main():
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(2)
    eps = cfg.rms_norm_eps
    recalls, hits8, hits12 = {}, {}, {}
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        has_router = hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "weight")
        pred = None
        if has_router:
            gw = em.read_tensor(smap, f"model.layers.{li}.mlp.gate.weight").float()          # [64, H]
            pw = em.read_tensor(smap, f"model.layers.{li}.post_attention_layernorm.weight").float()
            with torch.no_grad():
                logits = rms(h, pw, eps) @ gw.T                                              # [nwin, seq, 64]
                pred = torch.topk(logits, 12, dim=-1).indices                                # top-12 predicted
        truth = {}
        hooks = []
        if has_router:
            def hk(mod, inp):
                x = inp[0].detach().float()
                lg = x @ mod.weight.float().T.to(x.device)
                truth.setdefault("t", []).append(torch.topk(lg, TOPK, dim=-1).indices.cpu())
            hooks.append(layer.mlp.gate.register_forward_pre_hook(hk))
        layer.cuda()
        outs = []
        for b in range(nwin):
            with torch.no_grad():
                yb = layer(h[b:b + 1].cuda(), attention_mask=mask, position_ids=pos)[0]
            outs.append(yb.cpu())
        h = torch.cat(outs, 0)
        for hkh in hooks:
            hkh.remove()
        em.free_layer(layer)
        torch.cuda.empty_cache()
        if has_router and truth.get("t"):
            true6 = torch.cat(truth["t"], 0).reshape(-1, TOPK)                               # [tok, 6]
            p = pred.reshape(-1, 12)
            def hit(k):
                s = 0
                for a, b in ((true6, p[:, :k]),):
                    m = (a.unsqueeze(-1) == b.unsqueeze(1)).any(-1).float().mean().item()
                    s = m
                return s
            recalls[li], hits8[li], hits12[li] = hit(6), hit(8), hit(12)
            print(f"  layer {li:2d}: pred-top6 recall={recalls[li]:.3f}  prefetch-hit top8={hits8[li]:.3f} top12={hits12[li]:.3f}", flush=True)
    if recalls:
        avg = np.mean(list(recalls.values()))
        print(f"\nLOOKAHEAD (DeepSeek-V2-Lite, {len(recalls)} MoE layers): mean top6 recall={avg:.3f} "
              f"| top8 prefetch={np.mean(list(hits8.values())):.3f} | top12 prefetch={np.mean(list(hits12.values())):.3f} "
              f"| colibri GLM-5.2 reference: 0.716", flush=True)


if __name__ == "__main__":
    main()
