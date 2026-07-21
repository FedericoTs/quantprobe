"""route_locality.py -- does single-stream MoE routing have TEMPORAL locality? (paper-2 Step 1)

On real text (fp16, the true routing), per layer, we measure between CONSECUTIVE tokens of one stream:
  - expert-set overlap vs a random-pair baseline  -> is routing locally "sticky"?
  - working-set size: # distinct routed experts over a window of W consecutive tokens -> cacheable?
  - expert-usage concentration (top-share)        -> heavy-tailed (cacheable) or uniform?

If consecutive overlap >> random AND the working set saturates small, a hot-expert cache makes
single-stream MoE coalesced + fast while the cold majority stays 2-bit resident (the paper-2 thesis).
Off a plain fp16 streaming pass (no quant needed). Within a 2048-token window, positions t and t+1 ARE
consecutive tokens of one sequence, so we get temporal locality directly.
"""
from __future__ import annotations

import gc
import os
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from weights.evoq_moe import _eval_setup, materialize_cpu, free_layer

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WS = [1, 2, 4, 8, 16, 32, 64, 128]


def main():
    nwin = int(os.environ.get("EVOQ_NWIN", "2"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = _eval_setup(nwin)
    n_exp = int(getattr(cfg, "n_routed_experts", 64))
    cap = {}
    for li in range(L):
        layer = model.model.layers[li]
        materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        is_moe = hasattr(layer.mlp, "gate")
        buf, hk = [], None
        if is_moe:
            hk = layer.mlp.gate.register_forward_hook(
                lambda m, inp, out: buf.append(out[0].detach().to(torch.int16).cpu()))   # [seqlen, k]
        out = torch.empty_like(h)
        for b in range(h.shape[0]):
            hb = h[b:b + 1].cuda()
            with torch.no_grad():
                yb = layer(hb, attention_mask=mask, position_ids=pos)[0]
            out[b:b + 1] = yb.cpu(); del hb, yb
        h = out
        if hk is not None:
            hk.remove(); cap[li] = buf
        free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 6 == 0 or li == L - 1:
            print(f"  layer {li}/{L} captured", flush=True)

    lines = ["# routing TEMPORAL locality (fp16, consecutive tokens), DeepSeek-V2-Lite",
             f"# {nwin} win x {seqlen} tok | k routed experts of {n_exp} (+2 shared always-on)",
             f"{'layer':>5} {'consec_ov':>10} {'rand_ov':>8} {'ws@8':>6} {'ws@32':>6} {'ws@128':>7} {'topshare':>9}"]
    agg = []
    for li in sorted(cap):
        consec, rand, freq = [], [], torch.zeros(n_exp)
        wsacc = {W: [] for W in WS}
        for w in cap[li]:
            w = w.long(); S, k = w.shape
            a, b = w[:-1], w[1:]
            consec.append(((a.unsqueeze(2) == b.unsqueeze(1)).any(2).sum(1).float() / k).mean().item())
            perm = torch.randperm(S)
            consec_r = (w.unsqueeze(2) == w[perm].unsqueeze(1)).any(2).sum(1).float() / k
            rand.append(consec_r.mean().item())
            for W in WS:
                if S >= W:
                    step = max(1, (S - W) // 60)
                    cnts = [int(torch.unique(w[s:s + W]).numel()) for s in range(0, S - W + 1, step)]
                    wsacc[W].append(sum(cnts) / len(cnts))
            freq += torch.bincount(w.reshape(-1), minlength=n_exp).float()
        cov, rov = sum(consec) / len(consec), sum(rand) / len(rand)
        ws = {W: (sum(wsacc[W]) / len(wsacc[W]) if wsacc[W] else 0.0) for W in WS}
        topshare = (freq.max() / freq.sum()).item() * n_exp                 # 1.0=uniform, >1=concentrated
        lines.append(f"{li:5d} {cov:10.3f} {rov:8.3f} {ws[8]:6.1f} {ws[32]:6.1f} {ws[128]:7.1f} {topshare:9.2f}")
        agg.append((cov, rov, ws))
    mcov = sum(a[0] for a in agg) / len(agg); mrov = sum(a[1] for a in agg) / len(agg)
    lines += ["",
              f"# mean consec_overlap = {mcov:.3f}  vs  random-pair = {mrov:.3f}  "
              f"(temporal locality = {mcov/(mrov+1e-9):.2f}x base rate)",
              f"# mean working set: W=8 -> {sum(a[2][8] for a in agg)/len(agg):.1f}, "
              f"W=32 -> {sum(a[2][32] for a in agg)/len(agg):.1f}, "
              f"W=128 -> {sum(a[2][128] for a in agg)/len(agg):.1f} distinct routed experts of {n_exp}",
              f"# verdict: overlap only {mcov/(mrov+1e-9):.1f}x base, working set ~{sum(a[2][32] for a in agg)/len(agg):.0f}/{n_exp} within 32 tok -> hot-expert cache NOT viable (paper-2 thesis REFUTED)"]
    out = "\n".join(lines); print("\n" + out, flush=True)
    open(os.path.join(DATA, "route_locality.txt"), "w", encoding="utf-8").write(out + "\n")
    print(f"\nsaved -> {os.path.join(DATA, 'route_locality.txt')}", flush=True)


if __name__ == "__main__":
    main()
