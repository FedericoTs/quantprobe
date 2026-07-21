"""task_trim_probe.py -- can we TRIM a MoE for a specific task/domain (drop or down-bit unused experts)?

Measures domain-conditional expert usage on DeepSeek-V2-Lite: route one PROSE window and one CODE window,
count per-expert selections per layer. Verdicts per layer: how many of 64 experts cover 90% of routings
per domain (concentration), cross-domain overlap of the used sets (task-specificity), and the never-used
tail (trim candidates). Context: on MIXED text, 62/64 experts were touched in 128 tokens (locality dead) --
but task-CONDITIONAL concentration was never measured. fp16 streaming, ~10 min. RUN WITH main .venv.
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

TOPK = 6
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(1)
    prose = open(os.path.join(HERE, "data", "wikitext2_train.txt"), encoding="utf-8").read()[:14000]
    code = open(os.path.join(HERE, "evoq_moe.py"), encoding="utf-8").read()[:14000]
    hs = {}
    emb = em.read_tensor(smap, "model.embed_tokens.weight").float()
    for d, txt in (("prose", prose), ("code", code)):
        i = tok(txt, return_tensors="pt").input_ids[:, :seqlen]
        hs[d] = torch.nn.functional.embedding(i, emb)
    del emb

    counts = {d: {} for d in hs}
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        has_router = hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "weight")
        store = {}
        hk = None
        if has_router:
            def hook(mod, inp):
                x = inp[0].detach().float()
                lg = x @ mod.weight.float().T.to(x.device)
                store["top"] = torch.topk(lg, TOPK, dim=-1).indices.cpu().numpy().reshape(-1)
            hk = layer.mlp.gate.register_forward_pre_hook(hook)
        layer.cuda()
        for d in hs:
            with torch.no_grad():
                out = layer(hs[d][:1].cuda(), attention_mask=mask, position_ids=pos)[0]
            hs[d] = out.cpu()
            if has_router:
                c = np.bincount(store["top"], minlength=64)
                counts[d][li] = c
        if hk:
            hk.remove()
        em.free_layer(layer)
        torch.cuda.empty_cache()
        if li % 9 == 0:
            print(f"  layer {li}/{L}", flush=True)

    print("\nTASK-TRIM PROBE (DeepSeek-V2-Lite, 64 experts/layer, top-6, 2048 tok/domain)")
    print(f"  {'layer':6s} {'dom':6s} n90   never | used-set Jaccard prose~code")
    n90s, jacs, nevers = {d: [] for d in counts}, [], {d: [] for d in counts}
    for li in sorted(next(iter(counts.values())).keys()):
        tops = {}
        for d in counts:
            c = counts[d][li]
            order = np.argsort(-c); cum = np.cumsum(c[order]) / max(1, c.sum())
            n90 = int(np.searchsorted(cum, 0.90) + 1)
            never = int((c == 0).sum())
            tops[d] = set(np.where(c > 0)[0].tolist())
            n90s[d].append(n90); nevers[d].append(never)
            if li in (1, 13, 26):
                print(f"  L{li:<5d} {d:6s} {n90:3d}   {never:3d}", flush=True)
        a, b = tops["prose"], tops["code"]
        j = len(a & b) / max(1, len(a | b))
        jacs.append(j)
        if li in (1, 13, 26):
            print(f"         Jaccard={j:.2f}", flush=True)
    print(f"\n  MEANS over {len(jacs)} MoE layers:")
    for d in counts:
        print(f"    {d:6s}: n90={np.mean(n90s[d]):.1f}/64 experts for 90% of routings | never-used={np.mean(nevers[d]):.1f}")
    print(f"    used-set Jaccard prose~code = {np.mean(jacs):.2f}")
    print("\n  VERDICT: n90 << 64 AND low Jaccard -> task-trim viable; n90 near 64 or high overlap -> trim dead.")


if __name__ == "__main__":
    main()
