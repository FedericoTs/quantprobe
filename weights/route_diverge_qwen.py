"""route_diverge_qwen.py -- routing-divergence on Qwen1.5-MoE (generality mechanism check).

The Qwen port of route_diverge.py: two streaming passes (fp16 vs cached-quant) over the same tokens,
fp16 router both, a forward hook on each layer's mlp.gate capturing the top-4 selected experts. Qwen
routing is plain softmax+topk, so topk(router_logits) == the selected experts (softmax is monotonic).
Per MoE layer: overlap = mean |E_fp16 INTERSECT E_q| / 4, top-1 agreement, divergence = 1 - overlap.

Loads the per-layer quantized weights cached by evoq_moe_qwen.measure (EVOQ_CACHE). Run once per config
(carve-out vs uniform Ks) and contrast the two route_diverge_qwen_*.txt -> does the carve-out keep
routing stable while uniform collapses it, on a non-MLA MoE too?  EVOQ_TAG names the output.
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

from safetensors.torch import load_file as _sf_load

import weights.evoq_moe as em                                  # monkeypatched by the import below
from weights.evoq_moe_qwen import _eval_setup_qwen, TARGETS_Q  # noqa: F401 (import applies monkeypatch)
from weights.qtip_trellis import K_RATE

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _cfgstr():
    down_k = os.environ.get("EVOQ_DOWN_K", str(K_RATE)); attn_k = os.environ.get("EVOQ_ATTN_K", str(K_RATE))
    shared_k = os.environ.get("EVOQ_SHARED_K", str(K_RATE)); int8 = bool(os.environ.get("EVOQ_INT8_GS"))
    return f"qwen-moe routed-gate/up=2 DOWN_K={down_k} ATTN_K={attn_k} SHARED_K={shared_k} INT8_GS={int8}"


def _pass(model, L, smap, posemb, mask, pos, h0, quant, cdir):
    h = h0.clone(); cap = {}; cos, sin = posemb; tag = "quant" if quant else "fp16 "
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.cuda()
        if quant:
            cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))
            for name, mod in layer.named_modules():
                if name in cw:
                    mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        buf = []
        hk = layer.mlp.gate.register_forward_hook(                                 # router Linear -> [tok, 60]
            lambda m, inp, out: buf.append(out.detach().float().topk(4, dim=-1).indices.to(torch.int16).cpu()))
        out = torch.empty_like(h)
        for b in range(h.shape[0]):
            hb = h[b:b + 1].cuda()
            with torch.no_grad():
                yb = layer(hb, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin))[0]
            out[b:b + 1] = yb.cpu(); del hb, yb
        h = out
        hk.remove(); cap[li] = torch.cat(buf, 0)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        print(f"  [{tag}] layer {li}/{L} done", flush=True)
    return cap


def main():
    nwin = int(os.environ.get("EVOQ_NWIN", "2"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, posemb, mask, pos = _eval_setup_qwen(nwin)
    h0 = h.clone(); cfgstr = _cfgstr(); cdir = em.cache_dir(cfgstr)
    print(f"qwen routing-divergence [{cfgstr}]: {nwin} win x {seqlen} tok, cache {cdir}", flush=True)
    fp16 = _pass(model, L, smap, posemb, mask, pos, h0, False, cdir)
    quant = _pass(model, L, smap, posemb, mask, pos, h0, True, cdir)

    lines = [f"# qwen routing divergence vs fp16 | {cfgstr} | {nwin*seqlen} tok/layer | top-4 of 60 experts",
             f"{'layer':>5} {'overlap':>9} {'top1_agree':>11} {'divergence':>11}"]
    divs = []
    for li in sorted(fp16):
        a, b = fp16[li].long(), quant[li].long(); N, k = a.shape
        inter = (a.unsqueeze(2) == b.unsqueeze(1)).any(2).sum(1).float()
        ov = (inter / k).mean().item(); top1 = (a[:, 0] == b[:, 0]).float().mean().item()
        divs.append((li, 1 - ov)); lines.append(f"{li:5d} {ov:9.4f} {top1:11.4f} {1 - ov:11.4f}")
    if divs:
        half = len(divs) // 2
        early = sum(d for _, d in divs[:half]) / max(1, half)
        late = sum(d for _, d in divs[half:]) / max(1, len(divs) - half)
        lines += ["", f"# early-half mean div={early:.4f} late-half={late:.4f} "
                      f"ratio={late / max(early, 1e-9):.2f}x max={max(d for _, d in divs):.4f}"]
    out = "\n".join(lines); print("\n" + out, flush=True)
    tag = os.environ.get("EVOQ_TAG", "")
    fn = os.path.join(DATA, f"route_diverge_qwen{('_' + tag) if tag else ''}.txt")
    open(fn, "w", encoding="utf-8").write(out + "\n"); print(f"\nsaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
