"""route_diverge.py -- mechanism contrast: 2-bit expert-routing divergence by depth.

Two streaming passes over the SAME eval tokens from the SAME embeddings: fp16, then quantized
(the config in the EVOQ_*_K env vars). The router (mlp.gate) is fp16 in BOTH passes, so any change
in expert selection is caused only by upstream hidden-state drift -- not by quantizing the router.
A forward hook on each layer's MoEGate captures the exact topk_idx it chose. Per MoE layer:
  overlap    = mean |E_fp16 INTERSECT E_quant| / k   (1.0 = identical routing)
  top1_agree = fraction of tokens whose single highest-weighted expert matches
  divergence = 1 - overlap
If quantization re-routes experts and that compounds with depth, divergence rises with layer index.

The quant pass loads cached weights if present (evoq_moe.measure's per-layer cache), else QUANTIZES
the layer with the SAME trellis codec (k_for reads EVOQ_*_K) and caches it -> the contrast uses one
codec, only the bit-allocation differs. Run two configs and diff the two route_diverge_*.txt by depth:
  carve-out (cached) : EVOQ_DOWN_K=3 EVOQ_ATTN_K=4 EVOQ_SHARED_K=4 EVOQ_DENSE_K=4 EVOQ_INT8_GS=1
  uniform   (requant): EVOQ_DOWN_K=2 EVOQ_ATTN_K=2 EVOQ_SHARED_K=2 EVOQ_DENSE_K=2 EVOQ_INT8_GS=1
-> the mechanism figure: does protecting attention/shared keep routing stable while uniform collapses it?

Env: EVOQ_NWIN (def 2) ; EVOQ_MAXL caps layers (smoke) ; EVOQ_TAG names the output file.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from safetensors.torch import load_file as _sf_load, save_file as _sf_save

from weights.evoq_moe import (_eval_setup, materialize_cpu, free_layer, cache_dir,
                              _quantize_layer, TARGETS)
from weights.qtip_trellis import K_RATE

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _cfgstr():
    """Reproduce evoq_moe.measure's cache-key string exactly from the EVOQ_* env vars."""
    down_k = os.environ.get("EVOQ_DOWN_K", str(K_RATE)); attn_k = os.environ.get("EVOQ_ATTN_K", str(K_RATE))
    shared_k = os.environ.get("EVOQ_SHARED_K", str(K_RATE)); dense_k = os.environ.get("EVOQ_DENSE_K", str(K_RATE))
    int8 = bool(os.environ.get("EVOQ_INT8_GS")); awq = bool(int(os.environ.get("EVOQ_AWQ", "0")))
    awq_a = float(os.environ.get("EVOQ_AWQ_ALPHA", "0.5"))
    return (f"routed-gate/up=2 DOWN_K={down_k} ATTN_K={attn_k} SHARED_K={shared_k} "
            f"DENSE_K={dense_k} INT8_GS={int8} AWQ={awq}(a={awq_a})")


def _pass(model, L, smap, rot, mask, pos, h0, quant, cdir, rep, maxl):
    """One full streaming forward. Returns {layer: topk_idx[tot_tok, k]} for MoE layers. If quant,
    each TARGET linear is set to its cached quantized weight (cache hit) or freshly quantized+cached
    (cache miss); the router mlp.gate is never a TARGET -> stays fp16, identical to the fp16 pass."""
    h = h0.clone(); cap = {}; tag = "quant" if quant else "fp16 "
    for li in range(min(L, maxl)):
        layer = model.model.layers[li]
        materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        if quant:
            lc = os.path.join(cdir, f"layer_{li:02d}.safetensors")
            lm = os.path.join(cdir, f"layer_{li:02d}.json")
            if os.path.exists(lc) and os.path.exists(lm):                      # CACHE HIT
                cw = _sf_load(lc)
                for name, mod in layer.named_modules():
                    if name in cw:
                        mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
            else:                                                              # CACHE MISS: quantize + save
                _quantize_layer(layer, rep)
                cw = {name: mod.weight.detach().half().cpu() for name, mod in layer.named_modules()
                      if isinstance(mod, nn.Linear) and name.split(".")[-1] in TARGETS}
                _sf_save(cw, lc); json.dump({"li": li}, open(lm, "w"))
        is_moe = hasattr(layer.mlp, "gate"); buf, hk = [], None
        if is_moe:
            hk = layer.mlp.gate.register_forward_hook(
                lambda m, inp, out: buf.append(out[0].detach().to(torch.int16).cpu()))   # out[0]=topk_idx
        out = torch.empty_like(h)
        for b in range(h.shape[0]):                                            # bsz=1 (caps MLA memory)
            hb = h[b:b + 1].cuda()
            with torch.no_grad():
                yb = layer(hb, attention_mask=mask, position_ids=pos)[0]
            out[b:b + 1] = yb.cpu(); del hb, yb
        h = out
        if hk is not None:
            hk.remove(); cap[li] = torch.cat(buf, 0)
        free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        print(f"  [{tag}] layer {li}/{L} {'MoE' if is_moe else 'dense'} done", flush=True)
    return cap


def main():
    nwin = int(os.environ.get("EVOQ_NWIN", "2"))
    maxl = int(os.environ.get("EVOQ_MAXL", "999"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = _eval_setup(nwin)
    h0 = h.clone(); cfgstr = _cfgstr(); cdir = cache_dir(cfgstr)
    tag = os.environ.get("EVOQ_TAG", "")
    print(f"routing-divergence [{cfgstr}]: {nwin} win x {seqlen} tok, cache {cdir}", flush=True)
    rep = {"bits": 0.0, "q": 0, "kept": 0}
    fp16 = _pass(model, L, smap, rot, mask, pos, h0, False, cdir, rep, maxl)
    quant = _pass(model, L, smap, rot, mask, pos, h0, True, cdir, rep, maxl)

    lines = [f"# routing divergence vs fp16 | cfg: {cfgstr} | {nwin*seqlen} tok/layer | fp16 router both",
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
                      f"ratio={late / max(early, 1e-9):.2f}x",
                  f"# first MoE div={divs[0][1]:.4f} last={divs[-1][1]:.4f} "
                  f"max={max(d for _, d in divs):.4f}"]
    out = "\n".join(lines); print("\n" + out, flush=True)
    fn = os.path.join(DATA, f"route_diverge{('_' + tag) if tag else ''}.txt")
    open(fn, "w", encoding="utf-8").write(out + "\n"); print(f"\nsaved -> {fn}", flush=True)


if __name__ == "__main__":
    main()
