"""expert_bits.py -- is 2-bit the floor, or are the experts FUNCTIONALLY over-provisioned?

The experts are maximal-entropy as WEIGHTS (at the Gaussian RD bound). But weight-information !=
functional-information: the rank-robustness law says high-rank tensors average quantization noise away, so
the FUNCTION may need fewer bits than the weights carry. Test: quantize ONLY the routed experts to K bits
(rest fp16) and sweep K. If ppl damage grows MUCH slower than the bits drop, the experts are over-
provisioned -> we can push below 2-bit.

FAST screen: per-output-channel uniform absmax (K>=2) / binary mean-magnitude (K=1), done on GPU. This is
a CONSERVATIVE codec (above the trellis RD bound), so if the function survives THIS at K bits, the optimal
trellis survives too. EVOQ_EXP_K in {3,2,1}. Rest of model fp16. EVOQ_NWIN windows.
"""
from __future__ import annotations
import gc, os, sys, time
import torch
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import weights.evoq_moe as em

FP16 = 6.3070


def fast_quant(W, K):
    if K <= 1:                                              # 1-bit: sign x per-row mean magnitude
        s = W.abs().mean(dim=1, keepdim=True)
        return torch.sign(W) * s
    qmax = 2 ** (K - 1) - 1                                 # K-bit signed, per-row absmax
    s = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
    return (W / s).round().clamp(-qmax - 1, qmax) * s


def run():
    K = int(os.environ.get("EVOQ_EXP_K", "2"))
    nwin = int(os.environ.get("EVOQ_NWIN", "151"))
    layspec = os.environ.get("EVOQ_EXP_LAYERS", "")          # "lo-hi" inclusive; "" = all layers
    lo, hi = (int(x) for x in layspec.split("-")) if layspec else (0, 10 ** 9)
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(nwin)
    scope = f"layers {lo}-{min(hi,L-1)}" if layspec else "all layers"
    print(f"EXPERT-BITS [FAST]: routed experts -> {K}-bit ({scope}, rest fp16) | {nwin} win", flush=True)
    t0 = time.time(); relmse = []
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        for name, mod in layer.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if ".experts." in name and name.endswith(("gate_proj", "up_proj", "down_proj")) and lo <= li <= hi:
                W = mod.weight.data
                wh = fast_quant(W, K)
                if li == 13 and len(relmse) < 3:
                    relmse.append((((W - wh) ** 2).sum() / (W ** 2).sum()).item())
                mod.weight.data = wh
        h = em._run_layer(layer, h, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 6 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    rm = sum(relmse) / len(relmse) if relmse else float("nan")
    line = (f"EXPERT-BITS [routed@{K}bit fast, {scope}, rest fp16]: ppl={ppl:.4f} (delta vs fp16 {ppl-FP16:+.4f}) | "
            f"weight rel-MSE@{K}bit={rm:.4f} (trellis@2bit=0.069) | fp16={FP16:.4f}")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
