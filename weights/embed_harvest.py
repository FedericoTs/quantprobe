"""embed_harvest.py -- the last untested lever: quantize embed_tokens + lm_head (the only fp16-kept
weight tensors) to K-bit per-128 uniform, on top of the carve-out. ~419M params each = 838MB fp16 ->
~0.6GB freed at 4-bit. embed_probe.py proved clustering buys ~0 net bits and these are ~4.6x more fragile
than experts (2-bit rel-MSE 0.316), so the safe point is ~4-bit; lm_head feeds the softmax directly, so
watch the 3-bit cliff. Reuses the carve-out cache for all layers. EVOQ_EMB_K bit-width.
"""
from __future__ import annotations
import gc, os, sys, time
import torch
import torch.nn.functional as F
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from safetensors.torch import load_file as _sf_load
import weights.evoq_moe as em

CARVE_CFG = "routed-gate/up=2 DOWN_K=3 ATTN_K=4 SHARED_K=4 DENSE_K=4 INT8_GS=True AWQ=False(a=0.5)"
FP16, CARVE = 6.3070, 6.9616


def quant_group(W, K, g=128):
    qmax = 2 ** (K - 1) - 1
    out = W.clone()
    for i in range(0, W.shape[1], g):
        blk = W[:, i:i + g]
        s = (blk.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-9)
        out[:, i:i + g] = (blk / s).round().clamp(-qmax - 1, qmax) * s
    return out


def run():
    K = int(os.environ.get("EVOQ_EMB_K", "4"))
    nwin = int(os.environ.get("EVOQ_NWIN", "151"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(nwin)
    cdir = em.cache_dir(CARVE_CFG)

    emb = em.read_tensor(smap, "model.embed_tokens.weight").float().cuda()
    embq = quant_group(emb, K)
    rm_emb = (((emb - embq) ** 2).sum() / (emb ** 2).sum()).item()
    h = F.embedding(ids, embq.cpu())                       # recompute initial hidden from quantized embed
    del emb, embq; torch.cuda.empty_cache()
    print(f"EMBED-HARVEST: embed+lm_head -> {K}-bit per-128, rest=carve-out | {nwin} win | embed rel-MSE={rm_emb:.4f}", flush=True)

    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))
        for name, mod in layer.named_modules():
            if isinstance(mod, torch.nn.Linear) and name in cw:
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        h = em._run_layer(layer, h, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 9 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)

    normw = em.read_tensor(smap, "model.norm.weight").float().cuda()
    headw = em.read_tensor(smap, "lm_head.weight").float().cuda()
    headq = quant_group(headw, K)
    rm_head = (((headw - headq) ** 2).sum() / (headw ** 2).sum()).item()
    del headw; torch.cuda.empty_cache()
    nlls, ntok = [], 0
    for b in range(nwin):
        hb = h[b].cuda()
        v = hb.pow(2).mean(-1, keepdim=True)
        hb = (hb * torch.rsqrt(v + cfg.rms_norm_eps)) * normw
        logits = hb @ headq.T
        nlls.append(F.cross_entropy(logits[:-1].float(), ids[b, 1:].cuda()) * (seqlen - 1))
        ntok += seqlen - 1
        del hb, logits
    ppl = torch.exp(torch.stack(nlls).sum() / ntok).item()
    line = (f"EMBED-HARVEST [embed+lm_head@{K}bit per-128, rest carve-out]: ppl={ppl:.4f} "
            f"(delta vs carve-out {ppl-CARVE:+.4f}) | carve-out={CARVE:.4f} fp16={FP16:.4f} | "
            f"embed rel-MSE={rm_emb:.4f} head rel-MSE={rm_head:.4f} | frees ~{419*2*(16-K)/16/1000:.2f}GB")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
