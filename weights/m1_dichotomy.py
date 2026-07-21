"""m1_dichotomy.py -- the rank-conditional rotation test (proves the incoherence bias).

Insert a random-orthogonal gauge R on a HIGH-rank intermediate (the shared-expert MLP, ~2816-dim) --
store down' = down @ R^T and feed it R@h via a pre-hook (bit-identical at FP16) -- then 2-bit quantize
down'. Compare the rotation's damage here to the +1623 it caused on the LOW-rank KV-latent. Prediction:
rotating a high-rank tensor is benign; rotating the low-rank bottleneck is catastrophic. Same operation,
opposite sign => incoherence rotation is rank-conditional.

EVOQ_DICHO_ROT in {identity, orth}. Full WikiText-2 set, int8 gs.
"""
from __future__ import annotations
import gc, os, sys, time
import numpy as np
import torch
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from safetensors.torch import load_file as _sf_load
import weights.evoq_moe as em

CARVE_CFG = "routed-gate/up=2 DOWN_K=3 ATTN_K=4 SHARED_K=4 DENSE_K=4 INT8_GS=True AWQ=False(a=0.5)"
FP16, CARVE = 6.3070, 6.9616


def rand_orth(n, seed):
    rng = np.random.default_rng(1000 + seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q.astype(np.float32)


def _prehook(Rt):
    def h(m, args):
        return (args[0] @ Rt.to(args[0].device, args[0].dtype),)
    return h


def run():
    rot = os.environ.get("EVOQ_DICHO_ROT", "identity")
    nwin = int(os.environ.get("EVOQ_NWIN", "151"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rotemb, mask, pos = em._eval_setup(nwin)
    cdir = em.cache_dir(CARVE_CFG)
    print(f"DICHOTOMY: shared-MLP intermediate rotation={rot}, shared down_proj@2bit | {nwin} win", flush=True)
    t0 = time.time(); interm_rank = None
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rotemb
        layer.cuda()
        cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))
        hk = None
        for name, mod in layer.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if "shared_experts" in name and name.endswith("down_proj"):
                Wd = mod.weight.detach().float().cpu().numpy()       # [hidden, interm]
                interm = Wd.shape[1]
                R = rand_orth(interm, li) if rot == "orth" else np.eye(interm, dtype=np.float32)
                wh, _ = em.trellis_quant(Wd @ R.T, K=2)              # down' = down @ R^T, then 2-bit
                mod.weight.data = torch.from_numpy(wh).to(mod.weight.device)
                hk = mod.register_forward_pre_hook(_prehook(torch.from_numpy(R.T)))   # feed it R@h
                if li == 1 and interm_rank is None:
                    s = np.linalg.svd(Wd.astype(np.float64), compute_uv=False); p = s**2/(s**2).sum()
                    interm_rank = float(np.exp(-(p*np.log(p+1e-30)).sum()))
                    print(f"  shared down: interm={interm}, eff_rank={interm_rank:.0f} "
                          f"(vs KV-latent composed eff_rank 394)", flush=True)
            elif name in cw:
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        h = em._run_layer(layer, h, mask, pos)
        if hk is not None:
            hk.remove()
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 9 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    line = (f"DICHOTOMY [shared-MLP rot={rot}, down@2bit]: ppl={ppl:.4f} (delta vs carve-out {ppl-CARVE:+.4f}) "
            f"| HIGH-rank intermediate (eff~{interm_rank}) | compare to KV-latent (low-rank) rot=hadamard +1623")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
