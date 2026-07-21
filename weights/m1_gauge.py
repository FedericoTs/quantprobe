"""m1_gauge.py -- "Fragility is Fungible" test (the 6->10 hinge).

Insert an invertible gauge R in the KV-latent, between the down-proj kv_a (writes the d_c=512 latent c_KV)
and the up-proj kv_b (reads it): store kv_a' with R applied to its first 512 (c_KV) output rows, and kv_b'
with R^{-1} applied to its 512 input columns. At FP16 the model is BIT-IDENTICAL (R^{-1}R cancels) -- the
ONLY thing that changes is the basis the 2-bit quantization noise lands in. Then quantize the gauged pair
to 2-bit inside the deployed carve-out and measure Delta-ppl vs the gauge family.

CONFIRMS the gauge law (-> a 10): whitening relocates the +5.27 collapse toward 0; anti-whitening exceeds
+5.27; targeted whitening beats generic Hadamard. REFUTES (-> honest 6): Delta flat across all R (the
+5.27 is an intrinsic information-theoretic floor). The 'fp16kv' arm keeps the latent at 16-bit = the
quality upper bound and a direct test of the "keep the bottleneck at 16-bit" idea.

EVOQ_GAUGE in {fp16kv, identity, hadamard, svd, diagbalance, anti}. Full WikiText-2 set, int8 gs.
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
D_C = 512  # DeepSeek-V2-Lite kv_lora_rank; kv_a out = 512 c_KV + 64 k_rope = 576


def _hadamard(n):
    H = np.ones((1, 1), np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float32)


def make_gauge(kind, ka, kb, d=D_C):
    if kind == "hadamard":
        return _hadamard(d)
    if kind == "svd":                                   # rotate latent to kv_b's right-singular basis
        return np.linalg.svd(kb.astype(np.float64), full_matrices=False)[2].astype(np.float32)
    a = np.linalg.norm(ka[:d], axis=1) + 1e-8           # down-proj row norm per latent dim
    b = np.linalg.norm(kb, axis=0) + 1e-8               # up-proj col norm per latent dim
    if kind == "diagbalance":                           # SmoothQuant-style: balance a and b
        return np.diag(np.sqrt(b / a)).astype(np.float32)
    if kind == "anti":                                  # concentrate (should HURT)
        return np.diag(np.sqrt(a / b)).astype(np.float32)
    return np.eye(d, dtype=np.float32)                   # identity


def run():
    gauge = os.environ.get("EVOQ_GAUGE", "identity")
    fp16kv = (gauge == "fp16kv")
    nwin = int(os.environ.get("EVOQ_NWIN", "151"))
    cfg, tok, smap, model, L, seqlen, nwin, ids, h, rot, mask, pos = em._eval_setup(nwin)
    cdir = em.cache_dir(CARVE_CFG)
    print(f"GAUGE TEST [{gauge}] | cache {cdir} | {nwin} win", flush=True)
    t0 = time.time()
    for li in range(L):
        layer = model.model.layers[li]
        em.materialize_cpu(layer, smap, f"model.layers.{li}.")
        layer.self_attn.rotary_emb = rot
        layer.cuda()
        cw = _sf_load(os.path.join(cdir, f"layer_{li:02d}.safetensors"))
        ka = layer.self_attn.kv_a_proj_with_mqa.weight.detach().float().cpu().numpy()
        kb = layer.self_attn.kv_b_proj.weight.detach().float().cpu().numpy()
        R = Rinv = None
        if not fp16kv:
            R = make_gauge(gauge, ka, kb)
            Rinv = np.linalg.inv(R.astype(np.float64)).astype(np.float32)
            if li == 0:                                  # causal control: FP16 composed map unchanged
                kbn = kb @ Rinv
                diff = np.abs(kbn @ (R @ ka[:D_C]) - kb @ ka[:D_C]).max()
                cn = np.linalg.norm(kbn, axis=0)
                flat = float(np.exp(np.log(cn + 1e-9).mean()) / (cn.mean() + 1e-9))
                print(f"  [check] FP16 composed-map max|diff|={diff:.2e} (should be ~0); "
                      f"kv_b col-norm flatness={flat:.3f} (1=flat)", flush=True)
        for name, mod in layer.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            if name.endswith("kv_a_proj_with_mqa"):
                if fp16kv:
                    mod.weight.data = torch.from_numpy(ka).to(mod.weight.device)
                else:
                    kan = ka.copy(); kan[:D_C] = R @ ka[:D_C]
                    wh, _ = em.trellis_quant(kan, K=2)
                    mod.weight.data = torch.from_numpy(wh).to(mod.weight.device)
            elif name.endswith("kv_b_proj"):
                if fp16kv:
                    mod.weight.data = torch.from_numpy(kb).to(mod.weight.device)
                else:
                    wh, _ = em.trellis_quant(kb @ Rinv, K=2)
                    mod.weight.data = torch.from_numpy(wh).to(mod.weight.device)
            elif name in cw:
                mod.weight.data = cw[name].to(mod.weight.device, torch.float32)
        h = em._run_layer(layer, h, mask, pos)
        em.free_layer(layer); gc.collect(); torch.cuda.empty_cache()
        if li % 9 == 0 or li == L - 1:
            print(f"  layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)
    ppl = em._finish_ppl(h, smap, cfg, ids, seqlen, nwin)
    tag = "kv@fp16 (bottleneck kept 16-bit)" if fp16kv else f"kv@2bit gauge={gauge}"
    line = (f"GAUGE TEST [{gauge}]: {tag} -> ppl={ppl:.4f} (delta vs carve-out {ppl-CARVE:+.4f}) | "
            f"carve-out={CARVE:.4f} fp16={FP16:.4f} | ref: kv@2bit identity gauge = +5.27")
    print("\n" + line, flush=True); em._save(line)


if __name__ == "__main__":
    run()
