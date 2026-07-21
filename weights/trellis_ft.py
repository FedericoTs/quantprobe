"""trellis_ft -- BLOCK-LOCAL fine-tuning of the 2-bit QTIP trellis, on a single GTX 1060 6GB.

The 'make history with low compute' push: 2-bit fine-tuning normally needs datacenter GPUs because you
hold the whole model + activations; BLOCK-LOCAL distillation (one transformer layer at a time) fits 6GB
by construction. We FREEZE the Viterbi trellis assignment (no re-Viterbi in the loop -- 450s/layer makes
that infeasible) and learn the differentiable continuous params:
  * shared codebook C[2^L]  (currently FIXED random-Gaussian -> adapting it to data is the main lever)
  * per-group-row scale gs[out, ng]  (init = gain*std)
  * AWQ per-channel scale s[in]      (init = AWQ scale)
minimizing the block-output distillation loss ||layer_q(X) - layer_fp16(X)||^2 / ||layer_fp16(X)||^2.

Differentiable dequant of a linear:  W_hat = ( FWHT_per128( C[idx] * gs ) / sqrt(G) * signs ) / s
where idx = the frozen Viterbi state path (indexes C); FWHT = matmul by the 128x128 Hadamard (linear,
differentiable); the C[idx] gather scatter-adds gradient into the 4096 codebook entries.

Recipe (feasibility workflow wwtiyhjgs): AdamW betas(0.9,0.95) wd0, lr 1e-3 (gs,s) / 2e-3 (C), K=100
steps/block, renormalize C to unit-std each step (degeneracy guard). Calib = wikitext2_train (disjoint
from the wikitext2_test eval). GATE 0 = FT one block, block-MSE must drop >=30% below frozen-PTQ.

Run:  EVOQ_FT_GATE=0 .venv/Scripts/python -m weights.trellis_ft gate0
"""
from __future__ import annotations
import math, os, sys, gc
import numpy as np, torch
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from weights.quant_sota import _fwht_rows
from weights.qtip_trellis import viterbi_quant, _recons, G, L_BITS, K_RATE
from weights.evoq_llama import (shard_map, read_tensor, meta_model, causal_mask, _rotary, MDIR, LINEAR7)
from transformers import AutoConfig, AutoTokenizer
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _hadamard(n):
    """n x n +-1 Hadamard matrix (Sylvester), n a power of 2. FWHT_rows(M) = M @ H (then /sqrt(n))."""
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


@torch.no_grad()
def frozen_state(W, signs, awq_s, K):
    """Run the PTQ pipeline and return the FROZEN trellis assignment + scales for FT.
    Returns: path [ng,T] int64 (codebook indices), gs0 [ng,1] float (gain*std init), (rows, cols)."""
    rows, cols = W.shape
    Wq = W if awq_s is None else (W * awq_s[None, :]).astype(np.float32)
    N = Wq.reshape(rows, -1, G).reshape(-1, G)
    R = _fwht_rows(N * signs) / math.sqrt(G)
    std = R.std(1, keepdims=True); std[std == 0] = 1.0
    X = (R / std).astype(np.float32)
    Xq, path = viterbi_quant(X, L_BITS, K, return_path=True)              # path = Viterbi states = C indices
    g = (X * Xq).sum(1, keepdims=True) / np.maximum((Xq * Xq).sum(1, keepdims=True), 1e-9)
    gs0 = (g * std).astype(np.float32)
    return path.astype(np.int64), gs0, rows, cols


class DQLinear(torch.nn.Module):
    """Differentiable dequant linear: weight = derotate(C[frozen_path] * gs) / s. Shares C across the layer."""
    def __init__(self, path, gs0, signs, awq0, rows, cols, C, bias, Ht):
        super().__init__()
        ng = rows * (cols // G)
        self.register_buffer("path", torch.from_numpy(path).to(DEV))            # [ng, G] frozen indices
        self.register_buffer("signs", torch.from_numpy(signs.astype(np.float32)).to(DEV))  # [G]
        self.gs = torch.nn.Parameter(torch.from_numpy(gs0).to(DEV))             # [ng,1] learnable
        self.s = torch.nn.Parameter(torch.from_numpy(awq0.astype(np.float32)).to(DEV))  # [cols] learnable
        self.C = C                                                              # shared codebook leaf [ncode]
        self.rows, self.cols, self.ng = rows, cols, cols // G
        self.Ht = Ht                                                            # [G,G] Hadamard
        self.bias = bias

    def weight_hat(self):
        Xq = self.C[self.path]                                                  # [ng_total, G] gather (grad->C)
        Rq = Xq * self.gs                                                       # [ng_total, G]
        back = (Rq @ self.Ht) / math.sqrt(G) * self.signs.view(1, -1)          # de-rotate (FWHT)
        wh = back.reshape(self.rows, self.cols) / self.s.view(1, -1)
        return wh

    def forward(self, x):
        return F.linear(x, self.weight_hat(), self.bias)


def _build_block_ft(li, smap, cfg, AWQ_A=0.5, down_k=3, ncode=None):
    """Materialize layer li (fp16), capture per-linear AWQ scale, freeze trellis state, wrap linears in
    DQLinear sharing one codebook C. Returns (layer, C, list_of_dqlinears)."""
    from weights.quant_lab import _awq_scale
    ncode = ncode or (1 << L_BITS)
    layer = meta_model(cfg).model.layers[li]; layer.to_empty(device=DEV)
    prefix = f"model.layers.{li}."
    for nn_ in ("input_layernorm", "post_attention_layernorm"):
        getattr(layer, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().to(DEV))
    return layer  # caller fills weights + wraps (kept simple; see ft_layer)


@torch.no_grad()
def _calib_hidden(cfg, tok, smap, ntok, seqlen=512):
    """fp16 hidden state entering layer 0, for ntok//seqlen sequences of wikitext2_train (FT calib)."""
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2_train.txt"),
               encoding="utf-8").read()
    ids = tok(txt, return_tensors="pt").input_ids[:, :ntok]
    nseq = ids.shape[1] // seqlen
    ids = ids[:, :nseq * seqlen].reshape(nseq, seqlen)
    h = F.embedding(ids, read_tensor(smap, "model.embed_tokens.weight").float()).to(DEV)
    return h, ids, seqlen, nseq


def ft_layer(li, layer_in, cfg, smap, rot, K_steps=100, lr_c=2e-3, lr_s=1e-3, down_k=3, verbose=True):
    """Block-local FT of ONE decoder layer. layer_in [nseq, seqlen, hid] = fp16 input. Returns the layer
    (with learned DQLinears) + (mse0, mse_final) block-output relative MSE. Pure block-local, fits 6GB."""
    from weights.quant_lab import _awq_scale
    prefix = f"model.layers.{li}."
    seqlen = layer_in.shape[1]
    pos = torch.arange(seqlen, device=DEV).unsqueeze(0); cos_sin = rot(layer_in, pos)
    mask = causal_mask(seqlen).to(DEV)
    # fp16 teacher layer + output
    fp16_layer = meta_model(cfg).model.layers[li]; fp16_layer.to_empty(device=DEV)
    for nn_ in ("input_layernorm", "post_attention_layernorm"):
        getattr(fp16_layer, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().to(DEV))
    n2m = dict(fp16_layer.named_modules()); Wf = {}
    for nm, mod in n2m.items():
        if nm.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
            Wf[nm] = read_tensor(smap, prefix + nm + ".weight").float()
            mod.weight.data.copy_(Wf[nm].to(DEV))
    with torch.no_grad():
        Y = torch.cat([fp16_layer(layer_in[i:i+1], attention_mask=mask, position_embeddings=cos_sin)[0]
                       for i in range(layer_in.shape[0])], 0)             # fp16 teacher output
        # capture per-linear AWQ E|x| via the fp16 forward (hooks)
        eabs = {}
        hooks = [n2m[nm].register_forward_pre_hook(
            (lambda key: (lambda m, inp: eabs.__setitem__(key, inp[0].detach().float().reshape(-1, inp[0].shape[-1]).abs().mean(0))))(nm))
            for nm in Wf]
        fp16_layer(layer_in[0:1], attention_mask=mask, position_embeddings=cos_sin)
        for h in hooks: h.remove()
    # freeze trellis state + wrap with shared codebook C
    Ht = torch.from_numpy(_hadamard(G)).to(DEV)
    C = torch.nn.Parameter(_recons(L_BITS).detach().clone().to(DEV))           # shared codebook leaf
    signs = (np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1)
    dqs = []
    ftlayer = meta_model(cfg).model.layers[li]; ftlayer.to_empty(device=DEV)
    for nn_ in ("input_layernorm", "post_attention_layernorm"):
        getattr(ftlayer, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().to(DEV))
    fn2m = dict(ftlayer.named_modules())
    for nm in Wf:
        W = Wf[nm].numpy()
        awq0 = _awq_scale({nm: eabs[nm].cpu().numpy()}, nm, 0.5).astype(np.float32)
        K = down_k if nm.endswith("down_proj") else K_RATE
        path, gs0, rows, cols = frozen_state(W, signs, awq0, K)
        bias = fn2m[nm].bias.detach() if getattr(fn2m[nm], "bias", None) is not None else None
        dq = DQLinear(path, gs0, signs, awq0, rows, cols, C, bias, Ht).to(DEV)
        dqs.append(dq)
        parent = fn2m[nm.rsplit(".", 1)[0]]; setattr(parent, nm.rsplit(".", 1)[-1], dq)
    opt = torch.optim.AdamW([{"params": [C], "lr": lr_c},
                             {"params": [d.gs for d in dqs] + [d.s for d in dqs], "lr": lr_s}],
                            betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, K_steps, eta_min=1e-5)
    nseq = layer_in.shape[0]; n_ho = max(1, nseq // 4)                          # held-out split (generalization)
    tr = list(range(nseq - n_ho)); ho = list(range(nseq - n_ho, nseq))
    denom = (Y ** 2).mean()

    def block_mse(idx):                                                        # full-batch rel-MSE over seqs idx
        with torch.no_grad():
            e = sum(((ftlayer(layer_in[i:i+1], attention_mask=mask, position_embeddings=cos_sin)[0]
                      - Y[i:i+1]) ** 2).mean() for i in idx) / len(idx)
        return float(e / denom)

    ho_ptq = block_mse(ho)                                                     # held-out MSE at PTQ init (pre-FT)
    mse0 = block_mse(tr)
    import time as _time
    for step in range(K_steps):
        _ts = _time.time(); opt.zero_grad()
        lv = 0.0
        for i in tr:                                                           # grad-ACCUMULATE per seq (peak=1 seq graph)
            l = ((ftlayer(layer_in[i:i+1], attention_mask=mask, position_embeddings=cos_sin)[0]
                  - Y[i:i+1]) ** 2).mean() / denom / len(tr)
            l.backward(); lv += float(l)                                       # backward frees each seq's graph
        opt.step(); sched.step()
        with torch.no_grad():                                                  # degeneracy guard: unit-std C
            C.data = (C.data - C.data.mean()) / C.data.std()
        torch.cuda.synchronize()
        if verbose and (step % 20 == 0 or step == K_steps - 1 or os.environ.get("EVOQ_FT_TIME")):
            print(f"    step {step:3d}  train rel-MSE {lv:.5f}  ({_time.time()-_ts:.1f}s, {len(tr)} seqs)", flush=True)
    loss = lv
    ho_ft = block_mse(ho)
    del fp16_layer; gc.collect(); torch.cuda.empty_cache()
    return ftlayer, Y, mse0, float(loss), ho_ptq, ho_ft                        # Y = fp16 teacher output (advances stream)


def gate0():
    """GATE 0: FT layer 0 (single block) for K steps; PASS if block rel-MSE drops >=30%. Minutes, low heat."""
    cfg = AutoConfig.from_pretrained(MDIR); cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    smap = shard_map(); rot = _rotary(cfg).to(DEV)
    ntok = int(os.environ.get("EVOQ_FT_NTOK", "4096"))
    h, ids, seqlen, nseq = _calib_hidden(cfg, tok, smap, ntok)
    li = int(os.environ.get("EVOQ_FT_LAYER", "0"))
    print(f"GATE 0/1: FT layer {li}, {nseq}x{seqlen} calib tokens (train+held-out), K=100 steps", flush=True)
    _, _Y, mse0, mse1, ho_ptq, ho_ft = ft_layer(li, h, cfg, smap, rot, K_steps=int(os.environ.get("EVOQ_FT_K", "100")))
    tr_drop = (mse0 - mse1) / mse0 * 100
    ho_drop = (ho_ptq - ho_ft) / ho_ptq * 100
    # PASS needs BOTH: train drops (loop works) AND held-out drops (generalizes, not overfitting the tie)
    verdict = ("PASS -> FT loop works AND generalizes (held-out drops) -> proceed to full 7B-FT" if (tr_drop >= 30 and ho_drop >= 15)
               else "OVERFIT WARNING -> train drops but held-out flat/up -> would TIE on ppl; do NOT run full" if (tr_drop >= 30 and ho_drop < 5)
               else f"PARTIAL (train {tr_drop:.0f}%, held-out {ho_drop:.0f}%)")
    print(f"\nGATE: train rel-MSE {mse0:.5f}->{mse1:.5f} ({tr_drop:.1f}% drop) | "
          f"HELD-OUT {ho_ptq:.5f}->{ho_ft:.5f} ({ho_drop:.1f}% drop)\n  {verdict}", flush=True)


def measure7b_ft():
    """FULL block-local FT of Llama-2-7B on the 1060, then WikiText-2 eval. Phase 1: stream the fp16
    teacher; FT each layer block-local (ft_layer); bake the FT'd dequantized weights to disk (D:, ~14GB
    temp; avoids holding 32 layers' index-paths in RAM). Phase 2: stream wikitext2_test through the baked
    FT'd layers -> ppl. The 'make history' run: 2-bit FT end-to-end on a 6GB consumer GPU.
    Env: EVOQ_FT_K(100) EVOQ_DOWN_K(3) EVOQ_FT_NTOK(4096) EVOQ_NWIN(16) EVOQ_FT_QMAX(L)."""
    import time, glob
    cfg = AutoConfig.from_pretrained(MDIR); cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    smap = shard_map(); rot = _rotary(cfg).to(DEV); L = cfg.num_hidden_layers
    K = int(os.environ.get("EVOQ_FT_K", "100")); down_k = int(os.environ.get("EVOQ_DOWN_K", "3"))
    qmax = int(os.environ.get("EVOQ_FT_QMAX", str(L)))                          # FT only first qmax layers (smoke)
    tmp = os.path.join("D:" + os.sep, "evo-compress-data", "ft_tmp"); os.makedirs(tmp, exist_ok=True)
    ntok = int(os.environ.get("EVOQ_FT_NTOK", "4096"))
    h, _ids, seqlen, nseq = _calib_hidden(cfg, tok, smap, ntok)
    print(f"FT PHASE 1: block-local FT {min(qmax,L)}/{L} layers, K={K}, down_k={down_k}, {nseq}x{seqlen} calib", flush=True)
    t0 = time.time()
    for li in range(L):
        prefix = f"model.layers.{li}."
        if li < qmax:
            ftlayer, Y, m0, m1, h0, h1 = ft_layer(li, h, cfg, smap, rot, K_steps=K, lr_c=2e-3, lr_s=1e-3,
                                                  down_k=down_k, verbose=False)
            with torch.no_grad():
                blob = {}
                for nm, mod in ftlayer.named_modules():
                    if isinstance(mod, DQLinear): blob[nm + ".w"] = mod.weight_hat().detach().half().cpu()
                for nn_ in ("input_layernorm", "post_attention_layernorm"):
                    blob[nn_] = getattr(ftlayer, nn_).weight.detach().half().cpu()
            torch.save(blob, os.path.join(tmp, f"layer{li:02d}.pt"))
            h = Y; del ftlayer, Y
            print(f"  FT layer {li:2d}/{L} ({time.time()-t0:.0f}s) train {m0:.4f}->{m1:.4f} | heldout {h0:.4f}->{h1:.4f}", flush=True)
        else:                                                                  # smoke: remaining layers stay fp16
            lay = meta_model(cfg).model.layers[li]; lay.to_empty(device=DEV)
            for nn_ in ("input_layernorm", "post_attention_layernorm"):
                getattr(lay, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().to(DEV))
            n2m = dict(lay.named_modules())
            for nm, mod in n2m.items():
                if nm.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
                    mod.weight.data.copy_(read_tensor(smap, prefix + nm + ".weight").float().to(DEV))
            pos = torch.arange(h.shape[1], device=DEV).unsqueeze(0); cs = rot(h, pos); mk = causal_mask(h.shape[1]).to(DEV)
            with torch.no_grad():
                h = torch.cat([lay(h[i:i+1], attention_mask=mk, position_embeddings=cs)[0] for i in range(h.shape[0])], 0)
            del lay
        gc.collect(); torch.cuda.empty_cache()
    del h
    # PHASE 2: eval on wikitext2_test through the baked FT'd layers
    from weights.wikitext2_ppl import get_wikitext2_test_ids
    seqlen_e, nwin = 2048, int(os.environ.get("EVOQ_NWIN", "16"))
    ids_all = get_wikitext2_test_ids(tok); nwin = min(nwin, ids_all.numel() // seqlen_e)
    ids = ids_all[:, :nwin * seqlen_e].reshape(nwin, seqlen_e)
    emb = read_tensor(smap, "model.embed_tokens.weight").float()
    he = F.embedding(ids, emb).to(DEV); del emb
    pos = torch.arange(seqlen_e, device=DEV).unsqueeze(0); cos_sin = rot(he, pos); mask = causal_mask(seqlen_e).to(DEV)
    print(f"FT PHASE 2: eval {nwin} windows on wikitext2_test", flush=True)
    for li in range(L):
        prefix = f"model.layers.{li}."
        lay = meta_model(cfg).model.layers[li]; lay.to_empty(device=DEV)
        fpath = os.path.join(tmp, f"layer{li:02d}.pt")
        if os.path.exists(fpath):                                              # FT'd layer (baked weights)
            blob = torch.load(fpath, map_location=DEV)
            for nn_ in ("input_layernorm", "post_attention_layernorm"):
                getattr(lay, nn_).weight.data.copy_(blob[nn_].float())
            n2m = dict(lay.named_modules())
            for nm, mod in n2m.items():
                if nm.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
                    mod.weight.data.copy_(blob[nm + ".w"].float())
        else:                                                                  # fp16 (smoke remainder)
            for nn_ in ("input_layernorm", "post_attention_layernorm"):
                getattr(lay, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().to(DEV))
            for nm, mod in dict(lay.named_modules()).items():
                if nm.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
                    mod.weight.data.copy_(read_tensor(smap, prefix + nm + ".weight").float().to(DEV))
        with torch.no_grad():
            he = torch.cat([lay(he[b:b+1], attention_mask=mask, position_embeddings=cos_sin)[0] for b in range(nwin)], 0)
        del lay; gc.collect(); torch.cuda.empty_cache()
    normw = read_tensor(smap, "model.norm.weight").float().to(DEV)
    headw = read_tensor(smap, "lm_head.weight").float().to(DEV)
    v = he.pow(2).mean(-1, keepdim=True); he = (he * torch.rsqrt(v + cfg.rms_norm_eps)) * normw
    nlls, nt = [], 0
    for b in range(nwin):
        logits = he[b] @ headw.T
        nlls.append(F.cross_entropy(logits[:-1].float(), ids[b, 1:].to(DEV)) * (seqlen_e - 1)); nt += seqlen_e - 1
    ppl = torch.exp(torch.stack(nlls).sum() / nt).item()
    tag = f"K={K} down_k={down_k} ftlayers={min(qmax,L)}"
    line = (f"\nLlama-2-7B QTIP-trellis FINE-TUNED (block-local, {tag}): WikiText-2 ppl = {ppl:.4f} "
            f"@ ~2.44b ({nwin} win) | PTQ baseline 6.454 | QuIP#-FT 6.19 | QTIP-FT 5.86 | fp16 5.47")
    print(line, flush=True)
    try:
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trellis_results.txt"), "a",
             encoding="utf-8").write(line + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate0"
    {"gate0": gate0, "measure7b_ft": measure7b_ft}.get(mode, gate0)()
