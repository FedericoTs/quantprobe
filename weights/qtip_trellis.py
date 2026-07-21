"""QTIP fixed-rate bitshift trellis -> Llama-2-7B / WikiText-2 head-to-head (PTQ target ~6.8 vs
champion 9.63; 5.86 needs fine-tuning we can't do on a 1060). Preflight PASSED 3/3 (D_trellis 0.069
= Gaussian D(R=2) bound, ratio 0.59 vs scalar). This does the full-model PPL.

trellis_quant(W): per-group signed-FWHT, normalize each group-row to UNIT STD, fixed-rate (L,K=2,V=1)
bitshift Viterbi quantize the T=128 within-group sequence against a conservative random-Gaussian
unit-std code (3INST only improves), reconstruct, de-rotate. NO outliers (QTIP-parity). Honest
b/w = K + 16/G (std fp16/group) ~= 2.125. L=12 suffices (preflight hit the D(R) bound at L=12).

measure_7b: stream Llama-2-7B layers (6GB-safe), quantize each of the 7 linears on the fly, eval
WikiText-2 (standard seqlen-2048 protocol, fp16 ref 5.47).

Run:  python -m weights.qtip_trellis preflight | mse05 | measure7b   [EVOQ_TL=12 EVOQ_NWIN=16]
"""
from __future__ import annotations
import math, os, sys, time, gc
import numpy as np, torch
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from weights.quant_sota import _fwht_rows
from weights.evoq_llama import (shard_map, read_tensor, meta_model, materialize_layer,
                                causal_mask, _rotary, MDIR)
from transformers import AutoConfig, AutoTokenizer

G = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"
L_BITS = int(os.environ.get("EVOQ_TL", "12"))
K_RATE = 2


def _recons_3inst(L):
    """QTIP's real 3INST computed code: LCG -> mask/xor -> two fp16-reinterpreted halves summed ~N(0,1).
    Properly Gaussian + tail-covering (vs random-LUT gaps), zero-mean-normalized. Lookup-free on sm_61."""
    state = np.arange(1 << L, dtype=np.uint64)
    x = (state * np.uint64(89226354) + np.uint64(64248484)) & np.uint64(0xFFFFFFFF)
    m = ((1 << 15) | ((1 << 12) - 1)); m = (m << 16) | m
    bits = (x & np.uint64(m)) ^ np.uint64(996162400)
    top = (bits >> np.uint64(16)).astype(np.uint16).view(np.float16).astype(np.float32)
    bot = (bits & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float32)
    r = top + bot
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)     # fp16 inf/nan guard
    r = r - r.mean()
    return torch.tensor(r / r.std()).to(DEV)


def _recons(L, seed=0):
    if os.environ.get("EVOQ_CODE", "random") == "3inst":
        return _recons_3inst(L)
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(1 << L, generator=g).to(DEV)
    r = r - r.mean()                                          # ZERO-MEAN (DC bias compounds over 32 layers -> divergence)
    return r / r.std()


@torch.no_grad()
def viterbi_quant(Xnp, L=L_BITS, K=K_RATE, seed=0, chunk=int(os.environ.get("EVOQ_VCHUNK", "512")),
                  return_path=False):
    """Fixed-rate bitshift Viterbi over T=128 sequences (rows of Xnp [B,128], unit-std).
    Returns reconstruction Xq [B,128]. Verified indexing (de Bruijn predecessors).
    return_path=True also returns the chosen L-bit STATE path [B,T] (uint16) -- this is what the
    deployable trellis_run decoder serializes into the 2-bit bitstream (state low-K bits = symbol)."""
    nstate = 1 << L; KV = K; ncand = 1 << KV; npred = 1 << (L - KV)
    recons = _recons(L, seed)
    sumdelta = (torch.arange(ncand, device=DEV) << (L - KV)).view(1, -1)
    base = (torch.arange(nstate, device=DEV) >> KV)[::ncand].view(-1, 1)
    state_cand = base + sumdelta                              # [npred, ncand]
    B, T = Xnp.shape
    out = np.empty((B, T), np.float32)
    path = np.empty((B, T), np.uint16) if return_path else None
    for c0 in range(0, B, chunk):
        Xt = torch.tensor(Xnp[c0:c0 + chunk].T, device=DEV)   # [T, b]
        b = Xt.shape[1]
        cost = (recons.view(1, -1) - Xt[0].view(-1, 1)) ** 2  # [b, nstate]
        if os.environ.get("EVOQ_FIXED_START"):                # shift-register starts at 0 -> state_0 in {0..ncand-1}
            cost[:, ncand:] = 1e30                            # => decoder needs NO stored init (payload honestly K bits)
        back = torch.empty(T, b, npred, dtype=torch.int16, device=DEV)
        for t in range(1, T):
            se = (recons.view(1, -1) - Xt[t].view(-1, 1)) ** 2
            bestv, besti = cost[:, state_cand].min(-1)        # [b, npred]
            back[t] = besti.to(torch.int16)
            cost = se + bestv.repeat_interleave(ncand, dim=1)
        final = torch.empty(T, b, dtype=torch.long, device=DEV)
        final[T - 1] = cost.argmin(-1)
        for t in range(T - 1, 0, -1):
            grp = final[t] >> KV
            chosen = back[t].gather(1, grp.view(-1, 1)).squeeze(1).long()
            final[t - 1] = state_cand[grp, chosen]
        out[c0:c0 + chunk] = recons[final].T.cpu().numpy()
        if return_path:
            path[c0:c0 + chunk] = final.T.to(torch.int32).cpu().numpy().astype(np.uint16)
    return (out, path) if return_path else out


def trellis_quant(W, L=L_BITS, seed=0, p_out=0.0, awq_s=None, gain=True, K=K_RATE):
    """rotate -> unit-std/group-row -> Viterbi -> de-rotate. Options to handle Llama's activation
    outliers: awq_s (AWQ per-channel scale, applied before rotation, undone after); p_out (0.5%% exact
    outliers); gain (per-group-row variance-matching to undo quantization variance-shrinkage).
    K = trellis rate (bits/weight; 2=default 2-bit, 3=3-bit) for per-tensor mixed-precision allocation."""
    if os.environ.get("EVOQ_IDENTITY"):
        return W.astype(np.float32), 16.0
    rows, cols = W.shape; ng = cols // G
    Wq = W if awq_s is None else (W * awq_s[None, :]).astype(np.float32)
    if p_out > 0:
        thr = np.quantile(np.abs(Wq), 1.0 - p_out); mask = np.abs(Wq) >= thr
        base = Wq.copy(); base[mask] = 0.0
    else:
        mask = None; base = Wq
    signs = (np.random.default_rng(seed).integers(0, 2, G).astype(np.float32) * 2 - 1)
    N = base.reshape(rows, -1, G).reshape(-1, G)
    R = _fwht_rows(N * signs) / math.sqrt(G)
    std = R.std(1, keepdims=True); std[std == 0] = 1.0
    X = (R / std).astype(np.float32)
    Xq = viterbi_quant(X, L, K)
    g = ((X * Xq).sum(1, keepdims=True) / np.maximum((Xq * Xq).sum(1, keepdims=True), 1e-9)
         if gain else np.ones((X.shape[0], 1), np.float32))      # per-group-row variance-match gain
    gs = (g * std).astype(np.float32)                            # combined per-group-row scale (side-info)
    gs_bits = 16.0
    if os.environ.get("EVOQ_INT8_GS"):                           # int8 side-info (per-tensor affine, ~0.4% step)
        lo, hi = float(gs.min()), float(gs.max())
        gsq = np.round((gs - lo) / (hi - lo + 1e-12) * 255.0)
        gs = (lo + gsq / 255.0 * (hi - lo)).astype(np.float32)
        gs_bits = 8.0
    Rq = Xq * gs
    back = _fwht_rows(Rq) / math.sqrt(G) * signs
    wh = back.reshape(rows, cols).astype(np.float32)
    if mask is not None:
        wh[mask] = Wq[mask]
    if awq_s is not None:
        wh = (wh / awq_s[None, :]).astype(np.float32)
    bw = K + gs_bits / G + (p_out * (32 + 16) if p_out > 0 else 0.0) + (16.0 / rows if awq_s is not None else 0.0)
    return wh, bw


@torch.no_grad()
def companding_probe():
    """OUTPUT-AWARE headroom probe (cheap, no full run). On REAL Llama-7B activations for the hard
    linears, ask: does output-aware tuning beat our current trellis+AWQ(0.5)+gain+outliers config?
    Tests (a) AWQ-alpha sweep, (b) an OUTPUT-SPACE per-output-row gain (closed-form, +16b/row side-info)
    vs the current WEIGHT-space variance gain. This is the cheap GLVQ-companding-direction signal:
    if best output-MSE << baseline -> there's real headroom for learned companding -> commit a 7B run.
    If not -> the codec is output-optimal for this lever -> pivot to second-model validation."""
    from weights.evoq_llama import LINEAR7
    from weights.quant_lab import _awq_scale
    import torch.nn.functional as F
    cfg = AutoConfig.from_pretrained(MDIR); cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    model = meta_model(cfg); rot = _rotary(cfg).cuda(); smap = shard_map()
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2_train.txt"),
               encoding="utf-8").read()
    ids = tok(txt, return_tensors="pt").input_ids[:, :512]
    h = F.embedding(ids, read_tensor(smap, "model.embed_tokens.weight").float()).cuda()
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0); cos_sin = rot(h, pos)
    mask = causal_mask(ids.shape[1]).cuda()
    TGT = {16: ("down_proj", "gate_proj"), 8: ("down_proj",)}
    grabbed = {}
    for li in range(17):
        prefix = f"model.layers.{li}."; layer = model.model.layers[li]; layer.to_empty(device="cuda")
        for nn_ in ("input_layernorm", "post_attention_layernorm"):
            getattr(layer, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().cuda())
        n2m = dict(layer.named_modules()); hooks = []
        if li in TGT:
            def mk(key):
                def hk(m, inp): grabbed[key] = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu().numpy()
                return hk
            for p in TGT[li]:
                for nm, mod in n2m.items():
                    if nm.endswith(p): hooks.append(mod.register_forward_pre_hook(mk(f"{li}.{p}")))
        for nm, mod in n2m.items():
            if nm.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
                mod.weight.data.copy_(read_tensor(smap, prefix + nm + ".weight").float().cuda())
        h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        for hk in hooks: hk.remove()
        layer.to_empty(device="meta"); gc.collect()

    def omse(W, wh, Xt):                                   # output-MSE on real activations (GPU)
        Wt = torch.from_numpy(W).cuda(); wt = torch.from_numpy(wh).cuda()
        ref = Xt @ Wt.T; ap = Xt @ wt.T
        return float(((ref - ap) ** 2).mean()), ref, ap

    print(f"{'tensor':<16}{'a=0':>8}{'a.25':>8}{'a.5(cur)':>9}{'a.75':>8}{'a=1':>8}{'+out-gain':>11}  (out-MSE x1e3)")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for key, X in grabbed.items():
        li, p = key.split(".")
        W = read_tensor(smap, f"model.layers.{li}.mlp.{p}.weight").float().numpy()[:1024]   # row-subsample (cheap, representative out-MSE)
        eabs = np.abs(X).mean(0); Xt = torch.from_numpy(X).cuda()
        row = []
        whs = {}
        for a in alphas:
            s = _awq_scale({p: eabs}, p, a).astype(np.float32)
            wh, _ = trellis_quant(W, p_out=0.005, awq_s=s)
            m, ref, ap = omse(W, wh, Xt); row.append(m * 1e3); whs[a] = (wh, ref, ap)
        # output-space per-output-row gain on the current (a=0.5) config: c_i = <ref_i,ap_i>/<ap_i,ap_i>
        wh5, ref, ap = whs[0.5]
        c = (ref * ap).sum(0) / torch.clamp((ap * ap).sum(0), min=1e-9)     # [out]
        ogain_mse = float(((ref - ap * c.view(1, -1)) ** 2).mean()) * 1e3
        best_a = alphas[int(np.argmin(row))]
        print(f"{key:<16}" + "".join(f"{v:>8.4f}" if i != 2 else f"{v:>9.4f}" for i, v in enumerate(row))
              + f"{ogain_mse:>11.4f}   best-a={best_a}", flush=True)
    print("\nVERDICT: if a!=0.5 or +out-gain gives out-MSE meaningfully < the a=0.5(cur) column -> "
          "output-aware headroom exists -> build learned per-channel companding + 7B run. "
          "If a=0.5 ~ best and out-gain ~ no-op -> codec is output-optimal -> pivot to 2nd-model validation.")


@torch.no_grad()
def mse_out7b():
    """Diagnose the 7B failure: capture REAL Llama-7B activations (stream first layers), compare
    output-MSE of trellis variants [plain / +gain / +AWQ / +AWQ+out] vs champion(scalar+AWQ+out)."""
    from weights.evoq_llama import LINEAR7
    from weights.quant_lab import _awq_scale
    from weights.noise_shaping import champion_codec, out_mse
    import torch.nn.functional as F
    cfg = AutoConfig.from_pretrained(MDIR); cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    model = meta_model(cfg); rot = _rotary(cfg).cuda(); smap = shard_map()
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2_train.txt"),
               encoding="utf-8").read()
    ids = tok(txt, return_tensors="pt").input_ids[:, :512]
    h = F.embedding(ids, read_tensor(smap, "model.embed_tokens.weight").float()).cuda()
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0); cos_sin = rot(h, pos)
    mask = causal_mask(ids.shape[1]).cuda()
    TGT = {16: ("down_proj", "gate_proj"), 8: ("down_proj",)}
    grabbed = {}
    for li in range(17):
        prefix = f"model.layers.{li}."; layer = model.model.layers[li]; layer.to_empty(device="cuda")
        for nn_ in ("input_layernorm", "post_attention_layernorm"):
            getattr(layer, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().cuda())
        n2m = dict(layer.named_modules()); hooks = []
        if li in TGT:
            def mk(key):
                def hk(m, inp): grabbed[key] = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu().numpy()
                return hk
            for p in TGT[li]:
                for nm, mod in n2m.items():
                    if nm.endswith(p): hooks.append(mod.register_forward_pre_hook(mk(f"{li}.{p}")))
        for nm, mod in n2m.items():
            if nm.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
                mod.weight.data.copy_(read_tensor(smap, prefix + nm + ".weight").float().cuda())
        h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        for hk in hooks: hk.remove()
        layer.to_empty(device="meta"); gc.collect()
    print(f"{'tensor':<18}{'champ(s+awq+o)':>15}{'trel':>9}{'+gain':>9}{'+awq':>9}{'+awq+o':>10}  (out-MSE x1e3)")
    for key, X in grabbed.items():
        li, p = key.split("."); W = read_tensor(smap, f"model.layers.{li}.mlp.{p}.weight").float().numpy()
        s = _awq_scale({p: np.abs(X).mean(0)}, p, 0.5).astype(np.float32)
        whc, _ = champion_codec(W, p, {p: X}, {p: np.abs(X).mean(0)}, 0.03, False)
        configs = {"trel": trellis_quant(W, gain=False)[0], "+gain": trellis_quant(W, gain=True)[0],
                   "+awq": trellis_quant(W, awq_s=s, gain=True)[0], "+awq+o": trellis_quant(W, awq_s=s, p_out=0.005, gain=True)[0]}
        vals = {k: out_mse(W, v, X) * 1e3 for k, v in configs.items()}
        mc = out_mse(W, whc, X) * 1e3
        print(f"{key:<18}{mc:>15.4f}{vals['trel']:>9.4f}{vals['+gain']:>9.4f}{vals['+awq']:>9.4f}{vals['+awq+o']:>10.4f}", flush=True)
    print("\nVERDICT: pick the trellis config with out-MSE <= champion; that's the fix for the 7B run.")


def mse_out():
    """CHEAP output-MSE gate on REAL 0.5B activations (the metric that predicts ppl, which I skipped).
    Compares champion(scalar+outliers) vs trellis-no-outlier vs trellis+0.5%-outlier. Decides whether
    outliers rescue the trellis before committing a 4h 7B re-run."""
    from transformers import AutoTokenizer
    from weights.quant_lab import CFG, build_model, load_fp16, calibrate
    from weights.noise_shaping import capture, champion_codec, out_mse
    tok = AutoTokenizer.from_pretrained(CFG); m = build_model(); load_fp16(m); calib = calibrate(m, tok)
    Lm = m.config.num_hidden_layers
    names = [f"model.layers.{Lm//2}.mlp.down_proj", f"model.layers.{Lm//2}.mlp.gate_proj",
             f"model.layers.{Lm//4}.mlp.down_proj"]
    acts = capture(m, tok, names); sd = m.state_dict()
    print(f"{'tensor':<24}{'scalar+out':>12}{'trel-noout':>12}{'trel+out':>11}  (output-MSE x1e3)")
    print("-" * 60)
    for nm in names:
        W = sd[nm + ".weight"].float().numpy(); X = acts[nm]
        whs, _ = champion_codec(W, nm + ".weight", acts, calib, 0.03, False)   # scalar ECVQ + outliers ~2b
        wt0, _ = trellis_quant(W, p_out=0.0)
        wt1, _ = trellis_quant(W, p_out=0.005)
        ms = out_mse(W, whs, X) * 1e3; m0 = out_mse(W, wt0, X) * 1e3; m1 = out_mse(W, wt1, X) * 1e3
        print(f"{nm.split('.',2)[-1][:23]:<24}{ms:>12.4f}{m0:>12.4f}{m1:>11.4f}", flush=True)
    print("\nVERDICT: if trel+out << trel-noout and <= scalar+out -> outliers rescue the trellis; "
          "re-run 7B with p_out=0.005. If trel+out still >> scalar+out -> codebook (random-Gaussian) "
          "is the problem -> need 3INST or range-matched code.")


def mse05():
    """quick weight-MSE sanity on 0.5B (vs the 7B preflight) + a single-tensor scalar baseline."""
    from weights.quant_lab import WPATH
    from safetensors import safe_open
    from weights.codec_zoo import _ecvq_levels, _nearest_idx
    with safe_open(WPATH, framework="pt") as f:
        W = f.get_tensor("model.layers.12.mlp.down_proj.weight").float().numpy()
    rows, cols = W.shape; ng = cols // G
    signs = (np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1)
    R = _fwht_rows(W.reshape(rows, -1, G).reshape(-1, G) * signs) / math.sqrt(G)
    std = R.std(1, keepdims=True); std[std == 0] = 1; X = (R / std).astype(np.float32)
    samp = X.ravel()[np.random.default_rng(1).integers(0, X.size, 20000)]
    lv = _ecvq_levels(samp, 4, 0.0); Ds = float(np.mean((X.ravel() - lv[_nearest_idx(X.ravel(), lv)]) ** 2))
    Xq = viterbi_quant(X[:4096]); Dt = float(((X[:4096] - Xq) ** 2).mean())
    print(f"0.5B down_proj: D_scalar {Ds:.4f}  D_trellis {Dt:.4f}  ratio {Dt/Ds:.3f}")


def measure05():
    """CHEAP 0.5B held-out ppl of the FIXED trellis (zero-mean code + outliers), vs champion-scalar+out
    @2.16b=60.35. Validates the fix before a 4h 7B re-run."""
    from transformers import AutoTokenizer
    from weights.quant_lab import CFG, build_model, load_fp16, ppl, load_quant
    tok = AutoTokenizer.from_pretrained(CFG); m = build_model(); load_fp16(m)
    fp16 = ppl(m, tok); load_fp16(m)
    for p_out in (0.0, 0.005):
        load_fp16(m)
        bpw = load_quant(m, lambda a, k: (lambda wh, bw: (wh, bw * a.size))(*trellis_quant(a, p_out=p_out)))
        pp = ppl(m, tok)
        print(f"0.5B trellis p_out={p_out}: ppl {pp:.4f} @ {bpw:.3f}b  (fp16 {fp16:.4f}; champ-scalar+out@2.16b=60.35)", flush=True)


@torch.no_grad()
def measure7b():
    """Streaming Llama-2-7B trellis ppl with OPTIONAL output bias-correction (EVOQ_BC=1, the fix for the
    linear error-compounding) + AWQ (EVOQ_AWQ=1). Carries a calib hidden state to compute per-linear
    E[x] (signed mean, for bias = (W-Wq)@E[x]) and E|x| (for AWQ) from quantized-prev inputs (sequential)."""
    from weights.evoq_llama import LINEAR7
    from weights.wikitext2_ppl import get_wikitext2_test_ids
    from weights.quant_lab import _awq_scale
    import torch.nn.functional as F
    cfg = AutoConfig.from_pretrained(MDIR); cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    model = meta_model(cfg); rot = _rotary(cfg).cuda(); L = cfg.num_hidden_layers
    smap = shard_map()
    BC = bool(int(os.environ.get("EVOQ_BC", "0"))); AWQ = bool(int(os.environ.get("EVOQ_AWQ", "0")))
    OGAIN = bool(int(os.environ.get("EVOQ_OGAIN", "0")))            # output-space per-output-row gain (calib)
    AWQ_A = float(os.environ.get("EVOQ_AWQ_ALPHA", "0.5"))
    DOWN_K = int(os.environ.get("EVOQ_DOWN_K", str(K_RATE)))         # mixed-precision: down_proj trellis rate (2 or 3)
    ALLOC = None                                                     # per-tensor water-fill allocation {name:{K,p_out}}
    if os.environ.get("EVOQ_ALLOC"):
        import json
        ALLOC = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", os.environ["EVOQ_ALLOC"])))
    p_out = float(os.environ.get("EVOQ_POUT", "0")); qmax = int(os.environ.get("EVOQ_QMAX", "99"))
    seqlen, nwin = 2048, int(os.environ.get("EVOQ_NWIN", "16"))
    ids_all = get_wikitext2_test_ids(tok); nwin = min(nwin, ids_all.numel() // seqlen)
    ids = ids_all[:, :nwin * seqlen].reshape(nwin, seqlen)
    emb = read_tensor(smap, "model.embed_tokens.weight").float()
    h = F.embedding(ids, emb).cuda()
    # calib hidden state (disjoint wikitext train) for bias/AWQ stats
    ctxt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2_train.txt"), encoding="utf-8").read()
    cids = tok(ctxt, return_tensors="pt").input_ids[:, :512]
    hc = F.embedding(cids, emb).cuda() if (BC or AWQ or OGAIN) else None; del emb
    pos = torch.arange(seqlen, device="cuda").unsqueeze(0); cos_sin = rot(h, pos)
    mask = causal_mask(seqlen).cuda()
    cpos = torch.arange(cids.shape[1], device="cuda").unsqueeze(0) if hc is not None else None
    ccs = rot(hc, cpos) if hc is not None else None
    cmask = causal_mask(cids.shape[1]).cuda() if hc is not None else None
    t0 = time.time(); bws = []; qsize = 0
    for li in range(L):
        prefix = f"model.layers.{li}."
        layer = model.model.layers[li]; layer.to_empty(device="cuda")
        for nn_ in ("input_layernorm", "post_attention_layernorm"):
            getattr(layer, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().cuda())
        name2mod = dict(layer.named_modules())
        Wf = {}
        for name, mod in name2mod.items():
            if name.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
                Wf[name] = read_tensor(smap, prefix + name + ".weight").float()
                mod.weight.data.copy_(Wf[name].cuda())          # load fp16 first
        ex, eabs, xfull = {}, {}, {}
        if hc is not None:                                      # capture E[x], E|x| (and full x for OGAIN) from calib
            hooks = []
            def mk(nm):
                def hk(m, inp):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    ex[nm] = x.mean(0); eabs[nm] = x.abs().mean(0)
                    if OGAIN: xfull[nm] = x                     # [Ntok, in] for output-space gain
                return hk
            for name, mod in name2mod.items():
                if name in Wf: hooks.append(mod.register_forward_pre_hook(mk(name)))
            layer(hc, attention_mask=cmask, position_embeddings=ccs)
            for hk in hooks: hk.remove()
        for name, mod in name2mod.items():                      # quantize
            if name not in Wf: continue
            W = Wf[name].numpy()
            if li < qmax:
                s = (_awq_scale({name: eabs[name].cpu().numpy()}, name, AWQ_A).astype(np.float32) if AWQ else None)
                if ALLOC is not None:                                  # per-tensor water-fill allocation
                    a = ALLOC.get(f"{li}.{name}", {"K": K_RATE, "p_out": p_out}); kt, pt = a["K"], a["p_out"]
                else:                                                  # else mixed-precision down_proj rule / uniform
                    kt = DOWN_K if name.endswith("down_proj") else K_RATE; pt = p_out
                wh, bw = trellis_quant(W, p_out=pt, awq_s=s, K=kt); bws.append(bw * W.size); qsize += W.size
            else:
                wh = W.astype(np.float32)
            wht = torch.from_numpy(wh).cuda()
            if OGAIN and li < qmax:                             # output-space per-output-row gain c_i (calib): +16b/row
                xc = xfull[name]; Wt = Wf[name].cuda()
                ref = xc @ Wt.T; ap = xc @ wht.T                # [Ntok, out]
                c = (ref * ap).sum(0) / torch.clamp((ap * ap).sum(0), min=1e-9)
                wht = wht * c.view(-1, 1)
                del Wt, ref, ap
            mod.weight.data.copy_(wht)
            if BC and li < qmax:                                # bias correction: b = (W - wh) @ E[x]
                b = (Wf[name].cuda() - mod.weight.data) @ ex[name]
                mod.bias = torch.nn.Parameter(b.detach())
        h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        if hc is not None: hc = layer(hc, attention_mask=cmask, position_embeddings=ccs)[0]
        layer.to_empty(device="meta"); gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0: print(f"  layer {li}/{L} ({time.time()-t0:.0f}s) BC={BC} AWQ={AWQ}", flush=True)
    normw = read_tensor(smap, "model.norm.weight").float().cuda()
    headw = read_tensor(smap, "lm_head.weight").float().cuda()
    v = h.pow(2).mean(-1, keepdim=True); h = (h * torch.rsqrt(v + cfg.rms_norm_eps)) * normw
    nlls, ntok = [], 0
    for b in range(nwin):
        logits = h[b] @ headw.T
        nlls.append(F.cross_entropy(logits[:-1].float(), ids[b, 1:].cuda()) * (seqlen - 1)); ntok += seqlen - 1
    ppl = torch.exp(torch.stack(nlls).sum() / ntok).item()
    bw_nom = (sum(bws) / qsize) if qsize else (K_RATE + 16.0 / G + (p_out * 48 if p_out else 0.0))  # real weighted avg b/w
    line = (f"\nLlama-2-7B QTIP-trellis L={L_BITS} BC={BC} AWQ={AWQ}(a={AWQ_A}) OGAIN={OGAIN} DOWN_K={DOWN_K} pout={p_out}: "
            f"WikiText-2 ppl = {ppl:.4f} @ ~{bw_nom:.3f}b ({nwin} win) | vs champion 9.63 | QTIP-noFT ~6.8 | fp16 5.47")
    print(line, flush=True)
    try:                                                          # never lose a 4h result to a print bug again
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trellis_results.txt"), "a",
             encoding="utf-8").write(line + "\n")
    except Exception:
        pass


@torch.no_grad()
def sens_db():
    """Build the per-tensor SENSITIVITY DB for mixed-precision water-fill (cycle-21 plan). For every
    linear (all 32 layers x 7 types), stream fp16, capture real calib activations, quantize a ROW-SUBSAMPLE
    at K=2 (+AWQ), and record outmse2 = mean_calib ||(W-Wq_K2)@x||^2. Since all tensors sit at the D(R)
    bound, the K2->K3 gain ∝ 0.75*outmse2, so outmse2 IS the sensitivity ranking. Writes data/sens_db.json
    {name: {outmse2, numel, type, layer}}. Row-subsample (EVOQ_SROWS, default 512) keeps it ~1-2h.
    Run: EVOQ_AWQ=1 .venv/Scripts/python -m weights.qtip_trellis sens_db"""
    import json
    from weights.evoq_llama import LINEAR7
    from weights.quant_lab import _awq_scale
    import torch.nn.functional as F
    SROWS = int(os.environ.get("EVOQ_SROWS", "512")); AWQ = bool(int(os.environ.get("EVOQ_AWQ", "1")))
    cfg = AutoConfig.from_pretrained(MDIR); cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(MDIR, use_fast=False)
    model = meta_model(cfg); rot = _rotary(cfg).cuda(); smap = shard_map(); L = cfg.num_hidden_layers
    ctxt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2_train.txt"), encoding="utf-8").read()
    cids = tok(ctxt, return_tensors="pt").input_ids[:, :512]
    h = F.embedding(cids, read_tensor(smap, "model.embed_tokens.weight").float()).cuda()
    pos = torch.arange(cids.shape[1], device="cuda").unsqueeze(0); cos_sin = rot(h, pos)
    mask = causal_mask(cids.shape[1]).cuda()
    db = {}; t0 = time.time()
    for li in range(L):
        prefix = f"model.layers.{li}."; layer = model.model.layers[li]; layer.to_empty(device="cuda")
        for nn_ in ("input_layernorm", "post_attention_layernorm"):
            getattr(layer, nn_).weight.data.copy_(read_tensor(smap, prefix + nn_ + ".weight").float().cuda())
        n2m = dict(layer.named_modules()); Wf = {}; xcap = {}
        for nm, mod in n2m.items():
            if nm.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
                Wf[nm] = read_tensor(smap, prefix + nm + ".weight").float()
                mod.weight.data.copy_(Wf[nm].cuda())
        hooks = []
        def mk(nm):
            def hk(m, inp): xcap[nm] = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            return hk
        for nm in Wf: hooks.append(n2m[nm].register_forward_pre_hook(mk(nm)))
        h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        for hk in hooks: hk.remove()
        for nm in Wf:
            Wfull = Wf[nm].numpy(); W = Wfull[:SROWS]
            s = (_awq_scale({nm: xcap[nm].abs().mean(0).cpu().numpy()}, nm, 0.5).astype(np.float32) if AWQ else None)
            wh, _ = trellis_quant(W, K=2, awq_s=s, p_out=0.0)
            xc = xcap[nm]; dW = (torch.from_numpy(W).cuda() - torch.from_numpy(wh).cuda())
            outmse2 = float(((xc @ dW.T) ** 2).mean())
            db[f"{li}.{nm}"] = {"outmse2": outmse2, "numel": int(Wfull.size),
                                "type": nm.split(".")[-1], "layer": li, "rows": int(Wfull.shape[0])}
        layer.to_empty(device="meta"); gc.collect(); torch.cuda.empty_cache()
        if li % 7 == 0: print(f"  sens layer {li}/{L} ({time.time()-t0:.0f}s)", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sens_db.json")
    json.dump(db, open(path, "w"), indent=1)
    rank = sorted(db.items(), key=lambda kv: -kv[1]["outmse2"])
    print(f"\nsens_db -> {path} ({len(db)} tensors). TOP-10 most sensitive (outmse2 x1e3):")
    for k, v in rank[:10]: print(f"  {k:<28} {v['outmse2']*1e3:8.4f}  ({v['type']})", flush=True)
    print("BOTTOM-5 (least sensitive):")
    for k, v in rank[-5:]: print(f"  {k:<28} {v['outmse2']*1e3:8.4f}  ({v['type']})", flush=True)


def make_alloc(target_bw=2.37, out_name=None):
    """Greedy water-fill from data/sens_db.json -> data/alloc_<bw>.json {name:{K,p_out}}.
    Start all at K=2,p_out=0 (2.125 b/w base). Two levers, gain ∝ 0.75*outmse2 per the D(R)-bound argument:
      - K2->K3: cost 1.0*numel bits, gain (K-bump) ~ 0.75*outmse2
      - p_out 0->0.005: cost 0.24*numel bits, gain (outlier) ~ smaller; only fund on top-sensitivity tensors
    Greedy by gain/cost, skip-and-continue until the avg-b/w budget is hit."""
    import json
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    db = json.load(open(os.path.join(base, "sens_db.json")))
    total = sum(v["numel"] for v in db.values())
    budget_bits = (target_bw - (2.0 + 16.0 / G)) * total       # bits above the K=2,p_out=0 base (2.125)
    alloc = {k: {"K": 2, "p_out": 0.0} for k in db}
    # K3 only on eligible types: local-output-MSE is a valid sensitivity rank WITHIN the residual-path
    # FFN (down/gate/up) but NOT vs softmax-attenuated attention -> default-restrict to FFN, and only
    # give outliers to eligible tensors (attention outliers are low ppl-value).
    k3_types = set(os.environ.get("EVOQ_K3_TYPES", "down_proj,gate_proj,up_proj").split(","))
    cands = []
    for k, v in db.items():
        if v["type"] not in k3_types:
            continue                                           # ineligible: stays K=2, p_out=0
        g = 0.75 * v["outmse2"]                                # K-bump gain proxy (whole-tensor)
        cands.append((g / (1.0 * v["numel"]), 1.0 * v["numel"], k, "K3", g))
        cands.append((0.35 * v["outmse2"] / (0.24 * v["numel"]), 0.24 * v["numel"], k, "OUT", 0))  # outlier ~half a K-bump
    cands.sort(key=lambda c: -c[0])
    spent = 0.0
    for gpb, cost, k, lever, _ in cands:
        if spent + cost > budget_bits:
            continue                                           # skip-and-continue (let cheap tensors fill tail)
        if lever == "K3" and alloc[k]["K"] == 2:
            alloc[k]["K"] = 3; spent += cost
        elif lever == "OUT" and alloc[k]["p_out"] == 0.0:
            alloc[k]["p_out"] = 0.005; spent += cost
    realized = (2.0 + 16.0 / G) + spent / total
    nK3 = sum(1 for a in alloc.values() if a["K"] == 3); nout = sum(1 for a in alloc.values() if a["p_out"] > 0)
    out_name = out_name or f"alloc_{target_bw:.2f}.json".replace(".", "p", 1)
    json.dump(alloc, open(os.path.join(base, out_name), "w"), indent=1)
    print(f"alloc -> {out_name}: target {target_bw:.3f} realized ~{realized:.3f} b/w | "
          f"{nK3}/{len(alloc)} tensors @K=3, {nout} w/ outliers")
    from collections import Counter
    print("  K=3 by type:", dict(Counter(a_k.split('.')[-1] for a_k, a in alloc.items() if a["K"] == 3)))


@torch.no_grad()
def measure7b_qwen():
    """GENERALITY test: the SAME trellis codec on Qwen2.5-7B (different arch family: GQA, attention
    QKV bias, RoPE theta 1e6, FFN 18944) / WikiText-2, to show the result isn't Llama-specific.
    Reuses evoq_7b's Qwen streaming infra + materialize_layer (loads biases too). QMAX=0 = fp16 baseline
    (also a cheap full-pipeline smoke). Env knobs same as measure7b (EVOQ_AWQ/POUT/OGAIN/AWQ_ALPHA/QMAX/NWIN)."""
    from weights.evoq_7b import (shard_map as q_shard_map, read_tensor as q_read_tensor,
                                 meta_model as q_meta_model, materialize_layer as q_materialize, MDIR as Q_MDIR,
                                 causal_mask as q_causal_mask)
    from weights.wikitext2_ppl import get_wikitext2_test_ids
    from weights.quant_lab import _awq_scale
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    import torch.nn.functional as F
    LINEAR7 = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    AWQ = bool(int(os.environ.get("EVOQ_AWQ", "0"))); OGAIN = bool(int(os.environ.get("EVOQ_OGAIN", "0")))
    AWQ_A = float(os.environ.get("EVOQ_AWQ_ALPHA", "0.5")); p_out = float(os.environ.get("EVOQ_POUT", "0"))
    qmax = int(os.environ.get("EVOQ_QMAX", "99"))
    cfg = AutoConfig.from_pretrained(Q_MDIR); cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(Q_MDIR); smap = q_shard_map()
    model = q_meta_model(cfg); L = cfg.num_hidden_layers
    rot = Qwen2RotaryEmbedding(config=cfg).cuda()
    seqlen, nwin = 2048, int(os.environ.get("EVOQ_NWIN", "8"))
    ids_all = get_wikitext2_test_ids(tok); nwin = min(nwin, ids_all.numel() // seqlen)
    ids = ids_all[:, :nwin * seqlen].reshape(nwin, seqlen)
    emb = q_read_tensor(smap, "model.embed_tokens.weight").float()
    h = F.embedding(ids, emb).cuda()
    use_calib = (AWQ or OGAIN) and qmax > 0
    ctxt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wikitext2_train.txt"), encoding="utf-8").read()
    cids = tok(ctxt, return_tensors="pt").input_ids[:, :512]
    hc = F.embedding(cids, emb).cuda() if use_calib else None; del emb
    pos = torch.arange(seqlen, device="cuda").unsqueeze(0); cos_sin = rot(h, pos)
    mask = q_causal_mask(seqlen).cuda()
    cpos = torch.arange(cids.shape[1], device="cuda").unsqueeze(0) if hc is not None else None
    ccs = rot(hc, cpos) if hc is not None else None
    cmask = q_causal_mask(cids.shape[1]).cuda() if hc is not None else None
    t0 = time.time(); bws = []
    for li in range(L):
        layer = model.model.layers[li]
        q_materialize(layer, smap, f"model.layers.{li}.")        # all params (incl q/k/v bias) fp32 CPU
        layer.cuda()
        n2m = dict(layer.named_modules()); ex, eabs, xfull = {}, {}, {}
        if hc is not None:
            hooks = []
            def mk(nm):
                def hk(m, inp):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    eabs[nm] = x.abs().mean(0)
                    if OGAIN: xfull[nm] = x
                return hk
            for name, mod in n2m.items():
                if name.split(".")[-1] in LINEAR7 and hasattr(mod, "weight"):
                    hooks.append(mod.register_forward_pre_hook(mk(name)))
            layer(hc, attention_mask=cmask, position_embeddings=ccs)
            for hk in hooks: hk.remove()
        for name, mod in n2m.items():
            if name.split(".")[-1] not in LINEAR7 or not hasattr(mod, "weight"): continue
            if li < qmax:
                W = mod.weight.data.cpu().numpy()
                s = (_awq_scale({name: eabs[name].cpu().numpy()}, name, AWQ_A).astype(np.float32) if AWQ else None)
                wh, bw = trellis_quant(W, p_out=p_out, awq_s=s); bws.append(bw * W.size)
                wht = torch.from_numpy(wh).cuda()
                if OGAIN:
                    xc = xfull[name]; Wt = mod.weight.data.float()
                    ref = xc @ Wt.T; ap = xc @ wht.T
                    c = (ref * ap).sum(0) / torch.clamp((ap * ap).sum(0), min=1e-9)
                    wht = wht * c.view(-1, 1); del Wt, ref, ap
                mod.weight.data.copy_(wht)
        h = layer(h, attention_mask=mask, position_embeddings=cos_sin)[0]
        if hc is not None: hc = layer(hc, attention_mask=cmask, position_embeddings=ccs)[0]
        layer.to_empty(device="meta"); gc.collect(); torch.cuda.empty_cache()
        if li % 7 == 0: print(f"  qwen layer {li}/{L} ({time.time()-t0:.0f}s) AWQ={AWQ} OGAIN={OGAIN} qmax={qmax}", flush=True)
    normw = q_read_tensor(smap, "model.norm.weight").float().cuda()
    headw = q_read_tensor(smap, "lm_head.weight").float().cuda()
    v = h.pow(2).mean(-1, keepdim=True); h = (h * torch.rsqrt(v + cfg.rms_norm_eps)) * normw
    nlls, ntok = [], 0
    for b in range(nwin):
        logits = h[b] @ headw.T
        nlls.append(F.cross_entropy(logits[:-1].float(), ids[b, 1:].cuda()) * (seqlen - 1)); ntok += seqlen - 1
    ppl = torch.exp(torch.stack(nlls).sum() / ntok).item()
    bw_nom = (K_RATE + 16.0 / G + (p_out * 48 if p_out else 0.0)) if qmax > 0 else 16.0
    line = (f"\nQwen2.5-7B QTIP-trellis L={L_BITS} AWQ={AWQ}(a={AWQ_A}) OGAIN={OGAIN} pout={p_out} qmax={qmax}: "
            f"WikiText-2 ppl = {ppl:.4f} @ ~{bw_nom:.3f}b ({nwin} win) | GENERALITY vs Llama-2-7B trellis +1.4 over fp16")
    print(line, flush=True)
    try:
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trellis_results.txt"), "a",
             encoding="utf-8").write(line + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "measure7b"
    {"mse05": mse05, "mse_out": mse_out, "mse_out7b": mse_out7b, "measure05": measure05,
     "companding_probe": companding_probe, "measure7b": measure7b, "measure7b_qwen": measure7b_qwen,
     "sens_db": sens_db,
     "make_alloc": lambda: make_alloc(float(os.environ.get("EVOQ_TARGET_BW", "2.37")))}.get(mode, measure7b)()
