"""evoq_gemma.py -- stream-and-quantize Gemma 4 12B (dense) on a 6GB GPU. RUN WITH .venv-gemma:
    .venv-gemma/Scripts/python -u -m weights.evoq_gemma

Replicates Gemma4TextModel.forward (dual-RoPE + alternating sliding/full masks + per-layer norms + final
logit soft-cap) but streams each of the 48 decoder layers (materialize from the single safetensors ->
optional trellis-quant -> forward all eval windows -> free), so a 12B model fits 6GB. fp16 baseline first.

EVOQ_GEMMA_MODE in {fp16, carveout(2b MLP+4b attn), mlp2, all2}. EVOQ_NWIN, EVOQ_SEQ.
"""
from __future__ import annotations
import gc, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from collections import UserDict
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

MDIR = "D:/evo-compress-data/gemma-4-12b"
ST = os.path.join(MDIR, "model.safetensors")
LP = "model.language_model."
MLP = ("gate_proj", "up_proj", "down_proj"); ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def bits_for(mode, short, li=0):
    if mode == "fp16":
        return 16
    if mode == "flipped":                                 # depth-aware: protect the fragile LATE band
        if short in MLP:
            return 4 if li >= 36 else 2
        return 4
    if mode == "carveout":
        return 2 if short in MLP else 4
    if mode == "mlp2":
        return 2 if short in MLP else 16
    if mode == "all2":
        return 2 if short in (MLP + ATTN) else 16
    return 16


def q_tensor(W, K):
    if K >= 16:
        return W
    import weights.evoq_moe as em
    wh, _ = em.trellis_quant(np.ascontiguousarray(W.astype(np.float32)), K=K)
    return np.asarray(wh, np.float32).reshape(W.shape)


_HAD = {}


def _hadamard(g, device):
    if g not in _HAD:
        Hm = torch.ones(1, 1)
        while Hm.shape[0] < g:
            Hm = torch.cat([torch.cat([Hm, Hm], 1), torch.cat([Hm, -Hm], 1)], 0)
        _HAD[g] = (Hm / g ** 0.5)
    return _HAD[g].to(device)


def fasthad_2bit(W, g=128):
    """Fast conservative 2-bit: per-group signed-Hadamard rotation + Lloyd-Max grid (GPU, seconds).
    rel-MSE ~0.118 (vs trellis 0.069) -- uniformly worse, so band results are conservative."""
    out, inn = W.shape
    Hm = _hadamard(g, W.device)
    Wr = (W.reshape(out, inn // g, g) @ Hm)
    s = Wr.std(dim=-1, keepdim=True).clamp_min(1e-8)
    x = Wr / s
    q = torch.where(x.abs() < 0.9816, 0.4528 * torch.sign(x), 1.5104 * torch.sign(x))
    return ((q * s) @ Hm.T).reshape(out, inn)


def run():
    mode = os.environ.get("EVOQ_GEMMA_MODE", "fp16")
    band = os.environ.get("EVOQ_GEMMA_BAND", "")            # "lo-hi": MLP@2bit-fasthad ONLY in these layers
    blo, bhi = (int(v) for v in band.split("-")) if band else (-1, -1)
    nwin = int(os.environ.get("EVOQ_NWIN", "4"))
    seqlen = int(os.environ.get("EVOQ_SEQ", "1024"))
    cfg = AutoConfig.from_pretrained(MDIR)
    tcfg = cfg.text_config
    H, L = tcfg.hidden_size, tcfg.num_hidden_layers
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model = model.to(torch.float32).eval()
    lm = model.model.language_model

    tok = AutoTokenizer.from_pretrained(MDIR)
    from weights.wikitext2_ppl import get_wikitext2_test_ids
    ids_all = get_wikitext2_test_ids(tok)
    nwin = min(nwin, ids_all.numel() // seqlen)
    ids = ids_all[:, :nwin * seqlen].reshape(nwin, seqlen)
    print(f"EVOQ-GEMMA [{mode}]: {nwin}x{seqlen} WikiText-2 | hidden={H} layers={L}", flush=True)

    f = safe_open(ST, framework="pt")
    g = lambda k: f.get_tensor(k)

    # --- non-layer modules: scaled embed, final norm, tied lm_head, fresh rotary ---
    lm.embed_tokens.to_empty(device="cpu")
    lm.embed_tokens.weight.data.copy_(g(LP + "embed_tokens.weight").float())
    lm.embed_tokens.embed_scale = torch.tensor(H ** 0.5)                 # Gemma normalizer
    lm.norm.to_empty(device="cpu"); lm.norm.weight.data.copy_(g(LP + "norm.weight").float())
    headw = g("lm_head.weight").float() if "lm_head.weight" in f.keys() else lm.embed_tokens.weight.data
    normw = lm.norm.weight.data
    rot = Gemma4TextRotaryEmbedding(tcfg).cuda()

    # --- initial hidden states (scaled embed), on CPU ---
    with torch.no_grad():
        h = lm.embed_tokens(ids)                                         # [nwin, seq, H]
    pos = torch.arange(seqlen, device="cuda").unsqueeze(0)

    # --- masks + rope, per layer_type, built once on a batch-1 sample ---
    sample = h[:1].cuda()
    mask_kwargs = dict(config=tcfg, inputs_embeds=sample, attention_mask=None, past_key_values=None,
                       position_ids=pos)
    masks = {"full_attention": create_causal_mask(**mask_kwargs),
             "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs)}
    pe = {lt: rot(sample, pos, lt) for lt in set(tcfg.layer_types)}
    del sample; torch.cuda.empty_cache()

    t0 = time.time()
    for li in range(L):
        lt = tcfg.layer_types[li]
        layer = lm.layers[li]
        layer.to_empty(device="cpu")
        sd = layer.state_dict()
        nq = 0
        for name in list(sd.keys()):
            W = g(f"{LP}layers.{li}.{name}").float()
            short = name.split(".")[-2] if name.endswith(".weight") else ""
            K = bits_for(mode, short, li)
            if K < 16 and W.ndim == 2 and short in (MLP + ATTN):
                W = torch.from_numpy(q_tensor(W.numpy(), K)); nq += 1
            sd[name].copy_(W)
        layer.cuda()
        if blo <= li <= bhi:                                # band mode: fasthad-2bit the MLP on GPU
            for name, mod in layer.named_modules():
                if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in MLP:
                    mod.weight.data = fasthad_2bit(mod.weight.data); nq += 1
        outs = []
        for b in range(nwin):
            with torch.no_grad():
                yb = layer(h[b:b + 1].cuda(), shared_kv_states=UserDict(),
                           position_embeddings=pe[lt], attention_mask=masks[lt],
                           position_ids=pos, past_key_values=None)
            outs.append((yb[0] if isinstance(yb, tuple) else yb).cpu())
        h = torch.cat(outs, 0)
        layer.to_empty(device="meta"); gc.collect(); torch.cuda.empty_cache()
        if li % 8 == 0 or li == L - 1:
            print(f"  layer {li}/{L} (q={nq}, {lt[:4]}) ({time.time()-t0:.0f}s)", flush=True)

    # --- final norm + lm_head + logit soft-cap ---
    cap = tcfg.final_logit_softcapping
    lm.norm.cuda(); hw = headw.cuda()
    nlls, ntok = [], 0
    for b in range(nwin):
        with torch.no_grad():
            hb = lm.norm(h[b].cuda())                                    # model's own final RMSNorm
        logits = hb @ hw.T
        if cap:
            logits = cap * torch.tanh(logits / cap)
        nlls.append(F.cross_entropy(logits[:-1].float(), ids[b, 1:].cuda()) * (seqlen - 1)); ntok += seqlen - 1
    ppl = float(torch.exp(torch.stack(nlls).sum() / ntok))
    tag = mode + (f" band={band}" if band else "")
    line = f"EVOQ-GEMMA [{tag}]: ppl={ppl:.4f} | Gemma-4-12B dense, {nwin}x{seqlen} WikiText-2"
    print("\n" + line, flush=True)
    open(os.path.join(os.path.dirname(__file__), "data", "gemma_gate.log"), "a", encoding="utf-8").write(line + "\n")


if __name__ == "__main__":
    run()
