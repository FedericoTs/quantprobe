"""BREAKTHROUGH TEST #1 -- FUNCTIONAL NOISE-SHAPING (error-feedback ECVQ along the contraction
axis, scored against the OUTPUT covariance). The 2-bit cliff is the Shannon+0.25 wall for
reconstructing EACH WEIGHT under MSE -- but the model only sees y = W@x (a linear functional).
Noise-shaping (Inose-Yasuda delta-sigma) provably beats per-sample quant bounds for a FILTERED
readout, and a GEMV is a low-pass integrator over the contraction axis.

Activation-side identity (matches evoq/champion): y_o = sum_g amax_og * sum_k xrr_gk * lv[idx_ogk],
where xrr_g = FWHT((x_g/awq_s_g)*signs)/sqrt(G). So the rotated-domain quant error e = Rn - lv[idx]
contributes (xrr . e) to the output. Minimizing OUTPUT MSE => shape e to be small in the
xrr-covariance metric H_g = E[xrr_g xrr_g^T].

3-WAY CONTROL (a negative is as clean as a positive):
  (a) PLAIN   : per-coord nearest level (min weight-MSE). baseline.
  (b) SIGMA-DELTA : 1st-order error-feedback along the 128 axis (fixed integrator, NO H) -- pure DSP.
  (c) GPTQ-H  : GPTQ error propagation using the REAL rotated-activation Hessian H_g -- output-aware.
  (d) GPTQ-I  : GPTQ with H=I (sanity; should ~= plain).
METRIC: true OUTPUT MSE = ||(W - Wq) @ X^T||_F^2 / (out*N) on REAL cached activations (the functional
distortion), at ISO-RATE (same K levels). If (b)/(c) beat (a), the mechanism is real;
(c)>>(b) means it's the activation off-diagonal (what Hadamard did NOT whiten).

Run:  python -m weights.noise_shaping
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoTokenizer
from weights.quant_lab import CFG, build_model, load_fp16, calibrate, _awq_scale, CALIB_TEXT
from weights.quant_sota import _fwht_rows

G = 128


def lloyd_levels(samples, K, iters=25, seed=0):
    """1-D Lloyd-Max: K centroids minimizing MSE on `samples` (the iso-rate codebook)."""
    s = np.sort(samples.astype(np.float64))
    q = np.quantile(s, (np.arange(K) + 0.5) / K)
    for _ in range(iters):
        edges = (q[:-1] + q[1:]) / 2
        idx = np.searchsorted(edges, s)
        newq = q.copy()
        for k in range(K):
            m = idx == k
            if m.any():
                newq[k] = s[m].mean()
        if np.allclose(newq, q):
            break
        q = newq
    return q.astype(np.float32)


def nearest(vals, lv):
    """nearest level index for each value (vals any shape)."""
    d = np.abs(vals[..., None] - lv[None, ...])
    return d.argmin(-1).astype(np.int64)


def capture(model, tok, names):
    acts, hooks = {}, []
    name2mod = dict(model.named_modules())

    def mk(k):
        def h(mod, inp):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            acts.setdefault(k, []).append(x)
        return h
    for k in names:
        hooks.append(name2mod[k].register_forward_pre_hook(mk(k)))
    ids = tok(CALIB_TEXT, return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        model(ids)
    for h in hooks:
        h.remove()
    return {k: torch.cat(v, 0).numpy() for k, v in acts.items()}


def rotate(W, s, seed=0):
    """champion rotation (no outliers, to isolate the mechanism): Ws=W*s, per-group signed-FWHT, amax-norm."""
    rows, cols = W.shape
    Ws = (W * s[None, :]).astype(np.float32)
    signs = (np.random.default_rng(seed).integers(0, 2, G).astype(np.float32) * 2 - 1)
    N = Ws.reshape(rows, -1, G).reshape(-1, G)
    R = _fwht_rows(N * signs) / math.sqrt(G)
    amax = np.abs(R).max(1); amax[amax == 0] = 1.0
    Rn = (R / amax[:, None]).reshape(rows, -1, G)            # [rows, ng, G]
    return Rn, amax.reshape(rows, -1), signs, (rows, cols)


def derotate(idx, lv, amax, signs, shape):
    rows, cols = shape
    ng = cols // G
    u = lv[idx] * amax[:, :, None]                           # [rows, ng, G]
    back = _fwht_rows(u.reshape(-1, G)) / math.sqrt(G) * signs
    return back.reshape(rows, cols)                          # de-rotated weight (in W*s space)


def xrr_of(X, s, signs):
    """rotated activations: per group, FWHT((x/s)*signs)/sqrt(G).  X [N, cols] -> [N, ng, G]."""
    N, cols = X.shape
    ng = cols // G
    z = (X / s[None, :]).reshape(N, ng, G) * signs[None, None, :]
    xr = _fwht_rows(z.reshape(-1, G)) / math.sqrt(G)
    return xr.reshape(N, ng, G)


def quant_plain(Rn, lv):
    return nearest(Rn, lv)


def quant_sigmadelta(Rn, lv):
    """1st-order error feedback along the G axis (per row,group). Pure DSP, no H."""
    rows, ng, g = Rn.shape
    idx = np.empty((rows, ng, g), np.int64)
    carry = np.zeros((rows, ng), np.float64)
    for k in range(g):
        v = Rn[:, :, k] + carry
        ik = nearest(v.astype(np.float32), lv)
        idx[:, :, k] = ik
        carry = v - lv[ik]
    return idx


def _gptq_block(Wb, H, lv, damp=0.1, act_order=True, clip=2.0):
    """Stabilized GPTQ on one [out, g] block with Hessian H [g,g]; quantize to nearest level in lv.
    Stabilizers for low-bit (fix the deep-bit blow-up): (1) ACT-ORDER -- quantize high-Hessian coords
    first; (2) stronger group DAMPING (default 0.1); (3) CLIP the running residual to clip*level-range
    so the error-feedback cannot run away when the codebook is too coarse."""
    out, g = Wb.shape
    H = H.copy().astype(np.float64)
    order = np.argsort(-np.diag(H)) if act_order else np.arange(g)
    inv = np.argsort(order)
    H = H[order][:, order]
    W = Wb.astype(np.float64)[:, order].copy()
    d = damp * np.mean(np.diag(H)) + 1e-8
    H[np.diag_indices(g)] += d
    try:
        Hinv = np.linalg.cholesky(np.linalg.inv(H)).T   # upper triangular factor
    except np.linalg.LinAlgError:
        Hinv = np.eye(g)
    lo, hi = clip * lv.min(), clip * lv.max()
    idx = np.empty((out, g), np.int64)
    for j in range(g):
        ij = nearest(W[:, j].astype(np.float32), lv)
        idx[:, j] = ij
        err = (W[:, j] - lv[ij]) / Hinv[j, j]
        if j + 1 < g:
            W[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
            np.clip(W[:, j + 1:], lo, hi, out=W[:, j + 1:])   # prevent runaway feedback
    return idx[:, inv]                                          # un-permute to original order


def quant_gptq(Rn, lv, Hgroups):
    rows, ng, g = Rn.shape
    idx = np.empty((rows, ng, g), np.int64)
    for gi in range(ng):
        idx[:, gi, :] = _gptq_block(Rn[:, gi, :], Hgroups[gi], lv)
    return idx


def out_mse(W, wh, X):
    """true output MSE = ||(W - wh) @ X^T||_F^2 / (out*N) on real activations (functional distortion).
    NOTE W,wh here are in the W*s (scaled) domain and X is raw; but (W-wh) is the scaled-weight error and
    the model computes (W*s) @ (x/s) = W@x, so use X/s consistently. We pass Xs = X (raw) and compare in
    scaled space by also scaling X: handled by caller passing X already (errors in W*s vs x/s cancel s)."""
    E = (W - wh) @ X.T
    return float((E ** 2).mean())


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model(); load_fp16(model)
    calib = calibrate(model, tok)
    # target the largest FFN tensors in a few mid layers
    L = model.config.num_hidden_layers
    targets = []
    for li in (L // 4, L // 2, 3 * L // 4):
        targets += [f"model.layers.{li}.mlp.down_proj", f"model.layers.{li}.mlp.gate_proj"]
    acts = capture(model, tok, targets)
    sd = model.state_dict()
    print(f"{'tensor':<34}{'K':>3}{'plain':>11}{'sigΔ':>11}{'gptqH':>11}{'gptqI':>11}   (output-MSE, x1e3; lower=better)")
    print("-" * 96)
    summary = {2: [], 3: []}
    for name in targets:
        W = sd[name + ".weight"].float().numpy()
        s = _awq_scale(calib, name + ".weight", 0.5).astype(np.float32)
        X = acts[name]                                          # [N, in] raw activations
        Xs = (X / s[None, :]).astype(np.float32)                # divide by AWQ scale (we quantize W*s)
        Ws = (W * s[None, :]).astype(np.float32)
        Rn, amax, signs, shape = rotate(W, s)
        # rotated-activation Hessian per group from real activations
        xr = xrr_of(X, s, signs)                                # [N, ng, G]
        N, ng, _ = xr.shape
        Hg = np.einsum("ngk,ngl->gkl", xr, xr) / N              # [ng, G, G]
        rng = np.random.default_rng(1)
        samp = Rn.reshape(-1)[rng.integers(0, Rn.size, 30000)]
        for K, bits in ((4, 2), (8, 3)):
            lv = lloyd_levels(samp, K)
            recon = {}
            recon["plain"] = derotate(quant_plain(Rn, lv), lv, amax, signs, shape)
            recon["sigd"] = derotate(quant_sigmadelta(Rn, lv), lv, amax, signs, shape)
            recon["gptqH"] = derotate(quant_gptq(Rn, lv, Hg), lv, amax, signs, shape)
            recon["gptqI"] = derotate(quant_gptq(Rn, lv, np.tile(np.eye(G), (ng, 1, 1))), lv, amax, signs, shape)
            # output MSE in W*s / x/s space (== W@x space): (Ws - recon) @ Xs^T
            m = {k: out_mse(Ws, v, Xs) * 1e3 for k, v in recon.items()}
            base = m["plain"]
            summary[bits].append((m["sigd"] / base, m["gptqH"] / base, m["gptqI"] / base))
            print(f"{name.split('.',2)[-1][:33]:<34}{K:>3}{m['plain']:>11.4f}{m['sigd']:>11.4f}"
                  f"{m['gptqH']:>11.4f}{m['gptqI']:>11.4f}", flush=True)
    print("\n=== MEAN OUTPUT-MSE RATIO vs PLAIN (lower=better; <1 means the method helps) ===")
    for bits in (2, 3):
        r = np.array(summary[bits])
        print(f"  {bits}-bit:  sigma-delta {r[:,0].mean():.3f}   GPTQ-realH {r[:,1].mean():.3f}   GPTQ-I {r[:,2].mean():.3f}")
    print("\nVERDICT GUIDE: GPTQ-realH << 1 (e.g. <0.85) at 2-bit => functional noise-shaping is REAL and "
          "largest at the cliff -> build the vectorized streaming version + scale to Llama-2-7B. "
          "sigma-delta<1 too => pure DSP shaping also helps. GPTQ-I ~= 1.0 validates the control "
          "(identity Hessian = no cross-coord info). All ~=1.0 => Hadamard already whitened the off-diagonal -> kill.")


def _codec(W, key, acts, calib, K, use_gptq):
    """Quantize one tensor: rotate -> fit K-level Lloyd -> (GPTQ-realH | plain) -> derotate -> /s.
    Returns (wh in W space, honest bits at fixed rate log2(K) + fp16 amax/group + levels)."""
    s = _awq_scale(calib, key, 0.5).astype(np.float32)
    Rn, amax, signs, shape = rotate(W, s)
    rows, cols = shape; ng = cols // G
    rng = np.random.default_rng(1)
    samp = Rn.reshape(-1)[rng.integers(0, Rn.size, min(30000, Rn.size))]
    lv = lloyd_levels(samp, K)
    if use_gptq:
        X = acts[key[:-len(".weight")]]
        xr = xrr_of(X, s, signs); N = xr.shape[0]
        Hg = np.einsum("ngk,ngl->gkl", xr, xr) / N
        idx = quant_gptq(Rn, lv, Hg)
    else:
        idx = quant_plain(Rn, lv)
    wh = derotate(idx, lv, amax, signs, shape) / s[None, :]
    bits = math.log2(K) * (rows * cols) + 16.0 * (rows * ng) + K * 16
    return wh.astype(np.float32), bits


def ppl_validate():
    """Whole-0.5B held-out ppl: plain ECVQ-2bit vs rotated-GPTQ-ECVQ-2bit (and 3-bit). Does the
    output-MSE halving translate to perplexity?  ISO-RATE (same K)."""
    from safetensors import safe_open
    from weights.quant_lab import WPATH, quant_keys, load_quant, ppl
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model(); load_fp16(model)
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
    acts = capture(model, tok, [k[:-len(".weight")] for k in qk])
    print(f"\nfp16 held-out ppl = {fp16:.4f}\n")
    print(f"{'scheme':<26}{'bits/wt':>9}{'ppl':>9}{'d_vs_plain':>12}")
    print("-" * 56)
    rows = []
    for K, bits in ((4, 2), (8, 3)):
        res = {}
        for tag, ug in (("plain", False), ("GPTQ-realH", True)):
            load_fp16(model)
            bpw = load_quant(model, lambda a, k: _codec(a, k, acts, calib, K, ug))
            p = ppl(model, tok)
            res[tag] = p
            print(f"{f'{bits}-bit {tag}':<26}{bpw:>9.3f}{p:>9.4f}"
                  f"{(p - res.get('plain', p)):>+12.4f}", flush=True)
        rows.append((bits, res['plain'], res['GPTQ-realH']))
    print("\n=== VERDICT ===")
    for bits, pl, gp in rows:
        d = gp - pl
        v = ("CONFIRMED: GPTQ-realH recovers ppl" if d < -0.05 else
             "NULL: MSE gain did NOT translate to ppl" if d > -0.02 else "marginal")
        print(f"  {bits}-bit: plain {pl:.4f} -> GPTQ-realH {gp:.4f}  (d={d:+.4f})  {v}")
    print("  (champion scalar ECVQ cliffs hard at 2-bit; if GPTQ-realH recovers it, scale to Llama-2-7B vs QTIP 5.86)")


def _entropy_bw(idx, K):
    h = np.bincount(idx.reshape(-1), minlength=K).astype(float)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


def champion_codec(W, key, acts, calib, lam, use_gptq, p_out=0.005):
    """FULL champion: AWQ + 0.5% fp16 outliers + per-group signed-FWHT + entropy-constrained ECVQ.
    plain nearest vs GPTQ-realH assignment. Returns (wh in W space, honest b/w)."""
    from weights.codec_zoo import _ecvq_levels
    s = _awq_scale(calib, key, 0.5).astype(np.float32)
    rows, cols = W.shape; ng = cols // G
    Ws = (W * s[None, :]).astype(np.float32)
    thr = np.quantile(np.abs(Ws), 1.0 - p_out)
    mask = np.abs(Ws) >= thr
    base = Ws.copy(); base[mask] = 0.0
    signs = (np.random.default_rng(0).integers(0, 2, G).astype(np.float32) * 2 - 1)
    N = base.reshape(rows, -1, G).reshape(-1, G)
    R = _fwht_rows(N * signs) / math.sqrt(G)
    amax = np.abs(R).max(1); amax[amax == 0] = 1.0
    Rn = (R / amax[:, None]).reshape(rows, ng, G)
    rng = np.random.default_rng(1)
    samp = Rn.reshape(-1)[rng.integers(0, Rn.size, min(20000, Rn.size))]
    lv = _ecvq_levels(samp, 64, lam).astype(np.float32)
    K = len(lv)
    if use_gptq:
        X = acts[key[:-len(".weight")]]
        xr = xrr_of(X, s, signs)
        Hg = np.einsum("ngk,ngl->gkl", xr, xr) / xr.shape[0]
        idx = quant_gptq(Rn, lv, Hg)
    else:
        idx = quant_plain(Rn, lv)
    amax2 = amax.reshape(rows, ng)
    wh = derotate(idx, lv, amax2, signs, (rows, cols))
    wh = wh.reshape(rows, cols)
    wh[mask] = Ws[mask]                                   # restore exact outliers
    wh = (wh / s[None, :]).astype(np.float32)
    n_out = int(mask.sum())
    bits = _entropy_bw(idx, K) * (rows * cols) + 16.0 * (rows * ng) + n_out * (32 + 16) + K * 16
    return wh, bits


def ppl_stack():
    """Does GPTQ output-aware feedback STACK on the FULL champion (entropy-ECVQ + outliers)?
    champion-plain vs champion+GPTQ at lambdas spanning ~2-3 b/w, 0.5B held-out ppl."""
    from safetensors import safe_open
    from weights.quant_lab import WPATH, quant_keys, load_quant, ppl
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model(); load_fp16(model)
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
    acts = capture(model, tok, [k[:-len(".weight")] for k in qk])
    print(f"\nfp16 held-out ppl = {fp16:.4f}  (champion FULL codec: outliers+entropy+AWQ)\n")
    print(f"{'scheme':<28}{'b/w':>8}{'ppl':>9}{'d_GPTQ':>9}")
    print("-" * 54)
    for lam in (0.03, 0.02, 0.008):
        res = {}
        for tag, ug in (("plain", False), ("+GPTQ", True)):
            load_fp16(model)
            bpw = load_quant(model, lambda a, k: champion_codec(a, k, acts, calib, lam, ug))
            p = ppl(model, tok); res[tag] = (bpw, p)
            dd = (p - res['plain'][1]) if tag == '+GPTQ' else 0.0
            print(f"{f'lam{lam} champ {tag}':<28}{bpw:>8.3f}{p:>9.4f}{dd:>+9.4f}", flush=True)
        d = res['+GPTQ'][1] - res['plain'][1]
        v = "STACKS (GPTQ helps champion)" if d < -0.05 else "REDUNDANT w/ outliers" if d > -0.02 else "marginal"
        print(f"    => lam{lam} (~{res['plain'][0]:.2f}b): {v}  (d={d:+.4f})\n", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "mse"
    {"ppl": ppl_validate, "stack": ppl_stack, "mse": main}.get(mode, main)()
