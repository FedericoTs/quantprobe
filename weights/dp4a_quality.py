"""Campaign2 dp4a QUALITY oracle -- does the speed-kernel's int8 numerics cost quality?

The F0/S1 kernels are BIT-EXACT but plateau at ~34%% util (instruction-issue bound:
~12 instr / 4 weights in the fp32+LUT inner loop). llama.cpp clears 0.50 util via dp4a:
4 weights per int8 instruction. To adopt dp4a we must (a) int8-snap the <=12-level ECVQ
codebook (static, per-tensor) and (b) Q8-quantize the FWHT'd activation per group (dynamic,
Q8_1-style). Both are LOSSY. This oracle measures the INCREMENTAL held-out ppl cost on the
real 0.5B, faithfully -- via the activation-side identity the kernel actually computes:

    y_j = sum_g amax[j,g] * ( sum_{k in g} xrr[g,k] * lv[idx[j,g,k]] ),
    xrr[g] = FWHT( x[g] * signs / awq_s[g] ) / sqrt(G)      (computed once per token)

Exact mode must reproduce the champion ppl (validates the identity). dp4a mode applies
int8 codebook + per-(token,group) Q8 to xrr, int32-accumulates (exact in fp32 for these
magnitudes), rescales. Outliers stay an EXACT sparse sidecar in BOTH modes (cancel in delta).

Run:  python -m weights.dp4a_quality
"""
from __future__ import annotations

import gc
import math
import sys

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from weights.evoq import encode_tensor, unpack6_t, fwht_inplace_t
from weights.quant_lab import (CFG, WPATH, build_model, calibrate, load_fp16, ppl,
                               quant_keys, _awq_scale)
from transformers import AutoTokenizer
from safetensors import safe_open

G = 128


class SimActLinear(nn.Module):
    """Activation-side identity forward of one evoq tensor, with optional dp4a (int8) numerics.
    mode in {'exact','dp4a'}: exact = fp32 identity (== champion); dp4a = int8 cb + Q8 xrr."""

    def __init__(self, comp, bias, mode="exact"):
        super().__init__()
        self.mode = mode
        self.rows, self.cols = int(comp["rows"]), int(comp["cols"])
        self.ng = self.cols // G
        as_t = lambda v: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        idx = unpack6_t(as_t(comp["packed"]), int(comp["n_idx"])).view(self.rows, self.ng, G)
        self.register_buffer("idx", idx, persistent=False)               # [rows,ng,G] long
        self.register_buffer("lv", as_t(comp["lv"]).float(), persistent=False)
        self.register_buffer("amax", as_t(comp["amax"]).float().view(self.rows, self.ng), persistent=False)
        self.register_buffer("signs", as_t(comp["signs"]).float(), persistent=False)   # [G] +-1
        self.register_buffer("awq_s", as_t(comp["awq_s"]).float(), persistent=False)    # [cols]
        # exact outlier sidecar -> dense correction matrix (shared by both modes)
        Wout = torch.zeros(self.rows * self.cols, dtype=torch.float32)
        Wout[as_t(comp["out_pos"]).long()] = as_t(comp["out_val"]).float()
        self.register_buffer("Wout", Wout.view(self.rows, self.cols), persistent=False)
        self.bias = bias
        # int8-snapped codebook (per-tensor static)
        self.cb_scale = float(self.lv.abs().max().item()) / 127.0
        self.register_buffer("lv_q", torch.round(self.lv / self.cb_scale).clamp(-127, 127), persistent=False)

    def forward(self, x):
        orig = x.shape
        in_dtype = x.dtype
        x = x.reshape(-1, self.cols).float()                  # [B, cols]
        B = x.shape[0]
        # z = x * signs / awq_s  (signs tiled over groups), then per-group FWHT / sqrt(G)
        z = (x / self.awq_s).view(B, self.ng, G) * self.signs.view(1, 1, G)
        xrr = fwht_inplace_t(z.contiguous()) / math.sqrt(G)   # [B, ng, G]
        y = x.new_zeros(B, self.rows)
        if self.mode == "dp4a":
            act_scale = xrr.abs().amax(-1, keepdim=True) / 127.0          # [B,ng,1] per (token,group)
            act_scale = act_scale.clamp_min(1e-12)
            xrr_q = torch.round(xrr / act_scale).clamp(-127, 127)         # int8 activations
            for g in range(self.ng):
                wq = self.lv_q[self.idx[:, g, :]]                         # [rows,G] int8 levels
                acc = xrr_q[:, g, :] @ wq.t()                            # [B,rows] exact int32 (fp32)
                scale = (self.cb_scale * self.amax[:, g]).view(1, self.rows) * act_scale[:, g, :]
                y += acc * scale
        else:
            for g in range(self.ng):
                w = self.lv[self.idx[:, g, :]]                            # [rows,G] fp32
                acc = xrr[:, g, :] @ w.t()                               # [B,rows]
                y += acc * self.amax[:, g].view(1, self.rows)
        y = y + x @ self.Wout.t()                                        # exact outlier sidecar (both modes)
        if self.bias is not None:
            y = y + self.bias.view(1, -1)
        return y.reshape(*orig[:-1], self.rows).to(in_dtype)


def build_sim(model, calib, mode):
    """Encode every quantized Linear and swap in a SimActLinear(mode)."""
    name2mod = dict(model.named_modules())
    with safe_open(WPATH, framework="pt") as f:
        qk = quant_keys(f)
        for k in sorted(qk):
            mod_name = k[: -len(".weight")]
            mod = name2mod[mod_name]
            W = f.get_tensor(k).float().numpy()
            s = _awq_scale(calib, k, 0.5)
            comp = encode_tensor(W, s, self_check=False)
            bias = mod.bias.detach().float() if getattr(mod, "bias", None) is not None else None
            sim = SimActLinear(comp, bias, mode=mode)
            # replace module in parent
            parent = name2mod[mod_name.rsplit(".", 1)[0]] if "." in mod_name else model
            setattr(parent, mod_name.rsplit(".", 1)[-1], sim)
    return model


def main():
    tok = AutoTokenizer.from_pretrained(CFG)
    model = build_model(); load_fp16(model)
    calib = calibrate(model, tok)
    fp16 = ppl(model, tok)
    print(f"fp16 held-out ppl = {fp16:.4f}\n", flush=True)

    print(f"{'mode':<34}{'ppl':>10}{'d_vs_fp16':>11}{'d_vs_exact':>12}")
    print("-" * 67)

    # 1) EXACT activation-side identity -- must reproduce the champion (validates the math)
    load_fp16(model); build_sim(model, calib, "exact")
    p_exact = ppl(model, tok)
    print(f"{'evoq exact (act-side, bit-exact)':<34}{p_exact:>10.4f}{p_exact-fp16:>+11.4f}{0.0:>+12.4f}", flush=True)
    gc.collect()

    # 2) dp4a numerics: int8 codebook + Q8 activation
    model = build_model(); load_fp16(model)
    build_sim(model, calib, "dp4a")
    p_dp4a = ppl(model, tok)
    print(f"{'evoq dp4a (int8 cb + Q8 xrr)':<34}{p_dp4a:>10.4f}{p_dp4a-fp16:>+11.4f}{p_dp4a-p_exact:>+12.4f}", flush=True)

    print(f"\nchampion arena ppl ref = 4.6302 (seed 0). Held-out noise floor ~0.067 b/w-equiv.")
    dd = p_dp4a - p_exact
    verdict = ("FREE (within seed noise) -> dp4a SAFE, build the kernel" if dd < 0.03 else
               "MEASURABLE -> collect data: per-group int16 act, codebook-aware act-scale, or keep top-k groups fp" if dd < 0.15 else
               "COSTLY -> dp4a needs error-feedback / hybrid; F0/S1 bit-exact stays primary")
    print(f"dp4a incremental d_ppl = {dd:+.4f}  => {verdict}")


if __name__ == "__main__":
    main()
