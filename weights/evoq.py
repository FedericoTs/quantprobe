"""evoq -- DIRECT-EXECUTION runtime for the evolved champion codec (rotation + AWQ + scalar ECVQ
+ outliers). Weights stay PACKED in memory (~6.5 bits/weight resident vs 16 fp16); each Linear
dequantizes on the fly in forward (LUT -> x amax -> inverse-FWHT per 128-group -> /AWQ -> scatter
outliers). The rotation is codec-internal, so NO online activation transform is needed.

Components:
  pack6/unpack6        -- 4x 6-bit indices in 3 bytes (K<=64 levels)
  fwht_t               -- torch FWHT matching quant_sota._fwht_rows exactly
  encode_tensor        -- mirror of codec_zoo ecvq()/_had_ecvq() emitting components (+self-check)
  QuantLinear          -- nn.Module holding packed buffers; dequant in forward
  save/load_container  -- safetensors + json meta (mmap-friendly)

Verification gates: (1) component dequant == reference wh bit-near-exactly per tensor;
(2) full 0.5B runtime ppl == arena champion ppl (4.6302 @ seed 0).

Run unit test:  python -m weights.evoq
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

G = 128
P_OUT = 0.005
KPOOL = 64
LAM = 0.008          # champion operating point
SEED = 0


# ----------------------------------------------------------------- bit packing (6-bit, K<=64)
def pack6(idx: np.ndarray) -> np.ndarray:
    """idx: 1-D uint8/int array of 6-bit values, len % 4 == 0 -> packed uint8 (3 bytes / 4 vals)."""
    assert idx.size % 4 == 0
    q = idx.astype(np.uint32).reshape(-1, 4)
    v = q[:, 0] | (q[:, 1] << 6) | (q[:, 2] << 12) | (q[:, 3] << 18)
    out = np.empty((q.shape[0], 3), np.uint8)
    out[:, 0] = v & 0xFF
    out[:, 1] = (v >> 8) & 0xFF
    out[:, 2] = (v >> 16) & 0xFF
    return out.reshape(-1)


def unpack6_t(packed: torch.Tensor, n: int) -> torch.Tensor:
    """packed uint8 tensor -> n int64 indices (torch, device-agnostic, vectorized)."""
    b = packed.view(-1, 3).to(torch.int32)
    v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    out = torch.stack((v & 63, (v >> 6) & 63, (v >> 12) & 63, (v >> 18) & 63), dim=1)
    return out.reshape(-1)[:n].to(torch.long)


# ----------------------------------------------------------------- FWHT (torch, matches numpy ref)
def fwht_t(x: torch.Tensor) -> torch.Tensor:
    """FWHT along last axis (out-of-place; small inputs / tests)."""
    y = x.clone()
    fwht_inplace_t(y)
    return y


def fwht_inplace_t(x: torch.Tensor) -> torch.Tensor:
    """In-place FWHT along last axis; identical operand order to quant_sota._fwht_rows.
    Peak extra memory = x.numel()/2 (one half-size temp per stage, freed between stages)."""
    g = x.shape[-1]
    h = 1
    while h < g:
        v = x.view(-1, g // (2 * h), 2, h)
        a = v[:, :, 0, :]
        b = v[:, :, 1, :]
        t = a - b                       # must precede a.add_
        a.add_(b)
        b.copy_(t)
        h *= 2
    return x


# ----------------------------------------------------------------- encoder (mirrors codec_zoo)
def encode_tensor(W: np.ndarray, awq_s: np.ndarray, lam: float = LAM, seed: int = SEED,
                  self_check: bool = True):
    """Mirror of codec_zoo.ecvq(): Ws = W * s; _had_ecvq(Ws, lam, G, 0.005); wh / s.
    Returns dict of components + (optionally) the reference dequant for the gate."""
    from weights.codec_zoo import _ecvq_levels, _nearest_idx
    from weights.quant_sota import _fwht_rows

    rows, cols = W.shape
    assert cols % G == 0, f"in_features {cols} not a multiple of {G} (no-pad assumption)"
    Ws = (W * awq_s[None, :]).astype(np.float32)

    n_out = int(round(Ws.size * P_OUT))
    thr = np.quantile(np.abs(Ws), 1.0 - P_OUT)
    mask = np.abs(Ws) >= thr
    base = Ws.copy()
    base[mask] = 0.0

    N = base.reshape(rows, -1, G).reshape(-1, G)
    signs = (np.random.default_rng(seed).integers(0, 2, G).astype(np.float32) * 2 - 1)
    R = _fwht_rows(N * signs) / np.sqrt(G)
    amax = np.abs(R).max(1)
    amax[amax == 0] = 1.0
    Rn = R / amax[:, None]
    rng = np.random.default_rng(seed + 1)
    samp = Rn.ravel()
    samp = samp[rng.integers(0, samp.size, min(20000, samp.size))]
    lv = _ecvq_levels(samp, KPOOL, lam)                       # float32, sorted, K<=64
    assert len(lv) >= 2, "degenerate ECVQ codebook (1 level) -- would corrupt silently"
    idx_raw = _nearest_idx(Rn.ravel(), lv)
    assert idx_raw.min() >= 0 and idx_raw.max() < min(64, len(lv)), "index out of 6-bit/codebook range"
    idx = idx_raw.astype(np.uint8)

    out_pos = np.flatnonzero(mask.ravel()).astype(np.int32)   # positions in ORIGINAL W layout (int32: max 67.9M < 2^31)
    out_val = W.ravel()[out_pos].astype(np.float32)           # original (unscaled) values

    comp = dict(
        packed=pack6(idx), n_idx=idx.size, K=len(lv),
        lv=lv.astype(np.float32), amax=amax.astype(np.float32),
        signs=signs.astype(np.int8), awq_s=awq_s.astype(np.float32),
        out_pos=out_pos, out_val=out_val, rows=rows, cols=cols,
    )
    if self_check:
        # reference dequant exactly as codec_zoo does it
        Rh = lv[idx.astype(np.int64)].reshape(Rn.shape) * amax[:, None]
        back = _fwht_rows(Rh) / np.sqrt(G) * signs
        wh = back.reshape(rows, cols)
        wh[mask] = Ws[mask]
        ref = (wh / awq_s[None, :]).astype(np.float32)
        got = dequant_np(comp)                                # fp32 path (bf16 round skipped)
        err = float(np.abs(ref - got).max())
        rel = err / (float(np.abs(ref).max()) + 1e-12)
        assert rel < 1e-5, f"component dequant mismatch: max abs {err:.3e} rel {rel:.3e}"
    return comp


def dequant_np(c) -> np.ndarray:
    """Numpy dequant (reference path for tests; fp32, no bf16 round)."""
    t = dequant_t({k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in c.items()},
                  torch.device("cpu"), torch.float32, bf16_round=False)
    return t.numpy()


def dequant_t(c, device, dtype, chunk_grouprows: int = 8192, bf16_round: bool = True) -> torch.Tensor:
    """Chunked torch dequant (used by QuantLinear.forward). Processes `chunk_grouprows`
    group-rows at a time through an in-place FWHT, capping transients at ~chunk*G*4B
    (default ~4MB) + the unavoidable full fp32 output. Weight values are bf16-rounded
    before the compute cast so all compute dtypes execute the identical codec."""
    rows, cols = int(c["rows"]), int(c["cols"])
    ng = cols // G
    M = rows * ng
    lv = c["lv"].to(device, torch.float32)
    amax = c["amax"].to(device, torch.float32)
    sg = (c["signs"].to(device, torch.float32) / math.sqrt(G))      # fold /sqrt(g) into signs
    s_g = c["awq_s"].to(device, torch.float32).view(ng, G)
    packed = c["packed"].to(device)
    out = torch.empty(rows * cols, dtype=torch.float32, device=device)
    for m0 in range(0, M, chunk_grouprows):
        m1 = min(m0 + chunk_grouprows, M)
        pb = packed[m0 * G // 4 * 3: m1 * G // 4 * 3]
        idx = unpack6_t(pb, (m1 - m0) * G)
        w = lv[idx].view(m1 - m0, G)
        w *= amax[m0:m1, None]
        fwht_inplace_t(w)
        w *= sg
        gi = torch.arange(m0, m1, device=device) % ng
        w /= s_g[gi]
        out[m0 * G: m1 * G] = w.reshape(-1)
    pos = c["out_pos"].to(device)
    out[pos.to(torch.long)] = c["out_val"].to(device, torch.float32)
    W = out.view(rows, cols)
    if bf16_round:                          # arena parity: weight VALUES bf16-rounded
        W = W.to(torch.bfloat16)
    return W.to(dtype)


# ----------------------------------------------------------------- runtime module
class QuantLinear(nn.Module):
    """Linear with packed evoq weights; dequantizes per forward (no persistent fp16 weight)."""

    def __init__(self, comp, bias: torch.Tensor | None, compute_dtype=torch.float32):
        super().__init__()
        self.rows, self.cols = int(comp["rows"]), int(comp["cols"])
        self.n_idx, self.K = int(comp["n_idx"]), int(comp["K"])
        self.compute_dtype = compute_dtype
        as_t = lambda v: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        self.register_buffer("packed", as_t(comp["packed"]), persistent=False)
        self.register_buffer("lv", as_t(comp["lv"]), persistent=False)
        self.register_buffer("amax", as_t(comp["amax"]), persistent=False)
        self.register_buffer("signs", as_t(comp["signs"]), persistent=False)
        self.register_buffer("awq_s", as_t(comp["awq_s"]), persistent=False)
        self.register_buffer("out_pos", as_t(comp["out_pos"]), persistent=False)
        self.register_buffer("out_val", as_t(comp["out_val"]), persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.float32), persistent=False)
        else:
            self.bias = None

    def _comp(self):
        return dict(packed=self.packed, n_idx=self.n_idx, lv=self.lv, amax=self.amax,
                    signs=self.signs, awq_s=self.awq_s, out_pos=self.out_pos,
                    out_val=self.out_val, rows=self.rows, cols=self.cols)

    def forward(self, x):
        dev = self.packed.device                  # dequant where the packed buffers live;
        x = x.to(dev)                             # move the (small) hidden state, not 200MB of components
        W = dequant_t(self._comp(), dev, self.compute_dtype)
        b = self.bias.to(dev, self.compute_dtype) if self.bias is not None else None
        return F.linear(x.to(self.compute_dtype), W, b)

    def extra_repr(self):
        return f"out={self.rows}, in={self.cols}, K={self.K}, packed6"


# ----------------------------------------------------------------- container io
def save_container(path: str, tensors: dict, meta: dict):
    """tensors: name -> dict of components. Stored as safetensors + sidecar json."""
    from safetensors.numpy import save_file
    flat = {}
    tmeta = {}
    for name, c in tensors.items():
        for k in ("packed", "lv", "amax", "signs", "awq_s", "out_pos", "out_val"):
            flat[f"{name}::{k}"] = np.ascontiguousarray(c[k])
        tmeta[name] = dict(rows=int(c["rows"]), cols=int(c["cols"]),
                           n_idx=int(c["n_idx"]), K=int(c["K"]))
    save_file(flat, path)
    with open(path + ".json", "w") as fh:
        json.dump(dict(meta=meta, tensors=tmeta), fh)


def load_container(path: str):
    """Returns (meta, dict name -> component dict of torch tensors [cpu])."""
    from safetensors import safe_open
    with open(path + ".json") as fh:
        info = json.load(fh)
    out = {}
    with safe_open(path, framework="pt") as f:
        for name, tm in info["tensors"].items():
            c = {k: f.get_tensor(f"{name}::{k}") for k in
                 ("packed", "lv", "amax", "signs", "awq_s", "out_pos", "out_val")}
            c.update(rows=tm["rows"], cols=tm["cols"], n_idx=tm["n_idx"], K=tm["K"])
            out[name] = c
    return info["meta"], out


# ----------------------------------------------------------------- unit test
def _unit_test():
    from safetensors import safe_open
    from weights.quant_lab import WPATH, quant_keys
    print("evoq unit test: pack/FWHT/encode/dequant vs reference codec_zoo path")
    # 1) pack roundtrip
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 64, 4096).astype(np.uint8)
    got = unpack6_t(torch.from_numpy(pack6(idx)), idx.size).numpy()
    assert np.array_equal(got, idx.astype(np.int64)), "pack6/unpack6 roundtrip failed"
    print("  pack6/unpack6 roundtrip: OK")
    # 2) FWHT equivalence
    from weights.quant_sota import _fwht_rows
    X = rng.standard_normal((37, G)).astype(np.float32)
    a = _fwht_rows(X)
    b = fwht_t(torch.from_numpy(X)).numpy()
    assert np.allclose(a, b, atol=1e-4), "fwht_t != _fwht_rows"
    print("  fwht_t == _fwht_rows: OK")
    # 3) encode/dequant self-check on two real tensors (incl. a GQA-shaped k_proj)
    with safe_open(WPATH, framework="pt") as f:
        keys = sorted(quant_keys(f))
        for k in (keys[40], [x for x in keys if "k_proj" in x][0]):
            W = f.get_tensor(k).float().numpy()
            s = np.ones(W.shape[1], np.float32) * 1.3  # nontrivial awq scale
            c = encode_tensor(W, s, self_check=True)   # asserts internally
            # QuantLinear forward vs direct matmul on the reference dequant
            ql = QuantLinear(c, None)
            x = torch.from_numpy(rng.standard_normal((3, W.shape[1])).astype(np.float32))
            y1 = ql(x)
            ct = {kk: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for kk, v in c.items()}
            Wq = dequant_t(ct, torch.device("cpu"), torch.float32)   # same bf16-rounded path as forward
            y2 = x @ Wq.T
            assert torch.allclose(y1, y2, atol=1e-4), f"QuantLinear forward mismatch on {k}"
            # resident b/w: 6 (idx) + 32/G (fp32 amax) + 0.5% * 64 (int32 pos + fp32 val) + awq/lv eps
            print(f"  encode+dequant+forward [{k.split('.')[-2]}  {W.shape}]: OK "
                  f"(resident ~{6 + 32/G + 0.005*64:.2f} b/w class)")
    print("ALL UNIT TESTS PASSED")


if __name__ == "__main__":
    _unit_test()
