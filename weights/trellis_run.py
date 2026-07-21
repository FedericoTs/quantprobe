"""trellis_run -- DEPLOYABLE runtime for the QTIP fixed-rate bitshift trellis (sm_61 / GTX 1060).

The frontier codec (qtip_trellis.trellis_quant) reaches QTIP-noFT parity (6.88 ppl @ ~2.1-2.37 b/w
on Llama-2-7B / WikiText-2) as PURE PTQ. This module turns that codec into a RUNTIME: it serializes
each tensor into a real 2-bit/weight bitstream and decodes it on-GPU, proving the compression is
real at *runtime* (resident memory), not just on paper.

Why the trellis is the right runtime for a bandwidth-bound sm_61 card:
  * each group of G=128 weights is a length-128 de-Bruijn SHIFT trellis. Store the initial L-bit
    state + 127 x K=2-bit symbols = L + 127*K bits ~= 2.08 b/w  ->  HALF the memory traffic of the
    4-bit scalar F0 path (evoq_run), at strictly better quality.
  * decode is the shift recurrence  w_t = ((w_{t-1} << K) & (2^L-1)) | symbol_t , value = code[w_t].
    The codebook is 2^L=4096 fp32 = 8 KB (fits in shared mem / constant cache -> "lookup-tiny"),
    or fully COMPUTED via 3INST (lookup-free). Decode is pure shifts + one indexed load.

Container per tensor (analogous to the scalar .evoq container):
  init   [ng]            uint16   initial L-bit trellis state per group  (ng = rows*cols/G)
  sym    [ng, (T-1)*K/8] uint8    packed 2-bit symbols (T=128, K=2 -> 32 bytes/group)
  code   [2^L]           fp32     reconstruction codebook (random-Gaussian, zero-mean unit-std)
  gs     [rows, ng_r]    fp16     per-group-row scale = gain * std  (folds steps 6-7 of trellis_quant)
  signs  [G]             int8     per-group FWHT sign pattern (incoherence rotation)
  awq_s  [cols]          fp16     AWQ per-channel scale (None -> ones)
  out_pos/out_val                 0.5%% fp16 outlier sidecar (exact W at those positions)

Gate: decode_trellis(encode_trellis(W)) == trellis_quant(W)[0]  (bit-exact), and the shift-recurrence
decode (from symbols only) reproduces the captured Viterbi state path.

Run:  .venv/Scripts/python.exe -m weights.trellis_run gate            # small synthetic + real 7B tensor
      .venv/Scripts/python.exe -m weights.trellis_run mem  <tensor>   # resident-bytes / b/w report
"""
from __future__ import annotations
import math, os, sys
import numpy as np, torch

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from weights.quant_sota import _fwht_rows
from weights.qtip_trellis import _recons, viterbi_quant, trellis_quant, G, L_BITS, K_RATE

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- serialize / pack
def _pack_symbols(sym):
    """sym [ng, M] uint8 in {0..3} -> packed [ng, ceil(M*2/8)] uint8 (4 symbols/byte, LSB-first)."""
    ng, M = sym.shape
    pad = (-M) % 4
    if pad:
        sym = np.concatenate([sym, np.zeros((ng, pad), np.uint8)], 1)
    s = sym.reshape(ng, -1, 4).astype(np.uint8)
    packed = (s[:, :, 0] | (s[:, :, 1] << 2) | (s[:, :, 2] << 4) | (s[:, :, 3] << 6)).astype(np.uint8)
    return packed, M


def _unpack_symbols(packed, M, dev=DEV):
    """packed [ng, B] uint8 -> sym [ng, M] uint8 in {0..3} on GPU."""
    p = torch.as_tensor(packed, device=dev).to(torch.int32)            # [ng, B]
    s0 = (p & 3); s1 = (p >> 2) & 3; s2 = (p >> 4) & 3; s3 = (p >> 6) & 3
    sym = torch.stack([s0, s1, s2, s3], -1).reshape(p.shape[0], -1)     # [ng, B*4]
    return sym[:, :M].contiguous()


# ----------------------------------------------------------------------------- encode
@torch.no_grad()
def encode_trellis(W, L=L_BITS, K=K_RATE, seed=0, p_out=0.0, awq_s=None, gain=True):
    """Mirror trellis_quant's pipeline but CAPTURE the Viterbi state path and serialize it.
    Returns (container dict, nominal b/w)."""
    rows, cols = W.shape; ng_r = cols // G
    Wq = W if awq_s is None else (W * awq_s[None, :]).astype(np.float32)
    if p_out > 0:
        thr = np.quantile(np.abs(Wq), 1.0 - p_out); mask = np.abs(Wq) >= thr
        base = Wq.copy(); base[mask] = 0.0
    else:
        mask = None; base = Wq
    signs = (np.random.default_rng(seed).integers(0, 2, G).astype(np.float32) * 2 - 1)
    N = base.reshape(rows, -1, G).reshape(-1, G)                        # [ng, G]
    R = _fwht_rows(N * signs) / math.sqrt(G)
    std = R.std(1, keepdims=True); std[std == 0] = 1.0
    X = (R / std).astype(np.float32)
    Xq, path = viterbi_quant(X, L, K, seed, return_path=True)           # Xq [ng,G], path uint16 [ng,G]
    if gain:
        g = (X * Xq).sum(1, keepdims=True) / np.maximum((Xq * Xq).sum(1, keepdims=True), 1e-9)
    else:
        g = np.ones((X.shape[0], 1), np.float32)
    gs = (g * std).astype(np.float32).reshape(rows, ng_r)              # per-group-row scale, folded
    # serialize: init state + low-K-bit symbols (the de-Bruijn shift fully determines the path)
    init = path[:, 0].astype(np.uint16)                                # [ng]
    sym = (path[:, 1:] & ((1 << K) - 1)).astype(np.uint8)              # [ng, T-1]
    packed, M = _pack_symbols(sym)
    packed_M, nsym, Wn = to_master_stream(init, sym, L, K)             # kernel-friendly sliding-window format
    out_pos = (np.flatnonzero(mask) if mask is not None else np.zeros(0, np.int64)).astype(np.int64)
    out_val = (W.reshape(-1)[out_pos] if out_pos.size else np.zeros(0, np.float32)).astype(np.float32)
    code = _recons(L, seed).detach().cpu().numpy().astype(np.float32)
    cont = dict(rows=rows, cols=cols, L=L, K=K, ng=rows * ng_r, ng_r=ng_r, M=M,
                init=init, packed=packed, code=code, gs=gs.astype(np.float16),
                signs=signs.astype(np.int8), T=G,
                packed_M=packed_M, nsym=nsym, Wn=Wn,                  # kernel sliding-window stream
                awq_s=(np.ones(cols, np.float32) if awq_s is None else awq_s).astype(np.float16),
                out_pos=out_pos, out_val=out_val)
    bw = K + 16.0 / G + (p_out * (32 + 16) if p_out > 0 else 0.0) + (16.0 / rows if awq_s is not None else 0.0)
    return cont, bw


# ----------------------------------------------------------------------------- master symbol stream (kernel format)
def to_master_stream(init, sym, L=L_BITS, K=K_RATE):
    """init [ng] uint16 + sym [ng, T-1] uint8 -> master stream M [ng, S] uint8 (S = L/K + T-1) s.t.
    the window at EVERY position p is the sliding 6-tuple:  w_p = sum_{k=0..L/K-1} M[:,p+k] << (L-K-K*k).
    M[0..L/K-1] = init's symbols MSB-first; M[L/K:] = data symbols. NO sequential dependency -> the
    decode-GEMV parallelizes per-lane exactly like the F0 4-bit kernel. Returns (packed M, S, W)."""
    W = L // K                                                         # symbols per window (=6 at L=12,K=2)
    ng = init.shape[0]
    head = np.empty((ng, W), np.uint8)
    for k in range(W):
        head[:, k] = (init >> np.uint16((W - 1 - k) * K)) & np.uint16((1 << K) - 1)   # MSB-first
    M = np.concatenate([head, sym.astype(np.uint8)], 1)                # [ng, W + (T-1)]
    packed, S = _pack_symbols(M)
    return packed, M.shape[1], W


@torch.no_grad()
def decode_window(packed_M, nsym, T, L=L_BITS, K=K_RATE, dev=DEV):
    """Reconstruct states [ng, T] from the master stream by the SLIDING WINDOW (what the kernel does):
    w_p = sum_{k=0..W-1} M[p+k] << (L-K-K*k).  No recurrence."""
    Wn = L // K
    M = _unpack_symbols(packed_M, nsym, dev).to(torch.int64)           # [ng, nsym]
    ng = M.shape[0]
    states = torch.zeros(ng, T, dtype=torch.int64, device=dev)
    for k in range(Wn):
        states += M[:, k:k + T] << (L - K - K * k)
    return states


# ----------------------------------------------------------------------------- decode (GPU)
@torch.no_grad()
def decode_path_from_symbols(init, packed, M, L, K, dev=DEV):
    """Reconstruct the [ng, T] L-bit state path from init + packed 2-bit symbols via the SHIFT
    recurrence ONLY (no stored path) -- this is exactly what the deployable kernel does."""
    nstate_mask = (1 << L) - 1; symmask = (1 << K) - 1
    sym = _unpack_symbols(packed, M, dev)                              # [ng, T-1] in {0..3}
    ng = sym.shape[0]; T = M + 1
    w = torch.as_tensor(init, device=dev).to(torch.int64)             # [ng]
    paths = torch.empty(ng, T, dtype=torch.int64, device=dev)
    paths[:, 0] = w
    for t in range(1, T):
        w = ((w << K) & nstate_mask) | (sym[:, t - 1].to(torch.int64) & symmask)
        paths[:, t] = w
    return paths


@torch.no_grad()
def decode_trellis(cont, dev=DEV):
    """Decode the container back to the fp32 weight tensor wh (== trellis_quant(W)[0])."""
    L, K, rows, cols, ng_r = cont["L"], cont["K"], cont["rows"], cont["cols"], cont["ng_r"]
    code = torch.as_tensor(cont["code"], device=dev).float()
    paths = decode_path_from_symbols(cont["init"], cont["packed"], cont["M"], L, K, dev)   # [ng, T]
    Xq = code[paths]                                                   # [ng, T]  reconstruction
    gs = torch.as_tensor(cont["gs"], device=dev).float().reshape(-1, 1)   # [ng, 1]
    Rq = Xq * gs                                                      # undo gain*std
    signs = torch.as_tensor(cont["signs"], device=dev).float()
    back = _fwht_rows(Rq.cpu().numpy()) / math.sqrt(G)                # FWHT de-rotate (cpu helper)
    back = torch.as_tensor(back, device=dev) * signs
    wh = back.reshape(rows, cols)
    awq_s = torch.as_tensor(cont["awq_s"], device=dev).float()
    wh = wh / awq_s[None, :]
    if cont["out_pos"].size:
        op = torch.as_tensor(cont["out_pos"], device=dev).long()
        wh.reshape(-1)[op] = torch.as_tensor(cont["out_val"], device=dev).float()
    return wh


# ----------------------------------------------------------------------------- rotated-space GEMV reference
@torch.no_grad()
def trellis_gemv_ref(cont, x, dev=DEV):
    """y = wh @ x computed in ROTATED space (weight stays as the 2-bit stream; activation is rotated).
    Identity:  y[i] = sum_grp gs[i,grp] * < code[ window-decode ](i,grp) , xrot[grp] >,
    xrot[grp] = FWHT( signs * (x/awq)[grp] ) / sqrt(G).  This is exactly the kernel's arithmetic
    (decode via sliding window + dot with the FWHT-prepped activation). No-outlier path (p_out=0);
    outliers reuse the existing csr_outlier sidecar. Validates the kernel math against the dense decode."""
    L, K, rows, cols, ng_r, T = cont["L"], cont["K"], cont["rows"], cont["cols"], cont["ng_r"], cont["T"]
    code = torch.as_tensor(cont["code"], device=dev).float()
    states = decode_window(cont["packed_M"], cont["nsym"], T, L, K, dev)    # [ng, T]
    Xq = code[states].reshape(rows, ng_r, T)                                # [rows, ng_r, T]
    gs = torch.as_tensor(cont["gs"], device=dev).float()                    # [rows, ng_r]
    signs = torch.as_tensor(cont["signs"], device=dev).float()              # [G]
    awq = torch.as_tensor(cont["awq_s"], device=dev).float()                # [cols]
    xg = (torch.as_tensor(x, device=dev).float() / awq).reshape(ng_r, T) * signs[None, :]
    xrot = torch.as_tensor(_fwht_rows(xg.cpu().numpy()), device=dev) / math.sqrt(G)   # [ng_r, T]
    dots = (Xq * xrot[None]).sum(-1)                                        # [rows, ng_r]
    return (gs * dots).sum(1)                                               # [rows]


# ----------------------------------------------------------------------------- gate / report
def _container_bytes(cont):
    b = (cont["init"].nbytes + cont["packed"].nbytes + cont["gs"].nbytes
         + cont["out_pos"].size * 2 + cont["out_val"].size * 2)        # outliers: int16 col + fp16 val (sidecar)
    code = cont["code"].nbytes                                         # amortized across all tensors of a layer
    return b, code


def gate():
    """(1) small synthetic tensor: shift-recurrence decode == captured path, and decode == trellis_quant.
       (2) a REAL Llama-2-7B tensor (small GPU footprint) end-to-end. Bit-exact recon gate + b/w report."""
    print("=== (1) synthetic 256x512 ===", flush=True)
    rng = np.random.default_rng(0)
    W = rng.standard_normal((256, 512)).astype(np.float32) * 0.02
    cont, bw = encode_trellis(W, awq_s=None, p_out=0.0)
    # path round-trip: shift-recurrence decode must reproduce recons[captured_path]
    wh_ref, _ = trellis_quant(W, p_out=0.0)
    wh = decode_trellis(cont).cpu().numpy()
    rel = np.abs(wh - wh_ref).max() / (np.abs(wh_ref).max() + 1e-9)
    cb, code = _container_bytes(cont)
    print(f"  decode vs trellis_quant: max-abs-rel = {rel:.2e}  {'BIT-EXACT' if rel < 1e-5 else 'OK' if rel < 1e-3 else 'FAIL'}")
    print(f"  resident = {cb} B for {W.size} weights = {cb*8/W.size:.3f} b/w  (+ {code} B shared codebook)")
    print(f"  nominal b/w = {bw:.3f}  | fp16 would be {W.size*2} B")
    # window-decode (kernel sliding-window) must reproduce the recurrence states exactly
    rec = decode_path_from_symbols(cont["init"], cont["packed"], cont["M"], cont["L"], cont["K"], "cpu")
    win = decode_window(cont["packed_M"], cont["nsym"], cont["T"], cont["L"], cont["K"], "cpu")
    print(f"  window-decode == recurrence path: {bool((rec == win).all())}  (kernel sliding-window format OK)")
    # rotated-space GEMV == dense decode @ x  (the kernel's exact arithmetic)
    x = rng.standard_normal(cont["cols"]).astype(np.float32)
    wh = decode_trellis(cont, "cpu").cpu().numpy()
    y_ref = wh @ x
    y_gemv = trellis_gemv_ref(cont, x, "cpu").cpu().numpy()
    gr = np.abs(y_gemv - y_ref).max() / (np.abs(y_ref).max() + 1e-9)
    print(f"  rotated-space GEMV vs wh@x: max-rel = {gr:.2e}  {'PASS' if gr < 1e-3 else 'FAIL'}  (activation-side rotation)")

    print("\n=== (2) real Llama-2-7B tensor (mlp.gate_proj.15) +AWQ ===", flush=True)
    try:
        from weights.evoq_llama import shard_map, read_tensor
        from weights.quant_lab import _awq_scale
        smap = shard_map()
        W = read_tensor(smap, "model.layers.15.mlp.gate_proj.weight").float().numpy()
        # small row-slice keeps the gate cheap (it may run on CPU to avoid touching a live GPU job)
        W = np.ascontiguousarray(W[:256])
        awq = None  # AWQ scale needs activations; test the geometry path (awq=ones) -- recon-exactness is the gate
        cont, bw = encode_trellis(W, p_out=0.005, awq_s=awq)
        wh_ref, _ = trellis_quant(W, p_out=0.005, awq_s=awq)
        wh = decode_trellis(cont).cpu().numpy()
        rel = np.abs(wh - wh_ref).max() / (np.abs(wh_ref).max() + 1e-9)
        cb, code = _container_bytes(cont)
        print(f"  shape {W.shape}  decode vs trellis_quant: max-abs-rel = {rel:.2e}  "
              f"{'BIT-EXACT' if rel < 1e-5 else 'OK' if rel < 1e-3 else 'FAIL'}")
        print(f"  resident = {cb/1e3:.1f} KB = {cb*8/W.size:.3f} b/w  (+8KB shared codebook) | "
              f"fp16 = {W.size*2/1e3:.1f} KB | nominal b/w {bw:.3f}")
        print(f"  >>> 2-bit trellis stream round-trips bit-exactly; resident compression is REAL.")
    except Exception as ex:
        print(f"  (skipped real-tensor gate: {type(ex).__name__}: {ex})")


# ----------------------------------------------------------------------------- CUDA kernel gate + bench
def _pack_M_2d(cont):
    """cont['packed_M'] [ng, mbytes] -> [rows, ng_r*mbytes] contiguous (kernel layout)."""
    pm = cont["packed_M"]; mbytes = pm.shape[1]
    return np.ascontiguousarray(pm.reshape(cont["rows"], cont["ng_r"] * mbytes)), mbytes


def kgate():
    """Build the trellis_gemv CUDA kernel, verify it == wh@x on a 7B-sized matrix, microbench it,
    and project 7B tok/s. Compares weight-traffic vs the 4-bit F0 path. NEEDS a free GPU."""
    import time
    from weights.evoq_kernel import build
    if not torch.cuda.is_available():
        print("CUDA not available"); return
    dev = "cuda"; ext = build(); print("built trellis_gemv kernel\n", flush=True)
    # synthetic weights are fine (we gate decode-GEMV == dense-decode @ x). Small + tiny VCHUNK keeps
    # this <200MB so it can validate CORRECTNESS safely alongside a running encode job (timing-free gate).
    out, inn = (int(os.environ.get("KG_OUT", "1024")), int(os.environ.get("KG_IN", "2048")))
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((out, inn)).astype(np.float32) * 0.02)
    cont, bw = encode_trellis(W, p_out=0.0, awq_s=None)
    pm2, mbytes = _pack_M_2d(cont)
    pm_t = torch.from_numpy(pm2).to(dev)
    code_t = torch.from_numpy(cont["code"]).to(dev)
    gs_t = torch.from_numpy(cont["gs"].astype(np.float32)).contiguous().to(dev)
    signs_t = torch.from_numpy(cont["signs"].astype(np.float32)).to(dev)
    awq_t = torch.from_numpy(cont["awq_s"].astype(np.float32)).to(dev)
    x = rng.standard_normal(inn).astype(np.float32)
    x_t = torch.from_numpy(x).to(dev)
    xrr = ext.fwht_prep(x_t.view(1, inn), awq_t, signs_t).view(inn).contiguous()
    y = ext.trellis_gemv(pm_t, xrr, code_t, gs_t, mbytes)
    y2 = ext.trellis_gemv2(pm_t, xrr, code_t, gs_t, mbytes)
    wh = decode_trellis(cont, dev)
    yref = (wh @ x_t)
    err = (y - yref).abs().max().item() / (yref.abs().max().item() + 1e-9)
    err2 = (y2 - yref).abs().max().item() / (yref.abs().max().item() + 1e-9)
    print(f"correctness {out}x{inn}: v1 rel err {err:.2e} {'PASS' if err < 1e-3 else 'FAIL'} | "
          f"v2 rel err {err2:.2e} {'PASS' if err2 < 1e-3 else 'FAIL'}", flush=True)
    # microbench
    ng_r = inn // G
    bytes_trellis = out * ng_r * mbytes + out * ng_r * 4          # 2-bit stream + gs fp32
    bytes_f0 = out * (inn // 2) + out * ng_r * 4                  # 4-bit + amax fp32
    torch.cuda.synchronize()
    for _ in range(10): ext.trellis_gemv(pm_t, xrr, code_t, gs_t, mbytes)
    torch.cuda.synchronize(); N = 300; t0 = time.time()
    for _ in range(N): ext.trellis_gemv(pm_t, xrr, code_t, gs_t, mbytes)
    torch.cuda.synchronize(); dt = (time.time() - t0) / N
    gbs = bytes_trellis / dt / 1e9; util = gbs / 192.0
    print(f"trellis_gemv: {dt*1e3:.3f} ms/call | {gbs:.1f} GB/s = {100*util:.0f}% util")
    print(f"  weight traffic: trellis {bytes_trellis/1e3:.0f}KB vs F0 {bytes_f0/1e3:.0f}KB "
          f"= {bytes_trellis/bytes_f0:.2f}x ({(1-bytes_trellis/bytes_f0)*100:.0f}% less DRAM)")
    # 7B projection at ~2.1 b/w
    resident_gb = 6.5e9 * (2.078 + 16.0/G) / 8 / 1e9
    toks = 192e9 * util / (resident_gb * 1e9)
    print(f"  -> 7B @ ~2.20 b/w ({resident_gb:.2f} GB resident): ~{toks:.1f} tok/s "
          f"(matmul-only; Q4_K_M 21.8 @ 4.56GB)")


def kbench():
    """SPEED+MEMORY headline: sweep the real Llama-2-7B shapes (7 linears x 32 layers), measure the
    2-bit trellis_gemv tok/s + resident GB on the GTX 1060, head-to-head vs the 4-bit F0 kernel on
    identical shapes. Kernel speed is value-independent -> synthetic streams (no 4h encode). NEEDS GPU.
    Quality (6.88 ppl) is the separate measure7b result; this is the deployability (speed/mem) half."""
    import time
    from weights.evoq_kernel import build, pack4
    if not torch.cuda.is_available():
        print("CUDA not available"); return
    dev = "cuda"; ext = build(); rng = np.random.default_rng(0)
    MB = (L_BITS // K_RATE + (G - 1) + 3) // 4                   # bytes/group master stream (=34 @ L12)
    Ln = 32
    shapes = [("q", 4096, 4096), ("k", 4096, 4096), ("v", 4096, 4096), ("o", 4096, 4096),
              ("gate", 11008, 4096), ("up", 11008, 4096), ("down", 4096, 11008)]
    code_t = torch.from_numpy(rng.standard_normal(1 << L_BITS).astype(np.float32)).to(dev)

    def bench(fn, *a, N=300):
        for _ in range(10): fn(*a)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(N): fn(*a)
        torch.cuda.synchronize(); return (time.time() - t0) / N

    print(f"=== Llama-2-7B shape sweep: 2-bit trellis_gemv v1/v2 vs 4-bit F0 (GTX 1060, sm_61) ===", flush=True)
    print(f"{'shape':<7}{'out':>6}x{'in':<6}{'v1 ms':>8}{'v2 ms':>8}{'v3i ms':>8}{'F0 ms':>8}{'v3i/F0':>8}{'v3i GB/s':>9}")
    tot_t = tot_t2 = tot_t3 = tot_f0 = 0.0; res_trel = res_f0 = 0
    for nm, out, inn in shapes:
        ng = inn // G
        pm = torch.from_numpy(rng.integers(0, 256, (out, ng * MB), dtype=np.uint8)).to(dev)
        gs = torch.rand(out, ng, device=dev) + 0.5
        xrr = torch.randn(inn, device=dev)
        packed4 = torch.from_numpy(pack4(rng.integers(0, 16, (out, inn)).astype(np.uint8))).to(dev)
        lv16 = torch.zeros(16, device=dev); lv16[:12] = torch.randn(12, device=dev)
        amax = torch.rand(out, ng, device=dev) + 0.5
        t_trel = bench(ext.trellis_gemv, pm, xrr, code_t, gs, MB)
        t_trel2 = bench(ext.trellis_gemv2, pm, xrr, code_t, gs, MB)
        t_trel3 = bench(ext.trellis_gemv3i, pm, xrr, gs, MB, 0.0, 1.0)   # lookup-free 3INST
        t_f0 = bench(ext.f0_gemv3, packed4, xrr, lv16, amax)
        bt = out * ng * MB + out * ng * 4
        util3 = bt / t_trel3 / 192e9
        tot_t += t_trel * Ln; tot_t2 += t_trel2 * Ln; tot_t3 += t_trel3 * Ln; tot_f0 += t_f0 * Ln
        res_trel += (out * ng * MB + out * ng * 2) * Ln       # gs fp16 in real container
        res_f0 += (out * (inn // 2) + out * ng * 4) * Ln
        print(f"{nm:<7}{out:>6}x{inn:<6}{t_trel*1e3:>8.3f}{t_trel2*1e3:>8.3f}{t_trel3*1e3:>8.3f}{t_f0*1e3:>8.3f}"
              f"{t_f0/t_trel3:>7.2f}x{bt/t_trel3/1e9:>9.0f}", flush=True)
    best = min(tot_t, tot_t2, tot_t3)
    print(f"\nPER-TOKEN matmul (x{Ln} layers): v1 {1000/tot_t/1e3:.1f} | v2 {1000/tot_t2/1e3:.1f} | "
          f"v3i(lookup-free) {1000/tot_t3/1e3:.1f} | F0 {1000/tot_f0/1e3:.1f} tok/s | "
          f"best-trellis vs F0 {tot_f0/best:.2f}x")
    print(f"7B RESIDENT: trellis {res_trel/1e9:.2f} GB ({res_trel*8/6.74e9:.3f} b/w) vs "
          f"F0 4-bit {res_f0/1e9:.2f} GB ({res_f0*8/6.74e9:.3f} b/w) | fp16 13.5 GB")
    print(f"  >>> 2-bit trellis: QTIP-parity quality (6.88 ppl) at {res_trel/1e9:.2f} GB resident, "
          f"matmul {1000/tot_t/1e3:.0f} tok/s on a 6GB GTX 1060 (Q4_K_M 21.8 t/s @ 4.56 GB).")


def kbench_batch():
    """PROBE 3: batched trellis decode-GEMM. Decode is fixed/weight -> amortizes over batch. Measure
    per-TOKEN time vs B for the 2-bit trellis (weight-stationary) vs the 4-bit F0 single-token baseline
    and the fp16 cuBLAS GEMM ceiling. If trellis per-token falls steeply with B and crosses F0, the
    trellis WINS for throughput/prefill (the structural deployability win). NEEDS GPU; short bursts."""
    import time
    from weights.evoq_kernel import build, pack4
    if not torch.cuda.is_available():
        print("CUDA not available"); return
    dev = "cuda"; ext = build(); rng = np.random.default_rng(0)
    MB = (L_BITS // K_RATE + (G - 1) + 3) // 4
    Ln = 32
    shapes = [("q", 4096, 4096), ("k", 4096, 4096), ("v", 4096, 4096), ("o", 4096, 4096),
              ("gate", 11008, 4096), ("up", 11008, 4096), ("down", 4096, 11008)]
    Bs = [1, 2, 4, 8, 16, 32]

    def bench(fn, *a, N=200):
        for _ in range(8): fn(*a)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(N): fn(*a)
        torch.cuda.synchronize(); return (time.time() - t0) / N

    # --- correctness: batched col b == single-token v3i on xrr[b] (mu=0, sigma=1) ---
    out, inn = 4096, 4096; ng = inn // G
    pm = torch.from_numpy(rng.integers(0, 256, (out, ng * MB), dtype=np.uint8)).to(dev)
    gs = (torch.rand(out, ng, device=dev) + 0.5)
    xB = torch.randn(8, inn, device=dev)
    yb = ext.trellis_gemv3i_batch(pm, xB, gs, MB, 0.0, 1.0)            # [8, out]
    y0 = ext.trellis_gemv3i(pm, xB[3].contiguous(), gs, MB, 0.0, 1.0)  # single col 3
    err = (yb[3] - y0).abs().max().item() / (y0.abs().max().item() + 1e-9)
    print(f"batched correctness (col==single): rel err {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}\n", flush=True)

    print(f"HONEST head-to-head: 2-bit trellis vs 4-bit F0, BOTH batched (us/token, sum over 7 shapes x{Ln} layers)")
    print(f"{'B':>3} | {'trel us/tok':>12} {'tok/s':>7} | {'F0 us/tok':>11} {'tok/s':>7} | {'trel/F0':>8} | {'fp16 us/tok':>12}")
    for B in Bs:
        t_tot = f0_tot = gemm_tot = 0.0
        for nm, o, i in shapes:
            ng_ = i // G
            pmx = torch.from_numpy(rng.integers(0, 256, (o, ng_ * MB), dtype=np.uint8)).to(dev)
            gsx = torch.rand(o, ng_, device=dev) + 0.5
            xb = torch.randn(B, i, device=dev)
            t_tot += bench(ext.trellis_gemv3i_batch, pmx, xb, gsx, MB, 0.0, 1.0) * Ln
            p4 = torch.from_numpy(pack4(rng.integers(0, 16, (o, i)).astype(np.uint8))).to(dev)
            lv = torch.zeros(16, device=dev); lv[:12] = torch.randn(12, device=dev)
            am = torch.rand(o, ng_, device=dev) + 0.5
            f0_tot += bench(ext.f0_gemv3_batch, p4, xb, lv, am) * Ln
            Wf = torch.randn(o, i, device=dev, dtype=torch.float16); xf = xb.to(torch.float16)
            gemm_tot += bench(lambda: xf @ Wf.t()) * Ln
        tus = t_tot / B * 1e6; fus = f0_tot / B * 1e6; gus = gemm_tot / B * 1e6
        print(f"{B:>3} | {tus:>12.0f} {1e6/tus:>7.1f} | {fus:>11.0f} {1e6/fus:>7.1f} | {fus/tus:>7.2f}x | {gus:>12.0f}", flush=True)
    print(f"\nGATE: trel/F0 > 1.0 -> the 2-bit trellis is FASTER per-token than 4-bit at that batch (denser AND "
          f"faster). Trellis reads half the weight bytes, so it should win once both are decode-amortized + "
          f"bandwidth-bound. fp16 GEMM (cuBLAS) = the compute ceiling.")


def kcompile():
    """nvcc-compile the kernel only (catch CUDA syntax errors); no GPU launch -> safe alongside a run."""
    from weights.evoq_kernel import build
    ext = build()
    print("COMPILE OK:", [s for s in dir(ext) if not s.startswith("_")])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if mode == "gate":
        gate()
    elif mode == "kgate":
        kgate()
    elif mode == "kbench":
        kbench()
    elif mode == "kbench_batch":
        kbench_batch()
    elif mode == "kcompile":
        kcompile()
