"""rot_probe.py -- DATA-FREE test of whether a GLOBAL rotation adds anything over the
trellis's existing per-group-128 signed-Hadamard, for Gemma-4-12B 2-bit MLP.

Three levers, all data-free, minutes, on REAL Gemma layer weights (streamed from the single
safetensors, no forward pass):

  L1 GLOBAL-vs-LOCAL RECON: does rotating the FULL input axis (15360 for down; 3840 for gate/up)
     by a Hadamard BEFORE the trellis change the achieved 2-bit rel-MSE vs the plain per-group-128
     trellis? Axiom-3 says the trellis already equalizes weight rel-MSE to ~0.068, so we expect
     ~no change -> that alone kills "global rotation as a weight-MSE lever". We measure it to be sure,
     AND we measure the thing global rotation is SUPPOSED to fix: cross-GROUP coherence (max column
     norm dispersion, and worst-group amax) which per-group Hadamard cannot touch.

  L2 GATE/UP JOINT (the VECTOR-quantization opening, axiom 3): gate_proj and up_proj share the SAME
     input (post-attn-norm residual). Stack them [2*interm, hidden] and ask: are their ROWS correlated
     across the two matrices (canonical correlation / shared right-singular subspace)? If the two
     projections share input-space structure, a joint rotation of their common input decorrelates
     redundancy a per-matrix codec cannot. Data-free tell = top canonical correlations >> random.

  L3 DOWN_PROJ INPUT-COHERENCE: down_proj's input is the GeGLU-gated hidden (gelu(gate*x) * (up*x)),
     which is NOT the residual stream and is heavy-tailed/sparse. A global rotation of down's input is
     ONLY function-preserving if applied to the gated hidden at runtime (a real inference-time rotate),
     unlike gate/up whose input rotation folds into the norm for free. We measure down's column-norm
     dispersion (kurtosis of per-column energy) = the data-free proxy for how much a global input
     rotation would flatten its amax field.

Prints, per lever, the number that decides go/no-go BEFORE any 6-8h trellis run.

Run: .venv-gemma/Scripts/python -u -m weights.rot_probe   (needs only numpy+safetensors+torch-cpu ok)
"""
from __future__ import annotations
import math, os, sys
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from safetensors import safe_open
from weights.qtip_trellis import trellis_quant, G
from weights.quant_sota import _fwht_rows

MDIR = "D:/evo-compress-data/gemma-4-12b"
ST = os.path.join(MDIR, "model.safetensors")
LP = "model.language_model."


def relmse(W, Wh):
    return float(((W - Wh) ** 2).sum() / (W ** 2).sum())


def had_full(W_in_axis):
    """Apply a single random-sign FWHT across the FULL input axis (last axis), padded to pow2.
    Returns rotated matrix (same shape up to pad) and the inverse info. Orthonormal."""
    rows, cols = W_in_axis.shape
    n = 1 << (cols - 1).bit_length()
    A = np.zeros((rows, n), np.float32); A[:, :cols] = W_in_axis
    signs = np.random.default_rng(0).integers(0, 2, n).astype(np.float32) * 2 - 1
    R = _fwht_rows(A * signs) / math.sqrt(n)
    return R, signs, cols


def coherence(W):
    """Per-column energy dispersion: kurtosis of column L2 norms + max/median ratio.
    This is what a GLOBAL rotation flattens and a per-group-128 rotation leaves cross-group."""
    cn = np.linalg.norm(W, axis=0)
    k = float(((cn - cn.mean()) ** 4).mean() / (cn.var() ** 2 + 1e-30) - 3)
    disp = float(cn.max() / (np.median(cn) + 1e-30))
    return k, disp


SROWS = int(os.environ.get("ROT_SROWS", "1024"))   # row-subsample: rel-MSE & coherence are row stats


def probe_recon(name, W):
    """L1: plain per-group trellis vs global-Hadamard-then-per-group-trellis. rel-MSE + coherence.
    Row-subsampled (rel-MSE per-group-row is an unbiased row statistic; coherence is per-column so
    uses full matrix for the column norms)."""
    Wf = np.ascontiguousarray(W.astype(np.float32))
    k0, d0 = coherence(Wf)                              # full-matrix column-norm dispersion
    Ws = np.ascontiguousarray(Wf[:SROWS])
    wh0, _ = trellis_quant(Ws, K=2)
    r0 = relmse(Ws, np.asarray(wh0, np.float32).reshape(Ws.shape))
    # global rotation of the INPUT axis first, then per-group trellis on the rotated matrix.
    R, signs, cols = had_full(Ws)
    kR, dR = coherence(had_full(Wf)[0])                 # coherence of fully-rotated matrix (all rows)
    whR, _ = trellis_quant(np.ascontiguousarray(R), K=2)
    rR = relmse(R, np.asarray(whR, np.float32).reshape(R.shape))
    print(f"  {name:22s}: plain rel-MSE={r0:.4f} (colnorm-kurt={k0:+6.1f} disp={d0:5.1f})  "
          f"| +global-Had rel-MSE={rR:.4f} (kurt={kR:+5.1f} disp={dR:4.1f})", flush=True)
    return r0, rR, k0, kR


def probe_gateup(g, up):
    """L2: canonical correlation between gate_proj and up_proj row spaces (shared input structure).
    High CC => a joint input rotation exposes cross-matrix redundancy a per-matrix codec misses."""
    g = g.astype(np.float64); up = up.astype(np.float64)
    # orthonormal bases of the two ROW spaces (rows live in shared input space R^hidden)
    Ug, _, _ = np.linalg.svd(g, full_matrices=False)   # not needed; we want input-space (right) vectors
    # right-singular vectors span the input directions each projection reads
    _, _, Vg = np.linalg.svd(g, full_matrices=False)   # [r, hidden]
    _, _, Vu = np.linalg.svd(up, full_matrices=False)
    r = min(64, Vg.shape[0], Vu.shape[0])
    M = Vg[:r] @ Vu[:r].T                               # [r,r], singular values = canonical correlations
    cc = np.linalg.svd(M, compute_uv=False)
    # random baseline: two random r-dim subspaces of R^hidden
    hid = g.shape[1]
    rng = np.random.default_rng(0)
    Qa, _ = np.linalg.qr(rng.standard_normal((hid, r))); Qb, _ = np.linalg.qr(rng.standard_normal((hid, r)))
    ccrand = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    print(f"  gate/up canon-corr top5={np.round(cc[:5],3)}  mean(top{r})={cc.mean():.3f}  "
          f"| random mean={ccrand.mean():.3f}  (ratio {cc.mean()/ccrand.mean():.2f}x)", flush=True)
    return float(cc.mean()), float(ccrand.mean())


def main():
    layers = [0, 5, 23, 47]     # depth spread (axiom 6: early fragile); 5 is first full-attn layer
    f = safe_open(ST, framework="pt")
    g = lambda k: f.get_tensor(k).float().numpy()
    print(f"ROT-PROBE Gemma-4-12B: global-rotation levers, DATA-FREE. groups G={G}\n")
    print("[L1] weight-recon: plain per-group-128 trellis vs +global-Hadamard input rotation")
    print("     (axiom-3: trellis already equalizes rel-MSE~0.068; a global rotation that LOWERS it")
    print("      or sharply cuts colnorm-kurt/disp is the only weight-space signal for the lever)")
    agg = []
    for li in layers:
        pre = f"{LP}layers.{li}."
        for short in ("gate_proj", "up_proj", "down_proj"):
            W = g(pre + f"mlp.{short}.weight")
            r0, rR, k0, kR = probe_recon(f"L{li} {short}", W)
            agg.append((r0, rR))
    r0m = np.mean([a for a, _ in agg]); rRm = np.mean([b for _, b in agg])
    print(f"\n  L1 VERDICT: mean plain rel-MSE={r0m:.4f} vs +global-Had={rRm:.4f} "
          f"(delta {rRm-r0m:+.4f}). {'GLOBAL ROTATION HELPS RECON' if rRm < r0m-0.002 else 'NO weight-recon gain (expected by axiom 3) -> global rotation is NOT a scalar-recon lever'}")

    print("\n[L2] gate/up JOINT rotation opening (shared residual input -> possible cross-matrix redundancy)")
    ratios = []
    for li in layers:
        pre = f"{LP}layers.{li}."
        gg = g(pre + "mlp.gate_proj.weight"); uu = g(pre + "mlp.up_proj.weight")
        cc, ccr = probe_gateup(gg, uu); ratios.append(cc / ccr)
    print(f"\n  L2 VERDICT: mean gate/up canon-corr ratio vs random = {np.mean(ratios):.2f}x. "
          f"{'CORRELATED -> joint VQ/rotation lever is real' if np.mean(ratios) > 1.5 else 'NEAR-RANDOM -> gate/up input subspaces independent -> joint rotation buys nothing'}")

    print("\n[L3] down_proj input coherence (GeGLU-gated hidden; global rotation would need runtime rotate)")
    for li in layers:
        W = g(f"{LP}layers.{li}.mlp.down_proj.weight")
        k, d = coherence(W)
        print(f"  L{li} down_proj input-colnorm: kurt={k:+7.1f}  max/median={d:6.1f}  "
              f"(high => a runtime global input rotation could flatten the amax field the per-group cannot)", flush=True)


if __name__ == "__main__":
    main()
