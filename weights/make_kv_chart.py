"""Chart I — the context term (Law 4 v2): decode speed vs KV depth, measured vs predicted.
Data: kv_depth_deep.log (clean, warm-up-controlled) + kv_depth_sweep.log (first sweep, run-order noise).
Curve: v1.1 law on 2016-xmp hybrid placement (KV on the VRAM tier, eta_kv = 0.70 single-point calibration).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans"})
TEAL, RED, SUB, INK = "#0f766e", "#d73027", "#8a8f98", "#14181f"

# ---- the law (mirror of quantprobe.plan.evaluate, hybrid on 2016-xmp, qwen3-30b @2.5) ----
def hybrid_tps(ctx, act_scale=1.0):
    ne, a, bits, ab = 1.2, 3.3, 2.5, 4.5
    act_ne = ne * ab / 8 * 1.15 * act_scale
    act_ex = (a - ne) * bits / 8 * 1.15 * act_scale
    vb, rb, geta, eta_r, eta_kv = 192, 48, 0.35, 0.38, 0.70
    kv_gb = ctx * 98304 / 1e9
    return 1 / (act_ne / (geta * vb) + act_ex / (eta_r * rb) + kv_gb / (eta_kv * vb))

# calibrate act_scale so d0 matches the measured warm baseline (file-calibrated box reality)
scale = 1.0
for _ in range(40):
    scale *= hybrid_tps(0, scale) / 20.02
MEAS_CLEAN = [(0, 20.02, 0.02), (16384, 16.12, 0.06)]
MEAS_NOISY = [(0, 17.68, 0.32), (2048, 20.19, 0.78), (8192, 18.41, 0.35)]

xs = np.linspace(0, 32768, 200)
ys = [hybrid_tps(x, scale) for x in xs]
naive = [1 / (1 / hybrid_tps(0, scale) + x * 98304 / 1e9 / 192) for x in xs]  # eta_kv=1 (bandwidth-only)

fig, ax = plt.subplots(figsize=(12, 6.75), dpi=170)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

ax.plot(xs, ys, color=TEAL, lw=2.6, label="Law 4 v2 (η$_{kv}$=0.70, single-point calibration)")
ax.plot(xs, naive, color=TEAL, lw=1.4, ls="--", alpha=0.55, label="bandwidth-only KV term (η$_{kv}$=1) — under-predicts the cost")
for x, y, e in MEAS_NOISY:
    ax.errorbar(x, y, yerr=e, fmt="o", mfc="white", mec=SUB, ecolor=SUB, ms=8, capsize=4, zorder=3)
for x, y, e in MEAS_CLEAN:
    ax.errorbar(x, y, yerr=e, fmt="o", color=RED, ms=10, capsize=5, zorder=4)
ax.annotate("warm baseline 20.02 ± 0.02", (0, 20.02), xytext=(1500, 21.1), color=INK, fontsize=11,
            arrowprops=dict(arrowstyle="-", color=SUB, lw=0.8))
ax.annotate("d16384: 16.12 ± 0.06  (−19.5%)\npre-registered −8…−15% → near-miss,\nresidual = attention compute on Pascal",
            (16384, 16.12), xytext=(17500, 17.6), color=INK, fontsize=11,
            arrowprops=dict(arrowstyle="-", color=SUB, lw=0.8))
ax.annotate("open = first sweep (run-order noise;\nfirst run after load reads low)", (2048, 20.19),
            xytext=(4200, 21.6), color=SUB, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=SUB, lw=0.8))
ax.set_xlabel("context depth (tokens of KV behind each generated token)", fontsize=12)
ax.set_ylabel("decode tok/s", fontsize=12)
ax.set_title("The context term (Law 4 v2) — every token re-reads the KV cache from its tier\n"
             "Qwen3-30B-A3B Q2_K, hybrid placement, 2016 desktop — term prompted by u/RogerAI--fyi", fontsize=13, pad=14)
ax.set_xlim(-800, 33500); ax.set_ylim(12, 23)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="lower left", fontsize=10.5, framealpha=0.95)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "x_chart_I_kvdepth.png")
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
