"""PAPER_MOE round-2 figures F6 (rank-conditional rotation), F7 (the bottleneck in three forms),
F8 (the density of trained MoEs). All numbers measured on the full WikiText-2 test set (seqlen 2048),
DeepSeek-V2-Lite. carve-out baseline 6.962, fp16 6.307, fp16-KV-latent 6.805."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN, RED, BLUE, GREY, ORANGE, PURPLE = "#1a9850", "#d73027", "#3060c0", "#777777", "#e08020", "#9467bd"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CARVE = 6.962

# ============== F6: incoherence is rank-conditional ==============
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.3))

# (a) gauge sweep: every rotation hurts, only fp16 helps
gauges = ["fp16\n(no quant)", "identity\n(native)", "diag\nbalance", "anti", "Hadamard", "SVD"]
gppl = [6.805, 12.23, 13.20, 18.89, 1630.0, 2786.0]
gcol = [GREEN, GREY, ORANGE, ORANGE, RED, RED]
axa.bar(gauges, gppl, color=gcol, edgecolor="black", lw=0.6)
axa.axhline(CARVE, color=GREY, ls="--", lw=1); axa.text(4.4, CARVE * 1.15, "carve-out 6.96", color=GREY, fontsize=8)
axa.set_yscale("log"); axa.set_ylabel("WikiText-2 ppl (log)")
axa.set_title("(a) Gauge sweep on the KV-latent: every rotation hurts\n(insert R between kv_a, kv_b — bit-identical at fp16)")
for i, p in enumerate(gppl):
    axa.text(i, p * 1.25, f"{p:.1f}" if p < 100 else f"{p:.0f}", ha="center", fontsize=8)

# (b) the dichotomy: rotation cost vs effective rank
axb.scatter([394], [1623], s=160, color=RED, marker="o", edgecolor="black", zorder=3, label="KV-latent (low-rank)")
axb.scatter([1168], [0.006], s=160, color=GREEN, marker="D", edgecolor="black", zorder=3, label="shared-MLP intermediate (high-rank)")
axb.annotate("eff_rank 394\n→ +1623 ppl", (394, 1623), textcoords="offset points", xytext=(12, -6), fontsize=9, color=RED)
axb.annotate("eff_rank 1168\n→ +0.006 ppl", (1168, 0.006), textcoords="offset points", xytext=(-30, 30), fontsize=9, color=GREEN)
axb.annotate("", xy=(1168, 0.02), xytext=(394, 1200), arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
axb.text(560, 5, "~270,000×\non rank alone", fontsize=9)
axb.set_yscale("log"); axb.set_xlabel("effective rank of the rotated tensor"); axb.set_ylabel("Δppl from the same orthogonal rotation (log)")
axb.set_title("(b) The dichotomy: incoherence is rank-conditional"); axb.set_xlim(250, 1350); axb.set_ylim(1e-3, 1e4)
axb.grid(alpha=0.3); axb.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(DATA, "fig_rank_robustness.png"), dpi=140, bbox_inches="tight")

# ============== F7: the bottleneck in three forms ==============
fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
EPS = 1e-2
wbits = [16, 4, 2];      wd = [0.0, 0.157, 5.43]          # weight-quant of the KV-latent (Δ vs fp16-KV 6.805)
cbits = [16, 8, 4, 3, 2]; cd = [0.0, 0.018, 4.87, 350.2, 1674.5]   # cache-quant of the c_KV activation
ax2.plot(wbits, [d + EPS for d in wd], "o-", color=BLUE, lw=1.8, ms=8, label="weight-quant (kv_a, kv_b)")
ax2.plot(cbits, [d + EPS for d in cd], "s-", color=PURPLE, lw=1.8, ms=8, label="cache-quant (c_KV activation)")
ax2.scatter([2], [1623], s=150, color=RED, marker="*", edgecolor="black", zorder=4, label="gauge rotation @2-bit (Hadamard)")
ax2.set_yscale("log"); ax2.set_xlabel("bits"); ax2.set_ylabel("Δppl above the fp16-KV-latent baseline (log)")
ax2.set_title("The same low-rank channel, three forms — all cliff, none rotatable\n(8-bit cache is free; below it, collapse)")
ax2.invert_xaxis(); ax2.grid(alpha=0.3); ax2.legend(fontsize=8.5, loc="center left")
ax2.set_ylim(5e-3, 5e3)
ax2.annotate("8-bit cache +0.018 (free)", xy=(8, 0.028), xytext=(11.5, 0.5), fontsize=8, color=PURPLE,
             arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.8))
fig2.tight_layout(); fig2.savefig(os.path.join(DATA, "fig_bottleneck_forms.png"), dpi=140, bbox_inches="tight")

# ============== F8: the density of trained MoEs ==============
fig3, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.2))

# (1) expert bit floor
a1.bar(["fp16", "2-bit\n(RD floor)", "1-bit\n(binary)"], [6.307, 6.962, 259.9], color=[GREY, GREEN, RED], edgecolor="black", lw=0.6)
a1.set_yscale("log"); a1.set_ylabel("WikiText-2 ppl (log)"); a1.set_ylim(5, 700)
a1.set_title("(a) Experts: 2-bit is the floor\nrel-MSE 0.069 = D(R=2); 1-bit collapses (+253)")
for i, p in enumerate([6.307, 6.962, 259.9]):
    a1.text(i, p * 1.3, f"{p:.1f}", ha="center", fontsize=8)

# (2) routing concentration
ks = ["top-1", "top-2", "top-3", "top-6"]; mass = [0.33, 0.53, 0.68, 1.0]
a2.bar(ks, mass, color=BLUE, edgecolor="black", lw=0.6)
a2.axhline(0.9, color=RED, ls="--", lw=1); a2.text(0.0, 0.92, "90% needs ~5.3 / 6 experts", color=RED, fontsize=8)
a2.set_ylabel("cumulative routing mass"); a2.set_ylim(0, 1.08)
a2.set_title("(b) Routing is flat\ndynamic top-k cannot cut bytes/token")
for i, m in enumerate(mass): a2.text(i, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)

# (3) depth gradient
bands = ["early ¼\n(1-7)", "early ½\n(1-13)", "late ½\n(14-26)", "late ¼\n(20-26)"]
dd = [66, 102, 5.6, 2.4]; dcol = [RED, RED, GREEN, GREEN]
a3.bar(bands, dd, color=dcol, edgecolor="black", lw=0.6)
a3.set_yscale("log"); a3.set_ylabel("Δppl from 1-bit experts in that band (log)"); a3.set_ylim(1.5, 260)
a3.set_title("(c) Depth gradient: early ≫ late\nearly layers load-bearing (~40×)")
for i, d in enumerate(dd): a3.text(i, d * 1.25, f"+{d:g}", ha="center", fontsize=8)
fig3.tight_layout(); fig3.savefig(os.path.join(DATA, "fig_density.png"), dpi=140, bbox_inches="tight")

print("saved fig_rank_robustness.png (F6), fig_bottleneck_forms.png (F7), fig_density.png (F8)")
