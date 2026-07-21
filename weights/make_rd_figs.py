"""PAPER_MOE figures F1 (gap collapse) and F4 (rate-distortion frontier as gap-ratio).
All numbers measured on the FULL WikiText-2 test set (151 windows, seqlen 2048). fp16 = 6.307.
F4 uses gap-ratio (quant ppl / own fp16) because absolute ppl is not comparable across papers
(MxMoE evaluates at seqlen 4096); the within-paper ratio cancels the context-length difference."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN, RED, BLUE, GREY, ORANGE, PURPLE = "#1a9850", "#d73027", "#3060c0", "#777777", "#e08020", "#9467bd"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FP16 = 6.307

# ---- F1: gap-collapse bar (full test set) ----
labels = ["uniform\n2-bit", "MxMoE\n(calib., sl4096)", "carve-out\n(data-free)", "+AWQ\n(data-light)", "fp16"]
ppls = [18.315, 7.01, 6.962, 6.768, 6.307]
colors = [RED, ORANGE, GREEN, GREEN, GREY]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
bars = ax.bar(labels, ppls, color=colors, edgecolor="black", lw=0.6)
ax.axhline(FP16, color=GREY, ls="--", lw=1)
for b, p in zip(bars, ppls):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.2, f"{p:.2f}", ha="center", fontsize=9)
ax.set_ylabel("WikiText-2 perplexity (lower = better)")
ax.set_title("Protecting attention + shared collapses the 2-bit gap ~18x\n(DeepSeek-V2-Lite, full test set, ~2.5 b/w)")
ax.set_ylim(6, 19.6)
ax.annotate("", xy=(2, 7.2), xytext=(0, 18.0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.3))
ax.text(0.95, 13.0, "~18x gap\ncollapse", fontsize=9, ha="center")
fig.tight_layout()
fig.savefig(os.path.join(DATA, "fig_gap_collapse.png"), dpi=140, bbox_inches="tight")

# ---- F4: rate-distortion frontier as GAP-RATIO vs whole-model b/w ----
# (b/w whole-model, gap-ratio = quant ppl / own fp16, label, color, marker)
pts = [
    (2.25, 1.184, "MxMoE (calibrated, sl4096)", ORANGE, "^"),
    (2.828, 1.099, "carve-out +AWQ @ low-bit", BLUE, "D"),
    (2.874, 1.104, "carve-out (data-free)", GREEN, "o"),
    (2.883, 1.073, "carve-out +AWQ", BLUE, "D"),
    (2.535, 1.206, "drop-writers (4.98 GB)", PURPLE, "v"),
]
fig2, ax2 = plt.subplots(figsize=(6.8, 4.4))
ax2.axhline(1.0, color=GREY, ls="--", lw=1)
ax2.text(2.17, 1.006, "fp16 = 1.00x", color=GREY, fontsize=8)
for x, y, lab, c, m in pts:
    ax2.scatter([x], [y], s=72, color=c, marker=m, edgecolor="black", lw=0.5, zorder=3)
    ax2.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8)
awq = sorted([(2.828, 1.099), (2.883, 1.073)])
ax2.plot([p[0] for p in awq], [p[1] for p in awq], color=BLUE, lw=1.2, ls="-", alpha=0.6, zorder=1)
ax2.set_xlabel("bits / weight (whole model)")
ax2.set_ylabel("gap-ratio = quant ppl / own fp16  (lower = better)")
ax2.set_title("Cross-paper frontier (gap-ratio): data-free + light-AWQ beat MxMoE")
ax2.set_xlim(2.15, 2.95); ax2.set_ylim(1.0, 1.25); ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(DATA, "fig_rate_distortion.png"), dpi=140, bbox_inches="tight")
print("saved fig_gap_collapse.png and fig_rate_distortion.png")
