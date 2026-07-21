"""X/Twitter chart: data-free 2-bit compression across 3 architectures on a 6GB GPU.
All numbers measured on WikiText-2. gap-ratio = 2-bit ppl / fp16 ppl (the cross-model quality cost)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INK, SUB, GREY = "#14181f", "#5b6572", "#c9ccd1"
GREEN, AMBER, RED = "#1a9850", "#e8901c", "#d73027"

models = ["DeepSeek-V2-Lite\n16B  ·  MoE", "Qwen2.5-7B\n7B  ·  dense", "Gemma 4 12B\nnaive 2-bit", "Gemma 4 12B\ndepth-aware 2-bit"]
fp16 = [6.31, 6.96, 7.37, 7.37]
twob = [6.96, 9.69, 14.06, 10.71]
gap = ["1.10×", "1.39×", "1.91×", "1.45×"]
acc = [GREEN, AMBER, RED, "#1d9e70"]

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GREY})
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=170)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

x = [0, 1.35, 2.7, 4.05]; w = 0.42
for i in range(4):
    ax.bar(x[i] - w/2, fp16[i], w, color=GREY, edgecolor="white", lw=1, zorder=3)
    ax.bar(x[i] + w/2, twob[i], w, color=acc[i], edgecolor="white", lw=1, zorder=3)
    ax.text(x[i] - w/2, fp16[i] + 0.18, f"{fp16[i]:.2f}", ha="center", va="bottom", fontsize=11, color=SUB, zorder=4)
    ax.text(x[i] + w/2, twob[i] + 0.18, f"{twob[i]:.2f}", ha="center", va="bottom", fontsize=12, color=acc[i], fontweight="bold", zorder=4)
    # hero gap-ratio
    ax.text(x[i], twob[i] + 1.35, gap[i], ha="center", va="bottom", fontsize=27, color=acc[i], fontweight="bold", zorder=5)
    ax.text(x[i], twob[i] + 1.05, "quality cost", ha="center", va="bottom", fontsize=9.5, color=SUB, zorder=5)

ax.set_xticks(x); ax.set_xticklabels(models, fontsize=13, color=INK, linespacing=1.5)
ax.set_ylim(0, 18.6); ax.set_yticks([])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GREY)
ax.tick_params(axis="x", length=0, pad=10)

# legend (inline)
ax.bar(-0.9, 0, color=GREY, label="fp16  (original)")
ax.bar(-0.9, 0, color=INK, label="2-bit  (ours, data-free)")
leg = ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=1.1, bbox_to_anchor=(0.005, 0.98))

# trend cues
ax.annotate("", xy=(2.7, 17.6), xytext=(0, 17.6), arrowprops=dict(arrowstyle="-|>", color=SUB, lw=1.4, alpha=0.6))
ax.text(1.35, 17.75, "sparsity buffers the error  →  dense gets harder", ha="center", va="bottom", fontsize=11, color=SUB, style="italic")
ax.annotate("", xy=(4.05, 12.6), xytext=(2.92, 15.4), arrowprops=dict(arrowstyle="-|>", color="#1d9e70", lw=1.6))
ax.text(3.62, 15.0, "protect the 12 most\nfragile layers: gap halved", ha="center", va="bottom", fontsize=10.5, color="#0f6e56", style="italic")

# titles
fig.text(0.055, 0.955, "Three LLMs on one 6 GB GPU at 2 bits/weight — then the gap, cut in half",
         fontsize=19.5, fontweight="bold", color=INK)
fig.text(0.055, 0.905, "Data-free 2-bit quantization — no calibration, no fine-tuning. A 24 GB model fits in ~4.5 GB and runs on a 2016 GTX 1060.",
         fontsize=12.5, color=SUB)
fig.text(0.055, 0.028, "WikiText-2 perplexity (lower = better)  ·  gap-ratio = 2-bit ÷ fp16  ·  rank-aware carve-out, data-free",
         fontsize=10.5, color=SUB)
fig.text(0.945, 0.028, "@federico_sciuca", fontsize=11, color=GREY, ha="right", fontweight="bold")

fig.subplots_adjust(left=0.055, right=0.955, top=0.85, bottom=0.175)
out = os.path.join(DATA, "x_compression_chart.png")
fig.savefig(out, dpi=170, facecolor="white")
print("saved", out)
