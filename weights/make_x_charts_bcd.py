"""X-thread charts B (dichotomy), C (inverted depth), D (placement beats budget). Same style as chart A."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SUB, GREY, LGREY = "#14181f", "#5b6572", "#9b9a92", "#c9ccd1"
GREEN, AMBER, RED, TEAL, DTEAL = "#1a9850", "#e8901c", "#d73027", "#1d9e70", "#0f6e56"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
plt.rcParams.update({"font.family": "DejaVu Sans"})
FOOT = "WikiText-2 perplexity (lower = better)  ·  data-free 2-bit  ·  one GTX 1060 6 GB"


def frame(title, sub):
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=170)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    fig.text(0.055, 0.955, title, fontsize=20, fontweight="bold", color=INK)
    fig.text(0.055, 0.905, sub, fontsize=12.5, color=SUB)
    fig.text(0.055, 0.028, FOOT, fontsize=10.5, color=SUB)
    fig.text(0.945, 0.028, "@federico_sciuca", fontsize=11, color=LGREY, ha="right", fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LGREY)
    fig.subplots_adjust(left=0.1, right=0.95, top=0.84, bottom=0.16)
    return fig, ax


# ---- B: the 270,000x dichotomy ----
fig, ax = frame("The same rotation. A 270,000× difference.",
                "Incoherence rotation — the field's universal quantization tool — is rank-conditional.")
ax.set_yscale("log"); ax.set_ylim(1e-3, 3e4); ax.set_xlim(0, 10)
ax.scatter([2.6], [0.006], s=650, color=TEAL, zorder=3, edgecolor="white", lw=2)
ax.scatter([7.4], [1623], s=650, color=RED, zorder=3, edgecolor="white", lw=2)
ax.annotate("", xy=(7.05, 900), xytext=(2.95, 0.011), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
ax.text(5.0, 3.2, "~270,000×\non effective rank alone", ha="center", fontsize=15, color=INK, fontweight="bold")
ax.text(2.6, 0.0016, "full-rank MLP  (eff. rank 1168)\n+0.006 ppl — harmless", ha="center", va="top", fontsize=12, color=DTEAL)
ax.text(7.4, 5200, "low-rank KV-latent  (eff. rank 394)\n+1623 ppl — catastrophic", ha="center", fontsize=12, color=RED)
ax.set_ylabel("Δ perplexity from the same orthogonal rotation (log)", fontsize=11, color=SUB)
ax.set_xticks([]); ax.tick_params(axis="y", labelsize=10, colors=SUB)
fig.text(0.055, 0.07, "Rotation helps high-rank tensors, destroys low-rank bottlenecks (MLA, LoRA, GQA).",
         fontsize=11, color=SUB, style="italic")
fig.savefig(os.path.join(DATA, "x_chart_B_dichotomy.png"), dpi=170, facecolor="white")

# ---- C: the inverted depth curve ----
fig, ax = frame("Everyone protects the early layers. Gemma's fragility lives at the end.",
                "Δ perplexity from quantizing ONLY that 12-layer band to 2-bit (Gemma 4 12B, rest fp16).")
bands = ["layers 0–11", "layers 12–23", "layers 24–35", "layers 36–47"]
vals = [2.14, 3.22, 3.16, 7.98]
cols = [LGREY, LGREY, LGREY, RED]
bars = ax.bar(bands, vals, color=cols, edgecolor="white", lw=1, width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"+{v}", ha="center", fontsize=13,
            color=RED if v > 7 else SUB, fontweight="bold" if v > 7 else "normal")
ax.set_ylim(0, 10.4); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=12, colors=INK, length=0, pad=8)
ax.annotate("where intuition (and weight\nstatistics) said to look", xy=(0, 2.6), xytext=(0.02, 5.6),
            fontsize=11, color=GREY, style="italic", ha="center",
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2, alpha=0.7))
ax.annotate("where a 30-minute functional test\nfound it — ~4× more fragile", xy=(3, 8.3), xytext=(2.35, 9.6),
            fontsize=11.5, color=RED, style="italic", ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
fig.text(0.055, 0.07, "Late-layer error feeds the output head directly — nothing downstream washes it out.",
         fontsize=11, color=SUB, style="italic")
fig.savefig(os.path.join(DATA, "x_chart_C_depthcurve.png"), dpi=170, facecolor="white")

# ---- D: placement beats budget ----
fig, ax = frame("Two files. Identical size. 2.25 perplexity apart.",
                "Stock llama.cpp, Gemma 4 12B, Q2_K base — only WHICH layers get the extra bits differs.")
names = ["uniform Q2_K", "first 12 layers @ Q4_K", "last 12 layers @ Q4_K"]
ppl = [14.41, 12.27, 10.02]; err = [0.43, 0.36, 0.28]
sizes = ["4.73 GB", "5.22 GB", "5.22 GB"]
cols = [GREY, AMBER, TEAL]
bars = ax.bar(names, ppl, yerr=err, color=cols, edgecolor="white", lw=1, width=0.5,
              error_kw=dict(ecolor=INK, lw=1.3, capsize=5))
for b, p, s, c in zip(bars, ppl, sizes, cols):
    ax.text(b.get_x() + b.get_width() / 2, p + 0.75, f"{p:.2f}", ha="center", fontsize=16, color=c, fontweight="bold")
    ax.text(b.get_x() + b.get_width() / 2, 0.55, s, ha="center", fontsize=11.5, color="white", fontweight="bold")
ax.set_ylim(0, 17.4); ax.set_yticks([]); ax.tick_params(axis="x", labelsize=12.5, colors=INK, length=0, pad=8)
ax.plot([0.85, 0.85, 2.15, 2.15], [15.6, 16.2, 16.2, 15.6], color=INK, lw=1.2)
ax.text(1.5, 16.5, "byte-identical files — placement is worth ~2× the budget",
        ha="center", fontsize=12.5, color=INK, fontweight="bold")
fig.text(0.055, 0.07, "Same bytes; the only change is which layers get them. Copy-paste llama-quantize recipe in the repo.",
         fontsize=11, color=SUB, style="italic")
fig.savefig(os.path.join(DATA, "x_chart_D_placement.png"), dpi=170, facecolor="white")
print("saved B, C, D")
