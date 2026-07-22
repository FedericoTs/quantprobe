"""Chart H: reading a benchmark I never saw — the law called Laguna S 2.1's decode within 1%,
and the spec-decode x MoE antagonism explains its decay under load."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SUB, LGREY = "#14181f", "#5b6572", "#c9ccd1"
TEAL, RED, AMBER = "#1d9e70", "#d73027", "#e8901c"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
plt.rcParams.update({"font.family": "DejaVu Sans"})

loads = [1, 2, 4, 8]
decode = [76.7, 47.5, 34.9, 24.5]          # their measured per-stream decode tok/s
lo = [76.7, 39, 24, 16]; hi = [117, 56, 45, 31]   # their ranges/peaks where given
BASE = 47.0                                 # my pre-registered base-decode prediction

fig, ax = plt.subplots(figsize=(12, 6.75), dpi=170)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
x = range(len(loads))

# speculation zone (above the base line) vs bandwidth-bound zone (below)
ax.axhspan(BASE, 130, color=TEAL, alpha=0.06)
ax.axhspan(0, BASE, color=SUB, alpha=0.05)
ax.axhline(BASE, color=INK, ls="--", lw=1.6)
ax.text(3.05, BASE + 1.5, "my pre-registered prediction:\n~47 tok/s base decode (no speculation)",
        fontsize=10.5, color=INK, va="bottom", ha="right", fontweight="bold")
ax.text(0.02, 112, "speculation zone — DFlash generates >1 token per memory pass",
        fontsize=10, color="#0f6e56", style="italic")
ax.text(0.02, 8, "bandwidth-bound zone — one token per pass (η ≈ 0.80)",
        fontsize=10, color=SUB, style="italic")

# their measured decode with ranges
for i in x:
    ax.plot([i, i], [lo[i], hi[i]], color=LGREY, lw=6, solid_capstyle="round", zorder=2)
ax.plot(x, decode, "-o", color=RED, lw=2.5, ms=11, mfc="white", mec=RED, mew=2.5, zorder=4)
for i, v in zip(x, decode):
    ax.annotate(f"{v}", (i, v), textcoords="offset points", xytext=(0, 14 if i else 16),
                fontsize=13, color=RED, fontweight="bold", ha="center")

# annotate the two proofs
ax.annotate("×1 = 76.7 ⇒ η 1.29, impossible single-pass\n→ the law DETECTS DFlash",
            (0, 76.7), textcoords="offset points", xytext=(58, -6), fontsize=10, color="#0f6e56")
ax.annotate("×2 per-stream 47.5\n= prediction within 1%", (1, 47.5),
            textcoords="offset points", xytext=(40, 24), fontsize=10.5, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
ax.annotate("speculative gain erodes under load —\nverify batch unions more experts (measured antagonism)",
            (3, 24.5), textcoords="offset points", xytext=(-6, 40), fontsize=10, color=RED,
            ha="right", arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

ax.set_xticks(list(x)); ax.set_xticklabels([f"×{l}\n{l}/{l} streams" for l in loads], fontsize=11, color=INK)
ax.set_ylim(0, 130); ax.set_ylabel("per-stream decode tok/s (their measurement)", fontsize=12, color=SUB)
ax.set_xlabel("")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=SUB); ax.grid(axis="y", alpha=0.2)
fig.text(0.055, 0.955, "Reading a benchmark I'd never seen: the law called it within 1%",
         fontsize=18.5, fontweight="bold", color=INK)
fig.text(0.055, 0.905, "Laguna S 2.1 (117.6B MoE, 8B active, NVFP4), single DGX Spark — their published numbers vs. my prediction from the config alone.",
         fontsize=11.5, color=SUB)
fig.text(0.055, 0.03, "tok/s = η(tier) × bandwidth ÷ active-bytes · η≈0.80 on GB10 (3rd independent confirmation) · speculation × MoE antagonism = the decay",
         fontsize=9.5, color=SUB)
fig.text(0.945, 0.03, "@federico_sciuca", fontsize=11, color=LGREY, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.07, right=0.965, top=0.855, bottom=0.14)
fig.savefig(os.path.join(DATA, "x_chart_H_laguna.png"), facecolor="white")
print("saved x_chart_H_laguna.png")
