"""Chart G: running the 744B at home — cost vs speed, measured vs predicted, and what placement buys free."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SUB, LGREY, TEAL, RED = "#14181f", "#5b6572", "#c9ccd1", "#1d9e70", "#d73027"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
plt.rcParams.update({"font.family": "DejaVu Sans"})

# (cost $, tok/s, label, measured?, dx, dy)
PTS = [
    (60,    0.07, "my 2016 desktop\n(2-bit, SATA stream)", False, 0, 14),
    (240,   0.5,  "+ NVMe", False, 0, 14),
    (1500,  1.8,  "128 GB DDR5\ncolibri int4 (measured)", True, -10, -34),
    (1500,  3.5,  "same box\n+ my 2-bit recipe", False, 12, 4),
    (2500,  9.0,  "used 200 GB/s\nworkstation", False, -8, 12),
    (6000,  28,   "2× ASUS GX10\n(744B fits: 2-bit only)", False, -14, 12),
    (10000, 45,   "Mac Studio 512 GB", False, -34, 12),
    (16000, 42.5, "4× DGX Spark\nW4 (measured)", True, 10, -30),
    (16000, 61,   "4× Spark\n+ my recipe", False, 12, 4),
]
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=170)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(40, 60000); ax.set_ylim(0.04, 150)

# the placement dividend: vertical arrows at fixed cost
for x, y0, y1 in [(1500, 1.8, 3.5), (16000, 42.5, 61)]:
    ax.annotate("", xy=(x, y1 * 0.82), xytext=(x, y0 * 1.2),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2))
ax.text(1500 * 1.25, 2.45, "same hardware,\nplacement only", fontsize=9.5, color=TEAL, style="italic")
ax.text(16000 * 1.25, 50, "×1.4–1.6 free", fontsize=9.5, color=TEAL, style="italic")

for c, t, lab, meas, dx, dy in PTS:
    if meas:
        ax.scatter([c], [t], s=200, color=INK, marker="s", zorder=3, edgecolor="white", lw=1.5)
    else:
        ax.scatter([c], [t], s=180, facecolor="white", edgecolor=TEAL, lw=2.4, zorder=3)
    ax.annotate(lab, (c, t), textcoords="offset points", xytext=(dx, dy), fontsize=9,
                color=INK if meas else "#0f6e56", ha="center",
                fontweight="bold" if meas else "normal")

ax.set_xlabel("hardware cost (USD, log) — beyond a PC you already own", fontsize=12, color=SUB)
ax.set_ylabel("GLM-5.2 744B decode tok/s (log)", fontsize=12, color=SUB)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=SUB); ax.grid(alpha=0.22, which="both")
ax.scatter([], [], s=140, color=INK, marker="s", label="measured (theirs)")
ax.scatter([], [], s=120, facecolor="white", edgecolor=TEAL, lw=2, label="predicted by the law (pre-registered)")
ax.legend(frameon=False, fontsize=10.5, loc="upper left")
fig.text(0.055, 0.955, "Running a 744B at home: what money buys — and what placement buys free",
         fontsize=18.5, fontweight="bold", color=INK)
fig.text(0.055, 0.905, "Hollow points: falsifiable predictions, published first. Green arrows: the probed-2-bit recipe on the same hardware.",
         fontsize=11.5, color=SUB)
fig.text(0.055, 0.03, "tok/s = η(tier) × bandwidth ÷ active-bytes · error ±25–40% at this range · quality: probe the fragile band first",
         fontsize=10, color=SUB)
fig.text(0.945, 0.03, "@federico_sciuca", fontsize=11, color=LGREY, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.07, right=0.965, top=0.86, bottom=0.15)
fig.savefig(os.path.join(DATA, "x_chart_G_744b.png"), facecolor="white")
print("saved x_chart_G_744b.png")
