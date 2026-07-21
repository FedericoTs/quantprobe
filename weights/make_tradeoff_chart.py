"""README chart F: the speed-memory trade-off with the capacity cliff (Qwen3-30B on the 2016 box).
Shows the law's core lesson: fewer bits = less memory AND more speed — until the RAM cliff."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, SUB, LGREY = "#14181f", "#5b6572", "#c9ccd1"
TEAL, RED, AMBER = "#1d9e70", "#d73027", "#e8901c"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
plt.rcParams.update({"font.family": "DejaVu Sans"})

# Qwen3-30B-A3B on the XMP'd 2016 box (t=30.5, a=3.3, ne=1.2; geta .35, vb 192, rb 48, db 0.45, ra 12)
t, a, ne = 30.5, 3.3, 1.2
geta, vb, rb, db, ra, etaR = 0.35, 192, 48, 0.45, 12, 0.38
BITS = [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5]
QUAL = {2.0: 1.10, 2.25: 1.085, 2.5: 1.07, 2.75: 1.06, 3.0: 1.05, 3.5: 1.035, 4.0: 1.025, 4.5: 1.02}
pts = []
for b in BITS:
    AB = max(b, 4.5)
    size = (ne * AB / 8 + (t - ne) * b / 8) * 1.08
    actNe, actEx = ne * AB / 8 * 1.15, (a - ne) * b / 8 * 1.15
    if size - ne * AB / 8 * 1.08 <= ra:                      # hybrid fits
        tps = 1 / (actNe / (geta * vb) + actEx / (etaR * rb))
    else:                                                     # disk streaming
        miss = max(0, 1 - (ra * 0.9) / size)
        tps = 0.95 / (actEx * miss / db + (actNe + actEx * (1 - miss)) / (etaR * rb))
    pts.append((b, size, tps))

fig, ax = plt.subplots(figsize=(12, 6.75), dpi=170)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
ax.plot(xs, ys, "-", color=TEAL, lw=2.5, zorder=2)
for b, s, v in pts:
    cur = abs(b - 2.5) < 0.01
    ax.scatter([s], [v], s=170 if cur else 80, color=RED if cur else TEAL, zorder=3, edgecolor="white", lw=1.5)
    if b in (2.0, 2.5, 3.0, 4.0):
        ax.annotate(f"{b:g}-bit\n×{QUAL[b]:.2f} quality", (s, v), textcoords="offset points",
                    xytext=(10, 12 if v > 5 else 16), fontsize=10.5, color=RED if cur else SUB,
                    fontweight="bold" if cur else "normal")
ax.axvline(ra, color=RED, ls="--", lw=1.6)
ax.text(ra + 0.25, 8.5, "RAM capacity\n(16 GB box)\n— the cliff", fontsize=11, color=RED)
ax.axvline(ra + 16, color=LGREY, ls=":", lw=1.4)
ax.text(ra + 16.25, 8.5, "with +16 GB RAM (~€30)\nthe cliff moves here", fontsize=10.5, color=SUB)
ax.annotate("measured: 19.30 ± 0.88\n(predicted 19)", (pts[2][1], pts[2][2]), textcoords="offset points",
            xytext=(-130, -34), fontsize=11, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
ax.set_xlim(6, 32); ax.set_ylim(0, 23)
ax.set_xlabel("model memory (GB) — Qwen3-30B-A3B at each bit-width", fontsize=12, color=SUB)
ax.set_ylabel("predicted decode tok/s (best placement)", fontsize=12, color=SUB)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.tick_params(colors=SUB); ax.grid(alpha=0.25)
fig.text(0.055, 0.955, "Fewer bits: less memory AND more speed — until the cliff",
         fontsize=19.5, fontweight="bold", color=INK)
fig.text(0.055, 0.91, "Bandwidth is the wall, so smaller weights run faster — until the model stops fitting in RAM and speed collapses to disk.",
         fontsize=12, color=SUB)
fig.text(0.055, 0.03, "Qwen3-30B-A3B, hybrid placement (attention→VRAM, experts→RAM), 2016 desktop · quality = gap-ratio with the depth-aware recipe",
         fontsize=10, color=SUB)
fig.text(0.945, 0.03, "@federico_sciuca", fontsize=11, color=LGREY, ha="right", fontweight="bold")
fig.subplots_adjust(left=0.07, right=0.96, top=0.86, bottom=0.11)
fig.savefig(os.path.join(DATA, "x_chart_F_tradeoff.png"), facecolor="white")
print("saved x_chart_F_tradeoff.png")
