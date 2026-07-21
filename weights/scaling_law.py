"""scaling_law.py -- the tiered decode-throughput law, fitted across our measured points + colibri's
published tiers: tok/s = eta(tier,codec) x BW_tier / bytes_active_per_token.
Chart: predicted vs measured over 7B->744B (5 orders of magnitude of total params), 3 memory tiers.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INK, SUB, GREY = "#14181f", "#5b6572", "#c9ccd1"
BLUE, TEAL, RED, AMBER = "#2a78d6", "#1baf7a", "#d73027", "#eda100"

# (label, tier, BW GB/s, active GB/token, measured tok/s, ours?)
# BW: GTX1060 VRAM 192 measured-class; our DDR4 ~35; SATA 0.45 measured; colibri DDR5 desktop ~60; 5070Ti PCIe-hybrid ~?
PTS = [
    ("7B Q4 · VRAM (1060)",        "VRAM", 192,  4.68,  22.77, True),
    ("7B Q2 · CPU",                "RAM",  35,   3.01,  7.18,  True),
    ("7B Q4 · CPU",                "RAM",  35,   4.68,  5.08,  True),
    ("30B-A3B Q2 · CPU",           "RAM",  35,   1.11,  12.62, True),
    ("16B-A2.4 IQ2 · exps=CPU",    "RAM",  35,   0.85,  11.81, True),
    ("110B-A12 IQ2 · SATA disk",   "disk", 0.45, 2.37,  0.19,  True),
    ("colibri 744B · 128GB DDR5",  "RAM",  60,   16.0,  1.8,   False),
    ("colibri 744B · 25GB cold",   "disk", 0.8,  10.0,  0.07,  False),
    ("4xDGX-Spark 744B · TP4",   "RAM",  1092, 19.8,  42.5,  False),
]

print("eta = tok/s x bytes / BW   (dimensionless utilization; law: eta ~ constant per tier)")
etas = {}
for lab, tier, bw, act, tps, ours in PTS:
    eta = tps * act / bw
    etas.setdefault(tier, []).append(eta)
    print(f"  {lab:28s} tier={tier:4s}  eta={eta:.2f}")
for t, v in etas.items():
    print(f"  >> {t}: eta {min(v):.2f}-{max(v):.2f} (mean {sum(v)/len(v):.2f})")

mean_eta = {t: sum(v) / len(v) for t, v in etas.items()}
fig, ax = plt.subplots(figsize=(9.5, 7), dpi=170)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.plot([0.03, 60], [0.03, 60], color=GREY, lw=1.2, ls="--", zorder=1)
COL = {"VRAM": BLUE, "RAM": TEAL, "disk": RED}
for lab, tier, bw, act, tps, ours in PTS:
    pred = mean_eta[tier] * bw / act
    ax.scatter([pred], [tps], s=170 if ours else 150, color=COL[tier],
               marker="o" if ours else "s", edgecolor="white", lw=1.5, zorder=3)
    off = (7, -13) if "16B" in lab else ((7, 6) if ours else (7, -12))
    ax.annotate(lab, (pred, tps), textcoords="offset points", xytext=off, fontsize=9.5,
                color=INK if ours else SUB)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.03, 60); ax.set_ylim(0.03, 60)
ax.set_xlabel("predicted tok/s  =  η(tier) × BW ÷ active bytes", fontsize=12, color=SUB)
ax.set_ylabel("measured tok/s", fontsize=12, color=SUB)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=SUB)
ax.grid(alpha=0.25, which="both")
fig.text(0.07, 0.955, "One law, 7B → 744B, on commodity hardware",
         fontsize=17, fontweight="bold", color=INK)
fig.text(0.07, 0.915, "tok/s = η · bandwidth ÷ active-bytes per token — circles = ours (2016 GTX 1060, DDR4, SATA) · squares = colibri (published)",
         fontsize=10.5, color=SUB)
leg = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=11, label=t) for t, c in COL.items()]
ax.legend(handles=leg, frameon=False, fontsize=11, loc="upper left")
fig.subplots_adjust(left=0.09, right=0.96, top=0.87, bottom=0.09)
fig.savefig(os.path.join(DATA, "x_chart_E_scalinglaw.png"), facecolor="white")
print("saved x_chart_E_scalinglaw.png")
