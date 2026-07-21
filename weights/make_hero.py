"""README hero as PNG: the three-tier placement map + the law. Replaces the fragile SVG."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, SUB = "#14181f", "#5b6572"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(12.4, 4.6), dpi=170)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(0, 124); ax.set_ylim(0, 46); ax.axis("off")

fig.text(0.045, 0.94, "Placement beats budget", fontsize=21, fontweight="bold", color=INK)
fig.text(0.045, 0.855, "Where bits sit — which layers, which memory tier — matters more than how many you have. Every number measured on my 2016 desktop.",
         fontsize=11.5, color=SUB)

TIERS = [
    dict(x=3,  w=36, edge="#d73027", fill="#fcebeb", tint="#f7c1c1", dark="#791f1f", mid="#993c1d",
         title="DISK · 0.45 GB/s (measured)", eta="η ≈ 0.9–1.0 — bandwidth-pure",
         rows=["cold experts · 2-bit (halve every read)", "110B streams at 0.19 tok/s — as predicted"],
         foot="lookahead prefetch: 99% of experts known early"),
    dict(x=44, w=38, edge="#1d9e70", fill="#e1f5ee", tint="#9fe1cb", dark="#085041", mid="#0f6e56",
         title="RAM · 34→48 GB/s (2133→XMP 3000)", eta="η ≈ 0.62 dense · 0.38 MoE",
         rows=["routed experts · 2-bit (the RD floor)", "fragile band · 4-bit ← the 30-min probe"],
         foot="30B MoE: 19.3 tok/s hybrid · batch-8: 22 tok/s"),
    dict(x=87, w=34, edge="#378add", fill="#e6f1fb", tint="#b5d4f4", dark="#0c447c", mid="#185fa5",
         title="VRAM · 192 GB/s (GTX 1060 6 GB)", eta="η ≈ 0.35 (low-bit decode: 0.04)",
         rows=["attention + norms · 4-bit, every token", "KV cache · 8-bit (free; below: cliff)"],
         foot="experts don't belong here on Pascal (+54% CPU)"),
]
for t in TIERS:
    ax.add_patch(FancyBboxPatch((t["x"], 10), t["w"], 27, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=t["fill"], ec=t["edge"], lw=1.6))
    ax.text(t["x"] + 2, 33.5, t["title"], fontsize=10.5, fontweight="bold", color=t["dark"])
    ax.text(t["x"] + 2, 30.3, t["eta"], fontsize=9, color=t["mid"])
    for i, r in enumerate(t["rows"]):
        y = 25.5 - i * 5.2
        ax.add_patch(FancyBboxPatch((t["x"] + 2, y - 1.6), t["w"] - 4, 3.9,
                                    boxstyle="round,pad=0.3,rounding_size=0.8", fc=t["tint"], ec="none"))
        ax.text(t["x"] + 3.2, y + 0.1, r, fontsize=8.6, color=t["dark"], va="center")
    ax.text(t["x"] + 2, 11.8, t["foot"], fontsize=8.2, color=t["mid"], style="italic")

for x0, x1, lab in [(39.6, 43.4, "prefetch"), (82.6, 86.4, "stream")]:
    ax.add_patch(FancyArrowPatch((x0, 23), (x1, 23), arrowstyle="-|>", mutation_scale=16, color=SUB, lw=1.6))
    ax.text((x0 + x1) / 2, 25.2, lab, fontsize=8, color=SUB, ha="center")

ax.add_patch(FancyBboxPatch((3, 1.5), 118, 5.6, boxstyle="round,pad=0.4,rounding_size=1.2",
                            fc=INK, ec="none"))
ax.text(6, 4.3, "tok/s  =  η(tier)  ×  bandwidth  ÷  active-bytes-per-token", fontsize=12.5,
        color="white", fontweight="bold", va="center")
ax.text(66, 4.3, "— one equation, 7B → 744B: my box, colibri's tiers, a 4× DGX Spark",
        fontsize=9.3, color="#c9ccd1", va="center")

fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.02)
fig.savefig(os.path.join(DATA, "hero_placement.png"), facecolor="white")
print("saved hero_placement.png")
