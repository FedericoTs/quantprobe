"""Figure F3: VRAM during live generation -- a 16B MoE resident on a 6 GB GTX 1060.
Measured (nvidia-smi -l 1 during generation): idle 794 MiB -> peak 5873 / 6144 MiB."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOTAL, DESKTOP, PEAK = 6144, 794, 5873
model = PEAK - DESKTOP

fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.barh([0], [DESKTOP], color="#999999", edgecolor="black", lw=0.6, label=f"desktop/overhead ({DESKTOP} MiB)")
ax.barh([0], [model], left=[DESKTOP], color="#1a9850", edgecolor="black", lw=0.6,
        label=f"16B MoE @ 2-bit ({model} MiB)")
ax.axvline(TOTAL, color="#d73027", ls="--", lw=1.6)
ax.text(TOTAL - 60, 0.42, f"{TOTAL} MiB capacity", color="#d73027", ha="right", fontsize=8.5)
ax.text(PEAK, -0.46, f"peak {PEAK} MiB during generation", ha="center", fontsize=8.5)
ax.set_xlim(0, 6500); ax.set_ylim(-0.7, 0.7); ax.set_yticks([])
ax.set_xlabel("GPU memory (MiB)")
ax.set_title("A 16-billion-parameter MoE generating text, resident on a 6 GB GTX 1060 (2016)")
ax.legend(fontsize=8, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(DATA, "fig_vram.png"), dpi=140, bbox_inches="tight")
print("saved fig_vram.png")
