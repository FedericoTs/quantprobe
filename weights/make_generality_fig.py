"""Generality figure (F5): expert-routing overlap vs depth, carve-out vs uniform, on BOTH
architectures (DeepSeek-V2-Lite MLA + Qwen1.5-MoE MHA). Shows the mechanism is architecture-general.
Data: route_diverge.txt / _uniform.txt (DeepSeek) and route_diverge_qwen_{carveout,uniform}.txt."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN, RED, GREY = "#1a9850", "#d73027", "#999999"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ds_layers = list(range(1, 27))
ds_carve = [0.9526, 0.9272, 0.9303, 0.9333, 0.9197, 0.9128, 0.9092, 0.9045, 0.9076, 0.9044, 0.9062,
            0.8914, 0.8780, 0.8816, 0.8778, 0.8699, 0.8851, 0.8660, 0.8719, 0.8482, 0.8560, 0.8566,
            0.8653, 0.8630, 0.8641, 0.8820]
ds_uni = [0.8518, 0.8236, 0.8228, 0.8208, 0.7870, 0.7668, 0.7441, 0.7386, 0.7106, 0.7258, 0.6911,
          0.6843, 0.6049, 0.6691, 0.6580, 0.6562, 0.6941, 0.6375, 0.6501, 0.5852, 0.6273, 0.6283,
          0.6434, 0.6156, 0.6532, 0.6984]
qw_layers = list(range(0, 24))
qw_carve = [0.9044, 0.9236, 0.8962, 0.8933, 0.8817, 0.8797, 0.9012, 0.9166, 0.9138, 0.9039, 0.9103,
            0.9102, 0.9177, 0.9161, 0.9246, 0.9213, 0.9156, 0.9020, 0.8892, 0.8728, 0.8676, 0.8591,
            0.8636, 0.8694]
qw_uni = [0.7169, 0.7852, 0.7478, 0.7781, 0.7505, 0.7120, 0.7559, 0.7884, 0.7922, 0.7555, 0.7511,
          0.7767, 0.7937, 0.7875, 0.8066, 0.8104, 0.8070, 0.7657, 0.7355, 0.6981, 0.6903, 0.6747,
          0.6713, 0.6920]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
panels = [(ax1, ds_layers, ds_carve, ds_uni, "DeepSeek-V2-Lite", "MLA attention · top-6 / 64 experts"),
          (ax2, qw_layers, qw_carve, qw_uni, "Qwen1.5-MoE-A2.7B", "MHA attention · top-4 / 60 experts")]
for ax, lay, carve, uni, title, sub in panels:
    ax.axhline(1.0, color=GREY, ls="--", lw=1, label="fp16 (identical)")
    ax.plot(lay, carve, "o-", color=GREEN, lw=2, ms=3.5, label="carve-out (ours)")
    ax.plot(lay, uni, "s-", color=RED, lw=2, ms=3.5, label="uniform 2-bit")
    ax.set_xlabel("layer (depth)"); ax.set_title(f"{title}\n{sub}", fontsize=9.5)
    ax.set_ylim(0.5, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower left")
ax1.set_ylabel("expert-routing overlap vs fp16")
fig.suptitle("The mechanism is architecture-general: protecting attention + shared keeps MoE routing "
             "stable under 2-bit\n(uniform collapses routing on both models; fp16 router in all cases)",
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(DATA, "fig_generality_routing.png"), dpi=140, bbox_inches="tight")
print("saved fig_generality_routing.png")
