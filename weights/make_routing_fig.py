"""Render the mechanism figure (PAPER_MOE.md F2): expert-routing overlap and top-1 agreement vs
depth, carve-out vs uniform 2-bit. Numbers from route_diverge.txt / route_diverge_uniform.txt."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

layers = list(range(1, 27))
carve_ov = [0.9526, 0.9272, 0.9303, 0.9333, 0.9197, 0.9128, 0.9092, 0.9045, 0.9076, 0.9044,
            0.9062, 0.8914, 0.8780, 0.8816, 0.8778, 0.8699, 0.8851, 0.8660, 0.8719, 0.8482,
            0.8560, 0.8566, 0.8653, 0.8630, 0.8641, 0.8820]
uni_ov = [0.8518, 0.8236, 0.8228, 0.8208, 0.7870, 0.7668, 0.7441, 0.7386, 0.7106, 0.7258,
          0.6911, 0.6843, 0.6049, 0.6691, 0.6580, 0.6562, 0.6941, 0.6375, 0.6501, 0.5852,
          0.6273, 0.6283, 0.6434, 0.6156, 0.6532, 0.6984]
carve_t1 = [0.9346, 0.8943, 0.8857, 0.8948, 0.8545, 0.8345, 0.8274, 0.8428, 0.8276, 0.8442,
            0.8354, 0.8333, 0.7891, 0.8025, 0.7896, 0.7969, 0.7930, 0.7783, 0.7883, 0.7349,
            0.7314, 0.8137, 0.7854, 0.7773, 0.7842, 0.8276]
uni_t1 = [0.8076, 0.7380, 0.7305, 0.7556, 0.6628, 0.6111, 0.5735, 0.5881, 0.4497, 0.5850,
          0.5571, 0.5745, 0.4600, 0.5444, 0.4980, 0.5278, 0.4983, 0.4324, 0.5276, 0.3933,
          0.3484, 0.5630, 0.5044, 0.4924, 0.5527, 0.6045]

GREEN, RED, GREY = "#1a9850", "#d73027", "#999999"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.axhline(1.0, color=GREY, ls="--", lw=1, label="fp16 (identical routing)")
ax1.plot(layers, carve_ov, "o-", color=GREEN, lw=2, ms=4, label="carve-out (ours, +0.59 ppl)")
ax1.plot(layers, uni_ov, "s-", color=RED, lw=2, ms=4, label="uniform 2-bit (+6.08 ppl)")
ax1.set_xlabel("layer (depth)"); ax1.set_ylabel(r"routing overlap  |E$_{fp16}\cap$E$_q$| / 6")
ax1.set_title("Expert-routing overlap vs fp16"); ax1.set_ylim(0.5, 1.02)
ax1.legend(fontsize=8, loc="lower left"); ax1.grid(alpha=0.3)
ax2.axhline(1.0, color=GREY, ls="--", lw=1)
ax2.plot(layers, carve_t1, "o-", color=GREEN, lw=2, ms=4, label="carve-out (ours)")
ax2.plot(layers, uni_t1, "s-", color=RED, lw=2, ms=4, label="uniform 2-bit")
ax2.set_xlabel("layer (depth)"); ax2.set_ylabel("top-1 expert agreement")
ax2.set_title("Top-1 expert agreement vs fp16"); ax2.set_ylim(0.3, 1.02)
ax2.legend(fontsize=8, loc="lower left"); ax2.grid(alpha=0.3)
fig.suptitle("Protecting attention + shared experts keeps MoE routing stable under 2-bit\n"
             "(DeepSeek-V2-Lite, WikiText-2; fp16 router in both)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fig_routing_divergence.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
