"""Generate the Transformer-for-LoRa-RFFI overview used in the dissertation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).with_name("transformer_rffi_overview.png")


def box(ax, xy, width, height, title, subtitle="", face="#F7FAFC", edge="#64748B"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.92, title,
            ha="center", va="center", fontsize=12, fontweight="bold", color="#172033")
    if subtitle:
        ax.text(x + width / 2, y + height * 0.82, subtitle,
                ha="center", va="center", fontsize=9.5, color="#475569")
    return patch


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                linewidth=1.7, color="#475569"))


fig, ax = plt.subplots(figsize=(16, 7.5), dpi=180)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.955, "Transformer processing pipeline for LoRa radio-frequency fingerprint identification",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#111827")
ax.text(0.5, 0.91, "Local time-frequency evidence is converted to tokens and related globally by self-attention",
        ha="center", va="center", fontsize=11, color="#475569")

# 1. Spectrogram
box(ax, (0.025, 0.24), 0.16, 0.56, "1. Channel-independent\nspectrogram", "102 x 62 x 1")
gx = np.linspace(0, 1, 102)[:, None]
gy = np.linspace(0, 1, 62)[None, :]
spec = 0.25 * np.sin(10 * gx + 5 * gy) + 0.4 * np.exp(-((gy - (0.15 + 0.7 * gx)) % 1) ** 2 / 0.012)
ax.imshow(spec.T, extent=(0.052, 0.158, 0.365, 0.665), origin="lower",
          aspect="auto", cmap="viridis", zorder=2)
ax.text(0.105, 0.305, "Time", ha="center", fontsize=9, color="#475569")
ax.text(0.043, 0.515, "Frequency", rotation=90, ha="center", va="center",
        fontsize=9, color="#475569")

# 2. Padding and patches
box(ax, (0.22, 0.24), 0.18, 0.56, "2. Pad and split", "104 x 64 -> 4 x 4 patches")
grid_x0, grid_y0, gw, gh = 0.255, 0.33, 0.11, 0.28
ax.add_patch(Rectangle((grid_x0, grid_y0), gw, gh, facecolor="#DCEAFE",
                       edgecolor="#2563EB", linewidth=1.5))
for i in range(1, 8):
    x = grid_x0 + gw * i / 8
    ax.plot([x, x], [grid_y0, grid_y0 + gh], color="white", linewidth=1)
for i in range(1, 6):
    y = grid_y0 + gh * i / 6
    ax.plot([grid_x0, grid_x0 + gw], [y, y], color="white", linewidth=1)
ax.add_patch(Rectangle((grid_x0 + gw * 6 / 8, grid_y0 + gh * 4 / 6),
                       gw * 2 / 8, gh * 2 / 6, facecolor="#FDBA74",
                       edgecolor="#EA580C", linewidth=1.3))
ax.text(0.31, 0.285, "26 x 16 = 416 tokens", ha="center", fontsize=9.5, color="#475569")

# 3. Embedding
box(ax, (0.435, 0.24), 0.16, 0.56, "3. Token embedding", "Linear projection + position")
token_y = [0.60, 0.53, 0.46, 0.39]
colors = ["#DBEAFE", "#DCFCE7", "#FFEDD5", "#FCE7F3"]
for idx, (y, c) in enumerate(zip(token_y, colors), 1):
    ax.add_patch(FancyBboxPatch((0.475, y), 0.08, 0.042,
                                boxstyle="round,pad=0.006", facecolor=c,
                                edgecolor="#64748B", linewidth=1))
    ax.text(0.515, y + 0.021, f"token {idx}" if idx < 4 else "...",
            ha="center", va="center", fontsize=9, color="#334155")
ax.text(0.515, 0.31, "+ learnable position", ha="center", fontsize=9.5, color="#475569")

# 4. Transformer encoder
box(ax, (0.63, 0.18), 0.205, 0.68, "4. Transformer encoder", "Repeated L times",
    face="#FAF5FF", edge="#7C3AED")
ax.add_patch(FancyBboxPatch((0.665, 0.50), 0.135, 0.09,
                            boxstyle="round,pad=0.01", facecolor="#EDE9FE",
                            edgecolor="#7C3AED", linewidth=1.3))
ax.text(0.7325, 0.545, "Multi-head\nself-attention", ha="center", va="center",
        fontsize=10.5, color="#3B0764")
ax.add_patch(FancyBboxPatch((0.665, 0.30), 0.135, 0.09,
                            boxstyle="round,pad=0.01", facecolor="#F3E8FF",
                            edgecolor="#7C3AED", linewidth=1.3))
ax.text(0.7325, 0.345, "Feed-forward\nnetwork", ha="center", va="center",
        fontsize=10.5, color="#3B0764")
arrow(ax, (0.7325, 0.495), (0.7325, 0.40))
ax.text(0.642, 0.445, "LayerNorm\n+ residual", ha="center", va="center",
        fontsize=8.5, color="#6D28D9")
ax.text(0.808, 0.445, "LayerNorm\n+ residual", ha="center", va="center",
        fontsize=8.5, color="#6D28D9")

# Global-attention motif
points = [(0.68, 0.655), (0.715, 0.69), (0.75, 0.645), (0.785, 0.69)]
for x, y in points:
    ax.scatter([x], [y], s=44, color="#7C3AED", zorder=4)
for i, p1 in enumerate(points):
    for p2 in points[i + 1:]:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#C4B5FD", linewidth=0.9, zorder=2)
ax.text(0.732, 0.615, "global token interaction", ha="center", fontsize=8.8, color="#6D28D9")

# 5. Classification
box(ax, (0.87, 0.24), 0.105, 0.56, "5. Device ID", "Pooling -> L2 norm")
ax.add_patch(FancyBboxPatch((0.89, 0.48), 0.065, 0.09,
                            boxstyle="round,pad=0.008", facecolor="#DCFCE7",
                            edgecolor="#16A34A", linewidth=1.3))
ax.text(0.9225, 0.525, "30-class\nsoftmax", ha="center", va="center",
        fontsize=10, color="#14532D")
ax.text(0.9225, 0.385, "Predicted\ntransmitter", ha="center", va="center",
        fontsize=9.5, color="#475569")

arrow(ax, (0.185, 0.52), (0.22, 0.52))
arrow(ax, (0.40, 0.52), (0.435, 0.52))
arrow(ax, (0.595, 0.52), (0.63, 0.52))
arrow(ax, (0.835, 0.52), (0.87, 0.52))

ax.text(0.5, 0.085,
        "Padding preserves boundary samples; smaller patches retain finer RF detail but increase the quadratic attention cost.",
        ha="center", va="center", fontsize=10.5, color="#475569")

plt.savefig(OUT, bbox_inches="tight", pad_inches=0.15, facecolor="white")
print(OUT)
