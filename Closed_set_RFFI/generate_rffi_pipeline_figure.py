"""Generate the dissertation figure explaining RF fingerprint formation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).with_name("rffi_fingerprint_formation.png")


def rounded_box(ax, x, y, w, h, label, face, edge="#334155"):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#475569",
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(10.2, 4.7), constrained_layout=True)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 4.7)
    ax.axis("off")

    y = 2.75
    w = 1.22
    h = 0.85
    xs = [0.25, 1.88, 3.51, 5.14, 6.77, 8.66]
    labels = [
        "Digital LoRa\nbaseband",
        "DAC and\nreconstruction",
        "I/Q modulator\nand oscillator",
        "Power amplifier\nand RF filter",
        "Wireless\nchannel",
        "Receiver, feature\nextraction, classifier",
    ]
    faces = ["#eef2ff", "#dbeafe", "#dbeafe", "#dbeafe", "#ffedd5", "#dcfce7"]

    for x, label, face in zip(xs, labels, faces):
        rounded_box(ax, x, y, w if x != xs[-1] else 1.3, h, label, face)

    for idx in range(len(xs) - 1):
        right = xs[idx] + (w if idx != len(xs) - 1 else 1.3)
        arrow(ax, right + 0.04, y + h / 2, xs[idx + 1] - 0.06, y + h / 2)

    callouts = [
        (2.49, "DAC and timing\nmismatch"),
        (4.12, "Oscillator and\nI/Q mismatch"),
        (5.75, "PA and filter\ndistortion"),
        (7.38, "Channel fading\nand noise"),
    ]
    for x, text in callouts:
        ax.plot([x, x], [2.75, 2.2], color="#64748b", linewidth=1.1)
        ax.text(x, 2.02, text, ha="center", va="top", fontsize=8.3, color="#1f2937")

    ax.plot([1.88, 6.36], [1.05, 1.05], color="#2563eb", linewidth=2.0)
    ax.plot([1.88, 1.88], [1.05, 1.25], color="#2563eb", linewidth=2.0)
    ax.plot([6.36, 6.36], [1.05, 1.25], color="#2563eb", linewidth=2.0)
    ax.text(
        4.12,
        0.72,
        "Device-specific RF fingerprint",
        ha="center",
        va="center",
        fontsize=10,
        color="#1d4ed8",
        fontweight="bold",
    )

    ax.plot([6.77, 8.0], [1.05, 1.05], color="#c2410c", linewidth=2.0)
    ax.plot([6.77, 6.77], [1.05, 1.25], color="#c2410c", linewidth=2.0)
    ax.plot([8.0, 8.0], [1.05, 1.25], color="#c2410c", linewidth=2.0)
    ax.text(
        7.385,
        0.72,
        "Channel nuisance",
        ha="center",
        va="center",
        fontsize=10,
        color="#9a3412",
        fontweight="bold",
    )

    ax.text(
        0.86,
        4.15,
        "Nominally identical devices transmit the same intended LoRa waveform",
        ha="left",
        va="center",
        fontsize=10.2,
        color="#334155",
    )
    arrow(ax, 4.9, 4.12, 4.9, 3.67)
    ax.text(
        9.31,
        4.15,
        "Predicted\ndevice identity",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#166534",
    )
    arrow(ax, 9.31, 3.58, 9.31, 3.65)

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
