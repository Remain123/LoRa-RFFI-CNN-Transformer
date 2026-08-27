"""Generate a publication-ready time-domain LoRa baseband figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).with_name("lora_time_domain_iq.png")


def main():
    bandwidth = 125_000.0
    spreading_factor = 7
    symbol_duration = 2**spreading_factor / bandwidth
    sample_rate = 2_000_000.0

    time = np.arange(int(round(symbol_duration * sample_rate))) / sample_rate
    chirp_rate = bandwidth / symbol_duration
    start_frequency = -bandwidth / 2.0
    phase = 2.0 * np.pi * (
        start_frequency * time + 0.5 * chirp_rate * time**2
    )
    signal = np.exp(1j * phase)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), constrained_layout=True)

    detail_mask = time <= 0.24e-3
    detail_time = time[detail_mask] * 1e3
    axes[0].plot(detail_time, signal.real[detail_mask], linewidth=1.35, label=r"$I(t)$")
    axes[0].plot(
        detail_time,
        signal.imag[detail_mask],
        linewidth=1.2,
        linestyle="--",
        label=r"$Q(t)$",
    )
    axes[0].set_xlim(detail_time[0], detail_time[-1])
    axes[0].set_ylim(-1.15, 1.15)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Normalised amplitude")
    axes[0].set_title("(a) In-phase and quadrature components of a LoRa up-chirp")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", frameon=True, ncol=2)

    axes[1].plot(time * 1e3, np.abs(signal), linewidth=2.0, label=r"Envelope $|s(t)|$")
    axes[1].fill_between(time * 1e3, 0, np.abs(signal), alpha=0.12)
    axes[1].set_xlim(0, symbol_duration * 1e3)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Normalised magnitude")
    axes[1].set_title("(b) Constant envelope over one LoRa symbol")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower right", frameon=True)

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
