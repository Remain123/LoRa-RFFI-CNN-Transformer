"""Generate the LoRa chirp figure used in the dissertation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


OUTPUT = Path(__file__).with_name("lora_chirp_time_frequency.png")


def main():
    bandwidth = 125_000.0
    spreading_factor = 7
    chips_per_symbol = 2**spreading_factor
    symbol_duration = chips_per_symbol / bandwidth
    sample_rate = 1_000_000.0

    samples_per_symbol = int(round(symbol_duration * sample_rate))
    time_symbol = np.arange(samples_per_symbol) / sample_rate
    normalized_time = time_symbol / symbol_duration

    up_frequency = ((normalized_time + 0.26) % 1.0 - 0.5) * bandwidth
    down_frequency = -((normalized_time + 0.26) % 1.0 - 0.5) * bandwidth

    symbol_indices = [0, 32, 64, 96]
    signal_parts = []
    for symbol_index in symbol_indices:
        fraction = (normalized_time + symbol_index / chips_per_symbol) % 1.0
        instantaneous_frequency = (fraction - 0.5) * bandwidth
        phase = 2.0 * np.pi * np.cumsum(instantaneous_frequency) / sample_rate
        signal_parts.append(np.exp(1j * phase))
    signal = np.concatenate(signal_parts)

    frequency, time_stft, spectrum = stft(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=256,
        noverlap=240,
        nfft=2048,
        return_onesided=False,
        boundary=None,
    )
    frequency = np.fft.fftshift(frequency)
    spectrum = np.fft.fftshift(spectrum, axes=0)
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1e-8))
    magnitude_db -= magnitude_db.max()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), constrained_layout=True)

    axes[0].plot(normalized_time, up_frequency / 1e3, linewidth=2.0, label="Up-chirp")
    axes[0].plot(
        normalized_time,
        down_frequency / 1e3,
        linewidth=2.0,
        linestyle="--",
        label="Down-chirp",
    )
    axes[0].axhline(0, color="0.55", linewidth=0.8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-bandwidth / 2e3 - 5, bandwidth / 2e3 + 5)
    axes[0].set_xlabel(r"Normalised time within one symbol, $t/T_s$")
    axes[0].set_ylabel("Instantaneous frequency (kHz)")
    axes[0].set_title("(a) Ideal LoRa chirps and frequency wrapping")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", frameon=True)

    mesh = axes[1].pcolormesh(
        time_stft * 1e3,
        frequency / 1e3,
        magnitude_db,
        shading="auto",
        cmap="viridis",
        vmin=-45,
        vmax=0,
    )
    for boundary in np.arange(1, len(symbol_indices)) * symbol_duration * 1e3:
        axes[1].axvline(boundary, color="white", linewidth=0.8, linestyle=":", alpha=0.9)
    axes[1].set_ylim(-bandwidth / 2e3, bandwidth / 2e3)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Baseband frequency (kHz)")
    axes[1].set_title("(b) Spectrogram of four cyclically shifted LoRa symbols")
    colorbar = fig.colorbar(mesh, ax=axes[1], pad=0.02)
    colorbar.set_label("Normalised magnitude (dB)")

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
