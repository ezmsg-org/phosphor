"""Demo: synthetic multi-frequency sine waves fed into SweepWidget or SpectrumWidget."""

import argparse
import random
import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from phosphor import (
    ScatterConfig,
    ScatterWidget,
    SpectrumConfig,
    SpectrumWidget,
    SweepConfig,
    SweepEvent,
    SweepWidget,
)


def main():
    parser = argparse.ArgumentParser(description="Phosphor renderer demo")
    parser.add_argument("--mode", choices=["sweep", "spectrum", "scatter"], default="sweep", help="Render mode")
    parser.add_argument("--channels", type=int, default=128, help="Total channels")
    parser.add_argument("--srate", type=float, default=30000.0, help="Sample rate (Hz)")
    parser.add_argument("--dur", type=float, default=2.0, help="Display duration (s)")
    parser.add_argument("--visible", type=int, default=64, help="Visible channels")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Each channel gets a different frequency (1–100 Hz, linearly spaced).
    frequencies = np.linspace(1.0, 100.0, args.channels)
    amplitudes = np.ones(args.channels, dtype=np.float32) * 100.0
    chunk_size = max(1, int(args.srate / 60))  # ~500 samples per timer tick
    sample_counter = 0

    if args.mode == "scatter":
        n_ch = args.channels
        # Generate positions on a unit circle
        angles = np.linspace(0, 2 * np.pi, n_ch, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
        labels = [f"E{i}" for i in range(n_ch)]
        config = ScatterConfig(
            positions=positions,
            cmap="viridis",
            modulate_color=True,
            modulate_size=True,
            channel_labels=labels,
        )
        widget = ScatterWidget(config)
        widget.setWindowTitle("Phosphor Demo — Scatter")
        phase = 0.0

        def push_chunk():
            nonlocal phase
            # Rotating wave around the circle
            values = (np.sin(angles + phase) * 0.5 + 0.5).astype(np.float32)
            widget.push_data(values)
            phase += 0.05

    elif args.mode == "spectrum":
        fft_size = int(args.srate)
        n_bins = fft_size // 2
        config = SpectrumConfig(
            n_channels=args.channels,
            srate=args.srate,
            n_bins=n_bins,
            n_visible=args.visible,
        )
        widget = SpectrumWidget(config)
        widget.setWindowTitle("Phosphor Demo — Spectrum")

        def push_chunk():
            nonlocal sample_counter
            t = (sample_counter + np.arange(fft_size)) / args.srate
            data = (np.sin(2.0 * np.pi * frequencies[None, :] * t[:, None]) * amplitudes[None, :]).astype(np.float32)
            # Compute FFT per channel → magnitude spectrum
            windowed = data * np.hanning(fft_size)[:, None]
            spectrum = np.abs(np.fft.rfft(windowed, axis=0))  # (fft_size//2 + 1, n_channels)
            magnitudes = spectrum[1 : n_bins + 1, :]  # drop DC, keep n_bins
            widget.push_data(magnitudes)
            sample_counter += fft_size

    else:
        config = SweepConfig(
            n_channels=args.channels,
            srate=args.srate,
            display_dur=args.dur,
            n_visible=args.visible,
        )
        widget = SweepWidget(config)
        widget.setWindowTitle("Phosphor Demo")
        push_counter = 0

        def push_chunk():
            nonlocal sample_counter, push_counter
            t = (sample_counter + np.arange(chunk_size)) / args.srate
            data = (np.sin(2.0 * np.pi * frequencies[None, :] * t[:, None]) * amplitudes[None, :]).astype(np.float32)
            widget.push_data(data)
            sample_counter += chunk_size
            push_counter += 1

            # Inject a random event with jitter (~every 0.3–0.7s)
            if push_counter % 30 == 0 and random.random() < 0.7:
                t_now = widget.sweep_buffer.elapsed_time
                jitter = random.uniform(-0.1, 0.1)
                t_event = max(0.0, t_now + jitter)
                if random.random() < 0.5:
                    # Full-height event
                    ev = SweepEvent(t_elapsed=t_event, color=(1.0, 1.0, 0.4))
                else:
                    # Per-channel event
                    ch = random.randint(0, args.channels - 1)
                    ev = SweepEvent(t_elapsed=t_event, channel=ch, color=(0.4, 1.0, 1.0))
                widget.push_events([ev])

    widget.resize(1200, 800)
    widget.show()

    timer = QTimer()
    timer.timeout.connect(push_chunk)
    timer.start(max(1, int(1000 / 60)))  # ~16 ms

    app.aboutToQuit.connect(timer.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
