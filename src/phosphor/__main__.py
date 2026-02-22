"""Demo: synthetic multi-frequency sine waves fed into SweepWidget."""

import argparse
import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from phosphor import SweepConfig, SweepWidget


def main():
    parser = argparse.ArgumentParser(description="Sweep renderer demo")
    parser.add_argument("--channels", type=int, default=128, help="Total channels")
    parser.add_argument("--srate", type=float, default=30000.0, help="Sample rate (Hz)")
    parser.add_argument("--dur", type=float, default=2.0, help="Display duration (s)")
    parser.add_argument("--visible", type=int, default=64, help="Visible channels")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    config = SweepConfig(
        n_channels=args.channels,
        srate=args.srate,
        display_dur=args.dur,
        n_visible=args.visible,
    )
    widget = SweepWidget(config)
    widget.setWindowTitle("Phosphor Demo")
    widget.resize(1200, 800)
    widget.show()

    # --- Synthetic data generation ---
    # Each channel gets a different frequency (1–100 Hz, linearly spaced).
    frequencies = np.linspace(1.0, 100.0, args.channels)
    amplitudes = np.ones(args.channels, dtype=np.float32) * 100.0
    chunk_size = max(1, int(args.srate / 60))  # ~500 samples per timer tick
    sample_counter = 0

    def push_chunk():
        nonlocal sample_counter
        t = (sample_counter + np.arange(chunk_size)) / args.srate
        # Shape: (chunk_size, n_channels)
        data = (np.sin(2.0 * np.pi * frequencies[None, :] * t[:, None]) * amplitudes[None, :]).astype(np.float32)
        widget.push_data(data)
        sample_counter += chunk_size

    timer = QTimer()
    timer.timeout.connect(push_chunk)
    timer.start(max(1, int(1000 / 60)))  # ~16 ms

    app.aboutToQuit.connect(timer.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
