#!/usr/bin/env python
"""
Scatter Demo — simulated scalp-map heatmap with the ScatterWidget.

Generates electrode positions on concentric rings (mimicking a 10-20 EEG
montage layout) and drives them with a rotating spatial wave so color and
size ripple outward from a wandering hotspot.

Usage:
    python scatter_demo.py                     # 64 electrodes, color+size
    python scatter_demo.py --channels 128      # more electrodes
    python scatter_demo.py --no-size           # color only
    python scatter_demo.py --cmap plasma       # different colormap
    python scatter_demo.py --fixed-range 0 1   # fixed vmin/vmax

Requires: phosphor
    pip install phosphor
"""

import argparse
import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from phosphor import ScatterConfig, ScatterWidget


def make_scalp_positions(n: int) -> np.ndarray:
    """Generate electrode positions on concentric rings like a scalp map.

    Returns an (n, 2) array with positions roughly within a unit circle.
    """
    positions = []

    # Place one electrode at the center (Cz-like)
    positions.append([0.0, 0.0])
    remaining = n - 1

    ring = 1
    while remaining > 0:
        radius = ring * 0.25
        # More electrodes on outer rings
        count = min(remaining, 6 * ring)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        # Offset alternate rings for nicer packing
        angles += (ring % 2) * np.pi / count
        for a in angles:
            positions.append([radius * np.cos(a), radius * np.sin(a)])
        remaining -= count
        ring += 1

    return np.array(positions[:n], dtype=np.float32)


def make_electrode_labels(n: int) -> list[str]:
    """Generate electrode labels (E0, E1, …)."""
    return [f"E{i}" for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description="Phosphor scatter/heatmap demo")
    parser.add_argument("--channels", type=int, default=64, help="Number of electrodes")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap name")
    parser.add_argument("--no-size", action="store_true", help="Disable size modulation")
    parser.add_argument(
        "--fixed-range",
        type=float,
        nargs=2,
        metavar=("VMIN", "VMAX"),
        default=None,
        help="Fixed value range (default: autoscale)",
    )
    parser.add_argument("--marker-size", type=float, default=14.0, help="Base marker size")
    parser.add_argument("--fps", type=float, default=60.0, help="Data push rate (Hz)")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    n_ch = args.channels
    positions = make_scalp_positions(n_ch)
    labels = make_electrode_labels(n_ch)

    vmin = args.fixed_range[0] if args.fixed_range else None
    vmax = args.fixed_range[1] if args.fixed_range else None

    config = ScatterConfig(
        positions=positions,
        cmap=args.cmap,
        modulate_color=True,
        modulate_size=not args.no_size,
        marker_size=args.marker_size,
        size_range=(4.0, 24.0),
        vmin=vmin,
        vmax=vmax,
        channel_labels=labels,
    )

    widget = ScatterWidget(config)
    widget.setWindowTitle(f"Phosphor Scatter Demo — {n_ch} electrodes")
    widget.resize(800, 800)
    widget.show()

    # --- Simulation state ---
    t = 0.0
    dt = 1.0 / args.fps

    def push_data():
        nonlocal t

        # Hotspot wanders in a figure-8 (Lissajous) pattern
        hx = 0.5 * np.sin(t * 0.7)
        hy = 0.5 * np.sin(t * 1.1)

        # Distance from each electrode to the hotspot
        dx = positions[:, 0] - hx
        dy = positions[:, 1] - hy
        dist = np.sqrt(dx**2 + dy**2)

        # Gaussian blob centered on the hotspot, plus a small amount of noise
        values = np.exp(-(dist**2) / 0.15) + np.random.randn(n_ch).astype(np.float32) * 0.03

        widget.push_data(values.astype(np.float32))
        t += dt

    timer = QTimer()
    timer.timeout.connect(push_data)
    timer.start(max(1, int(1000 / args.fps)))

    app.aboutToQuit.connect(timer.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
