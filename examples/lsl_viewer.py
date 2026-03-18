#!/usr/bin/env python
"""
LSL Viewer — GPU-accelerated real-time viewer for Lab Streaming Layer data.

Resolves the first available LSL stream and plots it using phosphor.

Usage:
    python lsl_viewer.py                  # auto-resolve first stream
    python lsl_viewer.py --name MyStream  # resolve stream by name
    python lsl_viewer.py --type EEG       # resolve stream by type
    python lsl_viewer.py --visible 32     # show 32 channels at a time
    python lsl_viewer.py --scatter        # scatter/heatmap (needs channel locations)

Requires: pylsl, phosphor
    pip install pylsl phosphor
"""

import argparse
import math
import sys

import numpy as np
import pylsl
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from phosphor import ScatterConfig, ScatterWidget, SweepConfig, SweepWidget

# Map pylsl channel formats to numpy dtypes
_LSL_DTYPES = {
    pylsl.cf_float32: np.float32,
    pylsl.cf_double64: np.float64,
    pylsl.cf_int32: np.int32,
    pylsl.cf_int16: np.int16,
    pylsl.cf_int8: np.int8,
    pylsl.cf_int64: np.int64,
}


def resolve_stream(name: str | None, type_: str | None) -> pylsl.StreamInfo:
    """Resolve a single LSL stream, blocking until one is found."""
    if name:
        print(f"Resolving stream with name '{name}'...")
        results = pylsl.resolve_byprop("name", name, timeout=pylsl.FOREVER)
    elif type_:
        print(f"Resolving stream with type '{type_}'...")
        results = pylsl.resolve_byprop("type", type_, timeout=pylsl.FOREVER)
    else:
        print("Resolving first available stream...")
        results = pylsl.resolve_streams(wait_time=pylsl.FOREVER)

    # Filter to numeric, regularly-sampled streams
    for info in results:
        if info.nominal_srate() != pylsl.IRREGULAR_RATE and info.channel_format() != pylsl.cf_string:
            return info

    # If nothing matched the filter, just return the first result
    return results[0]


def parse_channel_info(
    info: pylsl.StreamInfo,
) -> tuple[list[str] | None, np.ndarray | None]:
    """Parse channel labels and locations from LSL stream info.

    Returns ``(labels, positions)`` where *positions* is ``(n_channels, 3)``
    float32 or ``None`` if no locations are present.
    """
    n_ch = info.channel_count()
    chans = info.desc().child("channels")
    if chans.empty():
        return None, None

    labels: list[str] = []
    positions: list[list[float]] = []
    has_locations = False

    ch_elem = chans.first_child()
    while not ch_elem.empty():
        # Label
        label_val = ch_elem.child("label").child_value()
        labels.append(label_val if label_val else "")

        # Location — <location><X>val</X><Y>val</Y><Z>val</Z></location>
        loc_elem = ch_elem.child("location")
        if not loc_elem.empty():
            x_val = loc_elem.child("X").child_value()
            y_val = loc_elem.child("Y").child_value()
            z_val = loc_elem.child("Z").child_value()
            x = float(x_val) if x_val else 0.0
            y = float(y_val) if y_val else 0.0
            z = float(z_val) if z_val else 0.0
            positions.append([x, y, z])
            if x != 0.0 or y != 0.0 or z != 0.0:
                has_locations = True
        else:
            positions.append([0.0, 0.0, 0.0])

        ch_elem = ch_elem.next_sibling()

    if len(labels) != n_ch:
        return None, None

    # Fill empty labels with channel indices
    for i, lbl in enumerate(labels):
        if not lbl:
            labels[i] = f"Ch {i}"

    pos_array = np.array(positions, dtype=np.float32) if has_locations else None
    return labels, pos_array


def main():
    parser = argparse.ArgumentParser(description="LSL Viewer (phosphor)")
    parser.add_argument("--name", type=str, default=None, help="Resolve stream by name")
    parser.add_argument("--type", type=str, default=None, help="Resolve stream by type")
    parser.add_argument("--dur", type=float, default=2.0, help="Display duration in seconds")
    parser.add_argument("--visible", type=int, default=None, help="Visible channels (default: all)")
    parser.add_argument(
        "--scatter",
        action="store_true",
        help="Use scatter/heatmap view (requires channel locations in stream metadata)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    info = resolve_stream(args.name, args.type)
    srate = info.nominal_srate()
    n_channels = info.channel_count()
    n_visible = args.visible if args.visible else min(n_channels, 64)

    print(f"Stream: {info.name()} | {n_channels}ch @ {srate}Hz | type={info.type()}")

    inlet = pylsl.StreamInlet(
        info,
        max_buflen=int(max(args.dur * 2, 1)),
        processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter,
    )
    inlet.open_stream()

    # Retrieve full stream info (includes description XML with channel metadata)
    full_info = inlet.info()
    channel_labels, channel_positions = parse_channel_info(full_info)

    if channel_labels:
        print(f"Channel labels: {channel_labels[0]} … {channel_labels[-1]}")
    if channel_positions is not None:
        print(f"Channel locations: found for {n_channels} channels")

    # Pre-allocate pull buffer matching the stream's native dtype
    max_samples = max(1024, math.ceil(srate / 30) * 2)
    stream_dtype = _LSL_DTYPES.get(info.channel_format(), np.float32)
    pull_buffer = np.empty((max_samples, n_channels), dtype=stream_dtype, order="C")

    use_scatter = args.scatter
    if use_scatter and channel_positions is None:
        print("Warning: --scatter requested but no channel locations found; falling back to sweep view")
        use_scatter = False

    if use_scatter:
        config = ScatterConfig(
            positions=channel_positions,
            channel_labels=channel_labels,
        )
        widget = ScatterWidget(config)
        widget.resize(800, 800)
    else:
        config = SweepConfig(
            n_channels=n_channels,
            srate=srate,
            display_dur=args.dur,
            n_visible=n_visible,
            channel_labels=channel_labels,
        )
        widget = SweepWidget(config)
        widget.resize(1200, 800)

    widget.setWindowTitle(f"LSL: {info.name()} ({n_channels}ch @ {srate}Hz)")
    widget.show()

    def pull_and_push():
        _, timestamps = inlet.pull_chunk(
            timeout=0.0,
            max_samples=max_samples,
            dest_obj=pull_buffer,
        )
        if timestamps:
            n = len(timestamps)
            widget.push_data(pull_buffer[:n].astype(np.float32))

    timer = QTimer()
    timer.timeout.connect(pull_and_push)
    timer.start(max(1, int(1000 / 60)))  # ~60 Hz poll

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
