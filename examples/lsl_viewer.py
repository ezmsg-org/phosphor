#!/usr/bin/env python
"""
LSL Viewer — GPU-accelerated real-time viewer for Lab Streaming Layer data.

Resolves the first available LSL stream and plots it using phosphor.

Usage:
    python lsl_viewer.py                  # auto-resolve first stream
    python lsl_viewer.py --name MyStream  # resolve stream by name
    python lsl_viewer.py --type EEG       # resolve stream by type
    python lsl_viewer.py --visible 32     # show 32 channels at a time

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

from phosphor import SweepConfig, SweepWidget

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


def main():
    parser = argparse.ArgumentParser(description="LSL Viewer (phosphor)")
    parser.add_argument("--name", type=str, default=None, help="Resolve stream by name")
    parser.add_argument("--type", type=str, default=None, help="Resolve stream by type")
    parser.add_argument("--dur", type=float, default=2.0, help="Display duration in seconds")
    parser.add_argument("--visible", type=int, default=None, help="Visible channels (default: all)")
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

    # Pre-allocate pull buffer matching the stream's native dtype
    max_samples = max(1024, math.ceil(srate / 30) * 2)
    stream_dtype = _LSL_DTYPES.get(info.channel_format(), np.float32)
    pull_buffer = np.empty((max_samples, n_channels), dtype=stream_dtype, order="C")

    config = SweepConfig(
        n_channels=n_channels,
        srate=srate,
        display_dur=args.dur,
        n_visible=n_visible,
    )
    widget = SweepWidget(config)
    widget.setWindowTitle(f"LSL: {info.name()} ({n_channels}ch @ {srate}Hz)")
    widget.resize(1200, 800)
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
