"""GPU-accelerated real-time sweep plot widget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import QWidget

from .channel_plot import ChannelPlotWidget
from .constants import (
    CHANNEL_COLORS,
    CURSOR_COLOR,
    CURSOR_GAP_COLUMNS,
    DEFAULT_DISPLAY_DUR,
    DEFAULT_MAX_EVENTS,
    DEFAULT_N_COLUMNS,
    DEFAULT_N_VISIBLE,
    EVENT_POOL_SIZE,
    EVENT_THICKNESS,
)
from .sweep_buffer import SweepBuffer, SweepEvent
from .x_axis import XAxisWidget

__all__ = ["SweepConfig", "SweepWidget"]


@dataclass
class SweepConfig:
    n_channels: int
    srate: float
    display_dur: float = DEFAULT_DISPLAY_DUR
    n_columns: int = DEFAULT_N_COLUMNS
    n_visible: int = DEFAULT_N_VISIBLE
    channel_labels: list[str] | None = None
    max_events: int = DEFAULT_MAX_EVENTS
    # ``top_down`` (default) puts channel index 0 at the top of the canvas
    # — what most scientific viewers do. ``bottom_up`` puts it at the bottom.
    channel_order: str = "top_down"


class SweepWidget(ChannelPlotWidget):
    """Embeddable QWidget that renders a GPU-accelerated multichannel sweep plot.

    Usage::

        widget = SweepWidget(SweepConfig(n_channels=128, srate=30000.0))
        widget.show()
        widget.push_data(np.random.randn(500, 128).astype(np.float32))
    """

    def __init__(self, config: SweepConfig, parent: QWidget | None = None):
        super().__init__(
            n_channels=config.n_channels,
            n_visible=min(config.n_visible, config.n_channels),
            channel_labels=config.channel_labels,
            parent=parent,
        )
        self._config = config

        # Time axis labels (sweep-specific, added below the canvas)
        self._time_axis = XAxisWidget(config.display_dur, unit="s", parent=self)
        self.layout().addWidget(self._time_axis)

        # CPU-side buffer
        self.sweep_buffer = SweepBuffer(
            n_channels=config.n_channels,
            srate=config.srate,
            display_dur=config.display_dur,
            n_columns=config.n_columns,
            n_visible=min(config.n_visible, config.n_channels),
            max_events=config.max_events,
            channel_order=config.channel_order,
        )
        self._buffer = self.sweep_buffer

        # Create initial graphics
        self._cached_version = -1
        self._multi_line = None
        self._z_offset_scale = 1.0
        self._cursor_line = None
        self._event_multi_line = None
        self._events_visible_count = 0
        self._setup_graphics()

        # Start rendering
        self._init_rendering()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_data(self, data: np.ndarray, timestamps=None) -> None:
        """Push new samples. *data* shape: ``(n_samples, n_channels)``, float32.

        *timestamps* — ``None``, a scalar (time of first sample), or an
        iterable of length *n_samples* (per-sample times).
        """
        self.sweep_buffer.push_data(data, timestamps)

    def push_events(self, events: list[SweepEvent]) -> None:
        """Push discrete events for overlay rendering."""
        self.sweep_buffer.push_events(events)

    def update_config(self, config: SweepConfig) -> None:
        """Update configuration at runtime (e.g. on reconnect)."""
        self._config = config
        self._channel_labels = config.channel_labels
        buf = self.sweep_buffer
        if config.n_channels != buf.n_channels:
            buf.set_n_channels(config.n_channels)
        if config.srate != buf.srate:
            buf.set_srate(config.srate)
        new_vis = min(config.n_visible, config.n_channels)
        if new_vis != buf.n_visible:
            buf.set_n_visible(new_vis)
        if config.display_dur != buf.display_dur:
            buf.set_display_dur(config.display_dur)
            self._time_axis.set_range(config.display_dur)
        self._update_range_label()

    # ------------------------------------------------------------------
    # Graphics setup
    # ------------------------------------------------------------------

    def _setup_graphics(self) -> None:
        """Create or recreate MultiLineGraphic and cursor line."""
        subplot = self._subplot

        # Delete old graphics
        if self._multi_line is not None:
            subplot.delete_graphic(self._multi_line)
            self._multi_line = None
        if self._cursor_line is not None:
            subplot.delete_graphic(self._cursor_line)
            self._cursor_line = None
        if self._event_multi_line is not None:
            subplot.delete_graphic(self._event_multi_line)
            self._event_multi_line = None
        self._events_visible_count = 0

        buf = self.sweep_buffer
        data = buf.get_multiline_data()

        # Build per-channel colors (cycling through CHANNEL_COLORS)
        n_vis = buf.n_visible
        colors = [CHANNEL_COLORS[i % len(CHANNEL_COLORS)][:3] for i in range(n_vis)]

        self._multi_line = subplot.add_multi_line(
            data,
            colors=colors,
            z_offset_scale=self._z_offset_scale,
            thickness=1.5,
        )

        # Cursor: vertical line at the sweep position.
        # Span the MultiLine's y-extent with small margin.
        sweep_x = buf.sweep_col / max(buf.n_columns - 1, 1) * buf.display_dur
        cursor_color = CURSOR_COLOR[:3]
        gap_w = CURSOR_GAP_COLUMNS / max(buf.n_columns - 1, 1) * buf.display_dur
        y_bottom = 0.0
        y_top = (n_vis - 1) * self._z_offset_scale
        margin = max((y_top - y_bottom) * 0.05, 0.5)
        self._cursor_y_min = y_bottom - margin
        self._cursor_y_max = y_top + margin
        self._cursor_line = subplot.add_line(
            np.array(
                [[sweep_x, self._cursor_y_min, 0], [sweep_x, self._cursor_y_max, 0]],
                dtype=np.float32,
            ),
            colors=cursor_color,
            thickness=max(1.0, gap_w * 2),
        )

        self._setup_event_pool()
        self._cached_version = buf.version

    def _setup_event_pool(self) -> None:
        """Pre-allocate a single MultiLineGraphic for event ticks."""
        data = np.zeros((EVENT_POOL_SIZE, 2, 3), dtype=np.float32)
        self._event_multi_line = self._subplot.add_multi_line(
            data,
            thickness=EVENT_THICKNESS,
        )

    # ------------------------------------------------------------------
    # Rendering (animation callback)
    # ------------------------------------------------------------------

    def _update_graphics(self) -> None:
        buf = self.sweep_buffer

        if buf.version != self._cached_version:
            # Version changed (scroll, resize, display_dur change) → full rebuild
            self._setup_graphics()
            return

        # Incremental update from dirty columns
        result = buf.get_dirty_multiline_range()
        if result is not None:
            data_slice, col_start, n_cols = result
            idx_start = col_start * 2
            idx_end = (col_start + n_cols) * 2
            self._multi_line.data[:, idx_start:idx_end] = data_slice

        # Update cursor x-position; y spans the MultiLine extent (not camera,
        # which would create a feedback loop with auto_scale).
        sweep_x = buf.sweep_col / max(buf.n_columns - 1, 1) * buf.display_dur
        self._cursor_line.data[0] = [sweep_x, self._cursor_y_min, 0]
        self._cursor_line.data[1] = [sweep_x, self._cursor_y_max, 0]

        self._update_event_graphics()

    def _update_event_graphics(self) -> None:
        """Update the event tick MultiLineGraphic from visible events."""
        buf = self.sweep_buffer
        visible_events = buf.get_visible_events()
        ml = self._event_multi_line

        data = np.zeros((EVENT_POOL_SIZE, 2, 3), dtype=np.float32)
        # Color stride: 2 data vertices + 1 NaN separator per line
        color_stride = ml.colors.value.shape[0] // EVENT_POOL_SIZE
        colors = np.zeros((EVENT_POOL_SIZE * color_stride, 4), dtype=np.float32)
        n_active = 0

        for ev, x_pos in visible_events:
            if n_active >= EVENT_POOL_SIZE:
                break

            if ev.channel is None:
                y_min = self._cursor_y_min
                y_max = self._cursor_y_max
            else:
                vis_start = buf.channel_offset
                vis_end = buf.channel_offset + buf.n_visible
                if ev.channel < vis_start or ev.channel >= vis_end:
                    continue
                vis_idx = ev.channel - buf.channel_offset
                y_center = vis_idx * self._z_offset_scale
                y_min = y_center - 0.45 * self._z_offset_scale
                y_max = y_center + 0.45 * self._z_offset_scale

            data[n_active, 0] = [x_pos, y_min, 0]
            data[n_active, 1] = [x_pos, y_max, 0]
            ci = n_active * color_stride
            colors[ci : ci + 2] = (*ev.color, 1.0)
            n_active += 1

        ml.data[:] = data
        ml.colors[:] = colors
        self._events_visible_count = n_active

    def _apply_auto_scale(self) -> None:
        """Set camera bounds from known data layout (avoids GPU readback)."""
        buf = self.sweep_buffer
        cam = self._subplot.camera
        cam.width = buf.display_dur
        cam.height = self._cursor_y_max - self._cursor_y_min
        cam.world.position = (
            buf.display_dur / 2,
            (self._cursor_y_min + self._cursor_y_max) / 2,
            (buf.n_visible - 1) / 2,
        )

    # ------------------------------------------------------------------
    # Keyboard controls (sweep-specific keys)
    # ------------------------------------------------------------------

    def _on_key_down(self, key: str) -> None:
        if key == ",":
            self._time_zoom(0.5)
        elif key == ".":
            self._time_zoom(2.0)
        else:
            super()._on_key_down(key)

    def _on_ctrl_scroll(self, delta: float) -> None:
        factor = 0.5 if delta > 0 else 2.0
        self._time_zoom(factor)

    def _time_zoom(self, factor: float) -> None:
        buf = self.sweep_buffer
        buf.set_display_dur(buf.display_dur * factor)
        self._time_axis.set_range(buf.display_dur)
        self._update_range_label()
