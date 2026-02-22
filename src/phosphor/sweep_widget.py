"""GPU-accelerated real-time sweep plot widget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from .channel_plot import ChannelPlotWidget
from .constants import DEFAULT_DISPLAY_DUR, DEFAULT_N_COLUMNS, DEFAULT_N_VISIBLE
from .gpu_renderer import GPURenderer
from .sweep_buffer import SweepBuffer
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
        )
        self._buffer = self.sweep_buffer

        # Create GPU renderer eagerly so the wgpu context exists
        # before the first draw (rendercanvas's _draw_and_present
        # cancels the draw if context is None).
        self.gpu_renderer = GPURenderer(self.canvas)

        # Start rendering
        self._init_rendering()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_data(self, data: np.ndarray) -> None:
        """Push new samples. *data* shape: ``(n_samples, n_channels)``, float32."""
        self.sweep_buffer.push_data(data)

    def update_config(self, config: SweepConfig) -> None:
        """Update configuration at runtime (e.g. on reconnect)."""
        self._config = config
        self._channel_labels = config.channel_labels
        buf = self.sweep_buffer
        if config.n_channels != buf.n_channels:
            buf.n_channels = config.n_channels
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
    # Rendering
    # ------------------------------------------------------------------

    def _draw_frame(self) -> None:
        self.gpu_renderer.update_and_draw(self.sweep_buffer)

    # ------------------------------------------------------------------
    # Keyboard controls (sweep-specific keys)
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> None:
        buf = self.sweep_buffer

        if key == Qt.Key.Key_Comma:
            buf.set_display_dur(buf.display_dur / 2.0)  # halve duration
            self._time_axis.set_range(buf.display_dur)
            self._update_range_label()
        elif key == Qt.Key.Key_Period:
            buf.set_display_dur(buf.display_dur * 2.0)  # double duration
            self._time_axis.set_range(buf.display_dur)
            self._update_range_label()
        else:
            super()._handle_key(key)
