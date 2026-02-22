"""GPU-accelerated real-time sweep renderer for multichannel timeseries."""

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget
from rendercanvas.qt import QRenderWidget

from .constants import DEFAULT_DISPLAY_DUR, DEFAULT_MAX_FPS, DEFAULT_N_COLUMNS, DEFAULT_N_VISIBLE
from .gpu_renderer import GPURenderer
from .sweep_buffer import SweepBuffer

__all__ = ["SweepConfig", "SweepWidget"]


@dataclass
class SweepConfig:
    n_channels: int
    srate: float
    display_dur: float = DEFAULT_DISPLAY_DUR
    n_columns: int = DEFAULT_N_COLUMNS
    n_visible: int = DEFAULT_N_VISIBLE


class SweepWidget(QWidget):
    """Embeddable QWidget that renders a GPU-accelerated multichannel sweep plot.

    Usage::

        widget = SweepWidget(SweepConfig(n_channels=128, srate=30000.0))
        widget.show()
        widget.push_data(np.random.randn(500, 128).astype(np.float32))
    """

    def __init__(self, config: SweepConfig, parent: QWidget | None = None):
        super().__init__(parent)
        self._config = config

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Render canvas
        self.canvas = QRenderWidget(parent=self)
        layout.addWidget(self.canvas)

        # Keyboard focus: intercept key events on the canvas
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.installEventFilter(self)

        # CPU-side buffer
        self.sweep_buffer = SweepBuffer(
            n_channels=config.n_channels,
            srate=config.srate,
            display_dur=config.display_dur,
            n_columns=config.n_columns,
            n_visible=min(config.n_visible, config.n_channels),
        )

        # Create GPU renderer eagerly so the wgpu context exists
        # before the first force_draw() (rendercanvas's _draw_and_present
        # cancels the draw if context is None).
        self.gpu_renderer = GPURenderer(self.canvas)

        # Register draw callback and drive rendering with our own QTimer
        self.canvas.request_draw(self._draw_frame)
        self._draw_timer = QTimer(self)
        self._draw_timer.timeout.connect(lambda: self.canvas.force_draw())
        self._draw_timer.start(max(1, int(1000 / DEFAULT_MAX_FPS)))

    def closeEvent(self, event):
        self._draw_timer.stop()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_data(self, data: np.ndarray) -> None:
        """Push new samples. *data* shape: ``(n_samples, n_channels)``, float32."""
        self.sweep_buffer.push_data(data)

    def update_config(self, config: SweepConfig) -> None:
        """Update configuration at runtime (e.g. on reconnect)."""
        self._config = config
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

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_frame(self) -> None:
        self.gpu_renderer.update_and_draw(self.sweep_buffer)

    # ------------------------------------------------------------------
    # Keyboard controls
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.canvas and event.type() == QEvent.Type.KeyPress:
            self._handle_key(event.key())
            return True
        return super().eventFilter(obj, event)

    def _handle_key(self, key: int) -> None:
        buf = self.sweep_buffer

        if key == Qt.Key.Key_Up:
            buf.set_channel_offset(buf.channel_offset - 1)
        elif key == Qt.Key.Key_Down:
            buf.set_channel_offset(buf.channel_offset + 1)
        elif key == Qt.Key.Key_PageUp:
            buf.set_channel_offset(buf.channel_offset - buf.n_visible)
        elif key == Qt.Key.Key_PageDown:
            buf.set_channel_offset(buf.channel_offset + buf.n_visible)

        elif key == Qt.Key.Key_BracketLeft:
            # Halve visible channels
            buf.set_n_visible(max(1, buf.n_visible // 2))
        elif key == Qt.Key.Key_BracketRight:
            # Double visible channels
            buf.set_n_visible(min(buf.n_channels, buf.n_visible * 2))

        elif key == Qt.Key.Key_Minus:
            buf.adjust_y_scale(0.8)  # zoom out
        elif key == Qt.Key.Key_Equal:
            buf.adjust_y_scale(1.25)  # zoom in

        elif key == Qt.Key.Key_A:
            buf.toggle_autoscale()

        elif key == Qt.Key.Key_Comma:
            buf.set_display_dur(buf.display_dur / 2.0)  # halve duration
        elif key == Qt.Key.Key_Period:
            buf.set_display_dur(buf.display_dur * 2.0)  # double duration
