"""GPU-accelerated real-time sweep renderer for multichannel timeseries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QLabel, QToolTip, QVBoxLayout, QWidget
from rendercanvas.qt import QRenderWidget

from .constants import (
    BG_COLOR,
    CHANNEL_COLORS,
    DEFAULT_DISPLAY_DUR,
    DEFAULT_MAX_FPS,
    DEFAULT_N_COLUMNS,
    DEFAULT_N_VISIBLE,
)
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
    channel_labels: list[str] | None = None


_NICE_INTERVALS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)


class TimeAxisWidget(QWidget):
    """Lightweight time-axis labels drawn with QPainter below the render canvas."""

    def __init__(self, display_dur: float, parent: QWidget | None = None):
        super().__init__(parent)
        self._display_dur = display_dur
        self.setFixedHeight(24)
        bg = BG_COLOR
        self.setStyleSheet(f"background-color: rgb({int(bg[0]*255)},{int(bg[1]*255)},{int(bg[2]*255)});")

    def set_display_dur(self, dur: float) -> None:
        self._display_dur = dur
        self.update()

    def paintEvent(self, event) -> None:
        w = self.width()
        if w < 1 or self._display_dur <= 0:
            return

        # Pick a nice tick interval targeting ~5-8 ticks.
        best = _NICE_INTERVALS[-1]
        for iv in _NICE_INTERVALS:
            n_ticks = self._display_dur / iv
            if n_ticks <= 10:
                best = iv
                break
        interval = best

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        painter.setFont(font)
        pen_color = QColor(180, 180, 180)
        painter.setPen(pen_color)
        fm = painter.fontMetrics()

        # Determine decimal places from interval.
        if interval >= 1.0 and interval == int(interval):
            decimals = 1
        else:
            decimals = max(1, -int(np.floor(np.log10(interval))))

        t = 0.0
        tick_values = []
        while t <= self._display_dur + interval * 0.01:
            tick_values.append(t)
            t += interval

        for i, tv in enumerate(tick_values):
            x = int(tv / self._display_dur * w) if self._display_dur > 0 else 0
            # Tick mark
            painter.drawLine(x, 0, x, 4)
            # Label
            label = f"{tv:.{decimals}f}"
            if i == len(tick_values) - 1:
                label += "s"
            tw = fm.horizontalAdvance(label)
            lx = max(0, min(x - tw // 2, w - tw))
            painter.drawText(lx, 18, label)

        painter.end()


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

        # Time axis labels
        self._time_axis = TimeAxisWidget(config.display_dur, parent=self)
        layout.addWidget(self._time_axis)

        # Keyboard focus & mouse tracking: intercept events on the canvas
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.setMouseTracking(True)
        self.canvas.installEventFilter(self)

        # Channel range overlay label (parented to canvas so it floats on top)
        self._range_label = QLabel(self.canvas)
        self._range_label.setStyleSheet(
            "background: rgba(25,25,30,200); color: #b4b4b4;"
            " padding: 2px 6px; font-size: 9pt;"
            " font-family: 'Menlo', 'Consolas', 'DejaVu Sans Mono', monospace;"
        )
        self._range_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._range_label.show()

        # CPU-side buffer
        self.sweep_buffer = SweepBuffer(
            n_channels=config.n_channels,
            srate=config.srate,
            display_dur=config.display_dur,
            n_columns=config.n_columns,
            n_visible=min(config.n_visible, config.n_channels),
        )

        # Create GPU renderer eagerly so the wgpu context exists
        # before the first draw (rendercanvas's _draw_and_present
        # cancels the draw if context is None).
        self.gpu_renderer = GPURenderer(self.canvas)

        # Use rendercanvas's built-in scheduler for frame pacing
        self.canvas.set_update_mode("continuous", max_fps=DEFAULT_MAX_FPS)
        self.canvas.request_draw(self._draw_frame)

        # Initial range label text
        self._update_range_label()

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
            self._time_axis.set_display_dur(config.display_dur)
        self._update_range_label()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_frame(self) -> None:
        self.gpu_renderer.update_and_draw(self.sweep_buffer)

    # ------------------------------------------------------------------
    # Keyboard controls
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        if obj is self.canvas:
            if event.type() == QEvent.Type.KeyPress:
                self._handle_key(event.key())
                return True
            if event.type() == QEvent.Type.MouseMove:
                self._handle_mouse_move(event)
                return False  # don't swallow — let canvas process too
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_range_label()

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
            self._time_axis.set_display_dur(buf.display_dur)
        elif key == Qt.Key.Key_Period:
            buf.set_display_dur(buf.display_dur * 2.0)  # double duration
            self._time_axis.set_display_dur(buf.display_dur)

        self._update_range_label()

    # ------------------------------------------------------------------
    # Mouse hover tooltip
    # ------------------------------------------------------------------

    def _handle_mouse_move(self, event) -> None:
        h = self.canvas.height()
        if h < 1:
            return
        buf = self.sweep_buffer
        mouse_y = event.position().y()
        ndc_y = 1.0 - 2.0 * (mouse_y / h)
        ch_index = int((1.0 - ndc_y) * buf.n_visible / 2.0)
        ch_index = max(0, min(ch_index, buf.n_visible - 1))
        abs_ch = buf.channel_offset + ch_index

        labels = self._config.channel_labels
        label = labels[abs_ch] if labels and abs_ch < len(labels) else f"Ch {abs_ch}"

        rgba = CHANNEL_COLORS[ch_index % len(CHANNEL_COLORS)]
        hex_color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
        html = f'<span style="color:{hex_color}">\u25a0</span> {label}'
        QToolTip.showText(event.globalPosition().toPoint(), html, self.canvas)

    # ------------------------------------------------------------------
    # Channel range overlay
    # ------------------------------------------------------------------

    def _update_range_label(self) -> None:
        buf = self.sweep_buffer
        first = buf.channel_offset + 1  # 1-indexed for display
        last = buf.channel_offset + buf.n_visible
        total = buf.n_channels
        self._range_label.setText(f"Ch {first}\u2013{last} / {total}")
        self._range_label.adjustSize()
        # Position at bottom-left of canvas
        margin = 4
        y = self.canvas.height() - self._range_label.height() - margin
        self._range_label.move(margin, max(0, y))
