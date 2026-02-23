"""Base class for multichannel plot widgets with shared canvas, scrolling, and key handling."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QLabel, QToolTip, QVBoxLayout, QWidget
from rendercanvas.qt import QRenderWidget

from .constants import CHANNEL_COLORS

__all__ = ["ChannelPlotWidget"]


class ChannelPlotWidget(QWidget):
    """Base widget for GPU-rendered multichannel plots.

    Provides canvas setup, channel scrolling (↑↓ PgUp/PgDn [ ]),
    y-scale keys (- = A), tooltip on hover, and range label overlay.

    Subclass contract:
    - Set ``self._buffer`` to a buffer object before calling ``_init_rendering()``.
    Expected interface: ``.n_visible``, ``.n_channels``, ``.channel_offset``,
    ``.set_channel_offset(int)``, ``.set_n_visible(int)``,
    ``.adjust_y_scale(float)``, ``.toggle_autoscale()``.
    - Add axis widgets to ``self.layout()`` after ``super().__init__``.
    - Implement ``_draw_frame(self)`` (called by rendercanvas scheduler).
    - Override ``_handle_key(key)`` to intercept subclass keys, calling
    ``super()._handle_key(key)`` for common keys.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        n_visible: int,
        channel_labels: list[str] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._n_channels = n_channels
        self._channel_labels = channel_labels

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Render canvas
        self.canvas = QRenderWidget(parent=self)
        layout.addWidget(self.canvas)

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

        # _buffer is set by the subclass before calling _init_rendering()
        self._buffer = None

    # ------------------------------------------------------------------
    # Subclass hook: call after buffer + renderer are ready
    # ------------------------------------------------------------------

    def _init_rendering(self) -> None:
        """Start continuous rendering and initialize the range label.

        Call from subclass ``__init__`` after ``self._buffer`` and any
        renderer are fully constructed.
        """
        from .constants import DEFAULT_MAX_FPS

        self.canvas.set_update_mode("continuous", max_fps=DEFAULT_MAX_FPS)
        self.canvas.request_draw(self._draw_frame)
        self._update_range_label()

    # ------------------------------------------------------------------
    # Subclass must implement
    # ------------------------------------------------------------------

    def _draw_frame(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Event filter (keyboard + mouse)
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

    # ------------------------------------------------------------------
    # Keyboard controls (shared)
    # ------------------------------------------------------------------

    def _handle_key(self, key: int) -> None:
        buf = self._buffer

        if key == Qt.Key.Key_Up:
            buf.set_channel_offset(buf.channel_offset - 1)
        elif key == Qt.Key.Key_Down:
            buf.set_channel_offset(buf.channel_offset + 1)
        elif key == Qt.Key.Key_PageUp:
            buf.set_channel_offset(buf.channel_offset - buf.n_visible)
        elif key == Qt.Key.Key_PageDown:
            buf.set_channel_offset(buf.channel_offset + buf.n_visible)

        elif key == Qt.Key.Key_BracketLeft:
            buf.set_n_visible(max(1, buf.n_visible // 2))
        elif key == Qt.Key.Key_BracketRight:
            buf.set_n_visible(min(buf.n_channels, buf.n_visible * 2))

        elif key == Qt.Key.Key_Minus:
            buf.adjust_y_scale(0.8)  # zoom out
        elif key == Qt.Key.Key_Equal:
            buf.adjust_y_scale(1.25)  # zoom in

        elif key == Qt.Key.Key_A:
            buf.toggle_autoscale()

        self._update_range_label()

    # ------------------------------------------------------------------
    # Mouse hover tooltip
    # ------------------------------------------------------------------

    def _handle_mouse_move(self, event) -> None:
        h = self.canvas.height()
        if h < 1:
            return
        buf = self._buffer
        mouse_y = event.position().y()
        ndc_y = 1.0 - 2.0 * (mouse_y / h)
        ch_index = int((1.0 - ndc_y) * buf.n_visible / 2.0)
        ch_index = max(0, min(ch_index, buf.n_visible - 1))
        abs_ch = buf.channel_offset + ch_index

        labels = self._channel_labels
        label = labels[abs_ch] if labels and abs_ch < len(labels) else f"Ch {abs_ch}"

        rgba = CHANNEL_COLORS[ch_index % len(CHANNEL_COLORS)]
        hex_color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
        html = f'<span style="color:{hex_color}">\u25a0</span> {label}'
        QToolTip.showText(event.globalPosition().toPoint(), html, self.canvas)

    # ------------------------------------------------------------------
    # Channel range overlay
    # ------------------------------------------------------------------

    def _update_range_label(self) -> None:
        buf = self._buffer
        if buf is None:
            return
        first = buf.channel_offset + 1  # 1-indexed for display
        last = buf.channel_offset + buf.n_visible
        total = buf.n_channels
        self._range_label.setText(f"Ch {first}\u2013{last} / {total}")
        self._range_label.adjustSize()
        # Position at bottom-left of canvas
        margin = 4
        y = self.canvas.height() - self._range_label.height() - margin
        self._range_label.move(margin, max(0, y))
