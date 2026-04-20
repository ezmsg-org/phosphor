"""Base class for multichannel plot widgets backed by fastplotlib."""

from __future__ import annotations

import fastplotlib as fpl
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QToolTip, QVBoxLayout, QWidget

from .constants import CHANNEL_COLORS

__all__ = ["ChannelPlotWidget"]


class ChannelPlotWidget(QWidget):
    """Base widget for fastplotlib-rendered multichannel plots.

    Provides canvas setup, channel scrolling (scroll / ↑↓ / PgUp/PgDn / [ ]),
    amplitude zoom (Shift+scroll / - = A), and range label overlay.

    Subclass contract:

    - Set ``self._buffer`` before calling ``_init_rendering()``.
      Expected interface: ``.n_visible``, ``.n_channels``, ``.channel_offset``,
      ``.set_channel_offset(int)``, ``.set_n_visible(int)``.
    - Add axis widgets to ``self.layout()`` after ``super().__init__``.
    - Implement ``_update_graphics()`` (called every frame via animation callback).
    - Override ``_on_ctrl_scroll(delta)`` for time/freq zoom.
    - Override ``_on_key_down(key)`` for subclass-specific keys, calling
      ``super()._on_key_down(key)`` for common keys.
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
        self._autoscale_enabled = True

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # fastplotlib figure with a single subplot
        self._figure = fpl.Figure()
        self._subplot = self._figure[0, 0]

        # Get the Qt widget from fastplotlib and embed it
        self._fpl_widget = self._figure.show()
        layout.addWidget(self._fpl_widget)

        # Channel range overlay label (parented to fpl widget so it floats on top)
        self._range_label = QLabel(self._fpl_widget)
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
    # Subclass hook: call after buffer is ready
    # ------------------------------------------------------------------

    def _init_rendering(self) -> None:
        """Start rendering and register event handlers.

        Call from subclass ``__init__`` after ``self._buffer`` is set and
        initial graphics are created via ``_setup_graphics()``.
        """
        # Disable built-in pan/zoom controller, axes, and default title
        self._subplot.controller = None
        self._subplot.axes.visible = False
        self._subplot.title.visible = False
        self._subplot.camera.maintain_aspect = False

        # Register fpl event handlers on the subplot's pygfx renderer
        renderer = self._subplot.renderer
        renderer.add_event_handler(self._on_key_down_event, "key_down")
        renderer.add_event_handler(self._on_wheel_event, "wheel")
        renderer.add_event_handler(self._on_pointer_move_event, "pointer_move")

        # Register animation callback (wrap in lambda to avoid getfullargspec
        # issue with bound methods under `from __future__ import annotations`)
        self._figure.add_animations(lambda: self._animation_callback())

        self._update_range_label()

    # ------------------------------------------------------------------
    # Subclass must implement
    # ------------------------------------------------------------------

    def _update_graphics(self) -> None:
        """Update MultiLine data each frame. Called from animation callback."""
        raise NotImplementedError

    def _apply_auto_scale(self) -> None:
        """Set camera to fit content. Override in subclass for fast path."""
        self._subplot.auto_scale(maintain_aspect=False, zoom=1.0)

    def _on_ctrl_scroll(self, delta: float) -> None:
        """Handle Ctrl+scroll for time/freq zoom. Override in subclass."""

    # ------------------------------------------------------------------
    # Animation callback
    # ------------------------------------------------------------------

    def _animation_callback(self) -> None:
        self._update_graphics()
        if self._autoscale_enabled:
            self._apply_auto_scale()

    # ------------------------------------------------------------------
    # Event handlers (fpl native events)
    # ------------------------------------------------------------------

    def _on_key_down_event(self, event) -> None:
        self._on_key_down(event.key)

    def _on_key_down(self, key: str) -> None:
        """Handle keyboard events. Override in subclass, call super for common keys."""
        buf = self._buffer

        if key == "ArrowUp":
            buf.set_channel_offset(buf.channel_offset - 1)
        elif key == "ArrowDown":
            buf.set_channel_offset(buf.channel_offset + 1)
        elif key == "PageUp":
            buf.set_channel_offset(buf.channel_offset - buf.n_visible)
        elif key == "PageDown":
            buf.set_channel_offset(buf.channel_offset + buf.n_visible)

        elif key == "[":
            buf.set_n_visible(max(1, buf.n_visible // 2))
        elif key == "]":
            buf.set_n_visible(min(buf.n_channels, buf.n_visible * 2))

        elif key == "-":
            self._zoom_amplitude(0.8)
        elif key == "=":
            self._zoom_amplitude(1.25)

        elif key in ("a", "A"):
            self._autoscale_enabled = not self._autoscale_enabled
            if self._autoscale_enabled:
                self._apply_auto_scale()

        self._update_range_label()

    def _on_wheel_event(self, event) -> None:
        delta = event.dy

        if "Control" in getattr(event, "modifiers", ()):
            # Ctrl+scroll → time/freq zoom (subclass hook)
            self._on_ctrl_scroll(delta)
        elif "Shift" in getattr(event, "modifiers", ()):
            # Shift+scroll → amplitude zoom
            factor = 1.1 if delta > 0 else 0.9
            self._zoom_amplitude(factor)
        else:
            # Unmodified scroll → channel scroll
            buf = self._buffer
            step = 1 if delta < 0 else -1
            buf.set_channel_offset(buf.channel_offset + step)
            self._update_range_label()

    def _on_pointer_move_event(self, event) -> None:
        self._handle_mouse_move(event)

    # ------------------------------------------------------------------
    # Amplitude zoom helper
    # ------------------------------------------------------------------

    def _zoom_amplitude(self, factor: float) -> None:
        """Scale the waveform amplitude per row, leaving channel spacing fixed.

        Sweep buffers expose ``set_amplitude_scale`` for this purpose. Other
        buffer types fall back to the legacy camera scaling (which also
        rescales row offsets, an undesirable side effect).
        """
        buf = self._buffer
        if hasattr(buf, "set_amplitude_scale"):
            buf.set_amplitude_scale(buf.amplitude_scale * factor)
            return
        camera = self._subplot.camera
        camera.world.scale_y *= factor
        self._autoscale_enabled = False

    # ------------------------------------------------------------------
    # Mouse hover tooltip
    # ------------------------------------------------------------------

    def _handle_mouse_move(self, event) -> None:
        if self._multi_line is None:
            return
        buf = self._buffer

        # Convert screen position to world coordinates
        world = self._subplot.map_screen_to_world(event)
        if world is None:
            return
        wy = float(world[1])

        # Find the nearest line by comparing computed Y positions
        best_idx = 0
        best_dist = float("inf")
        for i in range(buf.n_visible):
            line_y = i * self._z_offset_scale
            dist = abs(wy - line_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        ch_index = best_idx
        abs_ch = buf.channel_offset + ch_index

        labels = self._channel_labels
        label = labels[abs_ch] if labels and abs_ch < len(labels) else f"Ch {abs_ch}"

        rgba = CHANNEL_COLORS[ch_index % len(CHANNEL_COLORS)]
        hex_color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
        html = f'<span style="color:{hex_color}">\u25a0</span> {label}'
        from PySide6.QtCore import QPoint

        QToolTip.showText(self._fpl_widget.mapToGlobal(QPoint(int(event.x), int(event.y))), html, self._fpl_widget)

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
        # Position at bottom-left of fpl widget
        margin = 4
        y = self._fpl_widget.height() - self._range_label.height() - margin
        self._range_label.move(margin, max(0, y))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_range_label()
