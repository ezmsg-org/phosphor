"""Base class for multichannel plot widgets backed by fastplotlib."""

from __future__ import annotations

import logging

import fastplotlib as fpl
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QToolTip, QVBoxLayout, QWidget

from .constants import CHANNEL_COLORS
from .overlays import ChannelLabelOverlay, ScaleBarOverlay

logger = logging.getLogger(__name__)

__all__ = ["ChannelPlotWidget"]


class ChannelPlotWidget(QWidget):
    """Base widget for fastplotlib-rendered multichannel plots.

    Provides canvas setup, channel scrolling (scroll / ↑↓ / PgUp/PgDn / [ ]),
    amplitude zoom (Shift+scroll / - =), and range label overlay.

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

        # Trace colours, indexed by *absolute channel* so a channel keeps its
        # colour as it scrolls -- see _channel_color. Subclasses override this
        # with a configured palette before building graphics.
        self._palette = list(CHANNEL_COLORS)

        # Overlays, created on demand. Kept as None until asked for so a plot
        # that never wants them pays nothing.
        self._label_overlay: ChannelLabelOverlay | None = None
        self._scale_bar_overlay: ScaleBarOverlay | None = None
        # Cached world->screen-y projection, keyed on the inputs that can change
        # it. map_world_to_screen is a pygfx round-trip; at 60 fps it is worth
        # not repeating while nothing has moved.
        self._projection: tuple[float, float] | None = None
        self._projection_key: tuple | None = None
        self._scale_bar_text: str = ""
        # Identity of the label list last pushed to the overlay, so a relabel
        # via update_config is picked up without copying the list every frame.
        self._label_source: object = None

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
        self._renderer = renderer
        renderer.add_event_handler(self._on_key_down_event, "key_down")
        self._mouse_enabled = True
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
        """Frame the camera on the current content. Override for a fast path.

        Called every frame, unconditionally. It is the only thing that moves
        the camera -- ``_init_rendering`` disables the pan/zoom controller --
        so skipping it does not hand control to the user, it strands the view.
        Time and amplitude zoom work by changing what the data occupies and
        letting this follow.
        """
        self._subplot.auto_scale(maintain_aspect=False, zoom=1.0)

    def _on_ctrl_scroll(self, delta: float) -> None:
        """Handle Ctrl+scroll for time/freq zoom. Override in subclass."""

    # ------------------------------------------------------------------
    # Animation callback
    # ------------------------------------------------------------------

    def _animation_callback(self) -> None:
        self._update_graphics()
        self._apply_auto_scale()
        self._sync_overlays()

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def set_channel_labels(self, labels: list[str] | None) -> None:
        """Replace the per-channel labels used by the tooltip and the overlay.

        Labels often arrive after the plot is built -- a source announces its
        channel names on its own schedule -- so this exists to avoid callers
        assigning to ``_channel_labels`` and hoping the overlay notices.
        """
        self._channel_labels = list(labels) if labels is not None else None
        self._push_channel_labels()

    @property
    def channel_labels_visible(self) -> bool:
        """Whether per-trace identifiers are currently drawn."""
        return self._label_overlay is not None and self._label_overlay.isVisible()

    def set_channel_labels_visible(self, visible: bool) -> None:
        """Show or hide per-trace identifiers drawn over the canvas.

        Uses the ``channel_labels`` the plot was configured with. Positioning
        and font sizing follow the live camera, so labels stay on their traces
        as the user scrolls, pages, or zooms.

        Worth having a way to turn off: the labels sit on opaque chips, so on a
        dense trace they cover signal, and once the user knows which channel is
        which they mostly want them gone.
        """
        if not visible:
            if self._label_overlay is not None:
                self._label_overlay.hide()
            return
        if self._label_overlay is None:
            self._label_overlay = ChannelLabelOverlay(self._fpl_widget)
            self._label_overlay.setGeometry(self._fpl_widget.rect())
        self._push_channel_labels()
        self._label_overlay.show()
        self._label_overlay.raise_()
        self._sync_overlays()

    def set_scale_bar_text(self, text: str | None) -> None:
        """Show a calibration bar one row-height tall, labelled *text*.

        The caller supplies the text because what a row-height means depends on
        the signal's units, which this widget does not know. ``None`` or an
        empty string hides the bar.
        """
        if not text:
            if self._scale_bar_overlay is not None:
                self._scale_bar_overlay.set_bar(0.0, "")
                self._scale_bar_overlay.hide()
            return
        if self._scale_bar_overlay is None:
            self._scale_bar_overlay = ScaleBarOverlay(self._fpl_widget)
            self._scale_bar_overlay.setGeometry(self._fpl_widget.rect())
        self._scale_bar_text = text
        self._scale_bar_overlay.show()
        self._scale_bar_overlay.raise_()
        self._sync_overlays()

    def _push_channel_labels(self) -> None:
        """Forward the plot's labels to the overlay when they have been replaced.

        Keyed on identity rather than value: update_config swaps the list, and
        comparing 256 strings every frame to notice would cost more than the
        feature does.
        """
        if self._label_overlay is None or self._channel_labels is self._label_source:
            return
        self._label_source = self._channel_labels
        self._label_overlay.set_labels(list(self._channel_labels or []))

    def _screen_y_projection(self) -> tuple[float | None, float | None]:
        """Affine ``screen_y = y0 + slope * world_y`` for the current camera.

        Returns ``(None, None)`` if the camera cannot be probed, which the
        overlay handles by falling back to its analytic layout.
        """
        subplot = getattr(self, "_subplot", None)
        if subplot is None:
            return None, None
        buf = self._buffer
        size = self._fpl_widget.size()
        camera = getattr(subplot, "camera", None)
        key = (
            size.width(),
            size.height(),
            getattr(buf, "n_visible", None),
            getattr(self, "_z_offset_scale", 1.0),
            float(camera.height) if camera is not None else None,
            float(camera.world.position[1]) if camera is not None else None,
        )
        if self._projection is not None and key == self._projection_key:
            return self._projection

        try:
            # x is irrelevant to screen-y for the axis-aligned ortho camera.
            y_at_0 = float(subplot.map_world_to_screen((0.0, 0.0, 0.0))[1])
            y_at_1 = float(subplot.map_world_to_screen((0.0, 1.0, 0.0))[1])
        except Exception:  # noqa: BLE001 - pygfx call with no enumerated failures
            # In a paint path: losing the labels' precise placement is better
            # than losing the frame.
            logger.debug("map_world_to_screen failed; overlays fall back to analytic layout", exc_info=True)
            return None, None

        self._projection = (y_at_0, y_at_1 - y_at_0)
        self._projection_key = key
        return self._projection

    def _sync_overlays(self) -> None:
        """Keep overlays sized to the canvas and in step with the view."""
        if self._label_overlay is None and self._scale_bar_overlay is None:
            return
        rect = self._fpl_widget.rect()
        for overlay in (self._label_overlay, self._scale_bar_overlay):
            if overlay is not None and overlay.size() != self._fpl_widget.size():
                overlay.setGeometry(rect)

        buf = self._buffer
        if buf is None:
            return
        y0, slope = self._screen_y_projection()
        z_scale = getattr(self, "_z_offset_scale", 1.0) or 1.0

        if self._label_overlay is not None:
            self._push_channel_labels()
            self._label_overlay.set_view(
                getattr(buf, "channel_offset", 0),
                getattr(buf, "n_visible", 0),
                self._is_top_down(),
                y0,
                slope,
                z_scale,
            )

        if self._scale_bar_overlay is not None:
            text = self._scale_bar_text
            # A row spans ±0.5 in normalized units, so half a row's world
            # distance is the bar the caller's text describes.
            length = 0.0 if slope is None else 0.5 * abs(slope)
            self._scale_bar_overlay.set_bar(length, text)

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

        self._update_range_label()

    def _on_wheel_event(self, event) -> None:
        delta = event.dy

        if "Control" in getattr(event, "modifiers", ()):
            # Ctrl+scroll → time/freq zoom (subclass hook)
            self._on_ctrl_scroll(delta)
        elif "Shift" in getattr(event, "modifiers", ()):
            # Shift+scroll → amplitude zoom
            self._zoom_amplitude(1.1 if self._shift_wheel_delta(event) > 0 else 0.9)
        else:
            # Unmodified scroll → channel scroll
            step = self._channel_scroll_step(delta)
            if step:
                buf = self._buffer
                buf.set_channel_offset(buf.channel_offset + step)
                self._update_range_label()

    # ------------------------------------------------------------------
    # Row <-> channel mapping
    # ------------------------------------------------------------------
    #
    # A buffer lays its visible rows out along world-y, and "row 0" is the
    # bottom of the canvas because that is what a plain arange of z-offsets
    # produces. ``channel_order="top_down"`` reverses which channel lands on
    # which row, so anything converting between a screen position and a channel
    # has to go through here -- reading it off as ``channel_offset + row``
    # silently inverts the answer whenever top_down is in force, which is the
    # sweep's default.

    def _is_top_down(self) -> bool:
        """Whether the first visible channel is drawn at the top of the canvas.

        A buffer that does not declare an order (the spectrum) offsets its rows
        with a plain arange, so *absent* means bottom-up rather than the sweep's
        default.
        """
        return getattr(self._buffer, "channel_order", "bottom_up") == "top_down"

    def _channel_at_row(self, row: int) -> int:
        """Absolute channel drawn at *row*, counting up from the canvas bottom."""
        if self._is_top_down():
            row = self._buffer.n_visible - 1 - row
        return self._buffer.channel_offset + row

    def _row_of_channel(self, channel: int) -> int:
        """Row a channel is drawn at, counting up from the canvas bottom."""
        row = channel - self._buffer.channel_offset
        if self._is_top_down():
            row = self._buffer.n_visible - 1 - row
        return row

    def _channel_color(self, channel: int) -> tuple[float, float, float]:
        """Palette colour for an absolute channel index, as RGB.

        Keyed to the channel rather than the on-screen row so a trace keeps its
        colour while the window scrolls past it -- the point of a colour here
        is to let the eye follow one channel, which a colour that belongs to
        the row actively defeats.
        """
        return tuple(self._palette[channel % len(self._palette)][:3])

    @staticmethod
    def _shift_wheel_delta(event) -> float:
        """Wheel motion for Shift+scroll, from whichever axis it arrived on.

        Holding Shift makes the OS report a mouse wheel as *horizontal*
        scrolling -- the convention that scrolls a document sideways -- so the
        motion arrives in dx with dy pinned at 0. Reading dy alone then makes
        every notch look negative and amplitude only ever zooms out. A trackpad
        reports both axes natively and is unaffected, which is why this shows
        up on a mouse only.
        """
        dy = getattr(event, "dy", 0.0) or 0.0
        return dy if dy else (getattr(event, "dx", 0.0) or 0.0)

    @staticmethod
    def _channel_scroll_step(delta: float) -> int:
        """Channel-offset change for one wheel notch, 0 for no vertical motion.

        Scrolling down moves the window down the channel list, so the traces
        travel with the fingers the way a document does. Split out from the
        handler so the direction can be tested without a canvas, since it is
        the kind of thing that is obvious in use and invisible in review.

        A horizontal trackpad swipe arrives as a wheel event carrying dx with
        dy at 0, which is why 0 has to mean *stay*: taking it as a direction
        makes sideways scrolling walk the channel window.
        """
        if delta == 0:
            return 0
        return 1 if delta > 0 else -1

    def _on_pointer_move_event(self, event) -> None:
        self._handle_mouse_move(event)

    def set_mouse_enabled(self, enabled: bool) -> None:
        """Enable or disable mouse-driven plot interactions (wheel, hover tooltip).

        Keyboard shortcuts and external controls (ChannelPlotControlsWidget)
        keep working either way.
        """
        if enabled == self._mouse_enabled:
            return
        self._mouse_enabled = enabled
        if enabled:
            self._renderer.add_event_handler(self._on_wheel_event, "wheel")
            self._renderer.add_event_handler(self._on_pointer_move_event, "pointer_move")
        else:
            self._renderer.remove_event_handler(self._on_wheel_event, "wheel")
            self._renderer.remove_event_handler(self._on_pointer_move_event, "pointer_move")

    def set_max_fps(self, fps: float | None) -> None:
        """Cap the canvas render rate.

        A scrolling trace reads smooth well below the display refresh, and
        render cost falls roughly in proportion, so this is the cheapest
        performance knob a caller has. ``None`` leaves whatever rendercanvas
        was already doing; ``fps <= 0`` uncaps (``update_mode="fastest"``).

        Safe to call before or after the canvas exists; a backend that predates
        ``set_update_mode`` is warned about once and otherwise ignored, since
        losing the cap is a performance regression rather than a broken plot.
        """
        if fps is None:
            return
        canvas = getattr(self._figure, "canvas", None)
        set_update_mode = getattr(canvas, "set_update_mode", None)
        if set_update_mode is None:
            logger.warning("Canvas has no set_update_mode; leaving the render rate at its default.")
            return
        try:
            if fps <= 0:
                set_update_mode("fastest")
            else:
                # "continuous" is the mode that honours max_fps; "ondemand"
                # ignores it and redraws only on events, which a live sweep
                # would then never do.
                set_update_mode("continuous", max_fps=float(fps))
        except Exception:
            logger.exception("Failed to set the canvas render rate; leaving it at its default.")

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

        abs_ch = self._channel_at_row(best_idx)

        labels = self._channel_labels
        label = labels[abs_ch] if labels and abs_ch < len(labels) else f"Ch {abs_ch}"

        rgb = self._channel_color(abs_ch)
        hex_color = f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"
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
