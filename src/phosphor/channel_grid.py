"""GPU-accelerated 2D grid of one value per channel.

Draws one square per channel at its real ``(x, y)`` position (the ``(x, y)`` is
the square's **lower-left corner**), sized per channel, and colored by value.
The squares are a single quad mesh, so they live in real world-units: the
camera's starting bounds are the grid's own bounds with aspect preserved, so
squares render square, and a channel reported as smaller than the array pitch
draws as a smaller square with a gap rather than filling its cell.

Colour comes from one of two places. By default a value is normalized over
``vmin``/``vmax`` and run through ``cmap``, which suits a quantity that varies
continuously. A caller that classifies its values instead -- into states with
names and fixed colours, as an impedance reading or a channel-quality flag
would be -- passes the colours it computed to :meth:`ChannelGridWidget.set_colors`
and the words to :meth:`ChannelGridWidget.set_annotations`. What counts as good
is domain vocabulary, so it stays with the domain; this module renders whatever
it is handed.

Falls back to a unit-spaced sequential layout when every channel shares one
position, which is what an unmapped source looks like.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cmap as cmap_lib
import fastplotlib as fpl
import numpy as np
from PySide6 import QtCore, QtWidgets

from .grid_layout import build_quad_mesh_arrays, resolve_cell_geometry

logger = logging.getLogger(__name__)

__all__ = [
    "ChannelGridConfig",
    "ChannelGridWidget",
]

RGBA = tuple[float, float, float, float]


@dataclass
class ChannelGridConfig:
    positions: np.ndarray
    """``(n_ch, 2)`` array of electrode ``(x, y)`` lower-left corners, in the
    same units as ``sizes`` (micrometers for a CMP/device-mapped source)."""

    sizes: np.ndarray | None = None
    """``(n_ch,)`` per-channel square side length. ``None`` (or a non-positive
    entry) falls back to the inferred electrode pitch so squares tile their
    cells, matching the old full-cell heatmap look."""

    channel_labels: list[str] | None = None
    cmap: str = "viridis_r"
    vmin: float = 0.0
    vmax: float = 1000.0
    nan_color: RGBA = (0.15, 0.15, 0.18, 1.0)
    """Colour for a channel with no value. Applies to the colormap path only --
    explicit colours from ``set_colors`` are used verbatim, since a caller that
    computes its own colours is the one that knows what missing should look
    like."""

    show_values: bool = False
    value_format: str = "{:.0f}"
    text_face_color: str = "white"
    text_outline_color: str = "black"
    text_outline_thickness: float = 0.4
    text_font_size: float = 13.0
    value_unit: str = ""
    invert_y: bool = True
    """Flip the layout so low-``y`` positions render at the TOP of the screen.
    Matches the "row 0 is the first row I read" expectation most people have of
    a grid, and the row-0-at-bottom convention of electrode maps. ``False``
    keeps low-``y`` at the bottom, the native y-up world."""


class ChannelGridWidget(QtWidgets.QWidget):
    """Embeddable grid: one square per channel at its electrode position,
    color-mapped by value, with optional value text and a channel tooltip."""

    def __init__(self, config: ChannelGridConfig, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        n_ch = config.positions.shape[0]

        # Per-channel display rectangles: (x0, y0, side) with (x0, y0) the
        # lower-left corner in display space (already y-flipped if requested),
        # plus square centers for value text and tooltip hit-testing.
        self._rects, self._centers = self._resolve_geometry(config, n_ch)

        self._values_per_ch = np.full(n_ch, np.nan, dtype=np.float32)
        self._dirty = True
        self._needs_autoscale = True
        self._show_values = bool(config.show_values)
        self._cmap = cmap_lib.Colormap(config.cmap)
        self._nan_rgba = self._rgba_f(config.nan_color)
        # Caller-supplied colours and per-channel words. None means "use the
        # colormap" and "say nothing beyond the number".
        self._explicit_rgba: np.ndarray | None = None
        self._annotations: list[str] | None = None

        qt_layout = QtWidgets.QVBoxLayout(self)
        qt_layout.setContentsMargins(0, 0, 0, 0)

        self._figure = fpl.Figure()
        self._subplot = self._figure[0, 0]
        self._fpl_widget = self._figure.show()
        qt_layout.addWidget(self._fpl_widget)

        positions, indices = self._build_mesh_arrays()
        init_colors = np.tile(self._nan_rgba, (positions.shape[0], 1))
        self._mesh = self._subplot.add_mesh(positions, indices, mode="basic", colors=init_colors)

        self._text_graphics: list = []
        if self._show_values:
            self._create_text_graphics()

        self._tooltip = QtWidgets.QLabel(self._fpl_widget)
        self._tooltip.setStyleSheet(
            "background: rgba(25,25,30,220); color: #e8e8e8;"
            " padding: 12px 24px; font-size: 36pt;"
            " font-family: 'Menlo','Consolas','DejaVu Sans Mono',monospace;"
            " border: 1px solid rgba(120,120,120,160);"
        )
        self._tooltip.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._tooltip.hide()

        self._subplot.controller = None
        self._subplot.axes.visible = False
        self._subplot.title.visible = False
        # Disable fastplotlib's built-in hover tooltip — it shows raw geometry
        # data, meaningless for our value-coded squares; we provide our own
        # channel-aware tooltip via _on_pointer_move.
        try:
            self._subplot.tooltip.enabled = False
        except AttributeError:
            pass
        self._figure.add_animations(lambda: self._animation_callback())

        renderer = self._subplot.renderer
        renderer.add_event_handler(self._on_pointer_move, "pointer_move")
        renderer.add_event_handler(self._on_pointer_leave, "pointer_leave")

    # ---- Public API ----------------------------------------------------

    def push_data(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float32).reshape(-1)
        if v.size != self._values_per_ch.size:
            logger.warning("push_data size %d != n_channels %d; ignoring.", v.size, self._values_per_ch.size)
            return
        self._values_per_ch = v
        self._dirty = True

    def set_show_values(self, show: bool) -> None:
        if show == self._show_values:
            return
        self._show_values = show
        if show and not self._text_graphics:
            self._create_text_graphics()
        for _, tg in self._text_graphics:
            tg.visible = show
        self._dirty = True

    def set_colors(self, rgba: np.ndarray | None) -> None:
        """Colour each channel explicitly, overriding the colormap.

        For values that mean something categorical rather than continuous -- a
        pass/fail, a quality state -- where the mapping is the caller's to
        decide. ``None`` hands colouring back to the colormap.
        """
        if rgba is None:
            self._explicit_rgba = None
            self._dirty = True
            return
        arr = np.asarray(rgba, dtype=np.float32).reshape(-1, 4)
        if arr.shape[0] != self._values_per_ch.size:
            logger.warning("set_colors got %d rows for %d channels; ignoring.", arr.shape[0], self._values_per_ch.size)
            return
        self._explicit_rgba = arr
        self._dirty = True

    def set_annotations(self, annotations: list[str] | None) -> None:
        """Per-channel words for the tooltip, after the value.

        The counterpart to :meth:`set_colors`: a caller that colours a channel
        by a state it named should be able to show that name on hover, without
        this module having to know the vocabulary.
        """
        if annotations is not None and len(annotations) != self._values_per_ch.size:
            logger.warning(
                "set_annotations got %d entries for %d channels; ignoring.",
                len(annotations),
                self._values_per_ch.size,
            )
            return
        self._annotations = list(annotations) if annotations is not None else None
        self._dirty = True

    def set_color_range(self, vmin: float, vmax: float) -> None:
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        if vmin == self._config.vmin and vmax == self._config.vmax:
            return
        self._config.vmin = float(vmin)
        self._config.vmax = float(vmax)
        self._dirty = True

    def clear_values(self) -> None:
        """Reset all squares to NaN. Use before changing acquisition source so
        stale values don't linger until the new source overwrites them."""
        self._values_per_ch[:] = np.nan
        self._dirty = True

    def close_figure(self) -> None:
        """Tear down the fastplotlib figure deterministically.

        Without this, rendercanvas's deferred Qt callbacks can fire after
        the C++ widget has been deleted, raising a shiboken RuntimeError
        at app exit (or whenever this widget is replaced).
        """
        figure = getattr(self, "_figure", None)
        if figure is None:
            return
        try:
            figure.close()
        except Exception:
            logger.exception("fastplotlib figure close raised; ignoring")
        self._figure = None

    @property
    def show_values(self) -> bool:
        return self._show_values

    @property
    def color_range(self) -> tuple[float, float]:
        return self._config.vmin, self._config.vmax

    # ---- Geometry ------------------------------------------------------

    @staticmethod
    def _resolve_geometry(config: ChannelGridConfig, n_ch: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel display rects ``(x0, y0, side)`` and centers ``(cx, cy)``.

        Thin wrapper over the shared :func:`resolve_cell_geometry`.
        """
        return resolve_cell_geometry(config.positions, config.sizes, n_ch, config.invert_y)

    def _build_mesh_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Vertex positions ``(n*4, 3)`` and triangle indices ``(n*2, 3)`` for
        the per-channel quads."""
        return build_quad_mesh_arrays(self._rects)

    # ---- Internals -----------------------------------------------------

    @staticmethod
    def _rgba_f(rgba: RGBA) -> np.ndarray:
        return np.asarray(rgba, dtype=np.float32)

    def _create_text_graphics(self) -> None:
        for ch_idx in range(self._values_per_ch.size):
            cx, cy = self._centers[ch_idx]
            tg = self._subplot.add_text(
                "",
                font_size=self._config.text_font_size,
                face_color=self._config.text_face_color,
                outline_color=self._config.text_outline_color,
                outline_thickness=self._config.text_outline_thickness,
                # z above the mesh (z=0) so the text isn't occluded.
                offset=(float(cx), float(cy), 1.0),
                anchor="middle-center",
            )
            tg.visible = self._show_values
            self._text_graphics.append((ch_idx, tg))

    def _animation_callback(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        rgba = self._channel_rgba()  # (n_ch, 4) float
        # One color per quad → repeat across the quad's 4 vertices.
        self._mesh.colors[:] = np.repeat(rgba, 4, axis=0)

        if self._show_values:
            fmt = self._config.value_format
            for ch_idx, tg in self._text_graphics:
                v = self._values_per_ch[ch_idx]
                tg.text = "" if np.isnan(v) else fmt.format(float(v))

        if self._needs_autoscale:
            self._needs_autoscale = False
            # Fit the camera to the electrode grid's own bounds, equal scale on
            # both axes so the squares render square (rectilinear aspect).
            self._subplot.auto_scale(maintain_aspect=True, zoom=0.95)

    def _channel_rgba(self) -> np.ndarray:
        if self._explicit_rgba is not None:
            return self._explicit_rgba
        return self._render_continuous()

    def _render_continuous(self) -> np.ndarray:
        v = self._values_per_ch
        cmap_span = max(self._config.vmax - self._config.vmin, 1e-12)
        norm = (v - self._config.vmin) / cmap_span
        norm_safe = np.nan_to_num(np.clip(norm, 0.0, 1.0), nan=0.0)
        rgba = self._cmap(norm_safe).astype(np.float32)
        rgba[np.isnan(v)] = self._nan_rgba
        return rgba

    def _channel_at(self, wx: float, wy: float) -> int:
        """Index of the square containing world point ``(wx, wy)``, or -1."""
        x0 = self._rects[:, 0]
        y0 = self._rects[:, 1]
        side = self._rects[:, 2]
        inside = (wx >= x0) & (wx <= x0 + side) & (wy >= y0) & (wy <= y0 + side)
        hits = np.flatnonzero(inside)
        return int(hits[0]) if hits.size else -1

    def _on_pointer_move(self, event) -> None:
        world = self._subplot.map_screen_to_world(event)
        if world is None:
            self._tooltip.hide()
            return
        ch_idx = self._channel_at(float(world[0]), float(world[1]))
        if ch_idx < 0:
            self._tooltip.hide()
            return

        self._tooltip.setText(self._tooltip_text(ch_idx))
        self._tooltip.adjustSize()
        x = int(getattr(event, "x", 0)) + 14
        y = int(getattr(event, "y", 0)) + 14
        max_x = self._fpl_widget.width() - self._tooltip.width() - 4
        max_y = self._fpl_widget.height() - self._tooltip.height() - 4
        self._tooltip.move(min(x, max_x), min(y, max_y))
        self._tooltip.show()

    def _tooltip_text(self, ch_idx: int) -> str:
        """What hovering a channel says: its name, its value, and whatever the
        caller called it."""
        labels = self._config.channel_labels
        label = labels[ch_idx] if labels and ch_idx < len(labels) else f"ch{ch_idx}"
        note = self._annotations[ch_idx] if self._annotations else ""
        value = float(self._values_per_ch[ch_idx])
        if np.isnan(value):
            return f"{label}\n{note}" if note else f"{label}\nno value"
        unit = f" {self._config.value_unit}" if self._config.value_unit else ""
        return f"{label}\n{value:.1f}{unit}  ({note})" if note else f"{label}\n{value:.1f}{unit}"

    def _on_pointer_leave(self, _event) -> None:
        self._tooltip.hide()
