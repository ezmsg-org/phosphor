"""CPU-painted 2D grid of one value per channel, with a colour bar.

The QPainter counterpart of :class:`~phosphor.channel_grid.ChannelGridWidget`:
same :class:`~phosphor.channel_grid.ChannelGridConfig`, same value/colour
methods, but flat cells painted on the CPU rather than a quad mesh on a wgpu
canvas. Use it when the process already runs a live GPU canvas -- a second
canvas in the same process can glitch, render black when its dock is floated,
and hang on close -- or wherever a GPU context is not worth having for a few
hundred flat rectangles.

On top of the GPU grid it adds a colour bar down the right edge, labelled with
``vmin``/``vmax`` and ``value_unit``, and a pannable/zoomable world-space view
(left-drag, wheel, double-click to refit) that :meth:`ChannelHeatmapWidget.set_viewport`
can link to another view such as :class:`~phosphor.trace_grid.TraceGridWidget`.
"""

from __future__ import annotations

import logging

import cmap as cmap_lib
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .channel_grid import ChannelGridConfig
from .grid_layout import resolve_cell_geometry

logger = logging.getLogger(__name__)

__all__ = ["ChannelHeatmapWidget"]

_BG_COLOR = QtGui.QColor(20, 20, 24)
_BORDER_COLOR = QtGui.QColor(48, 48, 56)
_MARGIN = 8
_TEXT_MIN_CELL_PX = 30.0  # only draw a cell's value when the cell is at least this wide
# The value scales with its cell so a 128-channel grid stays readable, within
# bounds: below the floor the digits turn to mush, and above the ceiling a
# sparse grid reads as a wall of numbers rather than a heatmap.
_TEXT_MIN_PT = 10.0
_TEXT_MAX_PT = 16.0
# "-12.3" is five glyphs, and a proportional digit runs about 0.55 em wide, so
# this is what keeps the widest value inside its cell.
_TEXT_CELL_FRACTION = 0.30
# The colour bar: a strip down the right edge, labelled at both ends.
_LEGEND_GAP = 10  # between the cells' area and the bar
_LEGEND_BAR_WIDTH = 12
_LEGEND_TEXT_GAP = 4
_LEGEND_STOPS = 32
_LEGEND_TEXT_COLOR = QtGui.QColor(200, 200, 210)


class ChannelHeatmapWidget(QtWidgets.QWidget):
    """Flat per-channel colour grid with a colour bar, painted with QPainter.

    Draws one rectangle per channel at its ``(x, y)`` position (lower-left
    corner), coloured by its value through ``config.cmap`` over
    ``config.vmin``/``config.vmax``, with ``config.outlines`` on top. The
    visible region is a world-space rect fit to the widget with aspect
    preserved; it defaults to the cells' bounds. Hovering a cell shows its
    label and value. With ``config.show_values`` each cell wide enough to hold
    it shows its value in ``config.value_format``, in ``text_face_color`` with a
    1 px ``text_outline_color`` halo; the size follows the cells, so
    ``text_font_size`` and ``text_outline_thickness`` do not apply here.
    """

    # Emitted on user pan/zoom with the new visible world rect
    # (xmin, xmax, ymin, ymax); mirrors TraceGridWidget.viewport_changed.
    viewport_changed = QtCore.Signal(object)

    def __init__(self, config: ChannelGridConfig, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._n_ch = config.positions.shape[0]
        self._show_values = bool(config.show_values)
        self._cmap = cmap_lib.Colormap(config.cmap)
        self._nan_color = QtGui.QColor.fromRgbF(*config.nan_color)
        self._values = np.full(self._n_ch, np.nan, dtype=np.float32)

        # World-space cell rects (x0, y0, width, height) with y up, flipped
        # like ChannelGridWidget's when config.invert_y is set.
        self._rects, _ = resolve_cell_geometry(config.positions, config.sizes, self._n_ch, config.invert_y)
        y_sign = -1.0 if config.invert_y else 1.0
        self._outlines = [
            (np.asarray(o.segments, dtype=np.float64).reshape(-1, 2, 2) * (1.0, y_sign), o)
            for o in config.outlines or ()
        ]
        x0, y0, width, height = self._rects.T
        self._wx0, self._wy0 = float(x0.min()), float(y0.min())
        self._wx1, self._wy1 = float((x0 + width).max()), float((y0 + height).max())

        # Visible world rect (xmin, xmax, ymin, ymax); None → full bounds.
        self._view_rect: tuple[float, float, float, float] | None = None
        # Last cursor pos during a left-button pan (widget coords), or None.
        self._pan_last: QtCore.QPointF | None = None
        # Cached transform from the most recent paint, for mouse↔world mapping:
        # (scale, ox, oy, vx0, vy1).
        self._xform: tuple[float, float, float, float, float] | None = None

        # Screen rects for the most recent paint, used for hover hit-testing.
        self._screen_rects: list[tuple[int, QtCore.QRectF]] = []
        # Where the colour bar was last painted.
        self._legend_bar = QtCore.QRectF()

        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    # ---- Public API ----------------------------------------------------

    def push_data(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float32).reshape(-1)
        if v.size != self._n_ch:
            logger.warning("push_data size %d != n_channels %d; ignoring.", v.size, self._n_ch)
            return
        self._values = v
        self.update()

    def set_color_range(self, vmin: float, vmax: float) -> None:
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        self._config.vmin, self._config.vmax = float(vmin), float(vmax)
        self.update()

    def set_show_values(self, show: bool) -> None:
        self._show_values = bool(show)
        self.update()

    def clear_values(self) -> None:
        """Reset all cells to NaN, e.g. before changing source."""
        self._values[:] = np.nan
        self.update()

    @property
    def show_values(self) -> bool:
        return self._show_values

    @property
    def color_range(self) -> tuple[float, float]:
        return self._config.vmin, self._config.vmax

    # ---- Pan/zoom viewport ---------------------------------------------

    def _default_view(self) -> tuple[float, float, float, float]:
        return (self._wx0, self._wx1, self._wy0, self._wy1)

    def _view(self) -> tuple[float, float, float, float]:
        return self._view_rect if self._view_rect is not None else self._default_view()

    def current_viewport(self) -> tuple[float, float, float, float]:
        return self._view()

    def set_viewport(self, rect: tuple[float, float, float, float]) -> None:
        """Show *rect* (world coords) to match a linked view. No signal echo."""
        xmin, xmax, ymin, ymax = rect
        self._view_rect = (min(xmin, xmax), max(xmin, xmax), min(ymin, ymax), max(ymin, ymax))
        self.update()

    def reset_view(self) -> None:
        """Refit to all cells (double-click) and broadcast the new view."""
        self._view_rect = None
        self.update()
        self.viewport_changed.emit(self._default_view())

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self._pan_last = None  # cancel any in-progress pan
        self.reset_view()

    # ---- Painting ------------------------------------------------------

    def _format(self, value: float) -> str:
        return self._config.value_format.format(float(value))

    def _colors(self) -> list[QtGui.QColor]:
        vmin, vmax = self.color_range
        span = max(vmax - vmin, 1e-12)
        norm = np.clip((self._values - vmin) / span, 0.0, 1.0)
        rgba = self._cmap(np.nan_to_num(norm, nan=0.0))
        out: list[QtGui.QColor] = []
        for i in range(self._n_ch):
            if np.isnan(self._values[i]):
                out.append(self._nan_color)
            else:
                r, g, b, _a = rgba[i]
                out.append(QtGui.QColor.fromRgbF(float(r), float(g), float(b), 1.0))
        return out

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), _BG_COLOR)

        # The colour bar takes a strip down the right edge; the cells get the rest.
        legend_w = self._legend_width(painter.fontMetrics())
        vx0, vx1, vy0, vy1 = self._view()
        world_w = max(vx1 - vx0, 1e-9)
        world_h = max(vy1 - vy0, 1e-9)
        avail_w = max(1.0, self.width() - 2 * _MARGIN - legend_w)
        avail_h = max(1.0, self.height() - 2 * _MARGIN)
        scale = min(avail_w / world_w, avail_h / world_h)
        draw_w = world_w * scale
        draw_h = world_h * scale
        ox = _MARGIN + (avail_w - draw_w) / 2.0
        oy = _MARGIN + (avail_h - draw_h) / 2.0
        # Cache the transform so mouse handlers map widget↔world identically.
        self._xform = (scale, ox, oy, vx0, vy1)

        colors = self._colors()
        self._screen_rects = []
        painter.setPen(QtGui.QPen(_BORDER_COLOR, 1))
        font = painter.font()
        font.setPointSizeF(self._text_point_size(scale))
        painter.setFont(font)

        for i in range(self._n_ch):
            x0, y0, width, height = self._rects[i]
            sx = ox + (x0 - vx0) * scale
            # Flip y: larger world-y maps to a smaller screen-y (top edge).
            sy = oy + (vy1 - (y0 + height)) * scale
            rect = QtCore.QRectF(sx, sy, width * scale, height * scale)
            painter.fillRect(rect, colors[i])
            painter.drawRect(rect)
            self._screen_rects.append((i, rect))

            v = self._values[i]
            if self._show_values and min(width, height) * scale >= _TEXT_MIN_CELL_PX and np.isfinite(v):
                self._draw_outlined_text(painter, rect, self._format(v))
                painter.setPen(QtGui.QPen(_BORDER_COLOR, 1))

        # Outlines, in the same world space as the cells.
        for segments, outline in self._outlines:
            painter.setPen(QtGui.QPen(QtGui.QColor.fromRgbF(*outline.color), outline.thickness))
            for (ax, ay), (bx, by) in segments:
                painter.drawLine(
                    QtCore.QPointF(ox + (ax - vx0) * scale, oy + (vy1 - ay) * scale),
                    QtCore.QPointF(ox + (bx - vx0) * scale, oy + (vy1 - by) * scale),
                )

        painter.setFont(self.font())
        self._paint_legend(painter, legend_w)
        painter.end()

    # ---- Colour bar ------------------------------------------------------

    def _legend_labels(self) -> tuple[str, str]:
        """The top (max) and bottom (min) labels of the colour bar."""
        unit = f" {self._config.value_unit}" if self._config.value_unit else ""
        vmin, vmax = self.color_range
        return f"{vmax:g}{unit}", f"{vmin:g}{unit}"

    def _legend_width(self, metrics: QtGui.QFontMetrics) -> float:
        """Width the colour bar and its labels take at the right edge."""
        text_w = max(metrics.horizontalAdvance(label) for label in self._legend_labels())
        return _LEGEND_GAP + _LEGEND_BAR_WIDTH + _LEGEND_TEXT_GAP + text_w

    def _paint_legend(self, painter: QtGui.QPainter, legend_w: float) -> None:
        """The colormap as a bar from ``vmin`` (bottom) to ``vmax`` (top), labelled at both ends.

        The bar runs the widget's height, less room for the end labels, rather
        than the cells' drawn height: the cells move with the pan and zoom and
        the scale should not.
        """
        metrics = painter.fontMetrics()
        half_text = metrics.height() / 2.0
        x = self.width() - _MARGIN - legend_w + _LEGEND_GAP
        top = _MARGIN + half_text
        bottom = self.height() - _MARGIN - half_text
        bar = QtCore.QRectF(x, top, _LEGEND_BAR_WIDTH, max(4.0, bottom - top))
        self._legend_bar = bar

        gradient = QtGui.QLinearGradient(bar.bottomLeft(), bar.topLeft())
        fractions = np.linspace(0.0, 1.0, _LEGEND_STOPS)
        for frac, (r, g, b, _a) in zip(fractions, self._cmap(fractions), strict=True):
            gradient.setColorAt(float(frac), QtGui.QColor.fromRgbF(float(r), float(g), float(b), 1.0))
        painter.fillRect(bar, gradient)
        painter.setPen(QtGui.QPen(_BORDER_COLOR, 1))
        painter.drawRect(bar)

        painter.setPen(_LEGEND_TEXT_COLOR)
        top_label, bottom_label = self._legend_labels()
        text_x = bar.right() + _LEGEND_TEXT_GAP
        text_w = legend_w  # more than enough: the labels sized it
        align = QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
        painter.drawText(QtCore.QRectF(text_x, bar.top() - half_text, text_w, 2 * half_text), align, top_label)
        painter.drawText(QtCore.QRectF(text_x, bar.bottom() - half_text, text_w, 2 * half_text), align, bottom_label)

    def _text_point_size(self, scale: float) -> float:
        """Point size for the cell values, from the smallest cell on screen.

        Sized off the smallest so the tightest cell still fits its number,
        rather than the average cell fitting and a few overflowing.
        """
        if not self._n_ch:
            return _TEXT_MIN_PT
        smallest_cell_px = float(np.min(self._rects[:, 2:4])) * scale
        return max(_TEXT_MIN_PT, min(_TEXT_MAX_PT, smallest_cell_px * _TEXT_CELL_FRACTION))

    def _draw_outlined_text(self, painter: QtGui.QPainter, rect: QtCore.QRectF, text: str) -> None:
        """Text with a 1 px halo, so it reads on any cell colour.

        Drawn as eight halo copies offset around centre then one face copy on
        top -- cheaper and crisper than a stroked QPainterPath at this small
        point size.
        """
        align = QtCore.Qt.AlignmentFlag.AlignCenter
        painter.setPen(QtGui.QColor(self._config.text_outline_color))
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            painter.drawText(rect.translated(dx, dy), align, text)
        painter.setPen(QtGui.QColor(self._config.text_face_color))
        painter.drawText(rect, align, text)

    # ---- Pan / zoom interaction ----------------------------------------

    def _screen_to_world(self, pt: QtCore.QPointF) -> tuple[float, float] | None:
        if self._xform is None:
            return None
        scale, ox, oy, vx0, vy1 = self._xform
        wx = vx0 + (pt.x() - ox) / scale
        wy = vy1 - (pt.y() - oy) / scale  # screen-y is flipped vs world-y
        return wx, wy

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pan_last = event.position()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pan_last = None

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        anchor = self._screen_to_world(event.position())
        if anchor is None:
            return
        # Wheel up (positive) zooms in → the view rect shrinks about the
        # cursor, so the cell under the cursor stays put.
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            return
        factor = 0.85**steps
        wx, wy = anchor
        vx0, vx1, vy0, vy1 = self._view()
        self._view_rect = (
            wx + (vx0 - wx) * factor,
            wx + (vx1 - wx) * factor,
            wy + (vy0 - wy) * factor,
            wy + (vy1 - wy) * factor,
        )
        self.update()
        self.viewport_changed.emit(self._view_rect)
        event.accept()

    # ---- Hover tooltip -------------------------------------------------

    def _tooltip_text(self, ch_idx: int) -> str:
        labels = self._config.channel_labels
        label = labels[ch_idx] if labels and ch_idx < len(labels) else f"ch{ch_idx}"
        v = self._values[ch_idx]
        if not np.isfinite(v):
            return f"{label}: n/a"
        unit = f" {self._config.value_unit}" if self._config.value_unit else ""
        return f"{label}: {self._format(v)}{unit}"

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._pan_last is not None and self._xform is not None:
            scale = self._xform[0]
            pos = event.position()
            dx = (pos.x() - self._pan_last.x()) / scale
            dy = (pos.y() - self._pan_last.y()) / scale
            self._pan_last = pos
            vx0, vx1, vy0, vy1 = self._view()
            # Drag-right moves content right → view shifts left; screen-y flip
            # means drag-down (dy>0) shifts the world view up.
            self._view_rect = (vx0 - dx, vx1 - dx, vy0 + dy, vy1 + dy)
            self.update()
            self.viewport_changed.emit(self._view_rect)
            return

        pos = event.position()
        for i, rect in self._screen_rects:
            if rect.contains(pos):
                QtWidgets.QToolTip.showText(event.globalPosition().toPoint(), self._tooltip_text(i), self)
                return
        QtWidgets.QToolTip.hideText()
