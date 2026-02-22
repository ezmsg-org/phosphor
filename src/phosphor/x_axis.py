"""Generalized x-axis label widget for sweep and spectrum plots."""

from __future__ import annotations

import math

from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from .constants import BG_COLOR

__all__ = ["XAxisWidget"]


def _nice_125(v: float) -> float:
    """Return the smallest 1-2-5 multiple >= *v*."""
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10**exp
    for m in (1.0, 2.0, 5.0, 10.0):
        candidate = m * base
        if candidate >= v * 0.999:
            return candidate
    return 10.0 * base  # pragma: no cover


class XAxisWidget(QWidget):
    """Lightweight x-axis labels drawn with QPainter below the render canvas."""

    def __init__(self, range_max: float, unit: str = "s", parent: QWidget | None = None):
        super().__init__(parent)
        self._range_max = range_max
        self._range_min: float = 0.0
        self._unit = unit
        self._log = False
        self.setFixedHeight(24)
        bg = BG_COLOR
        self.setStyleSheet(f"background-color: rgb({int(bg[0]*255)},{int(bg[1]*255)},{int(bg[2]*255)});")

    def set_range(self, range_max: float) -> None:
        self._range_max = range_max
        self.update()

    def set_log(self, enabled: bool, range_min: float | None = None) -> None:
        self._log = enabled
        if range_min is not None:
            self._range_min = range_min
        self.update()

    def paintEvent(self, event) -> None:
        w = self.width()
        if w < 1 or self._range_max <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font = QFont()
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        painter.setFont(font)
        pen_color = QColor(180, 180, 180)
        painter.setPen(pen_color)
        fm = painter.fontMetrics()

        if self._log and self._range_min > 0:
            self._paint_log(painter, fm, w)
        else:
            self._paint_linear(painter, fm, w)

        painter.end()

    # ------------------------------------------------------------------
    # Linear ticks
    # ------------------------------------------------------------------

    def _paint_linear(self, painter: QPainter, fm, w: int) -> None:
        interval = _nice_125(self._range_max / 8)

        if interval >= 1.0 and interval == int(interval):
            decimals = 0
        else:
            decimals = max(1, -int(math.floor(math.log10(interval))))

        t = 0.0
        tick_values: list[float] = []
        while t <= self._range_max + interval * 0.01:
            tick_values.append(t)
            t += interval

        for i, tv in enumerate(tick_values):
            x = int(tv / self._range_max * w) if self._range_max > 0 else 0
            painter.drawLine(x, 0, x, 4)
            if decimals == 0:
                label = f"{int(tv)}"
            else:
                label = f"{tv:.{decimals}f}"
            if i == len(tick_values) - 1:
                label += self._unit
            tw = fm.horizontalAdvance(label)
            lx = max(0, min(x - tw // 2, w - tw))
            painter.drawText(lx, 18, label)

    # ------------------------------------------------------------------
    # Logarithmic ticks
    # ------------------------------------------------------------------

    def _paint_log(self, painter: QPainter, fm, w: int) -> None:
        f_min = self._range_min
        f_max = self._range_max
        log_span = math.log10(f_max / f_min)

        # Generate 1-2-5 ticks across the log range
        tick_values: list[float] = []
        exp_lo = math.floor(math.log10(f_min))
        exp_hi = math.ceil(math.log10(f_max))
        for e in range(int(exp_lo), int(exp_hi) + 1):
            base = 10.0**e
            for m in (1.0, 2.0, 5.0):
                v = m * base
                if f_min * 0.999 <= v <= f_max * 1.001:
                    tick_values.append(v)

        for i, tv in enumerate(tick_values):
            frac = math.log10(tv / f_min) / log_span
            x = int(frac * w)
            painter.drawLine(x, 0, x, 4)
            label = self._format_freq(tv)
            if i == len(tick_values) - 1:
                label += self._unit
            tw = fm.horizontalAdvance(label)
            lx = max(0, min(x - tw // 2, w - tw))
            painter.drawText(lx, 18, label)

    @staticmethod
    def _format_freq(v: float) -> str:
        if v >= 1.0 and v == int(v):
            return f"{int(v)}"
        if v >= 1.0:
            return f"{v:.1f}"
        return f"{v:.2g}"
