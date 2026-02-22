"""Lightweight time-axis label widget for sweep plots."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from .constants import BG_COLOR

__all__ = ["TimeAxisWidget"]

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
