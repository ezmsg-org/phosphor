"""Optional Qt controls panel for ChannelPlotWidget.

Mirrors the keyboard shortcuts defined on ``ChannelPlotWidget`` (and
``SweepWidget``) as on-screen buttons + spinboxes, for users who'd rather
click than memorize keys.

Usage::

    plot = SweepWidget(SweepConfig(...))
    controls = ChannelPlotControlsWidget(plot)
    layout.addWidget(plot)
    layout.addWidget(controls)
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from .channel_plot import ChannelPlotWidget

__all__ = ["ChannelPlotControlsWidget"]


class ChannelPlotControlsWidget(QtWidgets.QWidget):
    """Compact horizontal toolbar of controls for a ChannelPlotWidget."""

    def __init__(
        self,
        plot: ChannelPlotWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._plot = plot
        buf = plot._buffer
        if buf is None:
            raise ValueError("ChannelPlotWidget must have a buffer before adding controls")

        # Stay tight vertically — this is meant to be a slim toolbar, not a
        # panel. Fix the height to the bare minimum.
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "ChannelPlotControlsWidget { font-size: 9pt; }"
            "ChannelPlotControlsWidget QToolButton {"
            "  padding: 0px 4px; min-height: 18px; max-height: 18px;"
            "  min-width: 18px; font-size: 9pt;"
            "}"
            "ChannelPlotControlsWidget QSpinBox {"
            "  padding: 0px 2px; min-height: 18px; max-height: 18px; font-size: 9pt;"
            "}"
            "ChannelPlotControlsWidget QLabel { font-size: 9pt; color: #9a9aa0; }"
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(3)

        # Channel scrolling: up/down by one, and page up/down by n_visible.
        layout.addWidget(self._make_label("Channel"))
        self._btn_ch_up = self._make_button("\u2191", "Scroll up one channel", self._on_ch_up)
        self._btn_ch_down = self._make_button("\u2193", "Scroll down one channel", self._on_ch_down)
        self._btn_pg_up = self._make_button("\u21e1", "Page up (n_visible)", self._on_page_up)
        self._btn_pg_down = self._make_button("\u21e3", "Page down (n_visible)", self._on_page_down)
        layout.addWidget(self._btn_ch_up)
        layout.addWidget(self._btn_ch_down)
        layout.addWidget(self._btn_pg_up)
        layout.addWidget(self._btn_pg_down)

        layout.addWidget(self._make_separator())

        # Number visible: spinbox plus halve/double shortcuts.
        layout.addWidget(self._make_label("Visible"))
        self._spin_visible = QtWidgets.QSpinBox()
        self._spin_visible.setRange(1, buf.n_channels)
        self._spin_visible.setValue(buf.n_visible)
        self._spin_visible.setKeyboardTracking(False)
        self._spin_visible.setFixedWidth(56)
        self._spin_visible.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self._spin_visible.editingFinished.connect(self._on_visible_committed)
        layout.addWidget(self._spin_visible)
        layout.addWidget(self._make_button("/2", "Halve visible channels", self._on_visible_halve))
        layout.addWidget(self._make_button("x2", "Double visible channels", self._on_visible_double))

        layout.addWidget(self._make_separator())

        # Amplitude zoom (per-row waveform scale).
        layout.addWidget(self._make_label("Amplitude"))
        layout.addWidget(self._make_button("\u2212", "Shrink amplitude", lambda: self._plot._zoom_amplitude(0.8)))
        layout.addWidget(self._make_button("+", "Grow amplitude", lambda: self._plot._zoom_amplitude(1.25)))

        # Time zoom — only present if the plot supports it (sweep / spectrum).
        if hasattr(plot, "_time_zoom"):
            layout.addWidget(self._make_separator())
            layout.addWidget(self._make_label("Time"))
            layout.addWidget(self._make_button("\u2212", "Zoom time out (longer span)", lambda: plot._time_zoom(2.0)))
            layout.addWidget(self._make_button("+", "Zoom time in (shorter span)", lambda: plot._time_zoom(0.5)))

        layout.addWidget(self._make_separator())

        # Autoscale toggle.
        self._btn_auto = QtWidgets.QToolButton()
        self._btn_auto.setText("Auto")
        self._btn_auto.setCheckable(True)
        self._btn_auto.setChecked(plot._autoscale_enabled)
        self._btn_auto.setToolTip("Toggle camera autoscale (key: A)")
        self._btn_auto.toggled.connect(self._on_auto_toggled)
        layout.addWidget(self._btn_auto)

        layout.addStretch(1)

        # Periodically resync widget state with the buffer (other inputs —
        # keyboard, mouse — also mutate it). 200 ms is plenty for a panel.
        self._sync_timer = QtCore.QTimer(self)
        self._sync_timer.setInterval(200)
        self._sync_timer.timeout.connect(self._sync_from_buffer)
        self._sync_timer.start()

    # ---- helpers ------------------------------------------------------

    def _make_button(self, text: str, tip: str, slot) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        return b

    def _make_label(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("color: #9a9aa0; font-size: 9pt;")
        return lbl

    def _make_separator(self) -> QtWidgets.QFrame:
        f = QtWidgets.QFrame()
        f.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        f.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return f

    # ---- channel scroll ----------------------------------------------

    def _on_ch_up(self) -> None:
        buf = self._plot._buffer
        buf.set_channel_offset(buf.channel_offset - 1)
        self._plot._update_range_label()

    def _on_ch_down(self) -> None:
        buf = self._plot._buffer
        buf.set_channel_offset(buf.channel_offset + 1)
        self._plot._update_range_label()

    def _on_page_up(self) -> None:
        buf = self._plot._buffer
        buf.set_channel_offset(buf.channel_offset - buf.n_visible)
        self._plot._update_range_label()

    def _on_page_down(self) -> None:
        buf = self._plot._buffer
        buf.set_channel_offset(buf.channel_offset + buf.n_visible)
        self._plot._update_range_label()

    # ---- n_visible ----------------------------------------------------

    def _on_visible_committed(self) -> None:
        buf = self._plot._buffer
        buf.set_n_visible(int(self._spin_visible.value()))
        self._plot._update_range_label()

    def _on_visible_halve(self) -> None:
        buf = self._plot._buffer
        buf.set_n_visible(max(1, buf.n_visible // 2))
        self._plot._update_range_label()

    def _on_visible_double(self) -> None:
        buf = self._plot._buffer
        buf.set_n_visible(min(buf.n_channels, buf.n_visible * 2))
        self._plot._update_range_label()

    # ---- autoscale ----------------------------------------------------

    def _on_auto_toggled(self, on: bool) -> None:
        self._plot._autoscale_enabled = on
        if on:
            self._plot._apply_auto_scale()

    # ---- periodic resync ---------------------------------------------

    def _sync_from_buffer(self) -> None:
        buf = self._plot._buffer
        if buf is None:
            return
        if self._spin_visible.value() != buf.n_visible:
            self._spin_visible.blockSignals(True)
            self._spin_visible.setRange(1, buf.n_channels)
            self._spin_visible.setValue(buf.n_visible)
            self._spin_visible.blockSignals(False)
        if self._btn_auto.isChecked() != self._plot._autoscale_enabled:
            self._btn_auto.blockSignals(True)
            self._btn_auto.setChecked(self._plot._autoscale_enabled)
            self._btn_auto.blockSignals(False)
