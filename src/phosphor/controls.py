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

from PySide6 import QtCore, QtGui, QtWidgets

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

        # Named insertion points, so a subclass can put its own widgets next to
        # the group they belong with. Without these the only way to extend this
        # toolbar is to scan the layout for a QLabel with the right text, which
        # breaks the moment a label is renamed or reordered.
        self._slots: dict[str, int] = {}

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
        self._mark_slot("channel", layout)

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
        self._mark_slot("visible", layout)

        layout.addWidget(self._make_separator())

        # Amplitude zoom (per-row waveform scale), plus an editable range and
        # an autoscale toggle where the buffer supports a fixed scale.
        has_range = hasattr(buf, "set_y_range")
        keys = "Keys: - shrink, = grow, Shift+scroll zoom"
        if has_range:
            keys += ", A toggle autoscale"
        amp_label = self._make_label("Amplitude")
        amp_label.setToolTip(keys)
        layout.addWidget(amp_label)
        layout.addWidget(
            self._make_button("\u2212", f"Shrink amplitude\n{keys}", lambda: self._plot._zoom_amplitude(0.8))
        )
        self._edit_range: QtWidgets.QLineEdit | None = None
        self._unit_label: QtWidgets.QLabel | None = None
        self._btn_auto: QtWidgets.QToolButton | None = None
        if has_range:
            layout.addWidget(self._make_label("\u00b1"))
            self._edit_range = QtWidgets.QLineEdit()
            self._edit_range.setFixedWidth(64)
            validator = QtGui.QDoubleValidator(self._edit_range)
            validator.setBottom(0.0)
            self._edit_range.setValidator(validator)
            self._edit_range.setToolTip(
                f"Each row spans \u00b1 this value. Type a value and press Enter to fix it"
                f" (turns autoscale off).\n{keys}"
            )
            self._edit_range.editingFinished.connect(self._on_range_committed)
            layout.addWidget(self._edit_range)
            self._unit_label = self._make_label("")
            self._unit_label.hide()
            layout.addWidget(self._unit_label)
        layout.addWidget(self._make_button("+", f"Grow amplitude\n{keys}", lambda: self._plot._zoom_amplitude(1.25)))
        if has_range:
            self._btn_auto = QtWidgets.QToolButton()
            self._btn_auto.setText("Auto")
            self._btn_auto.setCheckable(True)
            self._btn_auto.setChecked(buf.autoscale)
            self._btn_auto.setToolTip(f"Refit the amplitude range to the visible data\n{keys}")
            self._btn_auto.toggled.connect(buf.set_autoscale)
            layout.addWidget(self._btn_auto)
        self._mark_slot("amplitude", layout)

        # Time zoom — only present if the plot supports it (sweep / spectrum).
        if hasattr(plot, "_time_zoom"):
            layout.addWidget(self._make_separator())
            layout.addWidget(self._make_label("Time"))
            layout.addWidget(self._make_button("\u2212", "Zoom time out (longer span)", lambda: plot._time_zoom(2.0)))
            layout.addWidget(self._make_button("+", "Zoom time in (shorter span)", lambda: plot._time_zoom(0.5)))
            self._mark_slot("time", layout)

        layout.addWidget(self._make_separator())

        # Channel-label toggle. The labels sit on opaque chips, so on a dense
        # trace they cover signal; once the user knows which channel is which,
        # they mostly want them gone.
        self._btn_labels = QtWidgets.QToolButton()
        self._btn_labels.setText("Labels")
        self._btn_labels.setCheckable(True)
        self._btn_labels.setChecked(plot.channel_labels_visible)
        self._btn_labels.setToolTip("Show per-channel labels over the traces")
        self._btn_labels.toggled.connect(self._plot.set_channel_labels_visible)
        layout.addWidget(self._btn_labels)
        self._mark_slot("labels", layout)

        self.add_controls(layout)
        layout.addStretch(1)
        self._mark_slot("end", layout)

        # Periodically resync widget state with the buffer (other inputs —
        # keyboard, mouse — also mutate it). 200 ms is plenty for a panel.
        self._sync_timer = QtCore.QTimer(self)
        self._sync_timer.setInterval(200)
        self._sync_timer.timeout.connect(self._sync_from_buffer)
        self._sync_timer.start()

    # ---- extension points ---------------------------------------------

    def add_controls(self, layout: QtWidgets.QHBoxLayout) -> None:
        """Append subclass widgets, before the trailing stretch.

        Called once during ``__init__``. Override for controls that belong at
        the end of the bar; use :meth:`insert_after` for ones that belong beside
        an existing group.
        """

    def insert_after(self, slot: str, widget: QtWidgets.QWidget) -> None:
        """Insert *widget* immediately after a named group.

        Slots are ``"channel"``, ``"visible"``, ``"amplitude"``, ``"time"``
        (absent unless the plot supports time zoom), ``"labels"``, and
        ``"end"``. Later slots shift as earlier ones grow, so inserting is
        order-independent.

        :raises KeyError: for an unknown slot, naming the ones that exist --
            silently dropping the widget would be worse, and ``"time"``
            genuinely is conditional.
        """
        if slot not in self._slots:
            raise KeyError(f"unknown control slot {slot!r}; available: {sorted(self._slots)}")
        index = self._slots[slot]
        self.layout().insertWidget(index, widget)
        for name, pos in self._slots.items():
            if pos >= index:
                self._slots[name] = pos + 1

    def set_amplitude_unit(self, unit: str | None) -> None:
        """Label the amplitude range with the signal's unit; empty hides it."""
        if self._unit_label is None:
            return
        self._unit_label.setText(unit or "")
        self._unit_label.setVisible(bool(unit))

    def _mark_slot(self, name: str, layout: QtWidgets.QHBoxLayout) -> None:
        """Record the current end of *layout* as the named insertion point."""
        self._slots[name] = layout.count()

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

    # ---- amplitude range ---------------------------------------------

    def _on_range_committed(self) -> None:
        # editingFinished also fires on focus-out; an untouched field must not
        # turn autoscale off.
        if not self._edit_range.isModified():
            return
        try:
            value = float(self._edit_range.text())
        except ValueError:
            return
        if value > 0:
            self._plot._buffer.set_y_range(value)
        self._edit_range.clearFocus()  # hand the keys back to the plot
        self._sync_from_buffer()

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
        if self._edit_range is not None and not self._edit_range.hasFocus():
            text = f"{buf.y_range:.4g}"
            if self._edit_range.text() != text:
                self._edit_range.setText(text)
        if self._btn_auto is not None and self._btn_auto.isChecked() != buf.autoscale:
            self._btn_auto.blockSignals(True)
            self._btn_auto.setChecked(buf.autoscale)
            self._btn_auto.blockSignals(False)
        if self._btn_labels.isChecked() != self._plot.channel_labels_visible:
            self._btn_labels.blockSignals(True)
            self._btn_labels.setChecked(self._plot.channel_labels_visible)
            self._btn_labels.blockSignals(False)
