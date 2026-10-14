"""The controls bar's amplitude range field and autoscale toggle.

A stub stands in for the plot: the bar only touches the buffer and a few plot
methods, and a real SweepWidget needs a GPU.
"""

import numpy as np
import pytest

from phosphor.sweep_buffer import SweepBuffer


class StubPlot:
    channel_labels_visible = False

    def __init__(self):
        self._buffer = SweepBuffer(n_channels=4, srate=1000.0, display_dur=1.0, n_columns=10, n_visible=4)
        self._buffer.push_data(np.full((500, 4), 1.0, dtype=np.float32))
        self._buffer.get_multiline_data()

    def set_channel_labels_visible(self, visible):
        pass

    def _zoom_amplitude(self, factor):
        self._buffer.set_amplitude_scale(self._buffer.amplitude_scale * factor)

    def _update_range_label(self):
        pass


@pytest.fixture
def controls(qapp):
    from phosphor.controls import ChannelPlotControlsWidget

    return ChannelPlotControlsWidget(StubPlot())


def test_the_field_shows_the_range_and_typing_fixes_it(controls):
    buf = controls._plot._buffer
    controls._sync_from_buffer()
    assert controls._edit_range.text() == "1"

    controls._edit_range.setText("250")
    controls._edit_range.setModified(True)  # as typing would
    controls._on_range_committed()

    assert buf.y_range == pytest.approx(250.0)
    assert not buf.autoscale
    assert not controls._btn_auto.isChecked()


def test_leaving_an_untouched_field_keeps_autoscale_on(controls):
    """editingFinished fires on focus-out too."""
    controls._sync_from_buffer()
    controls._on_range_committed()
    assert controls._plot._buffer.autoscale


def test_the_tooltips_name_the_shortcuts(controls):
    for widget in (controls._edit_range, controls._btn_auto):
        tip = widget.toolTip()
        assert "- shrink" in tip and "= grow" in tip and "A toggle autoscale" in tip
