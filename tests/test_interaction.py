"""Mouse interaction and trace colouring.

Both are things that are obvious the moment you use the plot and invisible in
a diff: a scroll wheel that runs backwards, a modifier whose motion arrives on
the axis you did not read, a colour that belongs to the row instead of the
channel. The decisions are pulled out into small pure helpers precisely so
they can be pinned here without a canvas.
"""

import numpy as np

from phosphor.channel_plot import ChannelPlotWidget
from phosphor.sweep_widget import SweepWidget


class Wheel:
    """Stand-in for a rendercanvas wheel event."""

    def __init__(self, dx=0.0, dy=0.0):
        self.dx, self.dy = dx, dy


# ---- wheel direction -------------------------------------------------------


def test_scroll_wheel_moves_the_channel_window_down_on_a_down_notch(qapp):
    """Direction is easy to get backwards and only obvious in use."""
    step = ChannelPlotWidget._channel_scroll_step
    assert step(1.0) == 1, "a positive wheel delta should advance the window"
    assert step(-1.0) == -1
    assert step(0.0) == 0, "a horizontal swipe carries dy=0 and must not scroll"


def test_shift_scroll_reads_the_axis_the_os_actually_used(qapp):
    """Holding Shift makes a mouse wheel arrive as horizontal scrolling.

    dy is then pinned at 0, every notch reads as negative, and amplitude only
    ever zooms out -- on a mouse. A trackpad sends both axes and looks fine,
    which is what makes this easy to miss.
    """
    read = ChannelPlotWidget._shift_wheel_delta

    assert read(Wheel(dy=3.0)) == 3.0, "trackpad: vertical axis is used as-is"
    assert read(Wheel(dy=-3.0)) == -3.0
    assert read(Wheel(dx=3.0)) == 3.0, "mouse+Shift: motion arrives on dx"
    assert read(Wheel(dx=-3.0)) == -3.0, "and must still be able to be negative"
    # dy wins when both are present, so a diagonal trackpad swipe is vertical.
    assert read(Wheel(dx=-9.0, dy=1.0)) == 1.0
    assert read(Wheel()) == 0.0


def test_trace_colour_belongs_to_the_channel_not_the_row(qapp):
    """A colour that follows the row defeats the point of having one: the eye
    cannot follow a channel across a scroll if its colour changes under it."""
    plot = ChannelPlotWidget.__new__(ChannelPlotWidget)  # no canvas needed
    plot._palette = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]

    assert plot._channel_color(0) == (1.0, 0.0, 0.0)
    assert plot._channel_color(3) == (1.0, 0.0, 0.0), "palette cycles"
    # Channel 4 is the same colour whether it is drawn in row 0 or row 3.
    assert plot._channel_color(4) == (0.0, 1.0, 0.0)


# ---- colours stay with their channels --------------------------------------


class FakeColors:
    """fastplotlib keeps every line's vertices in one flat RGBA buffer."""

    def __init__(self, n_rows: int, stride: int):
        self.value = np.zeros((n_rows * stride, 4), dtype=np.float32)

    def __setitem__(self, key, value):
        self.value[key] = value


class FakeBuffer:
    def __init__(self, n_visible: int, channel_offset: int):
        self.n_visible, self.channel_offset = n_visible, channel_offset


def make_widget(n_visible: int, offset: int, stride: int) -> SweepWidget:
    """A SweepWidget with only the fields _apply_channel_colors touches.

    Built without __init__ because everything else it would construct needs a
    GPU canvas, and the arithmetic under test is pure numpy.
    """
    w = SweepWidget.__new__(SweepWidget)
    w._palette = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    w.sweep_buffer = FakeBuffer(n_visible, offset)
    w._multi_line = type("ML", (), {})()
    w._multi_line.colors = FakeColors(n_visible, stride)
    return w


def test_visible_colours_are_taken_from_absolute_channels():
    w = make_widget(n_visible=3, offset=4, stride=2)
    assert w._visible_colors() == [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]


def test_scrolling_rewrites_the_colour_buffer_row_by_row():
    """Each row owns a contiguous run of `stride` entries -- its vertices plus
    the NaN separator that terminates the line."""
    stride = 4
    w = make_widget(n_visible=3, offset=1, stride=stride)
    w._apply_channel_colors()

    written = w._multi_line.colors.value
    assert written.shape == (3 * stride, 4)
    for row, expected in enumerate([(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]):
        run = written[row * stride : (row + 1) * stride]
        assert np.all(run[:, :3] == expected), f"row {row} is not one solid colour"
        assert np.all(run[:, 3] == 1.0), "alpha must stay opaque"

    assert w._cached_offset == 1, "the applied offset is recorded, so it applies once"


def test_a_channel_keeps_its_colour_across_a_scroll():
    """The whole point: follow one trace as the window moves past it."""
    stride = 4
    before = make_widget(n_visible=3, offset=0, stride=stride)
    before._apply_channel_colors()
    # Channel 2 is drawn in row 2 here, and in row 0 after scrolling by two.
    colour_of_channel_2 = before._multi_line.colors.value[2 * stride, :3].copy()

    after = make_widget(n_visible=3, offset=2, stride=stride)
    after._apply_channel_colors()

    np.testing.assert_array_equal(after._multi_line.colors.value[0, :3], colour_of_channel_2)


def test_colour_rewrite_survives_a_single_visible_channel():
    """max(n_visible, 1) guards a division that would otherwise be by zero."""
    w = make_widget(n_visible=1, offset=7, stride=5)
    w._apply_channel_colors()
    assert np.all(w._multi_line.colors.value[:, :3] == (0.0, 1.0, 0.0))
