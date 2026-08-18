"""Canvas overlay geometry.

The overlays are simple to draw and easy to get subtly wrong: a label a few
pixels off its trace, or one that stops updating when the view scrolls, is the
kind of thing that survives a screenshot review. The maths is all in
``_row_geometry`` and the change detection, so that is what is pinned here.
"""

import pytest
from PySide6 import QtCore

from phosphor.overlays import (
    MAX_LABEL_FONT_PX,
    MIN_LABEL_FONT_PX,
    ChannelLabelOverlay,
    ScaleBarOverlay,
)


@pytest.fixture()
def labels(qapp) -> ChannelLabelOverlay:
    ov = ChannelLabelOverlay()
    ov.resize(400, 300)
    return ov


@pytest.fixture()
def bar(qapp) -> ScaleBarOverlay:
    ov = ScaleBarOverlay()
    ov.resize(400, 300)
    return ov


# ---- world -> screen mapping -----------------------------------------------


def test_measured_projection_is_used_verbatim(labels):
    """When the plot supplies the camera's mapping, use it -- do not re-derive."""
    labels.set_view(0, 4, True, y0=100.0, slope=-25.0, z_scale=1.0)
    assert labels._row_geometry(4, 300) == (100.0, -25.0)


def test_analytic_fallback_spans_the_canvas(labels):
    """With no camera mapping, rows plus a 5% margin must fill the height."""
    labels.set_view(0, 5, True, y0=None, slope=None, z_scale=1.0)
    y0, slope = labels._row_geometry(5, 300)

    # Rows sit at world-y 0..4 with margin max(4*0.05, 0.5) = 0.5 either side,
    # so world -0.5 maps to the bottom (y=300) and 4.5 to the top (y=0).
    assert y0 + slope * -0.5 == pytest.approx(300.0)
    assert y0 + slope * 4.5 == pytest.approx(0.0)
    assert slope < 0  # higher world-y is nearer the top


def test_degenerate_slope_falls_back(labels):
    """A zero slope would collapse every label onto one line."""
    labels.set_view(0, 4, True, y0=10.0, slope=0.0, z_scale=1.0)
    _, slope = labels._row_geometry(4, 300)
    assert slope != 0.0


def test_single_channel_fallback_does_not_divide_by_zero(labels):
    labels.set_view(0, 1, True, z_scale=1.0)
    y0, slope = labels._row_geometry(1, 300)
    assert slope == pytest.approx(-300.0)  # margin 0.5 either side of one row
    assert y0 == pytest.approx(150.0)


# ---- repaint gating --------------------------------------------------------


def test_set_view_repaints_only_on_change(labels, monkeypatch):
    calls = []
    monkeypatch.setattr(labels, "update", lambda: calls.append(1))

    labels.set_view(0, 4, True, 100.0, -25.0, 1.0)
    assert len(calls) == 1
    labels.set_view(0, 4, True, 100.0, -25.0, 1.0)
    assert len(calls) == 1, "identical view should not repaint"
    labels.set_view(1, 4, True, 100.0, -25.0, 1.0)
    assert len(calls) == 2, "scrolling by one channel must repaint"


def test_set_labels_repaints_only_on_change(labels, monkeypatch):
    labels.set_labels(["a", "b"])
    calls = []
    monkeypatch.setattr(labels, "update", lambda: calls.append(1))

    labels.set_labels(["a", "b"])
    assert calls == []
    labels.set_labels(["a", "c"])
    assert len(calls) == 1


def test_scale_bar_repaints_only_on_change(bar, monkeypatch):
    bar.set_bar(40.0, "100 uV")
    calls = []
    monkeypatch.setattr(bar, "update", lambda: calls.append(1))

    bar.set_bar(40.0, "100 uV")
    assert calls == []
    bar.set_bar(41.0, "100 uV")
    assert len(calls) == 1
    bar.set_bar(41.0, "200 uV")
    assert len(calls) == 2


# ---- painting is defensive -------------------------------------------------


def test_painting_with_no_data_is_a_no_op(labels, bar):
    """Rendering starts before the first view arrives; must not raise."""
    labels.render(labels.grab())  # no labels, no view
    bar.render(bar.grab())  # no bar length or text


def test_label_paint_survives_extremes(labels):
    """Very tall and very short rows both have to draw without raising."""
    labels.set_labels([f"ch{i}" for i in range(512)])

    labels.set_view(0, 1, True, 150.0, -300.0, 1.0)  # one enormous row
    labels.grab()

    labels.set_view(0, 512, True, 300.0, -0.6, 1.0)  # sub-pixel rows
    labels.grab()


def test_font_bounds_are_sane():
    assert 0 < MIN_LABEL_FONT_PX < MAX_LABEL_FONT_PX
    # Ordinary reading size. Bigger conveys nothing and the opaque chip behind
    # each label would cover the signal it refers to.
    assert MAX_LABEL_FONT_PX <= 16


def test_font_is_capped_on_tall_rows(qapp, monkeypatch):
    """Few visible channels means rows hundreds of pixels tall; the label must
    not scale with them."""
    from PySide6 import QtGui

    sizes = []
    original = QtGui.QPainter.setFont

    def spy(self, font):
        sizes.append(font.pixelSize())
        return original(self, font)

    monkeypatch.setattr(QtGui.QPainter, "setFont", spy)

    ov = ChannelLabelOverlay()
    ov.resize(400, 1000)
    ov.set_labels(["ch0", "ch1"])
    ov.set_view(0, 2, True, 1000.0, -500.0, 1.0)  # 500 px rows
    ov.grab()

    assert sizes and max(sizes) <= MAX_LABEL_FONT_PX


def test_font_bounds_are_overridable(qapp):
    ov = ChannelLabelOverlay(min_font_px=6, max_font_px=9)
    assert (ov._min_font_px, ov._max_font_px) == (6, 9)


def test_labels_are_indexed_by_absolute_channel(labels, monkeypatch):
    """Scrolling changes which labels are drawn without the caller reslicing."""
    drawn: list[str] = []
    labels.set_labels([f"ch{i}" for i in range(64)])
    labels.set_view(0, 4, True, 300.0, -75.0, 1.0)

    # Intercept the text calls rather than reading pixels.
    from PySide6 import QtGui

    original = QtGui.QPainter.drawText

    def spy(self, *args):
        if args and isinstance(args[-1], str):
            drawn.append(args[-1])
        return original(self, *args)

    monkeypatch.setattr(QtGui.QPainter, "drawText", spy)

    labels.grab()
    assert drawn == ["ch0", "ch1", "ch2", "ch3"]

    drawn.clear()
    labels.set_view(10, 4, True, 300.0, -75.0, 1.0)
    labels.grab()
    assert drawn == ["ch10", "ch11", "ch12", "ch13"]


def test_scale_bar_sits_in_the_lower_right(bar, monkeypatch):
    """Out of the way of the traces, and anchored at its foot so the bottom
    edge does not move when the amplitude scale changes."""
    from PySide6 import QtGui

    from phosphor.overlays import SCALE_BAR_BOTTOM_MARGIN, SCALE_BAR_RIGHT_MARGIN

    lines: list[tuple] = []
    original = QtGui.QPainter.drawLine

    def spy(self, *args):
        lines.append(args)
        return original(self, *args)

    monkeypatch.setattr(QtGui.QPainter, "drawLine", spy)

    w, h = bar.width(), bar.height()
    bar.set_bar(60.0, "100 uV")
    bar.grab()

    # First drawLine is the vertical bar: (x, y_top, x, y_bot).
    x, y_top, _, y_bot = lines[0]
    assert x == w - SCALE_BAR_RIGHT_MARGIN
    assert y_bot == h - SCALE_BAR_BOTTOM_MARGIN
    assert y_bot - y_top == pytest.approx(60.0, abs=1)
    assert y_top > h / 2, "bar should sit below the vertical midpoint"

    # Growing the bar moves its top, not its foot.
    lines.clear()
    bar.set_bar(120.0, "200 uV")
    bar.grab()
    _, y_top2, _, y_bot2 = lines[0]
    assert y_bot2 == y_bot
    assert y_top2 < y_top


# ---- projection cache ------------------------------------------------------


class _Viewport:
    def __init__(self, rect):
        self.rect = rect


class _Subplot:
    """Enough of a subplot to probe: a viewport, a camera, and a mapping that
    reads the viewport the way fastplotlib's own does."""

    def __init__(self, rect=(0.0, 0.0, 800.0, 600.0)):
        self.viewport = _Viewport(rect)
        self.camera = type("Cam", (), {"height": 16.5, "world": type("W", (), {"position": (0.0, 7.5, 0.0)})()})()

    def map_world_to_screen(self, pos):
        _, _, _, h = self.viewport.rect
        y_offset = self.viewport.rect[1]
        ndc_y = (pos[1] - self.camera.world.position[1]) / (self.camera.height / 2)
        return 0.0, y_offset + (1 - ndc_y) * 0.5 * h


def _probe(subplot):
    from phosphor.channel_plot import ChannelPlotWidget

    plot = ChannelPlotWidget.__new__(ChannelPlotWidget)  # no canvas needed
    plot._subplot = subplot
    # A canvas whose size never changes, so that keying on it -- which is the
    # mistake this pins -- would hold the cache shut rather than error out.
    plot._fpl_widget = type("W", (), {"size": staticmethod(lambda: QtCore.QSize(800, 600))})()
    plot._buffer = type("Buf", (), {"n_visible": 16})()
    plot._z_offset_scale = 1.0
    plot._projection = None
    plot._projection_key = None
    return plot


def test_projection_follows_the_viewport_not_the_canvas_size():
    """The cache has to key on what the mapping actually reads.

    Keying on the canvas widget's size instead works right up until something
    reshapes the viewport without resizing the canvas, at which point nothing
    is left to evict the entry -- a sweep's camera height follows n_visible
    alone, so it does not move on a resize either. The labels then keep a
    projection built for the old geometry and sit stretched or compressed
    against their traces indefinitely.
    """
    subplot = _Subplot((0.0, 0.0, 800.0, 600.0))
    plot = _probe(subplot)

    first = plot._screen_y_projection()
    assert plot._screen_y_projection() == first, "unchanged view should hit the cache"

    subplot.viewport.rect = (0.0, 40.0, 800.0, 520.0)
    second = plot._screen_y_projection()

    assert second != first
    y0, slope = second
    for world_y in (0.0, 7.5, 15.0):
        expected = subplot.map_world_to_screen((0.0, world_y, 0.0))[1]
        assert y0 + slope * world_y == pytest.approx(expected, abs=1e-6)
