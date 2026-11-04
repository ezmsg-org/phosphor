"""The CPU channel heatmap: cells, colour bar, tooltip and linked viewport."""

import numpy as np
import pytest

from phosphor.channel_grid import ChannelGridConfig
from phosphor.grid_layout import Outline


@pytest.fixture
def make_heatmap(qapp):
    from phosphor.channel_heatmap import ChannelHeatmapWidget

    widgets = []

    def make(**config_kwargs):
        positions = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        config_kwargs.setdefault("vmin", 0.0)
        config_kwargs.setdefault("vmax", 10.0)
        config_kwargs.setdefault("invert_y", False)
        widget = ChannelHeatmapWidget(ChannelGridConfig(positions=positions, **config_kwargs))
        widget.resize(320, 240)
        widgets.append(widget)
        return widget

    yield make
    for widget in widgets:
        widget.deleteLater()


def _rgb(image, x, y):
    color = image.pixelColor(int(x), int(y))
    return color.redF(), color.greenF(), color.blueF()


def test_the_bar_runs_from_the_minimum_at_the_bottom_to_the_maximum_at_the_top(make_heatmap):
    heatmap = make_heatmap(cmap="magma")
    image = heatmap.grab().toImage()
    bar = heatmap._legend_bar
    assert bar.height() > 100 and bar.right() < heatmap.width()
    low = heatmap._cmap(np.array([0.0]))[0][:3]
    high = heatmap._cmap(np.array([1.0]))[0][:3]
    assert np.allclose(_rgb(image, bar.center().x(), bar.bottom() - 2), low, atol=0.1)
    assert np.allclose(_rgb(image, bar.center().x(), bar.top() + 2), high, atol=0.1)


def test_the_bar_is_labelled_with_the_range_and_unit(make_heatmap):
    heatmap = make_heatmap(value_unit="dB")
    assert heatmap._legend_labels() == ("10 dB", "0 dB")
    heatmap.set_color_range(7.0, 2.5)  # swapped ends are put in order
    assert heatmap.color_range == (2.5, 7.0)
    assert heatmap._legend_labels() == ("7 dB", "2.5 dB")
    assert make_heatmap()._legend_labels() == ("10", "0")


def test_the_cells_stay_clear_of_the_bar(make_heatmap):
    heatmap = make_heatmap()
    heatmap.grab()
    bar = heatmap._legend_bar
    assert len(heatmap._screen_rects) == 4
    assert all(rect.right() < bar.left() for _index, rect in heatmap._screen_rects)


def test_cells_take_their_value_colour_and_missing_values_the_nan_colour(make_heatmap):
    heatmap = make_heatmap(cmap="viridis")
    heatmap.push_data(np.array([0.0, 10.0, np.nan, 5.0]))
    image = heatmap.grab().toImage()
    rects = dict(heatmap._screen_rects)
    # Sample off-centre so any value text does not land on the probe.
    probe = {i: (rect.left() + 3, rect.top() + 3) for i, rect in rects.items()}
    assert np.allclose(_rgb(image, *probe[0]), heatmap._cmap(np.array([0.0]))[0][:3], atol=0.05)
    assert np.allclose(_rgb(image, *probe[1]), heatmap._cmap(np.array([1.0]))[0][:3], atol=0.05)
    assert np.allclose(_rgb(image, *probe[2]), heatmap._config.nan_color[:3], atol=0.05)


def test_a_wrong_sized_update_is_ignored(make_heatmap):
    heatmap = make_heatmap()
    heatmap.push_data(np.arange(4.0))
    heatmap.push_data(np.arange(3.0))
    assert np.array_equal(heatmap._values, np.arange(4.0))
    heatmap.clear_values()
    assert np.isnan(heatmap._values).all()


def test_the_tooltip_names_the_channel_and_formats_its_value(make_heatmap):
    heatmap = make_heatmap(channel_labels=["a", "b", "c", "d"], value_format="{:.1f}", value_unit="dB")
    heatmap.push_data(np.array([1.25, np.nan, 3.0, 4.0]))
    assert heatmap._tooltip_text(0) == "a: 1.2 dB"
    assert heatmap._tooltip_text(1) == "b: n/a"


def test_low_y_is_at_the_bottom_unless_inverted(make_heatmap):
    upright = make_heatmap()
    upright.grab()
    rects = dict(upright._screen_rects)
    assert rects[0].top() > rects[2].top()  # channel 0 (y=0) below channel 2 (y=1)

    inverted = make_heatmap(invert_y=True)
    inverted.grab()
    rects = dict(inverted._screen_rects)
    assert rects[0].top() < rects[2].top()


def test_outlines_flip_with_the_cells(make_heatmap):
    # The line between the two rows, at y=1 in position space.
    outline = Outline(np.array([[[0.0, 1.0], [2.0, 1.0]]], dtype=np.float32))
    heatmap = make_heatmap(invert_y=True, outlines=(outline,))
    segments, _ = heatmap._outlines[0]
    # Flipped, it is still the bottom edge of the y=0 row.
    assert segments[0, 0, 1] == pytest.approx(heatmap._rects[0, 1])


def test_set_viewport_shows_the_rect_without_echoing_it(make_heatmap):
    heatmap = make_heatmap()
    seen = []
    heatmap.viewport_changed.connect(seen.append)
    assert heatmap.current_viewport() == (0.0, 2.0, 0.0, 2.0)
    heatmap.set_viewport((1.0, 0.0, 0.0, 1.0))
    assert heatmap.current_viewport() == (0.0, 1.0, 0.0, 1.0)
    assert seen == []
    heatmap.reset_view()
    assert seen == [(0.0, 2.0, 0.0, 2.0)]
