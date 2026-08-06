"""Colour and tooltip policy for the per-channel value grid.

The widget itself needs a GPU canvas, but the two decisions worth pinning do
not: which colour a channel gets, and what hovering it says. Both exist to keep
domain vocabulary out of this module -- a caller that decides what "good" means
supplies the colours and the word, and the grid renders them without an opinion.
"""

import numpy as np
import pytest

from phosphor.channel_grid import ChannelGridConfig, ChannelGridWidget


def make_grid(n_ch: int = 4, **config_kwargs) -> ChannelGridWidget:
    """A grid with only the state the colour/tooltip logic reads.

    Built without ``__init__`` because everything else it constructs needs a
    canvas; these paths are numpy and string formatting.
    """
    import cmap as cmap_lib

    config = ChannelGridConfig(positions=np.zeros((n_ch, 2)), **config_kwargs)
    w = ChannelGridWidget.__new__(ChannelGridWidget)
    w._config = config
    w._values_per_ch = np.full(n_ch, np.nan, dtype=np.float32)
    w._cmap = cmap_lib.Colormap(config.cmap)
    w._nan_rgba = np.asarray(config.nan_color, dtype=np.float32)
    w._explicit_rgba = None
    w._annotations = None
    w._dirty = False
    return w


# ---- colour ----------------------------------------------------------------


def test_values_are_normalized_over_the_configured_range():
    w = make_grid(3, vmin=0.0, vmax=100.0, cmap="viridis")
    w.push_data(np.array([0.0, 50.0, 100.0]))

    rgba = w._channel_rgba()
    assert rgba.shape == (3, 4)
    # Monotone through the colormap: the ends differ and the middle is between.
    assert not np.allclose(rgba[0], rgba[2])
    assert not np.allclose(rgba[1], rgba[0])


def test_a_channel_with_no_value_gets_the_nan_colour():
    w = make_grid(2, nan_color=(1.0, 0.0, 1.0, 1.0))
    w.push_data(np.array([5.0, np.nan]))
    np.testing.assert_allclose(w._channel_rgba()[1], [1.0, 0.0, 1.0, 1.0])


def test_values_outside_the_range_clamp_rather_than_wrap():
    """An out-of-range reading should saturate at the end of the scale, not
    reappear at the other end looking healthy."""
    w = make_grid(4, vmin=0.0, vmax=10.0)
    w.push_data(np.array([-50.0, 0.0, 10.0, 999.0]))
    rgba = w._channel_rgba()
    np.testing.assert_allclose(rgba[0], rgba[1])
    np.testing.assert_allclose(rgba[3], rgba[2])


def test_a_degenerate_range_does_not_divide_by_zero():
    w = make_grid(2, vmin=5.0, vmax=5.0)
    w.push_data(np.array([5.0, 5.0]))
    assert np.all(np.isfinite(w._channel_rgba()))


def test_explicit_colours_override_the_colormap():
    """The categorical path: the caller classified the values and owns the
    mapping, so the colormap must not get a second say."""
    w = make_grid(2)
    w.push_data(np.array([1.0, 2.0]))
    w.set_colors(np.array([[1.0, 0, 0, 1.0], [0, 1.0, 0, 1.0]]))

    np.testing.assert_allclose(w._channel_rgba(), [[1, 0, 0, 1], [0, 1, 0, 1]])
    # Including for a channel with no value: a caller that computes its own
    # colours is the one that decides what missing looks like.
    w.push_data(np.array([np.nan, 2.0]))
    np.testing.assert_allclose(w._channel_rgba()[0], [1, 0, 0, 1])


def test_clearing_explicit_colours_restores_the_colormap():
    w = make_grid(2, vmin=0.0, vmax=10.0)
    w.push_data(np.array([0.0, 10.0]))
    w.set_colors(np.zeros((2, 4)))
    w.set_colors(None)
    assert not np.allclose(w._channel_rgba()[0], w._channel_rgba()[1])


def test_wrongly_sized_colours_are_refused_not_applied():
    """Silently colouring the wrong channels is worse than not colouring."""
    w = make_grid(4)
    w.set_colors(np.ones((3, 4)))
    assert w._explicit_rgba is None


# ---- tooltip ---------------------------------------------------------------


def test_tooltip_names_the_channel_and_its_value():
    w = make_grid(2, channel_labels=["A-1", "A-2"], value_unit="kOhm")
    w.push_data(np.array([12.34, np.nan]))
    assert w._tooltip_text(0) == "A-1\n12.3 kOhm"


def test_tooltip_falls_back_to_an_index_when_unlabelled():
    w = make_grid(2)
    w.push_data(np.array([1.0, 2.0]))
    assert w._tooltip_text(1).startswith("ch1\n")


def test_tooltip_omits_a_unit_that_was_never_set():
    w = make_grid(1)
    w.push_data(np.array([7.0]))
    assert w._tooltip_text(0) == "ch0\n7.0"


def test_annotations_appear_beside_the_value():
    w = make_grid(2, channel_labels=["A-1", "A-2"], value_unit="kOhm")
    w.push_data(np.array([12.0, 900.0]))
    w.set_annotations(["good", "open"])
    assert w._tooltip_text(1) == "A-2\n900.0 kOhm  (open)"


def test_an_annotation_stands_in_for_a_missing_value():
    """A channel with no reading still has something worth saying about it."""
    w = make_grid(1, channel_labels=["A-1"])
    w.push_data(np.array([np.nan]))
    w.set_annotations(["unmeasured"])
    assert w._tooltip_text(0) == "A-1\nunmeasured"


def test_a_missing_value_says_so_without_an_annotation():
    w = make_grid(1)
    assert w._tooltip_text(0) == "ch0\nno value"


def test_wrongly_sized_annotations_are_refused():
    w = make_grid(3)
    w.set_annotations(["a", "b"])
    assert w._annotations is None


@pytest.mark.parametrize("value", [0.0, -1.5, 1e6])
def test_tooltip_formats_any_finite_value(value):
    w = make_grid(1, value_unit="uV")
    w.push_data(np.array([value]))
    assert w._tooltip_text(0).startswith("ch0\n")
