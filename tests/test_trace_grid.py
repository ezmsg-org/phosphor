"""The trace grid's drawing decisions, minus the canvas.

Three of them matter enough to pin: which slot is drawn how brightly (the ring
does not reorder, so brightness has to follow age), when the graphics have to be
recreated rather than written into (getting this wrong is what made the widget
tear down every retained waveform to draw one new one), and the shape of the
error band.
"""

import numpy as np
import pytest

from phosphor.decimate import plan_minmax_decimation
from phosphor.trace_grid import TraceGridConfig, TraceGridWidget
from phosphor.trace_grid_buffer import TraceGridBuffer


def make_widget(n_ch=2, n_samples=4, history=3, **config_kwargs) -> TraceGridWidget:
    """A widget holding only what the drawing helpers read.

    Built without ``__init__`` because the rest of it needs a GPU canvas.
    """
    config = TraceGridConfig(positions=np.zeros((n_ch, 2)), n_samples=n_samples, history=history, **config_kwargs)
    w = TraceGridWidget.__new__(TraceGridWidget)
    w._config = config
    w._n_ch = n_ch
    w._n_samples = n_samples
    w._show_individual = config.show_individual
    w._show_mean = config.show_mean
    w._show_error = config.show_error and config.track_statistics
    w._buffer = TraceGridBuffer(n_ch, n_samples, history, track_statistics=config.track_statistics)
    w._dec_plan = plan_minmax_decimation(n_samples, 10_000)  # inactive at these sizes
    w._x_line_dec = np.tile(np.arange(n_samples, dtype=np.float32), (n_ch, 1))
    w._indiv_ml = w._mean_ml = w._error_ml = None
    w._graphics_version = -1
    # _map_y is affine per channel; identity keeps these tests about layout.
    w._map_y = lambda a: np.asarray(a, dtype=np.float32)
    return w


def wave(value, n_ch=2, n_samples=4):
    return np.full((n_ch, n_samples), value, dtype=np.float32)


# ---- age-graded brightness --------------------------------------------------


def test_the_newest_waveform_is_the_brightest():
    w = make_widget(history=3)
    for v in (1.0, 2.0, 3.0):
        w._buffer.push(wave(v))

    alpha = w._slot_alpha()
    ages = w._buffer.ages
    assert alpha[ages == 0] > alpha[ages == 1] > alpha[ages == 2]


def test_brightness_follows_the_slot_that_holds_the_newest():
    """Nothing is reordered on write, so after a wrap the brightest slot is not
    slot 0. Keying on position instead of age would light up a stale waveform."""
    w = make_widget(history=3)
    for v in (1.0, 2.0, 3.0, 4.0):  # wraps: newest lands back in slot 0
        w._buffer.push(wave(v))

    brightest = int(np.argmax(w._slot_alpha()))
    assert w._buffer.traces[brightest][0, 0] == pytest.approx(4.0)


def test_slots_never_written_are_fully_transparent():
    w = make_widget(history=4)
    w._buffer.push(wave(1.0))

    alpha = w._slot_alpha()
    assert (alpha > 0).sum() == 1
    np.testing.assert_allclose(alpha[w._buffer.ages >= w._buffer.n_retained], 0.0)


def test_fading_can_be_switched_off():
    """Every retained waveform's age changes when one arrives, so fading costs
    a colour rewrite per arrival -- worth it at evoked rates, not spike rates."""
    w = make_widget(history=3, age_fade=False)
    for v in (1.0, 2.0, 3.0):
        w._buffer.push(wave(v))

    alpha = w._slot_alpha()
    assert len(set(alpha.tolist())) == 1, "all retained waveforms should look alike"


def test_slot_rows_are_contiguous_so_one_arrival_writes_one_block():
    w = make_widget(n_ch=8, history=4)
    rows = w._slot_rows(2)
    assert (rows.start, rows.stop) == (16, 24)


# ---- when to recreate the graphics -----------------------------------------


def test_a_new_waveform_does_not_change_the_graphics_shape():
    """The whole point of the split: numbers change, shapes do not."""
    w = make_widget(history=3)
    w._buffer.push(wave(1.0))
    shape = w._graphics_shape()

    w._buffer.push(wave(2.0))
    assert w._graphics_shape() == shape


@pytest.mark.parametrize(
    "change",
    [
        lambda w: w._buffer.set_history(9),
        lambda w: setattr(w, "_show_mean", not w._show_mean),
        lambda w: setattr(w, "_show_error", not w._show_error),
        lambda w: setattr(w, "_show_individual", not w._show_individual),
    ],
)
def test_anything_that_adds_or_removes_lines_does_change_it(change):
    w = make_widget(history=3)
    # Two, so there is a standard deviation and the error toggle has an effect.
    w._buffer.push(wave(1.0))
    w._buffer.push(wave(3.0))
    shape = w._graphics_shape()

    change(w)
    assert w._graphics_shape() != shape


# ---- mean and error band ----------------------------------------------------


def test_the_band_is_two_curves_per_channel():
    w = make_widget(n_ch=2, history=4, show_error=True)
    for v in (1.0, 3.0):
        w._buffer.push(wave(v))

    mean_pos, band_pos = w._summary_positions()
    assert mean_pos.shape[0] == 2, "one mean line per channel"
    assert band_pos.shape[0] == 4, "a lower and an upper line per channel"


def test_the_band_brackets_the_mean():
    w = make_widget(n_ch=1, history=4, show_error=True)
    for v in (1.0, 3.0):
        w._buffer.push(wave(v, n_ch=1))

    mean_pos, band_pos = w._summary_positions()
    mean_y = mean_pos[0, :, 1]
    lower, upper = band_pos[0, :, 1], band_pos[1, :, 1]
    assert np.all(lower < mean_y) and np.all(upper > mean_y)


def test_the_band_keeps_each_curve_on_its_own_channel_x():
    """Two curves per channel means x tiles rather than broadcasts; getting it
    wrong draws the band across the wrong cells."""
    w = make_widget(n_ch=2, history=4, show_error=True)
    for v in (1.0, 3.0):
        w._buffer.push(wave(v))

    _, band_pos = w._summary_positions()
    np.testing.assert_allclose(band_pos[0, :, 0], w._x_line_dec[0])
    np.testing.assert_allclose(band_pos[2, :, 0], w._x_line_dec[0])


def test_no_band_until_there_is_a_spread_to_draw():
    """One waveform has a mean but no standard deviation, and a band of NaN
    would take the view off screen."""
    w = make_widget(history=4, show_error=True)
    w._buffer.push(wave(1.0))

    mean_pos, band_pos = w._summary_positions()
    assert mean_pos is not None
    assert band_pos is None


def test_no_summary_at_all_before_anything_arrives():
    w = make_widget(show_error=True)
    assert w._summary_positions() == (None, None)


def test_the_band_is_refused_when_statistics_are_off():
    """Asking for a band around a mean that is never computed should resolve to
    'no band', not to a crash at draw time."""
    w = make_widget(history=3, show_error=True, track_statistics=False)
    for v in (1.0, 3.0):
        w._buffer.push(wave(v))

    assert w._show_error is False
    assert w._summary_positions() == (None, None)


def test_the_mean_spans_more_waveforms_than_are_drawn():
    """What the running accumulator buys: an evoked average of everything seen,
    overlaid with the handful worth looking at."""
    w = make_widget(n_ch=1, history=2)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        w._buffer.push(wave(v, n_ch=1))

    mean_pos, _ = w._summary_positions()
    np.testing.assert_allclose(mean_pos[0, :, 1], 3.0)  # mean of 1..5, not of 4..5


# ---- graphics actually get created ------------------------------------------


class FakeColors:
    """fastplotlib takes one colour per line, then holds one per vertex."""

    def __init__(self, n_lines, n_points):
        self.value = np.zeros((n_lines * n_points, 4), dtype=np.float32)

    def __setitem__(self, key, value):
        self.value[key] = value


class FakeGraphic:
    def __init__(self, data):
        self.data = np.asarray(data).copy()
        self.colors = FakeColors(self.data.shape[0], self.data.shape[1])
        self.visible = True


class FakeSubplot:
    """Records what a renderer would have been asked to draw."""

    def __init__(self):
        self.created: list[FakeGraphic] = []
        self.deleted: list[FakeGraphic] = []

    def add_multi_line(self, data, colors=None, thickness=None, **_):
        g = FakeGraphic(data)
        self.created.append(g)
        return g

    def delete_graphic(self, graphic):
        self.deleted.append(graphic)


def drivable(**kwargs) -> TraceGridWidget:
    w = make_widget(**kwargs)
    w._subplot = FakeSubplot()
    return w


def test_the_first_waveform_creates_the_graphics():
    """The regression that showed up as a grid of empty cells with epochs
    arriving: the widget is built before any data, so the first frame rebuilds
    against an empty buffer. If that state is indistinguishable from a
    populated one, every later arrival takes the in-place path, finds nothing
    to write into, and draws nothing -- for as long as the app runs.
    """
    w = drivable(history=3)

    w._refresh_lines()  # a frame before any data, as happens on startup
    assert w._subplot.created == [], "nothing to draw yet"

    w._buffer.push(wave(1.0))
    w._refresh_lines()

    assert w._subplot.created, "the first waveform must bring its graphics with it"
    assert w._indiv_ml is not None


def test_later_waveforms_write_in_place_instead_of_recreating():
    """The other half: once the graphics exist, arrivals must not rebuild."""
    w = drivable(history=3)
    w._buffer.push(wave(1.0))
    w._refresh_lines()
    created = len(w._subplot.created)

    for v in (2.0, 3.0, 4.0):
        w._buffer.push(wave(v))
        w._refresh_lines()

    assert len(w._subplot.created) == created, "arrivals should not recreate graphics"
    assert w._subplot.deleted == []


def test_the_newest_waveform_reaches_the_graphic():
    """An in-place path that silently wrote nowhere would look identical to a
    working one until you looked at the screen."""
    w = drivable(history=3, show_mean=False)
    w._buffer.push(wave(1.0))
    w._refresh_lines()

    w._buffer.push(wave(7.0))
    w._refresh_lines()

    ys = w._indiv_ml.data[..., 1]
    assert np.isclose(ys, 7.0).any(), "the value just pushed should be in the graphic"


def test_the_error_band_appears_once_there_is_a_spread():
    """It cannot be drawn from one waveform, so it arrives a frame later than
    the mean and needs its own place in the shape key."""
    w = drivable(history=4, show_error=True)

    w._buffer.push(wave(1.0))
    w._refresh_lines()
    assert w._error_ml is None

    w._buffer.push(wave(3.0))
    w._refresh_lines()
    assert w._error_ml is not None


def test_clearing_and_refilling_brings_the_graphics_back():
    """clear() drops to the empty state; refilling has to leave it again."""
    w = drivable(history=3)
    w._buffer.push(wave(1.0))
    w._refresh_lines()

    w._buffer.clear()
    w._refresh_lines()
    assert w._indiv_ml is None

    w._buffer.push(wave(2.0))
    w._refresh_lines()
    assert w._indiv_ml is not None
