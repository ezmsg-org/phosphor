"""Retention and running statistics for the trace grid.

Pure numpy, so all of it runs headlessly. Two properties carry the weight:
a waveform arriving must not touch the ones already retained (the ring), and
what gets averaged must not be limited by what gets drawn (the accumulator).
Both are the difference between a widget that works for one waveform a second
and one that works for hundreds.
"""

import numpy as np
import pytest

from phosphor.trace_grid_buffer import TraceGridBuffer


def make_buffer(n_channels=2, n_samples=4, history=3, **kwargs) -> TraceGridBuffer:
    return TraceGridBuffer(n_channels, n_samples, history, **kwargs)


def wave(value: float, n_channels=2, n_samples=4) -> np.ndarray:
    """A waveform whose every sample is ``value``, so a slot's identity is
    readable straight off its contents."""
    return np.full((n_channels, n_samples), value, dtype=np.float32)


# ---- the ring ---------------------------------------------------------------


def test_a_new_waveform_writes_one_slot_and_leaves_the_rest():
    """The whole point of the ring: arrivals cost one slot, not a copy of
    everything retained."""
    buf = make_buffer(history=3)
    buf.push(wave(1.0))
    before = buf.traces.copy()

    buf.push(wave(2.0))

    # equal_nan, or the never-written slots read as "changed": NaN != NaN.
    changed = [i for i in range(3) if not np.array_equal(buf.traces[i], before[i], equal_nan=True)]
    assert changed == [1], "exactly one slot should differ"


def test_ages_identify_the_newest_without_reordering():
    buf = make_buffer(history=3)
    for v in (1.0, 2.0, 3.0):
        buf.push(wave(v))

    ages = buf.ages
    newest_slot = int(np.flatnonzero(ages == 0)[0])
    assert buf.traces[newest_slot][0, 0] == pytest.approx(3.0)
    oldest_slot = int(np.flatnonzero(ages == 2)[0])
    assert buf.traces[oldest_slot][0, 0] == pytest.approx(1.0)


def test_an_age_at_or_above_the_retained_count_marks_an_empty_slot():
    """The invariant a renderer uses to skip slots that were never written,
    without needing a separate mask."""
    buf = make_buffer(history=4)
    buf.push(wave(1.0))
    buf.push(wave(2.0))

    filled = buf.ages < buf.n_retained
    assert filled.sum() == 2
    for slot in np.flatnonzero(~filled):
        assert np.all(np.isnan(buf.traces[slot])), "an unfilled slot must stay NaN"


def test_the_ring_wraps_and_keeps_the_newest():
    buf = make_buffer(history=3)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        buf.push(wave(v))

    assert buf.n_retained == 3
    retained = {float(buf.traces[i][0, 0]) for i in range(3)}
    assert retained == {3.0, 4.0, 5.0}
    np.testing.assert_allclose(buf.recent(1)[0], wave(5.0))


def test_recent_returns_newest_first():
    buf = make_buffer(history=3)
    for v in (1.0, 2.0, 3.0):
        buf.push(wave(v))
    np.testing.assert_allclose([r[0, 0] for r in buf.recent()], [3.0, 2.0, 1.0])


def test_recent_of_an_empty_buffer_is_empty_not_an_error():
    assert make_buffer().recent().shape == (0, 2, 4)


def test_a_wrongly_shaped_waveform_is_refused():
    """Retaining a mis-shaped trace would silently misalign every channel."""
    buf = make_buffer(n_channels=2, n_samples=4)
    with pytest.raises(ValueError, match=r"expects \(2, 4\)"):
        buf.push(np.zeros((4, 2), dtype=np.float32))


# ---- history changes --------------------------------------------------------


def test_growing_the_history_keeps_what_was_on_screen():
    buf = make_buffer(history=2)
    buf.push(wave(1.0))
    buf.push(wave(2.0))

    buf.set_history(5)

    assert buf.n_retained == 2
    np.testing.assert_allclose([r[0, 0] for r in buf.recent()], [2.0, 1.0])


def test_shrinking_the_history_drops_the_oldest():
    buf = make_buffer(history=4)
    for v in (1.0, 2.0, 3.0, 4.0):
        buf.push(wave(v))

    buf.set_history(2)

    assert buf.n_retained == 2
    np.testing.assert_allclose([r[0, 0] for r in buf.recent()], [4.0, 3.0])


def test_a_resized_ring_keeps_accepting_in_order():
    """The cursor has to land past the newest, or the next push overwrites it."""
    buf = make_buffer(history=4)
    for v in (1.0, 2.0, 3.0):
        buf.push(wave(v))
    buf.set_history(3)
    buf.push(wave(4.0))

    np.testing.assert_allclose([r[0, 0] for r in buf.recent()], [4.0, 3.0, 2.0])


def test_resizing_the_history_does_not_disturb_the_statistics():
    """How many are drawn is not how many are averaged."""
    buf = make_buffer(history=2)
    for v in (1.0, 2.0, 3.0, 4.0):
        buf.push(wave(v))
    mean_before, _ = buf.statistics()

    buf.set_history(8)

    mean_after, _ = buf.statistics()
    np.testing.assert_allclose(mean_after, mean_before)
    assert buf.n_seen == 4


def test_changing_the_history_bumps_the_version_but_a_push_does_not():
    """A renderer rebuilds its graphics on a shape change and writes in place
    otherwise; conflating the two is what makes a plot blank on every arrival."""
    buf = make_buffer(history=2)
    version, updates = buf.version, buf.updates

    buf.push(wave(1.0))
    assert buf.version == version, "a new waveform must not force a rebuild"
    assert buf.updates > updates

    buf.set_history(4)
    assert buf.version > version


# ---- running statistics -----------------------------------------------------


def test_the_mean_spans_every_waveform_not_just_the_retained_ones():
    """The reason the accumulator exists: an evoked response averages hundreds
    of sweeps while only a few are worth overlaying."""
    buf = make_buffer(history=2)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        buf.push(wave(v))

    mean, _ = buf.statistics()
    assert buf.n_retained == 2
    np.testing.assert_allclose(mean, wave(3.0))  # mean of 1..5, not of 4..5


def test_the_standard_deviation_is_the_population_spread():
    buf = make_buffer(history=10)
    for v in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
        buf.push(wave(v))

    _, std = buf.statistics()
    np.testing.assert_allclose(std, wave(2.0), rtol=1e-5)


def test_identical_waveforms_give_a_zero_spread_not_a_nan():
    """Var = E[x^2] - E[x]^2 cancels to a small negative here, and an unclamped
    square root would take the band off screen."""
    buf = make_buffer(history=4)
    for _ in range(4):
        buf.push(wave(1000.0))

    _, std = buf.statistics()
    np.testing.assert_allclose(std, 0.0, atol=1e-3)


def test_a_single_waveform_has_a_mean_but_no_spread():
    buf = make_buffer()
    buf.push(wave(3.0))
    mean, std = buf.statistics()
    np.testing.assert_allclose(mean, wave(3.0))
    assert np.all(np.isnan(std)), "one sample has no spread to report"


def test_channels_that_contributed_nothing_stay_nan():
    """A channel with no data is not a flat line at zero; drawing it as one
    invents a signal that was never recorded."""
    buf = make_buffer(n_channels=2, history=4)
    for v in (1.0, 3.0):
        trace = wave(v)
        trace[1] = np.nan  # channel 1 never contributes
        buf.push(trace)

    mean, std = buf.statistics()
    np.testing.assert_allclose(mean[0], 2.0)
    assert np.all(np.isnan(mean[1]))
    assert np.all(np.isnan(std[1]))


def test_a_missing_channel_does_not_drag_the_others_mean_down():
    """NaN counted as zero is the easy bug, and it looks plausible."""
    buf = make_buffer(n_channels=1, history=4)
    for v in (10.0, np.nan, 20.0):
        buf.push(wave(v, n_channels=1))

    mean, _ = buf.statistics()
    np.testing.assert_allclose(mean, 15.0)  # not 10.0, which averaging in a 0 would give


def test_statistics_are_absent_before_anything_arrives():
    assert make_buffer().statistics() is None


def test_statistics_can_be_switched_off():
    """A stack of action potentials from possibly-different units has no
    meaningful average, so nothing should be paid to compute one."""
    buf = make_buffer(track_statistics=False)
    buf.push(wave(1.0))
    assert buf.statistics() is None
    assert buf.n_retained == 1


def test_clearing_forgets_the_average_as_well_as_the_traces():
    """A caller clears at a boundary -- new source, changed conditioning -- and
    a mean carried across it would average two different things."""
    buf = make_buffer(history=3)
    for v in (1.0, 2.0):
        buf.push(wave(v))

    buf.clear()

    assert buf.n_retained == 0 and buf.n_seen == 0
    assert buf.statistics() is None
    assert np.all(np.isnan(buf.traces))

    buf.push(wave(9.0))
    mean, _ = buf.statistics()
    np.testing.assert_allclose(mean, wave(9.0))


def test_the_accumulator_holds_up_over_many_arrivals():
    """float32 sums lose the low bits of later arrivals; the accumulators are
    float64 for exactly this."""
    buf = make_buffer(n_channels=1, n_samples=1, history=2)
    rng = np.random.default_rng(0)
    values = rng.standard_normal(5000).astype(np.float32) * 100.0 + 1000.0
    for v in values:
        buf.push(np.full((1, 1), v, dtype=np.float32))

    mean, std = buf.statistics()
    np.testing.assert_allclose(mean[0, 0], values.mean(), rtol=1e-4)
    np.testing.assert_allclose(std[0, 0], values.std(), rtol=1e-3)
