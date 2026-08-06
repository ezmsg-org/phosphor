"""SweepBuffer's CPU-side reduction, including the envelope input mode.

``SweepBuffer`` is pure numpy and threading -- no canvas, no GPU -- so it can be
exercised headlessly, which is where the reduction logic actually lives.

The property that matters for envelope mode is *equivalence*: pushing a
pre-reduced (min, max) stream must draw what the raw signal would have drawn.
If that does not hold, moving decimation upstream changes what the user sees,
and the whole point was that it should not.
"""

import numpy as np
import pytest

from phosphor.sweep_buffer import SweepBuffer


def make_buffer(**kwargs) -> SweepBuffer:
    defaults = dict(n_channels=4, srate=1000.0, display_dur=1.0, n_columns=10, n_visible=4)
    defaults.update(kwargs)
    return SweepBuffer(**defaults)


def minmax_decimate(raw: np.ndarray, factor: int) -> np.ndarray:
    """(n, ch) -> (n // factor, ch, 2), the shape an upstream decimator emits."""
    n = raw.shape[0] // factor * factor
    buckets = raw[:n].reshape(-1, factor, raw.shape[1])
    return np.stack([buckets.min(axis=1), buckets.max(axis=1)], axis=-1)


# ---- raw mode is unchanged --------------------------------------------------


def test_raw_push_reduces_to_columns():
    buf = make_buffer()
    raw = np.zeros((1000, 4), dtype=np.float32)
    raw[15, 0] = 5.0  # column 0 spans samples 0..99
    raw[150, 1] = -3.0  # column 1 spans 100..199
    buf.push_data(raw)

    assert buf.display_maxs[0, 0] == pytest.approx(5.0)
    assert buf.display_mins[1, 1] == pytest.approx(-3.0)


def test_raw_push_still_accepts_1d():
    buf = make_buffer(n_channels=1, n_visible=1)
    buf.push_data(np.ones(1000, dtype=np.float32))
    assert buf.display_maxs.max() == pytest.approx(1.0)


# ---- envelope mode ----------------------------------------------------------


def test_envelope_matches_raw_for_the_same_signal():
    """The equivalence that justifies decimating upstream at all.

    Note the envelope buffer is configured at the *envelope* rate, not the raw
    one: ``srate`` describes the stream being pushed. Both buffers then span the
    same wall-clock second over the same ten columns, so each column covers the
    same 100 raw samples, and the two reductions must agree exactly.
    """
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((1000, 4)).astype(np.float32)
    factor = 10

    from_raw = make_buffer(srate=1000.0)
    from_raw.push_data(raw)

    from_env = make_buffer(srate=1000.0 / factor, envelope=True)
    from_env.push_data(minmax_decimate(raw, factor))

    np.testing.assert_allclose(from_env.display_mins, from_raw.display_mins)
    np.testing.assert_allclose(from_env.display_maxs, from_raw.display_maxs)


def test_envelope_srate_is_the_bucket_rate():
    """Sizing from the pre-decimation rate is the easy mistake: the ring would
    be `factor` times too long and the sweep would sit mostly empty."""
    buf = make_buffer(srate=100.0, display_dur=1.0, envelope=True)
    assert buf.total_raw_samples == 100
    assert buf.raw_buffer.shape == (100, 4, 2)


def test_envelope_preserves_a_spike_stride_decimation_would_lose():
    """The motivating case, end to end through the buffer."""
    raw = np.zeros((1000, 1), dtype=np.float32)
    raw[37, 0] = 100.0  # missed by raw[::10]

    buf = make_buffer(n_channels=1, n_visible=1, envelope=True)
    buf.push_data(minmax_decimate(raw, 10))

    assert buf.display_maxs.max() == pytest.approx(100.0)


def test_envelope_allocates_a_rank_3_raw_buffer():
    assert make_buffer(envelope=True).raw_buffer.shape == (1000, 4, 2)
    assert make_buffer().raw_buffer.shape == (1000, 4)


def test_envelope_rejects_raw_shaped_data():
    """A silent misread here would plot half the channels at wrong values."""
    buf = make_buffer(envelope=True)
    with pytest.raises(ValueError, match="expects .*n_channels, 2"):
        buf.push_data(np.zeros((100, 4), dtype=np.float32))


def test_raw_mode_rejects_envelope_shaped_data():
    buf = make_buffer()
    with pytest.raises(ValueError, match="Did you mean envelope=True"):
        buf.push_data(np.zeros((100, 4, 2), dtype=np.float32))


def test_envelope_channel_padding_keeps_the_pair_axis():
    """Fewer channels than configured pads the channel axis only."""
    buf = make_buffer(n_channels=4, n_visible=4, envelope=True)
    data = np.ones((100, 2, 2), dtype=np.float32)
    buf.push_data(data)
    # Channels 0-1 carry the pushed value; 2-3 were padded with zeros.
    assert buf.display_maxs[0, 0] == pytest.approx(1.0)
    assert buf.display_maxs[0, 3] == pytest.approx(0.0)


def test_envelope_channel_trimming():
    buf = make_buffer(n_channels=2, n_visible=2, envelope=True)
    buf.push_data(np.ones((100, 5, 2), dtype=np.float32))
    assert buf.display_maxs.shape[1] == 2


def test_envelope_wraps_the_ring_like_raw():
    """More samples than one sweep must wrap, not overflow."""
    buf = make_buffer(envelope=True)
    env = np.zeros((1500, 4, 2), dtype=np.float32)
    env[..., 1] = 2.0
    buf.push_data(env)  # 1.5x the 1000-sample ring
    assert buf.display_maxs.max() == pytest.approx(2.0)


def test_set_envelope_reallocates():
    buf = make_buffer()
    assert buf.raw_buffer.ndim == 2
    buf.set_envelope(True)
    assert buf.raw_buffer.ndim == 3
    assert buf.envelope
    # Idempotent: no reallocation, and the version does not churn.
    version = buf.version
    buf.set_envelope(True)
    assert buf.version == version


def test_envelope_scale_and_midpoint_use_both_bounds():
    """_compute_y_scale/_compute_ch_mid read display_mins/maxs, so an envelope
    must fill both -- a bug filling only one would autoscale to half range."""
    buf = make_buffer(n_channels=1, n_visible=1, envelope=True)
    env = np.zeros((100, 1, 2), dtype=np.float32)
    env[..., 0] = -4.0
    env[..., 1] = 4.0
    buf.push_data(env)

    assert buf.display_mins.min() == pytest.approx(-4.0)
    assert buf.display_maxs.max() == pytest.approx(4.0)
    # ±0.5 normalization over a ±4 range.
    assert buf._compute_y_scale() == pytest.approx(0.125)


# ---- scrolling keeps what it already drew ----------------------------------
#
# Scrolling by one channel used to reallocate every buffer, so 31 of 32 rows
# were thrown away and redrawn from nothing even though their data had not
# changed. The fix is that storage spans every channel and the visible window
# is a view over it.


def channel_ramp(n_samples: int, n_channels: int) -> np.ndarray:
    """Each channel holds its own index, so a row's identity is its value."""
    return np.tile(np.arange(n_channels, dtype=np.float32), (n_samples, 1))


def test_all_channels_are_buffered_not_just_the_visible_ones():
    buf = make_buffer(n_channels=64, n_visible=8)
    assert buf.raw_buffer.shape[1] == 64
    assert buf.display_mins.shape[1] == 64


def test_scrolling_does_not_reallocate_or_rebuild():
    """A rebuild is what blanks the plot; the graphic keeps its shape here."""
    buf = make_buffer(n_channels=64, n_visible=8)
    buf.push_data(channel_ramp(1000, 64))

    version, raw = buf.version, buf.raw_buffer
    buf.set_channel_offset(1)

    assert buf.version == version, "scrolling must not force a graphic rebuild"
    assert buf.raw_buffer is raw, "scrolling must not reallocate the ring"


def test_scrolled_rows_keep_their_data():
    """The 31-of-32 case: everything that was on screen stays on screen.

    Asserted on the reduced columns rather than the multiline's y values,
    because those are normalized to the visible window -- scrolling onto a
    larger channel legitimately rescales every row, which would swamp the
    thing being checked.
    """
    buf = make_buffer(n_channels=64, n_visible=8)
    buf.push_data(channel_ramp(1000, 64))

    before = buf.display_maxs.copy()
    buf.set_channel_offset(1)

    # Storage is untouched by scrolling; the window moved over it.
    np.testing.assert_array_equal(buf.display_maxs, before)
    # The seven rows that were already on screen are the same seven channels.
    np.testing.assert_allclose(buf.display_maxs[:, 1:8].max(axis=0), np.arange(1, 8))
    # And the newly exposed row carries real data, not zeros.
    assert buf.display_maxs[:, 8].max() == pytest.approx(8.0)
    assert buf.get_multiline_data().shape[0] == 8


def test_a_channel_scrolled_into_view_has_history_immediately():
    """The case no amount of copy-on-scroll could fix: the data has to have
    been kept while the channel was off screen."""
    buf = make_buffer(n_channels=64, n_visible=4)
    buf.push_data(channel_ramp(1000, 64))

    buf.set_channel_offset(60)  # jump well past anything ever displayed
    data = buf.get_multiline_data()

    assert data.shape[0] == 4
    # Channels 60..63 were never visible, yet their columns are populated.
    np.testing.assert_allclose(buf.display_maxs[:, 60:64].max(axis=0), [60.0, 61.0, 62.0, 63.0])


def test_changing_visible_count_keeps_history_too():
    """Paging with /2 and x2 has the same problem and the same fix."""
    buf = make_buffer(n_channels=64, n_visible=8)
    buf.push_data(channel_ramp(1000, 64))
    raw = buf.raw_buffer

    buf.set_n_visible(16)

    assert buf.raw_buffer is raw, "resizing the window must not reallocate"
    assert buf.get_multiline_data().shape[0] == 16
    np.testing.assert_allclose(buf.display_maxs[:, :16].max(axis=0), np.arange(16))


def test_normalization_follows_the_visible_window():
    """Amplitude scale must track what is on screen -- an off-screen channel
    ten times larger should not flatten everything the user is looking at."""
    buf = make_buffer(n_channels=8, n_visible=2)
    data = np.zeros((1000, 8), dtype=np.float32)
    data[:, 0:2] = 1.0
    data[:, 7] = 100.0  # far off screen
    buf.push_data(data)

    buf.get_multiline_data()
    assert buf._compute_y_scale() == pytest.approx(0.5)  # keyed to the visible 1.0

    buf.set_channel_offset(6)  # now channel 7 is visible
    assert buf._compute_y_scale() == pytest.approx(0.005)


def test_time_zoom_preserves_the_envelope_pair_axis():
    """_resize_display_dur rebuilt the ring at the wrong rank in envelope mode,
    so zooming time with the envelope on mis-sized the buffer."""
    buf = make_buffer(n_channels=4, n_visible=4, envelope=True)
    env = np.zeros((1000, 4, 2), dtype=np.float32)
    env[..., 0], env[..., 1] = -3.0, 3.0
    buf.push_data(env)

    buf.set_display_dur(0.5)

    assert buf.raw_buffer.ndim == 3
    assert buf.raw_buffer.shape[1:] == (4, 2)
    assert buf.display_maxs.max() == pytest.approx(3.0)
