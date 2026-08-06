"""Tests for the min/max envelope decimation primitive.

Pure-array display decimation shared by the plotting widgets (evoked trace grid
today). These assert the properties the widgets rely on: peak amplitude is
preserved (the reason min/max beats stride for evoked P/N peaks), output shapes
match across leading dims, empty (all-NaN) history slots survive as NaN, the
x-envelope stays monotone, and small waveforms pass through untouched.
"""

from __future__ import annotations

import numpy as np

from phosphor.decimate import (
    minmax_decimate,
    minmax_decimate_x,
    plan_minmax_decimation,
)


def test_preserves_peaks_where_stride_would_clip():
    n = 15000
    sig = np.zeros(n, dtype=np.float32)
    # Sharp P/N peaks on adjacent samples inside one bucket.
    sig[7001] = 120.0
    sig[7002] = -90.0
    plan = plan_minmax_decimation(n, 1500)
    assert plan.active and plan.n_out == 1500

    dec = minmax_decimate(sig, plan)
    assert dec.shape == (1500,)
    # Both extremes retained.
    assert dec.max() == 120.0
    assert dec.min() == -90.0
    # Plain stride lands between the peaks and misses at least one.
    stride = sig[:: plan.bucket_size]
    assert not (stride.max() == 120.0 and stride.min() == -90.0)


def test_multidim_shape_and_x_alignment():
    n = 15000
    plan = plan_minmax_decimation(n, 1500)
    arr = np.random.default_rng(0).standard_normal((11, 128, n)).astype(np.float32)
    dec = minmax_decimate(arr, plan)
    assert dec.shape == (11, 128, plan.n_out)

    x = np.broadcast_to(np.linspace(0.0, 0.5, n, dtype=np.float32), (128, n))
    xd = minmax_decimate_x(x, plan)
    assert xd.shape == (128, plan.n_out)
    # Time axis stays monotone non-decreasing (each bucket center repeated for
    # its min/max pair), so the drawn polyline never doubles back.
    assert np.all(np.diff(xd, axis=-1) >= 0)


def test_all_nan_bucket_stays_nan():
    n = 1200
    plan = plan_minmax_decimation(n, 200)
    assert plan.active
    row = np.full(n, np.nan, dtype=np.float32)
    dec = minmax_decimate(row, plan)
    assert dec.shape == (plan.n_out,)
    assert np.all(np.isnan(dec))


def test_passthrough_when_within_budget():
    n = 800
    plan = plan_minmax_decimation(n, 1500)
    assert not plan.active
    sig = np.arange(n, dtype=np.float32)
    out = minmax_decimate(sig, plan)
    # Untouched (same object, same shape) so full resolution reaches the GPU.
    assert out.shape == (n,)
    assert np.array_equal(out, sig)
    x = np.broadcast_to(sig, (4, n))
    assert np.array_equal(minmax_decimate_x(x, plan), x)


def test_dtype_preserved():
    n = 4000
    plan = plan_minmax_decimation(n, 500)
    sig = np.random.default_rng(1).standard_normal(n).astype(np.float32)
    assert minmax_decimate(sig, plan).dtype == np.float32
