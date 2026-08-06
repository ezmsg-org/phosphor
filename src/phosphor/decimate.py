"""Display-only min/max (envelope) decimation for waveform plotting.

A pure, stateless companion to the plotting widgets. When a waveform has far
more samples than the screen has pixels to draw it on, uploading every sample
to the GPU is wasteful — a Utah-array evoked grid at 128 ch × 11 history ×
15 000 samples is ~253 MB of vertex data, most of it invisible.

The fix is to draw an **envelope**: split the sample axis into ``n_out // 2``
equal buckets and keep each bucket's *min* and *max*. Unlike plain stride
decimation (``ys[..., ::k]``), which lands between samples and clips the P/N
peaks that *are* the evoked signal, min/max preserves peak amplitude at ~2
points per bucket.

This is the stateless form of what :class:`~phosphor.sweep_buffer.SweepBuffer`
does on a ring (``_recompute_columns`` + ``_build_multiline_array``). The sweep
reduces samples as they stream past; a grid of retained waveforms has the whole
array in hand and reduces it on demand, so it needs the same arithmetic without
the circular-buffer and thread-lock machinery.

Decimate raw amplitudes *before* any affine cell-mapping: min/max is
order-preserving under a monotone map, so the drawn envelope is identical, and
keeping it upstream avoids re-clipping surprises.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["minmax_decimate", "minmax_decimate_x", "plan_minmax_decimation"]


class MinMaxDecimationPlan:
    """Precomputed bucket geometry for repeated min/max decimation.

    Built once from ``(n_samples, max_points)``; reused every frame. ``active``
    is False when the waveform already fits the point budget (no decimation) —
    callers should pass arrays through unchanged in that case.
    """

    __slots__ = ("active", "bucket_size", "n_buckets", "n_out", "n_samples", "trim")

    def __init__(self, n_samples: int, max_points: int) -> None:
        self.n_samples = int(n_samples)
        n_buckets = max(1, int(max_points) // 2)
        bucket_size = self.n_samples // n_buckets if n_buckets else 0
        # Only decimate when there is something to gain: each bucket must
        # collapse >1 raw sample, else the envelope is the original signal.
        if self.n_samples <= int(max_points) or bucket_size <= 1:
            self.active = False
            self.n_buckets = 0
            self.bucket_size = 0
            self.trim = 0
            self.n_out = self.n_samples
            return
        self.active = True
        self.n_buckets = n_buckets
        self.bucket_size = bucket_size
        # Drop the ragged tail so every bucket is full and reshape is exact.
        self.trim = n_buckets * bucket_size
        self.n_out = 2 * n_buckets


def plan_minmax_decimation(n_samples: int, max_points: int) -> MinMaxDecimationPlan:
    """Build a reusable :class:`MinMaxDecimationPlan` for ``n_samples`` waveforms
    drawn with at most ``max_points`` vertices each."""
    return MinMaxDecimationPlan(n_samples, max_points)


def minmax_decimate(values: np.ndarray, plan: MinMaxDecimationPlan) -> np.ndarray:
    """Min/max envelope decimation along the last (sample) axis.

    *values* shape ``(..., n_samples)`` → ``(..., 2 * n_buckets)``: each bucket
    contributes its ``min`` then ``max``, in that fixed order. All-NaN buckets
    (e.g. unfilled history slots) yield NaN, which callers treat as "empty".

    Returns *values* unchanged when ``plan.active`` is False.
    """
    if not plan.active:
        return values
    if values.shape[-1] != plan.n_samples:
        raise ValueError(f"minmax_decimate: last axis {values.shape[-1]} != plan.n_samples {plan.n_samples}")
    lead = values.shape[:-1]
    vb = values[..., : plan.trim].reshape(*lead, plan.n_buckets, plan.bucket_size)
    with warnings.catch_warnings():
        # All-NaN buckets legitimately warn "All-NaN slice"; NaN is the intended
        # result, so silence it rather than mask (matches phosphor's handling).
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanmin(vb, axis=-1)
        hi = np.nanmax(vb, axis=-1)
    dec = np.stack([lo, hi], axis=-1).reshape(*lead, plan.n_out)
    return dec.astype(values.dtype, copy=False)


def minmax_decimate_x(x: np.ndarray, plan: MinMaxDecimationPlan) -> np.ndarray:
    """Decimate a monotone x coordinate to match :func:`minmax_decimate`.

    Each bucket is drawn as a (min, max) pair sharing one x — the bucket's
    center — so the polyline stays monotonic in time. *x* shape
    ``(..., n_samples)`` → ``(..., 2 * n_buckets)`` with each center repeated
    for its pair. Returns *x* unchanged when ``plan.active`` is False.
    """
    if not plan.active:
        return x
    if x.shape[-1] != plan.n_samples:
        raise ValueError(f"minmax_decimate_x: last axis {x.shape[-1]} != plan.n_samples {plan.n_samples}")
    lead = x.shape[:-1]
    xb = x[..., : plan.trim].reshape(*lead, plan.n_buckets, plan.bucket_size)
    centers = xb.mean(axis=-1)
    out = np.repeat(centers, 2, axis=-1)
    return out.astype(x.dtype, copy=False)
