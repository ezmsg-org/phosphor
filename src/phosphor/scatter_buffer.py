"""Thread-safe scalar accumulator for scatter/heatmap visualization."""

import threading

import numpy as np


class ScatterBuffer:
    """Accumulates per-channel scalar values and provides their mean.

    Data pushed between ``consume()`` calls is averaged. When ``vmin``/``vmax``
    are not fixed, they are tracked via EWMA so the color scale adapts slowly.
    """

    def __init__(self, n_channels: int, vmin: float | None = None, vmax: float | None = None):
        self.n_channels = n_channels
        self._lock = threading.Lock()
        self._accum = np.zeros(n_channels, dtype=np.float64)
        self._per_ch_count = np.zeros(n_channels, dtype=np.int64)
        self._current = np.zeros(n_channels, dtype=np.float32)
        self._dirty = False

        # vmin/vmax: fixed values or EWMA-tracked
        self._fixed_vmin = vmin
        self._fixed_vmax = vmax
        self._ew_min: float | None = None
        self._ew_max: float | None = None
        self._alpha = 0.05  # EWMA smoothing (slow adaptation)

    @property
    def vmin(self) -> float:
        if self._fixed_vmin is not None:
            return self._fixed_vmin
        return self._ew_min if self._ew_min is not None else 0.0

    @property
    def vmax(self) -> float:
        if self._fixed_vmax is not None:
            return self._fixed_vmax
        return self._ew_max if self._ew_max is not None else 1.0

    def push_data(self, data: np.ndarray) -> None:
        """Push scalar values. Shape ``(n_channels,)`` or ``(n_samples, n_channels)``."""
        if data.size == 0:
            return
        with self._lock:
            if data.ndim == 1:
                if data.shape[0] != self.n_channels:
                    return
                finite = np.isfinite(data)
                if finite.any():
                    vals = np.where(finite, data.astype(np.float64), 0.0)
                    self._accum += vals
                    self._per_ch_count += finite.astype(np.int64)
                    self._dirty = True
            elif data.ndim == 2:
                if data.shape[1] != self.n_channels:
                    return
                finite = np.isfinite(data)
                if finite.any():
                    vals = np.where(finite, data.astype(np.float64), 0.0)
                    self._accum += vals.sum(axis=0)
                    self._per_ch_count += finite.astype(np.int64).sum(axis=0)
                    self._dirty = True

    def consume(self) -> np.ndarray | None:
        """Return mean of accumulated data since last consume, or None if clean.

        NaN channels (no finite samples since last consume) are preserved as
        NaN in the output.  Updates EWMA vmin/vmax from finite values only.
        """
        with self._lock:
            if not self._dirty:
                return None
            has_data = self._per_ch_count > 0
            if not has_data.any():
                self._accum[:] = 0.0
                self._per_ch_count[:] = 0
                self._dirty = False
                return None

            mean = np.full(self.n_channels, np.nan, dtype=np.float32)
            mean[has_data] = (self._accum[has_data] / self._per_ch_count[has_data]).astype(np.float32)
            self._accum[:] = 0.0
            self._per_ch_count[:] = 0
            self._dirty = False
            self._current[:] = np.nan_to_num(mean, nan=0.0)

            # Update EWMA for auto-ranging (finite values only)
            finite_vals = mean[np.isfinite(mean)]
            if finite_vals.size > 0:
                frame_min = float(finite_vals.min())
                frame_max = float(finite_vals.max())
                a = self._alpha
                if self._ew_min is None:
                    self._ew_min = frame_min
                    self._ew_max = frame_max
                else:
                    self._ew_min = self._ew_min * (1 - a) + frame_min * a
                    self._ew_max = self._ew_max * (1 - a) + frame_max * a

            return mean
