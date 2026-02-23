"""CPU-side buffer for pre-computed spectrum magnitudes (no FFT, no smoothing)."""

import threading

import numpy as np

from .constants import (
    AUTOSCALE_N_SIGMA,
    AUTOSCALE_TIME_CONSTANT,
    CHANNEL_COLORS,
)


class SpectrumBuffer:
    """Thin storage buffer for pre-computed magnitude spectra.

    Duck-types the interface expected by :class:`GPURenderer` and
    :class:`ChannelPlotWidget` so both sweep and spectrum can share the
    same rendering pipeline.
    """

    def __init__(self, n_channels: int, n_bins: int, n_visible: int = 64):
        self.n_channels = n_channels
        self.n_bins = n_bins
        self.n_visible = min(n_visible, n_channels)
        self.channel_offset = 0

        self.autoscale_enabled = True
        self._manual_y_scale: float | None = None

        self._lock = threading.Lock()
        self._version = 0

        self._allocate()

    # ------------------------------------------------------------------
    # Alias so GPURenderer sees n_columns
    # ------------------------------------------------------------------

    @property
    def n_columns(self) -> int:
        return self.n_bins

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _allocate(self) -> None:
        self.display_mins = np.zeros((self.n_bins, self.n_visible), dtype=np.float32)
        self.display_maxs = np.zeros((self.n_bins, self.n_visible), dtype=np.float32)

        self.ew_mean: np.ndarray | None = None
        self.ew_sq_mean: np.ndarray | None = None

        self._dirty_start: int | None = None
        self._dirty_end: int | None = None

        self._version += 1

    # ------------------------------------------------------------------
    # Sweep-cursor stub (pushed offscreen)
    # ------------------------------------------------------------------

    @property
    def sweep_col(self) -> int:
        """Return a value outside the valid column range so the cursor quad
        is clipped offscreen."""
        return self.n_bins * 2

    # ------------------------------------------------------------------
    # Public mutators
    # ------------------------------------------------------------------

    def push_data(self, magnitudes: np.ndarray) -> None:
        """Push pre-computed magnitudes.  Shape ``(n_bins, n_channels)``, float32."""
        if magnitudes.size == 0:
            return

        with self._lock:
            n_ch = magnitudes.shape[1] if magnitudes.ndim > 1 else 1
            if magnitudes.ndim == 1:
                magnitudes = magnitudes[:, np.newaxis]

            # Channel count mismatch
            if n_ch < self.n_channels:
                magnitudes = np.pad(magnitudes, ((0, 0), (0, self.n_channels - n_ch)))
            elif n_ch > self.n_channels:
                magnitudes = magnitudes[:, : self.n_channels]

            # Bin count mismatch
            if magnitudes.shape[0] != self.n_bins:
                magnitudes = magnitudes[: self.n_bins]
                if magnitudes.shape[0] < self.n_bins:
                    magnitudes = np.pad(magnitudes, ((0, self.n_bins - magnitudes.shape[0]), (0, 0)))

            # Visible channel slice
            end_ch = min(self.channel_offset + self.n_visible, self.n_channels)
            vis = magnitudes[:, self.channel_offset : end_ch].astype(np.float32)
            if vis.shape[1] < self.n_visible:
                vis = np.pad(vis, ((0, 0), (0, self.n_visible - vis.shape[1])))

            self.display_mins[:] = vis
            self.display_maxs[:] = vis

            self._update_autoscale(vis)

            # Mark entire buffer dirty
            self._dirty_start = 0
            self._dirty_end = self.n_bins - 1
            self._version += 1

    def set_n_channels(self, n: int) -> None:
        with self._lock:
            if n != self.n_channels:
                self.n_channels = n
                self.n_visible = min(self.n_visible, self.n_channels)
                self.channel_offset = min(self.channel_offset, max(0, self.n_channels - self.n_visible))
                self._allocate()

    def set_channel_offset(self, offset: int) -> None:
        with self._lock:
            offset = max(0, min(offset, self.n_channels - self.n_visible))
            if offset != self.channel_offset:
                self.channel_offset = offset
                self._allocate()

    def set_n_visible(self, n: int) -> None:
        with self._lock:
            n = max(1, min(n, self.n_channels))
            if n != self.n_visible:
                self.n_visible = n
                self.channel_offset = min(self.channel_offset, self.n_channels - self.n_visible)
                self._allocate()

    def set_n_bins(self, n_bins: int) -> None:
        with self._lock:
            n_bins = max(1, n_bins)
            if n_bins != self.n_bins:
                self.n_bins = n_bins
                self._allocate()

    def adjust_y_scale(self, factor: float) -> None:
        with self._lock:
            self._manual_y_scale = self.y_scale * factor
            self.autoscale_enabled = False

    def toggle_autoscale(self) -> None:
        with self._lock:
            self.autoscale_enabled = not self.autoscale_enabled

    # ------------------------------------------------------------------
    # Properties and GPU data
    # ------------------------------------------------------------------

    @property
    def y_scale(self) -> float:
        if not self.autoscale_enabled and self._manual_y_scale is not None:
            return self._manual_y_scale
        return self._compute_y_scale()

    @property
    def version(self) -> int:
        return self._version

    def get_gpu_data(self) -> np.ndarray:
        with self._lock:
            gpu = np.empty((self.n_bins, self.n_visible, 2), dtype=np.float32)
            gpu[:, :, 0] = self.display_mins
            gpu[:, :, 1] = self.display_maxs
            self._dirty_start = None
            self._dirty_end = None
            return gpu.reshape(-1)

    def get_dirty_gpu_data(self) -> tuple[np.ndarray, int, int] | None:
        with self._lock:
            if self._dirty_start is None:
                return None

            start = self._dirty_start
            end = self._dirty_end
            self._dirty_start = None
            self._dirty_end = None

            if end >= start:
                n_cols = end - start + 1
                gpu = np.empty((n_cols, self.n_visible, 2), dtype=np.float32)
                gpu[:, :, 0] = self.display_mins[start : end + 1]
                gpu[:, :, 1] = self.display_maxs[start : end + 1]
                return gpu.reshape(-1), start, n_cols
            else:
                gpu = np.empty((self.n_bins, self.n_visible, 2), dtype=np.float32)
                gpu[:, :, 0] = self.display_mins
                gpu[:, :, 1] = self.display_maxs
                return gpu.reshape(-1), 0, self.n_bins

    def get_channel_params(self) -> np.ndarray:
        params = np.zeros((self.n_visible, 8), dtype=np.float32)
        ys = self.y_scale
        for i in range(self.n_visible):
            y_center = 1.0 - (2.0 * (i + 0.5)) / self.n_visible
            mean_off = float(self.ew_mean[i] * ys) if self.ew_mean is not None else 0.0
            params[i, 0] = y_center - mean_off
            c = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            params[i, 4] = c[0]
            params[i, 5] = c[1]
            params[i, 6] = c[2]
            params[i, 7] = c[3]
        return params.reshape(-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_autoscale(self, vis_data: np.ndarray) -> None:
        clean = np.nan_to_num(vis_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        batch_mean = np.mean(clean, axis=0)
        batch_sq_mean = np.mean(clean**2, axis=0)

        # Use a fixed time constant equivalent — treat each push as 1 update
        alpha = 1.0 - np.exp(-1.0 / AUTOSCALE_TIME_CONSTANT)

        if self.ew_mean is None:
            self.ew_mean = batch_mean.copy()
            self.ew_sq_mean = batch_sq_mean.copy()
        else:
            self.ew_mean += alpha * (batch_mean - self.ew_mean)
            self.ew_sq_mean += alpha * (batch_sq_mean - self.ew_sq_mean)

    def _compute_y_scale(self) -> float:
        if self.ew_mean is None:
            return 1.0 / max(self.n_visible, 1)
        var = self.ew_sq_mean - self.ew_mean**2
        var = np.maximum(var, 0.0)
        sigma = np.sqrt(var)
        mean_sigma = float(np.mean(sigma))
        if mean_sigma < 1e-12:
            return 1.0 / max(self.n_visible, 1)
        return 1.0 / (AUTOSCALE_N_SIGMA * mean_sigma * self.n_visible)
