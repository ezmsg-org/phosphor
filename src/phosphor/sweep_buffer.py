"""CPU-side circular buffer with incremental min/max downsampling and autoscale."""

import threading
import warnings

import numpy as np

from .constants import (
    AUTOSCALE_N_SIGMA,
    AUTOSCALE_TIME_CONSTANT,
    CHANNEL_COLORS,
)


class SweepBuffer:
    def __init__(
        self,
        n_channels: int,
        srate: float,
        display_dur: float,
        n_columns: int,
        n_visible: int,
    ):
        self.n_channels = n_channels
        self.srate = srate
        self.display_dur = display_dur
        self._configured_n_columns = n_columns
        self.n_columns = n_columns
        self.n_visible = min(n_visible, n_channels)
        self.channel_offset = 0

        self.autoscale_enabled = True
        self._manual_y_scale: float | None = None

        self._lock = threading.Lock()
        self._version = 0

        self._allocate()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _allocate(self):
        """Allocate / reallocate all internal buffers."""
        self.total_raw_samples = max(int(round(self.srate * self.display_dur)), 1)
        self.n_columns = min(self._configured_n_columns, self.total_raw_samples)
        self.samples_per_column = self.total_raw_samples / self.n_columns

        self.raw_buffer = np.zeros((self.total_raw_samples, self.n_visible), dtype=np.float32)
        self.display_mins = np.zeros((self.n_columns, self.n_visible), dtype=np.float32)
        self.display_maxs = np.zeros((self.n_columns, self.n_visible), dtype=np.float32)

        self.write_pos = 0
        self.sweep_col = 0

        self.ew_mean: np.ndarray | None = None
        self.ew_sq_mean: np.ndarray | None = None

        self._dirty_start: int | None = None
        self._dirty_end: int | None = None

        self._version += 1

    # ------------------------------------------------------------------
    # Public mutators (called from data-source thread or UI thread)
    # ------------------------------------------------------------------

    def push_data(self, data: np.ndarray) -> None:
        """Push new samples. data shape: (n_samples, n_channels). Thread-safe."""
        if data.size == 0:
            return

        with self._lock:
            n_samples = data.shape[0]
            n_ch = data.shape[1] if data.ndim > 1 else 1
            if data.ndim == 1:
                data = data[:, np.newaxis]

            # Handle channel count mismatch
            if n_ch < self.n_channels:
                data = np.pad(data, ((0, 0), (0, self.n_channels - n_ch)))
            elif n_ch > self.n_channels:
                data = data[:, : self.n_channels]

            # Select visible channels
            end_ch = min(self.channel_offset + self.n_visible, self.n_channels)
            vis_data = data[:, self.channel_offset : end_ch].astype(np.float32)
            if vis_data.shape[1] < self.n_visible:
                vis_data = np.pad(vis_data, ((0, 0), (0, self.n_visible - vis_data.shape[1])))

            # Truncate if more data than one full sweep
            if n_samples > self.total_raw_samples:
                vis_data = vis_data[-self.total_raw_samples :]
                n_samples = self.total_raw_samples

            # Update autoscale statistics
            self._update_autoscale(vis_data)

            # Track column range before writing
            first_col = self._col_for_pos(self.write_pos)

            # Write into circular raw buffer
            remaining = n_samples
            src_off = 0
            while remaining > 0:
                space = self.total_raw_samples - self.write_pos
                chunk = min(remaining, space)
                self.raw_buffer[self.write_pos : self.write_pos + chunk] = vis_data[src_off : src_off + chunk]
                self.write_pos = (self.write_pos + chunk) % self.total_raw_samples
                remaining -= chunk
                src_off += chunk

            # Determine last affected column
            last_pos = (self.write_pos - 1) % self.total_raw_samples
            last_col = self._col_for_pos(last_pos)

            # Recompute affected display columns
            if last_col >= first_col:
                self._recompute_columns(first_col, last_col - first_col + 1)
            else:
                self._recompute_columns(first_col, self.n_columns - first_col)
                self._recompute_columns(0, last_col + 1)

            # Update sweep cursor
            self.sweep_col = self._col_for_pos(self.write_pos)

            # Mark dirty
            self._mark_dirty(first_col, last_col)

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

    def set_display_dur(self, dur: float) -> None:
        with self._lock:
            dur = max(0.1, dur)
            if dur != self.display_dur:
                self.display_dur = dur
                self._allocate()

    def set_srate(self, srate: float) -> None:
        with self._lock:
            if srate != self.srate:
                self.srate = srate
                self._allocate()

    def adjust_y_scale(self, factor: float) -> None:
        """Multiply y_scale by factor and disable autoscale."""
        with self._lock:
            self._manual_y_scale = self.y_scale * factor
            self.autoscale_enabled = False

    def toggle_autoscale(self) -> None:
        with self._lock:
            self.autoscale_enabled = not self.autoscale_enabled

    # ------------------------------------------------------------------
    # Properties and GPU data (called from render/UI thread)
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
        """Full upload: interleave min/max into column-major flat array."""
        with self._lock:
            gpu = np.empty((self.n_columns, self.n_visible, 2), dtype=np.float32)
            gpu[:, :, 0] = self.display_mins
            gpu[:, :, 1] = self.display_maxs
            self._dirty_start = None
            self._dirty_end = None
            return gpu.reshape(-1)

    def get_dirty_gpu_data(self) -> tuple[np.ndarray, int, int] | None:
        """Partial upload for dirty column range.

        Returns (flat_data, col_start, n_cols) or None if nothing dirty.
        """
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
                # Wrapped — upload full buffer
                gpu = np.empty((self.n_columns, self.n_visible, 2), dtype=np.float32)
                gpu[:, :, 0] = self.display_mins
                gpu[:, :, 1] = self.display_maxs
                return gpu.reshape(-1), 0, self.n_columns

    def get_channel_params(self) -> np.ndarray:
        """Per-channel y_offset and RGBA color. Shape (n_visible * 8,) float32."""
        params = np.zeros((self.n_visible, 8), dtype=np.float32)
        # ys = self.y_scale
        for i in range(self.n_visible):
            y_center = 1.0 - (2.0 * (i + 0.5)) / self.n_visible
            params[i, 0] = y_center
            c = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
            params[i, 4] = c[0]
            params[i, 5] = c[1]
            params[i, 6] = c[2]
            params[i, 7] = c[3]
        return params.reshape(-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _col_for_pos(self, pos: int) -> int:
        col = int(pos * self.n_columns / self.total_raw_samples)
        return min(col, self.n_columns - 1)

    def _recompute_columns(self, first_col: int, n_cols: int) -> None:
        for i in range(n_cols):
            col = (first_col + i) % self.n_columns
            start = int(col * self.samples_per_column)
            end = int((col + 1) * self.samples_per_column)
            end = min(end, self.total_raw_samples)
            if start < end:
                chunk = self.raw_buffer[start:end]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    mins = np.nanmin(chunk, axis=0)
                    maxs = np.nanmax(chunk, axis=0)
                # Replace NaN results (all-NaN columns) with 0
                self.display_mins[col] = np.nan_to_num(mins, nan=0.0)
                self.display_maxs[col] = np.nan_to_num(maxs, nan=0.0)

    def _update_autoscale(self, new_data: np.ndarray) -> None:
        clean = np.nan_to_num(new_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
        batch_mean = np.mean(clean, axis=0)
        batch_sq_mean = np.mean(clean**2, axis=0)

        dt = new_data.shape[0] / max(self.srate, 1.0)
        alpha = 1.0 - np.exp(-dt / AUTOSCALE_TIME_CONSTANT)

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

    def _mark_dirty(self, first_col: int, last_col: int) -> None:
        if first_col <= last_col:
            if self._dirty_start is None:
                self._dirty_start = first_col
                self._dirty_end = last_col
            elif self._dirty_start <= self._dirty_end:
                self._dirty_start = min(self._dirty_start, first_col)
                self._dirty_end = max(self._dirty_end, last_col)
            else:
                self._dirty_start = 0
                self._dirty_end = self.n_columns - 1
        else:
            # Wrapping range
            if self._dirty_start is None:
                self._dirty_start = first_col
                self._dirty_end = last_col
            else:
                self._dirty_start = 0
                self._dirty_end = self.n_columns - 1
