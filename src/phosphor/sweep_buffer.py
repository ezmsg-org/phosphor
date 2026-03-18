"""CPU-side circular buffer with incremental min/max downsampling."""

import threading
import warnings

import numpy as np


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

    # ------------------------------------------------------------------
    # Properties and LineStack data (called from render/UI thread)
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    def _compute_y_scale(self) -> float:
        """Compute normalization scale from current buffer data.

        Uses the max absolute value across all display columns/channels so
        that normalized data fits within ±0.5, matching LineStack separation.
        Must be called while holding ``_lock``.
        """
        max_abs = max(float(np.abs(self.display_mins).max()), float(np.abs(self.display_maxs).max()))
        return 0.5 / max(max_abs, 1e-12)

    def _build_linestack_array(self, mins, maxs, col_indices, scale) -> np.ndarray:
        """Build a ``[n_visible, 2*n_cols, 3]`` array from min/max slices.

        Must be called while holding ``_lock``.
        """
        n_cols = mins.shape[0]
        out = np.zeros((self.n_visible, 2 * n_cols, 3), dtype=np.float32)
        col_x = col_indices.astype(np.float32) / max(self.n_columns - 1, 1) * self.display_dur
        out[:, 0::2, 0] = col_x[np.newaxis, :]
        out[:, 1::2, 0] = col_x[np.newaxis, :]
        out[:, 0::2, 1] = mins.T * scale
        out[:, 1::2, 1] = maxs.T * scale
        return out

    def get_linestack_data(self) -> np.ndarray:
        """Full data shaped ``[n_visible, 2*n_columns, 3]`` for fastplotlib LineStack.

        Y-coordinates are normalized so the max absolute value maps to ±0.5,
        matching LineStack separation=1.0 and preventing channel overlap.
        """
        with self._lock:
            self._y_scale = self._compute_y_scale()
            out = self._build_linestack_array(
                self.display_mins,
                self.display_maxs,
                np.arange(self.n_columns),
                self._y_scale,
            )
            self._dirty_start = None
            self._dirty_end = None
            return out

    def get_dirty_linestack_range(self) -> tuple[np.ndarray, int, int] | None:
        """Incremental update for dirty column range.

        Returns ``(data_slice, col_start, n_cols)`` or ``None`` if clean.
        If the scale changed significantly, returns a full-buffer update.
        """
        with self._lock:
            if self._dirty_start is None:
                return None

            # Check if scale drifted significantly
            new_scale = self._compute_y_scale()
            old_scale = getattr(self, "_y_scale", new_scale)
            if old_scale > 0 and abs(new_scale - old_scale) / old_scale > 0.2:
                # Full update with new scale
                self._y_scale = new_scale
                out = self._build_linestack_array(
                    self.display_mins,
                    self.display_maxs,
                    np.arange(self.n_columns),
                    self._y_scale,
                )
                self._dirty_start = None
                self._dirty_end = None
                return out, 0, self.n_columns

            start = self._dirty_start
            end = self._dirty_end
            self._dirty_start = None
            self._dirty_end = None

            if end >= start:
                n_cols = end - start + 1
                out = self._build_linestack_array(
                    self.display_mins[start : end + 1],
                    self.display_maxs[start : end + 1],
                    np.arange(start, end + 1),
                    old_scale,
                )
                return out, start, n_cols
            else:
                # Wrapped — full update
                self._y_scale = new_scale
                out = self._build_linestack_array(
                    self.display_mins,
                    self.display_maxs,
                    np.arange(self.n_columns),
                    self._y_scale,
                )
                return out, 0, self.n_columns

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
