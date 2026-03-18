"""CPU-side buffer for pre-computed spectrum magnitudes (no FFT, no smoothing)."""

import threading

import numpy as np


class SpectrumBuffer:
    """Thin storage buffer for pre-computed magnitude spectra.

    Duck-types the interface expected by :class:`ChannelPlotWidget` so both
    sweep and spectrum widgets can share the same base class.
    """

    def __init__(self, n_channels: int, n_bins: int, n_visible: int = 64):
        self.n_channels = n_channels
        self.n_bins = n_bins
        self.n_visible = min(n_visible, n_channels)
        self.channel_offset = 0

        self._lock = threading.Lock()
        self._version = 0

        self._allocate()

    # ------------------------------------------------------------------
    # Alias so widgets can use n_columns generically
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

        self._dirty_start: int | None = None
        self._dirty_end: int | None = None

        self._version += 1

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

    # ------------------------------------------------------------------
    # Properties and MultiLine data
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    def _compute_y_scale(self) -> float:
        """Compute normalization scale. Must be called while holding ``_lock``."""
        max_abs = max(float(np.abs(self.display_mins).max()), float(np.abs(self.display_maxs).max()))
        return 0.5 / max(max_abs, 1e-12)

    def _build_multiline_array(self, mins, maxs, bin_indices, freq_max, scale) -> np.ndarray:
        """Build a ``[n_visible, 2*n_bins, 3]`` array. Must hold ``_lock``."""
        n_bins = mins.shape[0]
        out = np.zeros((self.n_visible, 2 * n_bins, 3), dtype=np.float32)
        bin_x = bin_indices.astype(np.float32) / max(self.n_bins - 1, 1) * freq_max
        out[:, 0::2, 0] = bin_x[np.newaxis, :]
        out[:, 1::2, 0] = bin_x[np.newaxis, :]
        out[:, 0::2, 1] = mins.T * scale
        out[:, 1::2, 1] = maxs.T * scale
        out[:, :, 2] = np.arange(self.n_visible)[:, np.newaxis]
        return out

    def get_multiline_data(self, freq_max: float) -> np.ndarray:
        """Full data shaped ``[n_visible, 2*n_bins, 3]`` for fastplotlib MultiLineGraphic.

        Y-coordinates are normalized so the max absolute value maps to ±0.5.
        Z-coordinates encode channel index for z_offset_scale separation.
        """
        with self._lock:
            self._y_scale = self._compute_y_scale()
            out = self._build_multiline_array(
                self.display_mins, self.display_maxs, np.arange(self.n_bins), freq_max, self._y_scale
            )
            self._dirty_start = None
            self._dirty_end = None
            return out

    def get_dirty_multiline_range(self, freq_max: float) -> tuple[np.ndarray, int, int] | None:
        """Incremental update for dirty bin range.

        Returns ``(data_slice, bin_start, n_bins)`` or ``None`` if clean.
        If the scale changed significantly, returns a full-buffer update.
        """
        with self._lock:
            if self._dirty_start is None:
                return None

            new_scale = self._compute_y_scale()
            old_scale = getattr(self, "_y_scale", new_scale)

            if old_scale > 0 and abs(new_scale - old_scale) / old_scale > 0.2:
                self._y_scale = new_scale
                out = self._build_multiline_array(
                    self.display_mins, self.display_maxs, np.arange(self.n_bins), freq_max, self._y_scale
                )
                self._dirty_start = None
                self._dirty_end = None
                return out, 0, self.n_bins

            start = self._dirty_start
            end = self._dirty_end
            self._dirty_start = None
            self._dirty_end = None

            if end >= start:
                n_bins = end - start + 1
                out = self._build_multiline_array(
                    self.display_mins[start : end + 1],
                    self.display_maxs[start : end + 1],
                    np.arange(start, end + 1),
                    freq_max,
                    old_scale,
                )
                return out, start, n_bins
            else:
                self._y_scale = new_scale
                out = self._build_multiline_array(
                    self.display_mins, self.display_maxs, np.arange(self.n_bins), freq_max, self._y_scale
                )
                return out, 0, self.n_bins
