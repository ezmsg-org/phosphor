"""GPU-accelerated real-time spectrum plot widget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import QWidget

from .channel_plot import ChannelPlotWidget
from .constants import CHANNEL_COLORS, DEFAULT_N_VISIBLE
from .spectrum_buffer import SpectrumBuffer
from .x_axis import XAxisWidget

__all__ = ["SpectrumConfig", "SpectrumWidget"]


@dataclass
class SpectrumConfig:
    n_channels: int
    srate: float
    n_bins: int  # fft_size // 2
    n_visible: int = DEFAULT_N_VISIBLE
    channel_labels: list[str] | None = None


class SpectrumWidget(ChannelPlotWidget):
    """Embeddable QWidget that renders a GPU-accelerated multichannel spectrum plot.

    Usage::

        widget = SpectrumWidget(SpectrumConfig(n_channels=128, srate=30000.0, n_bins=512))
        widget.show()
        widget.push_data(magnitudes)  # shape (n_bins, n_channels)
    """

    def __init__(self, config: SpectrumConfig, parent: QWidget | None = None):
        super().__init__(
            n_channels=config.n_channels,
            n_visible=min(config.n_visible, config.n_channels),
            channel_labels=config.channel_labels,
            parent=parent,
        )
        self._config = config
        self._nyquist = config.srate / 2.0
        self._full_n_bins = config.n_bins
        self._display_freq_max = self._nyquist
        self._log_x = False
        self._freq_min = self._nyquist / self._full_n_bins  # frequency resolution

        # Frequency axis (Hz)
        self._freq_axis = XAxisWidget(self._nyquist, unit="Hz", parent=self)
        self.layout().addWidget(self._freq_axis)

        # CPU-side buffer — starts with full bin count
        self.spectrum_buffer = SpectrumBuffer(
            n_channels=config.n_channels,
            n_bins=config.n_bins,
            n_visible=min(config.n_visible, config.n_channels),
        )
        self._buffer = self.spectrum_buffer

        # Create initial graphics
        self._cached_version = -1
        self._multi_line = None
        self._z_offset_scale = 1.0
        self._setup_graphics()

        # Start rendering
        self._init_rendering()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push_data(self, magnitudes: np.ndarray) -> None:
        """Push pre-computed magnitudes.  Shape ``(n_bins, n_channels)``, float32."""
        if self._log_x:
            magnitudes = self._resample_to_log(magnitudes)
        else:
            # Slice to display range
            display_bins = self._linear_display_bins()
            if magnitudes.shape[0] > display_bins:
                magnitudes = magnitudes[:display_bins]
        self.spectrum_buffer.push_data(magnitudes)

    def update_config(self, config: SpectrumConfig) -> None:
        """Update configuration at runtime."""
        self._config = config
        self._channel_labels = config.channel_labels
        self._nyquist = config.srate / 2.0
        self._full_n_bins = config.n_bins
        self._freq_min = self._nyquist / self._full_n_bins
        self._display_freq_max = min(self._display_freq_max, self._nyquist)

        buf = self.spectrum_buffer
        if config.n_channels != buf.n_channels:
            buf.set_n_channels(config.n_channels)

        new_vis = min(config.n_visible, config.n_channels)
        if new_vis != buf.n_visible:
            buf.set_n_visible(new_vis)

        self._sync_display()

    # ------------------------------------------------------------------
    # Graphics setup
    # ------------------------------------------------------------------

    def _setup_graphics(self) -> None:
        """Create or recreate MultiLineGraphic."""
        subplot = self._subplot

        if self._multi_line is not None:
            subplot.delete_graphic(self._multi_line)
            self._multi_line = None

        buf = self.spectrum_buffer
        data = buf.get_multiline_data(self._display_freq_max)

        n_vis = buf.n_visible
        colors = [CHANNEL_COLORS[i % len(CHANNEL_COLORS)][:3] for i in range(n_vis)]

        self._multi_line = subplot.add_multi_line(
            data,
            colors=colors,
            z_offset_scale=self._z_offset_scale,
            thickness=1.5,
        )

        self._cached_version = buf.version

    # ------------------------------------------------------------------
    # Rendering (animation callback)
    # ------------------------------------------------------------------

    def _update_graphics(self) -> None:
        buf = self.spectrum_buffer

        if buf.version != self._cached_version:
            self._setup_graphics()
            return

        # Incremental update
        result = buf.get_dirty_multiline_range(self._display_freq_max)
        if result is not None:
            data_slice, bin_start, n_bins = result
            idx_start = bin_start * 2
            idx_end = (bin_start + n_bins) * 2
            self._multi_line.data[:, idx_start:idx_end] = data_slice

    # ------------------------------------------------------------------
    # Keyboard controls
    # ------------------------------------------------------------------

    def _on_key_down(self, key: str) -> None:
        if key == ",":
            self._freq_zoom(0.5)
        elif key == ".":
            self._freq_zoom(2.0)
        elif key in ("l", "L"):
            self._log_x = not self._log_x
            self._sync_display()
        else:
            super()._on_key_down(key)

    def _on_ctrl_scroll(self, delta: float) -> None:
        factor = 0.5 if delta > 0 else 2.0
        self._freq_zoom(factor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _freq_zoom(self, factor: float) -> None:
        new_max = self._display_freq_max * factor
        new_max = max(new_max, self._freq_min * 4)
        new_max = min(new_max, self._nyquist)
        if new_max != self._display_freq_max:
            self._display_freq_max = new_max
            self._sync_display()

    def _linear_display_bins(self) -> int:
        """Number of linear bins covering 0 .. _display_freq_max."""
        frac = self._display_freq_max / self._nyquist
        return max(1, int(round(self._full_n_bins * frac)))

    def _sync_display(self) -> None:
        """Synchronize buffer size and axis after a range or mode change."""
        if self._log_x:
            n_bins = self._full_n_bins
            self._freq_axis.set_log(True, self._freq_min)
        else:
            n_bins = self._linear_display_bins()
            self._freq_axis.set_log(False)

        self.spectrum_buffer.set_n_bins(n_bins)
        self._freq_axis.set_range(self._display_freq_max)
        self._update_range_label()

    def _resample_to_log(self, magnitudes: np.ndarray) -> np.ndarray:
        """Resample linearly-spaced bins into log-spaced positions."""
        # Source: linear bins covering freq_min .. nyquist
        src_n = magnitudes.shape[0]
        src_freqs = np.linspace(self._freq_min, self._nyquist, src_n)

        # Target: log-spaced bins covering freq_min .. display_freq_max
        dst_n = self._full_n_bins
        dst_freqs = np.geomspace(self._freq_min, self._display_freq_max, dst_n)

        # Fractional indices into the source array
        idx = np.interp(dst_freqs, src_freqs, np.arange(src_n))
        lo = np.floor(idx).astype(int)
        hi = np.minimum(lo + 1, src_n - 1)
        frac = (idx - lo).astype(np.float32)

        # Vectorized interpolation across all channels
        result = magnitudes[lo] * (1.0 - frac)[:, None] + magnitudes[hi] * frac[:, None]
        return result
