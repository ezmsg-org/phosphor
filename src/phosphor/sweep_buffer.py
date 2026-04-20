"""CPU-side circular buffer with incremental min/max downsampling."""

import threading
import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from .constants import DEFAULT_MAX_EVENTS


@dataclass
class SweepEvent:
    t_elapsed: float  # absolute elapsed seconds (same clock as data)
    channel: int | None = None  # None = full-height; int = specific channel (0-based)
    label: str = ""
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)


class SweepBuffer:
    def __init__(
        self,
        n_channels: int,
        srate: float,
        display_dur: float,
        n_columns: int,
        n_visible: int,
        max_events: int = DEFAULT_MAX_EVENTS,
        channel_order: str = "top_down",
        amplitude_scale: float = 1.0,
    ):
        self.n_channels = n_channels
        self.srate = srate
        self.display_dur = display_dur
        self._configured_n_columns = n_columns
        self.n_columns = n_columns
        self.n_visible = min(n_visible, n_channels)
        self.channel_offset = 0
        # ``top_down``: channel index 0 at the top of the plot, growing downward.
        # ``bottom_up``: channel 0 at the bottom (legacy / scientific default).
        self.channel_order = channel_order
        # Multiplier applied to displayed sample values *after* the data-driven
        # autoscale. >1 makes waves taller (may clip into adjacent rows);
        # <1 makes them flatter. Channel row positions are not affected.
        self._amplitude_scale = float(amplitude_scale)

        self._lock = threading.Lock()
        self._version = 0

        # Time tracking — _elapsed holds the current time in whatever basis
        # the caller uses (sample-counting by default, or external timestamps).
        self._elapsed: float = 0.0

        # Event storage
        self._events: deque[SweepEvent] = deque(maxlen=max_events)
        self._events_dirty: bool = False

        self._allocate()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _allocate(self):
        """Allocate / reallocate all internal buffers."""
        self._samples_since_alloc = 0
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
        self._events_dirty = True
        # Per-channel midpoint cache (refreshed on every full rebuild).
        self._ch_mid = np.zeros((self.n_visible, 1), dtype=np.float32)

        self._version += 1

    # ------------------------------------------------------------------
    # Public mutators (called from data-source thread or UI thread)
    # ------------------------------------------------------------------

    def push_data(self, data: np.ndarray, timestamps=None) -> None:
        """Push new samples. data shape: (n_samples, n_channels). Thread-safe.

        *timestamps* sets the time basis for event alignment:

        - ``None`` — elapsed time increments by ``n_samples / srate``.
        - scalar — time of the first sample; elapsed becomes
          ``scalar + n_samples / srate``.
        - iterable of length *n_samples* — per-sample times; elapsed
          becomes the last entry.
        """
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

            # Track elapsed time and write position
            self._samples_since_alloc += n_samples
            if timestamps is None:
                self._elapsed += n_samples / self.srate
            elif np.ndim(timestamps) == 0:
                # Scalar: time of first sample in chunk.
                self._elapsed = float(timestamps) + n_samples / self.srate
            else:
                # Per-sample array: last entry + one sample period so
                # _elapsed is the exclusive end, consistent with the
                # scalar and sample-counting cases.
                self._elapsed = float(timestamps[-1]) + 1.0 / self.srate

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
                self._resize_display_dur(dur)

    def set_srate(self, srate: float) -> None:
        with self._lock:
            if srate != self.srate:
                self.srate = srate
                self._allocate()

    # ------------------------------------------------------------------
    # Properties and MultiLine data (called from render/UI thread)
    # ------------------------------------------------------------------

    @property
    def version(self) -> int:
        return self._version

    def _compute_y_scale(self) -> float:
        """Compute normalization scale from current buffer data.

        Uses the max absolute value across all display columns/channels so
        that normalized data fits within ±0.5, matching MultiLine z_offset_scale separation.
        Must be called while holding ``_lock``.
        """
        max_abs = max(float(np.abs(self.display_mins).max()), float(np.abs(self.display_maxs).max()))
        return 0.5 / max(max_abs, 1e-12)

    def _compute_ch_mid(self, scale: float) -> np.ndarray:
        """Per-channel midpoint (post-scale) of the visible window's range.

        Subtracted before amplitude scaling so a channel zooms around its
        own visual center instead of around y=0 — keeps DC bias from
        translating channels as the user changes ``amplitude_scale``.
        Must be called while holding ``_lock``.
        """
        ch_min = self.display_mins.min(axis=0)
        ch_max = self.display_maxs.max(axis=0)
        return (((ch_min + ch_max) / 2) * scale).astype(np.float32).reshape(-1, 1)

    def _build_multiline_array(self, mins, maxs, col_indices, scale) -> np.ndarray:
        """Build a ``[n_visible, 2*n_cols, 3]`` array from min/max slices.

        Must be called while holding ``_lock``.
        """
        n_cols = mins.shape[0]
        out = np.zeros((self.n_visible, 2 * n_cols, 3), dtype=np.float32)
        col_x = col_indices.astype(np.float32) / max(self.n_columns - 1, 1) * self.display_dur
        out[:, 0::2, 0] = col_x[np.newaxis, :]
        out[:, 1::2, 0] = col_x[np.newaxis, :]
        # ``amplitude_scale`` lets the user scale the waveform alone — the
        # channel row positions (Z below) are unaffected, so big-amplitude
        # signals just clip into adjacent rows rather than rescaling the
        # whole canvas. We zoom around each channel's own midpoint so that
        # DC bias does not translate the line down/up as ``amp`` grows.
        amp = self._amplitude_scale
        ch_mid = self._ch_mid  # (n_visible, 1), already in post-_y_scale units
        out[:, 0::2, 1] = (mins.T * scale - ch_mid) * amp + ch_mid
        out[:, 1::2, 1] = (maxs.T * scale - ch_mid) * amp + ch_mid
        if self.channel_order == "top_down":
            # Channel 0 at the highest Z (drawn at the top of the canvas);
            # subsequent channels grow downward.
            z_indices = (self.n_visible - 1) - np.arange(self.n_visible)
        else:
            z_indices = np.arange(self.n_visible)
        out[:, :, 2] = z_indices[:, np.newaxis]
        return out

    # ------------------------------------------------------------------
    # Display-state setters (thread-safe, mark dirty)
    # ------------------------------------------------------------------

    @property
    def amplitude_scale(self) -> float:
        return self._amplitude_scale

    def set_amplitude_scale(self, scale: float) -> None:
        scale = max(float(scale), 1e-6)
        with self._lock:
            if scale == self._amplitude_scale:
                return
            self._amplitude_scale = scale
            # Force a full rebuild on the next animation frame.
            self._dirty_start = 0
            self._dirty_end = self.n_columns - 1

    def set_channel_order(self, order: str) -> None:
        if order not in ("top_down", "bottom_up"):
            raise ValueError(f"channel_order must be 'top_down' or 'bottom_up', got {order!r}")
        with self._lock:
            if order == self.channel_order:
                return
            self.channel_order = order
            self._dirty_start = 0
            self._dirty_end = self.n_columns - 1

    def get_multiline_data(self) -> np.ndarray:
        """Full data shaped ``[n_visible, 2*n_columns, 3]`` for fastplotlib MultiLineGraphic.

        Y-coordinates are normalized so the max absolute value maps to ±0.5,
        and Z-coordinates encode channel index for z_offset_scale separation.
        """
        with self._lock:
            self._y_scale = self._compute_y_scale()
            self._ch_mid = self._compute_ch_mid(self._y_scale)
            out = self._build_multiline_array(
                self.display_mins,
                self.display_maxs,
                np.arange(self.n_columns),
                self._y_scale,
            )
            self._dirty_start = None
            self._dirty_end = None
            return out

    def get_dirty_multiline_range(self) -> tuple[np.ndarray, int, int] | None:
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
                self._ch_mid = self._compute_ch_mid(self._y_scale)
                out = self._build_multiline_array(
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
                out = self._build_multiline_array(
                    self.display_mins[start : end + 1],
                    self.display_maxs[start : end + 1],
                    np.arange(start, end + 1),
                    old_scale,
                )
                return out, start, n_cols
            else:
                # Wrapped — full update
                self._y_scale = new_scale
                self._ch_mid = self._compute_ch_mid(self._y_scale)
                out = self._build_multiline_array(
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

    def _resize_display_dur(self, new_dur: float) -> None:
        """Resize buffer for new display duration, preserving data and write position.

        Instead of resetting ``write_pos`` to 0, the sweep cursor stays at the
        same time-offset and existing samples are remapped by age into the new
        buffer.  Must be called while holding ``_lock``.
        """
        old_total = self.total_raw_samples
        old_write_pos = self.write_pos
        old_raw = self.raw_buffer

        self.display_dur = new_dur
        new_total = max(int(round(self.srate * new_dur)), 1)
        new_n_columns = min(self._configured_n_columns, new_total)

        if new_total == old_total:
            # Raw buffer unchanged; just update x-axis metadata
            self.n_columns = new_n_columns
            self.samples_per_column = new_total / new_n_columns
            self.sweep_col = self._col_for_pos(self.write_pos)
            self._events_dirty = True
            self._version += 1
            return

        # New write position: same total-sample count, different modulus
        new_write_pos = self._samples_since_alloc % new_total

        # Create new (zeroed) buffer
        new_raw = np.zeros((new_total, self.n_visible), dtype=np.float32)

        # Copy data preserving sample ages
        available = min(self._samples_since_alloc, old_total)
        keep = min(available, new_total)

        if keep > 0:
            # Extract `keep` most recent samples from old buffer (oldest first)
            old_start = (old_write_pos - keep) % old_total
            if old_start + keep <= old_total:
                old_data = old_raw[old_start : old_start + keep]
            else:
                first = old_total - old_start
                old_data = np.concatenate([old_raw[old_start:], old_raw[: keep - first]])

            # Insert into new buffer ending at new_write_pos - 1
            new_start = (new_write_pos - keep) % new_total
            if new_start + keep <= new_total:
                new_raw[new_start : new_start + keep] = old_data
            else:
                first = new_total - new_start
                new_raw[new_start:] = old_data[:first]
                new_raw[: keep - first] = old_data[first:]

        # Update state
        self.total_raw_samples = new_total
        self.n_columns = new_n_columns
        self.samples_per_column = new_total / new_n_columns
        self.raw_buffer = new_raw
        self.write_pos = new_write_pos
        self.sweep_col = self._col_for_pos(new_write_pos)

        # Recompute display columns from new raw data
        self.display_mins = np.zeros((new_n_columns, self.n_visible), dtype=np.float32)
        self.display_maxs = np.zeros((new_n_columns, self.n_visible), dtype=np.float32)
        self._recompute_columns(0, new_n_columns)

        self._dirty_start = None
        self._dirty_end = None
        self._events_dirty = True
        self._version += 1

    # ------------------------------------------------------------------
    # Time and events
    # ------------------------------------------------------------------

    @property
    def elapsed_time(self) -> float:
        """Current elapsed time in seconds (thread-safe)."""
        with self._lock:
            return self._elapsed

    def push_events(self, events: list[SweepEvent]) -> None:
        """Add events to the store. Thread-safe."""
        with self._lock:
            self._events.extend(events)
            self._events_dirty = True

    def get_visible_events(self) -> list[tuple[SweepEvent, float]]:
        """Return visible events with their x-positions. Thread-safe.

        Returns list of ``(event, x_position)`` for events within the current
        display window.  The x-position is derived from the sample-based cursor
        position so it stays aligned with the sweep data regardless of the
        timestamp basis.
        """
        with self._lock:
            elapsed = self._elapsed
            dur = self.display_dur
            # cursor_time tracks the sweep cursor in seconds (sample-based),
            # independent of the external timestamp basis.
            cursor_time = self._samples_since_alloc / self.srate
            result = []
            # Iterate newest-first so the render pool (which has a fixed
            # capacity) keeps the most recent events when there are more
            # visible events than pool slots.
            for ev in reversed(self._events):
                age = elapsed - ev.t_elapsed
                if 0 <= age < dur:
                    x_pos = (cursor_time - age) % dur
                    result.append((ev, x_pos))
            self._events_dirty = False
            return result
